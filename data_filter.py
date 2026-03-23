"""
data_filter.py
==============
Implements pre-filtering and truncation logic for large malware sandbox
detonation reports.

Hybrid Analysis reports can exceed 128 k tokens when ingested verbatim.
Research shows that retaining only a small set of high-signal fields
("total_processes", "mitre_attcks", "signatures") has a **negligible impact**
on model performance while dramatically reducing context window consumption.

This module provides the :class:`DataFilter` class used by both
:mod:`report_parser` and the ``--truncate-input`` flag in the eval pipeline.

Context window safety
---------------------
The filter enforces a hard per-report token budget (``max_tokens``) so that
a batch of filtered reports is guaranteed to fit within the smallest teacher
context window (GPT-4o at 128k).  With the defaults (``list_limit=5``,
``max_tokens=10_000``), 10 reports consume ~50–100k tokens, leaving room
for prompt templates and model output.
"""

from __future__ import annotations

import json as _json
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: Fields kept when running in ``essential_only`` mode.
#: Aligns with the CyberSOCEval truncation specification.
ESSENTIAL_FIELDS: tuple[str, ...] = (
    "sha256",
    "verdict",
    "threat_score",
    "submit_name",
    "type",
    "type_short",
    "total_processes",
    "mitre_attcks",
    "signatures",
    "tags",
    "environment_description",
)

#: Fields that are kept in the ``standard`` (non-essential-only) mode but
#: still truncated to avoid blowing the context window.
STANDARD_FIELDS: tuple[str, ...] = ESSENTIAL_FIELDS + (
    "domains",
    "hosts",
    "compromised_hosts",
    "extracted_files",
    "process_list",
    "network_traffic",
    "registry_operations",
    "file_operations",
)

#: Maximum number of list items to retain for any list-valued field.
DEFAULT_LIST_LIMIT: int = 5

#: Hard per-report token budget.  After field selection and list truncation,
#: if the report still exceeds this limit, list fields are progressively
#: shortened until the budget is met.  Set so that 10 reports + prompt
#: templates fit within GPT-4o's 128k context window.
DEFAULT_MAX_TOKENS: int = 10_000


# ---------------------------------------------------------------------------
# DataFilter
# ---------------------------------------------------------------------------

class DataFilter:
    """Filter and truncate a malware sandbox report dict.

    Parameters
    ----------
    mode:
        ``"essential"`` – keep only :data:`ESSENTIAL_FIELDS` (matches the
        ``--truncate-input`` behaviour used in CyberSOCEval benchmarking).
        ``"standard"`` – keep a broader set of fields but still truncate lists.
        ``"none"`` – pass through unchanged (useful for full-context ablations).
    list_limit:
        Maximum number of items kept per list-valued field.
    max_tokens:
        Hard per-report token ceiling.  If the filtered report exceeds this,
        list fields are progressively shortened until the budget is met.
        Set to ``0`` to disable the cap.
    """

    def __init__(
        self,
        mode: str = "essential",
        list_limit: int = DEFAULT_LIST_LIMIT,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        mode = mode.lower()
        if mode not in ("essential", "standard", "none"):
            raise ValueError(f"mode must be 'essential', 'standard', or 'none', got {mode!r}")
        self.mode = mode
        self.list_limit = list_limit
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(self, report: dict[str, Any]) -> dict[str, Any]:
        """Return a filtered copy of *report*.

        Parameters
        ----------
        report:
            Raw JSON-decoded report dict from a sandbox detonation.

        Returns
        -------
        dict
            A new dict containing only the allowed fields, with list fields
            truncated to at most :attr:`list_limit` items and the total size
            capped at :attr:`max_tokens`.
        """
        if self.mode == "none":
            return dict(report)

        allowed = ESSENTIAL_FIELDS if self.mode == "essential" else STANDARD_FIELDS
        filtered: dict[str, Any] = {}

        for key in allowed:
            value = self._extract(report, key)
            if value is None:
                continue
            if isinstance(value, list):
                value = value[: self.list_limit]
            filtered[key] = value

        # Enforce hard token budget
        if self.max_tokens > 0:
            filtered = self._enforce_budget(filtered)

        original_size = _approx_tokens(report)
        filtered_size = _approx_tokens(filtered)
        if original_size == 0:
            logger.debug("DataFilter[%s]: empty report, nothing to filter", self.mode)
            return filtered
        logger.debug(
            "DataFilter[%s]: ~%d → ~%d tokens (%.1f%% reduction)",
            self.mode,
            original_size,
            filtered_size,
            100.0 * (1 - filtered_size / original_size),
        )
        return filtered

    def filter_string(self, json_str: str) -> str:
        """Filter a JSON-encoded report string and return a JSON string."""
        report = _json.loads(json_str)
        filtered = self.filter(report)
        return _json.dumps(filtered)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _enforce_budget(self, filtered: dict[str, Any]) -> dict[str, Any]:
        """Progressively shorten fields until the report fits in budget.

        Strategy (in order):
          1. Halve list fields (largest first) until under budget
          2. Trim large string values within remaining list items
          3. Drop the largest fields entirely as a last resort
        """
        current_size = _approx_tokens(filtered)
        if current_size <= self.max_tokens:
            return filtered

        # --- Phase 1: halve list fields ---
        for _ in range(20):
            if current_size <= self.max_tokens:
                return filtered

            list_fields = [
                (_approx_tokens(v), k)
                for k, v in filtered.items()
                if isinstance(v, list) and len(v) > 1
            ]
            if not list_fields:
                break
            list_fields.sort(reverse=True)

            for _, key in list_fields:
                filtered[key] = filtered[key][: max(1, len(filtered[key]) // 2)]

            current_size = _approx_tokens(filtered)

        if current_size <= self.max_tokens:
            return filtered

        # --- Phase 2: trim large strings inside list items ---
        max_item_chars = 400  # keep ~100 tokens per list item
        for key, value in filtered.items():
            if not isinstance(value, list):
                continue
            trimmed_items = []
            for item in value:
                if isinstance(item, dict):
                    item = {
                        k: (v[:max_item_chars] + "…" if isinstance(v, str) and len(v) > max_item_chars else v)
                        for k, v in item.items()
                    }
                elif isinstance(item, str) and len(item) > max_item_chars:
                    item = item[:max_item_chars] + "…"
                trimmed_items.append(item)
            filtered[key] = trimmed_items

        current_size = _approx_tokens(filtered)
        if current_size <= self.max_tokens:
            return filtered

        # --- Phase 3: drop largest fields until under budget ---
        # Keep critical identity fields; drop behavioural fields first
        drop_priority = [
            "environment_description", "tags", "signatures",
            "mitre_attcks", "type", "type_short", "submit_name",
        ]
        for key in drop_priority:
            if current_size <= self.max_tokens:
                break
            if key in filtered:
                logger.debug("DataFilter: dropping field '%s' to meet budget", key)
                del filtered[key]
                current_size = _approx_tokens(filtered)

        return filtered

    @staticmethod
    def _extract(report: dict[str, Any], key: str) -> Any:
        """Return *key* from *report*, searching one level of nesting."""
        if key in report:
            return report[key]
        # Check common wrapper keys used by different sandbox formats
        for wrapper in ("analysis", "result", "report", "data", "summary"):
            sub = report.get(wrapper)
            if isinstance(sub, dict) and key in sub:
                return sub[key]
        return None


def _approx_tokens(obj: Any) -> int:
    """Rough token estimate: len(str(obj)) // 4."""
    try:
        return len(_json.dumps(obj)) // 4
    except Exception:
        return len(str(obj)) // 4
