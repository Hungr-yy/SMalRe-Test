"""
data_filter.py
==============
Implements pre-filtering and truncation logic for large malware sandbox
detonation reports, aligned with the CyberSOCEval truncation specification.

CyberSOCEval's approach (from the PurpleLlama malware_analysis benchmark):
  1. Retain only high-signal fields (17 important keys)
  2. Replace 32+ char hex hash values with "hash" (major token savings)
  3. Truncate signature descriptions to 50 characters
  4. Strip MITRE ATT&CK entries to tactic, technique, attck_id only

This preserves all structurally important data (full process trees, all
signatures, all MITRE entries) while dramatically reducing token count
through hash deduplication and description trimming — no list-item dropping
or hard token caps needed.

Reference: CyberSOCEval (Meta/CrowdStrike), arXiv:2509.20166, Appendix D
Source: PurpleLlama/CybersecurityBenchmarks/benchmark/crwd_meta/malware_analysis.py
"""

from __future__ import annotations

import json as _json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (aligned with CyberSOCEval)
# ---------------------------------------------------------------------------

#: Fields kept when running in ``essential`` mode.
#: Matches the CyberSOCEval ``_truncate_report`` important_keys list.
ESSENTIAL_FIELDS: tuple[str, ...] = (
    "size",
    "type",
    "submit_name",
    "sha256",
    "av_detect",
    "vx_family",
    "threat_score",
    "threat_level",
    "verdict",
    "certificates_validation_message",
    "total_processes",
    "total_signatures",
    "file_metadata",
    "processes",
    "mitre_attcks",
    "network_mode",
    "signatures",
)

#: Extended fields for ``standard`` mode — adds network/file/registry detail.
STANDARD_FIELDS: tuple[str, ...] = ESSENTIAL_FIELDS + (
    "domains",
    "hosts",
    "compromised_hosts",
    "extracted_files",
    "process_list",
    "network_traffic",
    "registry_operations",
    "file_operations",
    "tags",
    "environment_description",
    "type_short",
)

#: Maximum characters for signature description fields.
SIGNATURE_DESCRIPTION_LEN: int = 50


#: MITRE ATT&CK keys to retain (everything else is stripped).
MITRE_KEEP_KEYS: tuple[str, ...] = ("tactic", "technique", "attck_id")

#: Regex matching 32+ character hexadecimal strings (MD5, SHA1, SHA256, etc.).
_HASH_RE = re.compile(r"\b[0-9a-fA-F]{32,}\b")


# ---------------------------------------------------------------------------
# Hash removal (CyberSOCEval approach)
# ---------------------------------------------------------------------------

def remove_hash_values(obj: Any) -> Any:
    """Recursively replace 32+ char hex strings with ``"hash"``.

    This is the single largest token-saving operation.  Hybrid Analysis
    reports contain thousands of hash values (file hashes, process hashes,
    certificate thumbprints) that are redundant for behavioural analysis.
    """
    if isinstance(obj, str):
        return _HASH_RE.sub("hash", obj)
    if isinstance(obj, dict):
        return {k: remove_hash_values(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [remove_hash_values(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# DataFilter
# ---------------------------------------------------------------------------

class DataFilter:
    """Filter and truncate a malware sandbox report dict.

    Implements the CyberSOCEval truncation strategy:
      1. Keep only important fields
      2. Replace hex hash values with ``"hash"``
      3. Truncate signature descriptions to 50 chars
      4. Strip MITRE ATT&CK entries to tactic/technique/attck_id

    Parameters
    ----------
    mode:
        ``"essential"`` – CyberSOCEval-aligned field set (17 keys).
        ``"standard"`` – broader field set with network/file/registry detail.
        ``"none"`` – pass through unchanged.
    """

    def __init__(self, mode: str = "essential") -> None:
        mode = mode.lower()
        if mode not in ("essential", "standard", "none"):
            raise ValueError(f"mode must be 'essential', 'standard', or 'none', got {mode!r}")
        self.mode = mode

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
            A new dict with only the allowed fields, hash values replaced,
            signature descriptions truncated, and MITRE entries condensed.
        """
        if self.mode == "none":
            return dict(report)

        allowed = ESSENTIAL_FIELDS if self.mode == "essential" else STANDARD_FIELDS

        # Step 1: Keep only important fields
        filtered: dict[str, Any] = {}
        for key in allowed:
            value = self._extract(report, key)
            if value is not None:
                filtered[key] = value

        # Step 2: Replace hex hash values with "hash"
        filtered = remove_hash_values(filtered)

        # Step 3: Truncate signature descriptions
        for sig in filtered.get("signatures", []):
            if isinstance(sig, dict) and "description" in sig:
                sig["description"] = sig["description"][:SIGNATURE_DESCRIPTION_LEN]

        # Step 4: Condense MITRE ATT&CK entries
        mitre = filtered.get("mitre_attcks", [])
        if mitre:
            filtered["mitre_attcks"] = [
                {k: entry[k] for k in MITRE_KEEP_KEYS if k in entry}
                for entry in mitre
                if isinstance(entry, dict)
            ]

        original_size = _approx_tokens(report)
        filtered_size = _approx_tokens(filtered)
        if original_size > 0:
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

    @staticmethod
    def _extract(report: dict[str, Any], key: str) -> Any:
        """Return *key* from *report*, searching one level of nesting."""
        if key in report:
            return report[key]
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
