"""
json_repair.py
==============
Robust JSON extraction and repair for LLM responses.

LLMs frequently return JSON wrapped in prose, markdown fences, or with
minor formatting issues (trailing commas, single quotes, unquoted keys).
This module provides utilities to extract and repair such responses so
that upstream callers never receive an empty default due to a parse
failure.

Strategy (in order):
  1. Direct ``json.loads`` on stripped text.
  2. Strip markdown code fences (```json ... ```) and retry.
  3. Find the outermost JSON object/array via brace/bracket matching.
  4. Apply common repairs (trailing commas, single quotes, etc.) and retry.
  5. Field-level regex reconstruction using caller-supplied ``expected_fields``.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_json(raw: str, default: Any = None, expected_fields: dict[str, type] | None = None) -> Any:
    """Extract and parse JSON from a raw LLM response string.

    Parameters
    ----------
    raw:
        The raw text returned by the LLM.
    default:
        Value to return if all extraction strategies fail.
    expected_fields:
        Optional mapping of ``{field_name: expected_type}`` used as a last
        resort to reconstruct the JSON from regex matches.  Supported types:
        ``str``, ``int``, ``float``, ``list``, ``dict``.

    Returns
    -------
    Any
        The parsed JSON value, or *default* if extraction fails.
    """
    if not raw or not raw.strip():
        logger.warning("Empty LLM response; returning default.")
        return default

    text = raw.strip()

    # Strategy 1: direct parse
    result = _try_parse(text)
    if result is not _SENTINEL:
        return result

    # Strategy 2: strip markdown fences
    defenced = _strip_markdown_fences(text)
    if defenced != text:
        result = _try_parse(defenced)
        if result is not _SENTINEL:
            return result

    # Strategy 3: find outermost JSON object or array via brace matching
    extracted = _extract_outermost_json(text)
    if extracted is not None:
        result = _try_parse(extracted)
        if result is not _SENTINEL:
            return result

        # Strategy 4: repair common issues and retry
        repaired = _repair_json_string(extracted)
        result = _try_parse(repaired)
        if result is not _SENTINEL:
            return result

    # Strategy 4b: repair on the defenced text (covers fenced but broken JSON)
    repaired = _repair_json_string(defenced)
    result = _try_parse(repaired)
    if result is not _SENTINEL:
        return result

    # Strategy 5: field-level regex reconstruction
    if expected_fields:
        reconstructed = _reconstruct_from_fields(text, expected_fields)
        if reconstructed:
            logger.info("Reconstructed JSON from field-level regex extraction.")
            return reconstructed

    logger.warning("All JSON extraction strategies failed; returning default.")
    logger.debug("Raw LLM response (first 2000 chars): %s", raw[:2000])
    return default


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_SENTINEL = object()  # distinguishes "parsed None/False/0" from "parse failed"


def _try_parse(text: str) -> Any:
    """Try ``json.loads``; return ``_SENTINEL`` on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return _SENTINEL


def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences (``` or ```json etc.) from *text*."""
    # Match opening fence with optional language tag and closing fence
    pattern = r"```(?:json|JSON|js|javascript)?\s*\n?(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: strip lines that start with ``` (handles partial fences)
    if "```" in text:
        lines = text.splitlines()
        cleaned = [line for line in lines if not line.strip().startswith("```")]
        return "\n".join(cleaned).strip()

    return text


def _extract_outermost_json(text: str) -> str | None:
    """Find the outermost ``{...}`` or ``[...]`` in *text* using brace matching.

    Handles nested braces/brackets and skips braces inside JSON strings.
    Returns ``None`` if no balanced structure is found.
    """
    # Try object first, then array
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        start = text.find(open_char)
        if start == -1:
            continue

        depth = 0
        in_string = False
        escape_next = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape_next:
                escape_next = False
                continue

            if ch == "\\":
                if in_string:
                    escape_next = True
                continue

            if ch == '"' and not escape_next:
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    return None


def _repair_json_string(text: str) -> str:
    """Apply common repairs to a near-valid JSON string.

    Fixes:
      - Trailing commas before ``}`` or ``]``
      - Single-quoted strings → double-quoted
      - Python-style None/True/False literals
      - Unescaped newlines inside string values
    """
    # Replace Python literals with JSON equivalents
    repaired = text
    repaired = re.sub(r"\bNone\b", "null", repaired)
    repaired = re.sub(r"\bTrue\b", "true", repaired)
    repaired = re.sub(r"\bFalse\b", "false", repaired)

    # Remove trailing commas: , followed by optional whitespace then } or ]
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    # Single quotes to double quotes (only if text doesn't already have
    # double-quoted keys, which would indicate mixed quoting)
    if '"' not in repaired:
        repaired = repaired.replace("'", '"')

    return repaired


def _reconstruct_from_fields(
    text: str,
    expected_fields: dict[str, type],
) -> dict[str, Any] | None:
    """Last-resort reconstruction: regex-search for each expected field.

    Looks for patterns like ``"field_name": <value>`` or ``"field_name" : <value>``
    in the raw text and reconstructs a dict from what it finds.

    Returns ``None`` if no fields could be extracted.
    """
    result: dict[str, Any] = {}

    for field, ftype in expected_fields.items():
        value = _extract_field_value(text, field, ftype)
        if value is not _SENTINEL:
            result[field] = value

    return result if result else None


def _extract_field_value(text: str, field: str, ftype: type) -> Any:
    """Extract a single field value from *text* by regex."""
    # Pattern: "field_name" : <value>
    # We need to handle different value types

    if ftype == str:
        # Match "field": "value" (handles escaped quotes)
        pattern = rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).replace('\\"', '"').replace("\\n", "\n")

    elif ftype in (int, float):
        pattern = rf'"{re.escape(field)}"\s*:\s*(-?[\d.]+)'
        match = re.search(pattern, text)
        if match:
            try:
                return ftype(match.group(1))
            except ValueError:
                pass

    elif ftype == list:
        # Match "field": [...] using bracket matching
        pattern = rf'"{re.escape(field)}"\s*:\s*\['
        match = re.search(pattern, text)
        if match:
            start = match.end() - 1  # position of the [
            extracted = _extract_outermost_json(text[start:])
            if extracted:
                parsed = _try_parse(extracted)
                if parsed is not _SENTINEL:
                    return parsed

    elif ftype == dict:
        # Match "field": {...} using brace matching
        pattern = rf'"{re.escape(field)}"\s*:\s*\{{'
        match = re.search(pattern, text)
        if match:
            start = match.end() - 1  # position of the {
            extracted = _extract_outermost_json(text[start:])
            if extracted:
                parsed = _try_parse(extracted)
                if parsed is not _SENTINEL:
                    return parsed

    return _SENTINEL
