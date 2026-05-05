"""Tests for json_repair.py – robust JSON extraction from LLM responses."""

import pytest

from json_repair import (
    extract_json,
    _strip_markdown_fences,
    _extract_outermost_json,
    _repair_json_string,
    _reconstruct_from_fields,
)


# -----------------------------------------------------------------------
# Strategy 1: Direct parse
# -----------------------------------------------------------------------

class TestDirectParse:
    def test_clean_object(self):
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_clean_array(self):
        assert extract_json('[1, 2, 3]') == [1, 2, 3]

    def test_with_whitespace(self):
        assert extract_json('  \n{"a": 1}\n  ') == {"a": 1}

    def test_empty_string_returns_default(self):
        assert extract_json("", default="fallback") == "fallback"

    def test_none_like_returns_default(self):
        assert extract_json("   ", default={}) == {}


# -----------------------------------------------------------------------
# Strategy 2: Markdown fence stripping
# -----------------------------------------------------------------------

class TestMarkdownFences:
    def test_json_fence(self):
        raw = '```json\n{"key": "value"}\n```'
        assert extract_json(raw) == {"key": "value"}

    def test_plain_fence(self):
        raw = '```\n{"key": "value"}\n```'
        assert extract_json(raw) == {"key": "value"}

    def test_fence_with_extra_text(self):
        raw = 'Here is the result:\n```json\n{"key": "value"}\n```\nDone.'
        assert extract_json(raw) == {"key": "value"}

    def test_strip_markdown_fences_helper(self):
        raw = '```json\n{"a": 1}\n```'
        assert _strip_markdown_fences(raw) == '{"a": 1}'


# -----------------------------------------------------------------------
# Strategy 3: Outermost brace extraction
# -----------------------------------------------------------------------

class TestBraceExtraction:
    def test_json_embedded_in_prose(self):
        raw = 'The answer is: {"question": "what?", "answer": "yes"} and that is all.'
        assert extract_json(raw) == {"question": "what?", "answer": "yes"}

    def test_array_embedded_in_prose(self):
        raw = 'Results: [1, 2, 3] end.'
        assert extract_json(raw) == [1, 2, 3]

    def test_nested_objects(self):
        raw = 'OK here: {"outer": {"inner": [1, 2]}} done'
        result = extract_json(raw)
        assert result == {"outer": {"inner": [1, 2]}}

    def test_braces_inside_strings_are_skipped(self):
        raw = 'Look: {"msg": "use {braces} here"} end'
        result = extract_json(raw)
        assert result == {"msg": "use {braces} here"}

    def test_extract_outermost_json_helper(self):
        text = 'prefix {"a": 1} suffix'
        assert _extract_outermost_json(text) == '{"a": 1}'

    def test_no_json_returns_none(self):
        assert _extract_outermost_json("no json here") is None


# -----------------------------------------------------------------------
# Strategy 4: Repair common issues
# -----------------------------------------------------------------------

class TestRepair:
    def test_trailing_comma_object(self):
        raw = '{"a": 1, "b": 2,}'
        assert extract_json(raw) == {"a": 1, "b": 2}

    def test_trailing_comma_array(self):
        raw = '[1, 2, 3,]'
        assert extract_json(raw) == [1, 2, 3]

    def test_python_none_true_false(self):
        raw = '{"a": None, "b": True, "c": False}'
        assert extract_json(raw) == {"a": None, "b": True, "c": False}

    def test_single_quotes_only(self):
        raw = "{'a': 1, 'b': 2}"
        assert extract_json(raw) == {"a": 1, "b": 2}

    def test_repair_combined(self):
        """Trailing comma + Python literals together."""
        raw = '{"val": True, "items": [1, None,],}'
        result = extract_json(raw)
        assert result == {"val": True, "items": [1, None]}

    def test_repair_helper(self):
        assert '"true"' not in _repair_json_string('{"a": True}')
        assert "true" in _repair_json_string('{"a": True}')


# -----------------------------------------------------------------------
# Strategy 5: Field-level reconstruction
# -----------------------------------------------------------------------

class TestFieldReconstruction:
    def test_extract_string_field(self):
        raw = 'blah blah "feedback": "model is weak at X" blah'
        result = extract_json(raw, default={}, expected_fields={"feedback": str})
        assert result["feedback"] == "model is weak at X"

    def test_extract_int_field(self):
        raw = 'some text "proficiency": 7 more text'
        result = extract_json(raw, default={}, expected_fields={"proficiency": int})
        assert result["proficiency"] == 7

    def test_extract_float_field(self):
        raw = '"score": 0.85 in the test'
        result = extract_json(raw, default={}, expected_fields={"score": float})
        assert result["score"] == 0.85

    def test_extract_list_field(self):
        # Brace extraction finds ["a","b","c"] as valid JSON (a list),
        # so this tests that strategy 3 works on a fragment
        raw = 'here is "strengths": ["a", "b", "c"] done'
        result = extract_json(raw, default={}, expected_fields={"strengths": list})
        assert result == ["a", "b", "c"]

    def test_extract_dict_field(self):
        # Brace extraction finds the dict fragment
        raw = 'the "metrics": {"accuracy": 0.9} is good'
        result = extract_json(raw, default={}, expected_fields={"metrics": dict})
        assert result == {"accuracy": 0.9}

    def test_multiple_fields_reconstruction(self):
        """When no valid JSON object/array exists, field-level regex kicks in."""
        raw = (
            "The evaluation shows:\n"
            '- "proficiency": 5\n'
            '- "feedback": "needs work on MITRE mapping"\n'
            "Overall the student scored poorly.\n"
        )
        result = extract_json(raw, default={}, expected_fields={
            "proficiency": int,
            "feedback": str,
        })
        assert result["proficiency"] == 5
        assert result["feedback"] == "needs work on MITRE mapping"

    def test_reconstruction_with_nested_fields(self):
        """Field-level reconstruction when brace extraction finds a non-useful fragment."""
        # Use _reconstruct_from_fields directly to test the last-resort path
        raw = (
            'I found "proficiency": 8 and "feedback": "great progress"\n'
        )
        result = _reconstruct_from_fields(raw, {
            "proficiency": int,
            "feedback": str,
        })
        assert result["proficiency"] == 8
        assert result["feedback"] == "great progress"

    def test_no_fields_found_returns_default(self):
        raw = "completely unrelated text"
        result = extract_json(raw, default={"fallback": True}, expected_fields={"x": str})
        assert result == {"fallback": True}

    def test_reconstruct_helper_returns_none_if_empty(self):
        assert _reconstruct_from_fields("nothing here", {"x": str}) is None


# -----------------------------------------------------------------------
# Realistic LLM output scenarios
# -----------------------------------------------------------------------

class TestRealisticLLMOutputs:
    def test_claude_prose_wrapped_json(self):
        raw = (
            "Here is my evaluation of the student's performance:\n\n"
            '```json\n'
            '{\n'
            '  "feedback": "The student struggles with MITRE ATT&CK mapping",\n'
            '  "proficiency": 3,\n'
            '  "metrics": {"exact_match_accuracy": 0.2, "avg_jaccard": 0.35},\n'
            '  "strengths": ["Basic malware identification"],\n'
            '  "weaknesses": [{"area_name": "MITRE mapping"}],\n'
            '  "breakdowns": {},\n'
            '  "dataset": [{"input": "test", "output": "answer"}]\n'
            '}\n'
            '```\n\n'
            "I hope this helps with the next training round."
        )
        result = extract_json(raw)
        assert result["proficiency"] == 3
        assert result["metrics"]["exact_match_accuracy"] == 0.2
        assert len(result["dataset"]) == 1

    def test_gpt_clean_json(self):
        raw = '{"question": {"text": "What is the malware family?"}, "answer": {"correct_answers": ["A", "C"]}}'
        result = extract_json(raw)
        assert result["question"]["text"] == "What is the malware family?"
        assert result["answer"]["correct_answers"] == ["A", "C"]

    def test_gemini_prose_then_json(self):
        raw = (
            "Based on my analysis, here is the exam question:\n\n"
            '{"question": {"text": "Identify the C2 domain", "options": {"A": "evil.com", "B": "good.com"}}, '
            '"answer": {"correct_answers": ["A"]}}'
        )
        result = extract_json(raw)
        assert "question" in result
        assert result["answer"]["correct_answers"] == ["A"]

    def test_student_model_messy_output(self):
        raw = "I think the answer is:\ncorrect_answers: A, C, E\n"
        # This won't parse as JSON at all, so with expected_fields=None
        # it returns the default
        result = extract_json(raw, default=None)
        assert result is None

    def test_partially_malformed_curriculum(self):
        """Trailing comma + Python booleans — a common LLM glitch."""
        raw = (
            '```json\n'
            '{\n'
            '  "feedback": "Needs improvement",\n'
            '  "proficiency": 4,\n'
            '  "metrics": {"exact_match_accuracy": 0.4,},\n'
            '  "strengths": [],\n'
            '  "weaknesses": [],\n'
            '  "breakdowns": {},\n'
            '  "dataset": [],\n'
            '}\n'
            '```'
        )
        result = extract_json(raw)
        assert result["proficiency"] == 4
        assert result["metrics"]["exact_match_accuracy"] == 0.4
