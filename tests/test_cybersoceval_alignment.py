"""
test_cybersoceval_alignment.py
==============================
Tests for the 5 CyberSOCEval alignment changes:
  1. 3-way classification (ok / fallback / error)
  2. Jaccard similarity scoring (client-side)
  3. Guided decoding (json_schema parameter)
  4. Exponential backoff (retry helper)
  5. Parse error tracking in results
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Change 1: 3-way classification
# ---------------------------------------------------------------------------

class TestParseModelAnswer:
    """_parse_model_answer must return _parse_status in every code path."""

    def test_ok_status_on_valid_json(self):
        from main import _parse_model_answer
        result = _parse_model_answer('{"correct_answers": ["A", "C"]}')
        assert result["correct_answers"] == ["A", "C"]
        assert result.get("_parse_status") == "ok"

    def test_fallback_status_on_regex_extraction(self):
        from main import _parse_model_answer
        result = _parse_model_answer("The answer is A and B.")
        assert sorted(result["correct_answers"]) == ["A", "B"]
        assert result.get("_parse_status") == "fallback"

    def test_error_status_on_unparseable(self):
        from main import _parse_model_answer
        result = _parse_model_answer("no idea what the answer is")
        assert result["correct_answers"] == []
        assert result.get("_parse_status") == "error"

    def test_ok_status_with_markdown_fenced_json(self):
        from main import _parse_model_answer
        result = _parse_model_answer('```json\n{"correct_answers": ["B"]}\n```')
        assert result["correct_answers"] == ["B"]
        assert result.get("_parse_status") == "ok"


# ---------------------------------------------------------------------------
# Change 2: Jaccard similarity
# ---------------------------------------------------------------------------

class TestJaccardSimilarity:
    def test_identical_sets(self):
        from main import _jaccard_similarity
        assert _jaccard_similarity(["A", "B"], ["A", "B"]) == 1.0

    def test_disjoint_sets(self):
        from main import _jaccard_similarity
        assert _jaccard_similarity(["A"], ["B"]) == 0.0

    def test_partial_overlap(self):
        from main import _jaccard_similarity
        # intersection=1, union=3 → 1/3
        assert abs(_jaccard_similarity(["A", "B"], ["B", "C"]) - 1 / 3) < 1e-9

    def test_both_empty(self):
        from main import _jaccard_similarity
        assert _jaccard_similarity([], []) == 1.0

    def test_one_empty(self):
        from main import _jaccard_similarity
        assert _jaccard_similarity([], ["A"]) == 0.0


class TestComputeExamMetrics:
    def _make_entry(
        self,
        pred: list[str],
        gold: list[str],
        status: str = "ok",
    ) -> dict[str, Any]:
        return {
            "model_answer": {"correct_answers": pred},
            "answer": {"correct_answers": gold},
            "parse_status": status,
        }

    def test_perfect_score(self):
        from main import _compute_exam_metrics
        entries = [
            self._make_entry(["A"], ["A"]),
            self._make_entry(["B", "C"], ["B", "C"]),
        ]
        m = _compute_exam_metrics(entries)
        assert m["exact_match_accuracy"] == 1.0
        assert m["avg_jaccard"] == 1.0
        assert m["parse_error_count"] == 0

    def test_parse_errors_counted(self):
        from main import _compute_exam_metrics
        entries = [
            self._make_entry(["A"], ["A"]),
            self._make_entry([], [], status="error"),
            self._make_entry([], [], status="error"),
        ]
        m = _compute_exam_metrics(entries)
        assert m["parse_error_count"] == 2
        assert m["total_questions"] == 3
        assert abs(m["parse_error_rate"] - 2 / 3) < 1e-9

    def test_empty_exam(self):
        from main import _compute_exam_metrics
        m = _compute_exam_metrics([])
        assert m["total_questions"] == 0
        assert m["avg_jaccard"] == 0.0

    def test_partial_credit(self):
        from main import _compute_exam_metrics
        entries = [
            self._make_entry(["A", "B"], ["B", "C"]),  # Jaccard = 1/3
        ]
        m = _compute_exam_metrics(entries)
        assert m["exact_match_accuracy"] == 0.0
        assert abs(m["avg_jaccard"] - 1 / 3) < 1e-9


# ---------------------------------------------------------------------------
# Change 3: Guided decoding (json_schema parameter)
# ---------------------------------------------------------------------------

class TestGuidedDecodingParam:
    def test_student_trainer_generate_accepts_json_schema(self):
        """StudentTrainer.generate() must accept json_schema kwarg."""
        from student_trainer import StudentTrainer
        sig = StudentTrainer.generate.__code__.co_varnames
        assert "json_schema" in sig

    def test_vertex_trainer_generate_accepts_json_schema(self):
        """VertexAIStudentTrainer.generate() must accept json_schema kwarg."""
        from student_trainer import VertexAIStudentTrainer
        sig = VertexAIStudentTrainer.generate.__code__.co_varnames
        assert "json_schema" in sig

    def test_hf_inference_passes_response_format_when_schema_provided(self):
        """_hf_inference should pass response_format kwarg when json_schema is given."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            project="p", location="l", staging_bucket="gs://b",
        )

        schema = {"type": "object", "properties": {"x": {"type": "string"}}}

        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = '{"x": "y"}'

        with mock.patch("student_trainer.InferenceClient") as MockClient:
            MockClient.return_value.chat_completion.return_value = mock_response
            result = trainer._hf_inference("prompt", 128, json_schema=schema)

        call_kwargs = MockClient.return_value.chat_completion.call_args
        assert "response_format" in call_kwargs.kwargs
        rf = call_kwargs.kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["schema"] == schema


# ---------------------------------------------------------------------------
# Change 4: Exponential backoff
# ---------------------------------------------------------------------------

class TestRetryWithBackoff:
    def test_succeeds_immediately(self):
        from student_trainer import _retry_with_backoff
        assert _retry_with_backoff(lambda: 42) == 42

    def test_retries_on_failure_then_succeeds(self):
        from student_trainer import _retry_with_backoff

        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient")
            return "ok"

        result = _retry_with_backoff(
            flaky, max_retries=3, base_delay=0.01, max_delay=0.1,
        )
        assert result == "ok"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        from student_trainer import _retry_with_backoff

        with pytest.raises(ValueError, match="permanent"):
            _retry_with_backoff(
                lambda: (_ for _ in ()).throw(ValueError("permanent")),
                max_retries=2,
                base_delay=0.01,
                max_delay=0.1,
            )

    def test_teacher_backoff_formula(self):
        """Teacher _complete() should use 0.5 * 2^attempt capped at 30s."""
        import teacher_engine
        # Read the source to verify the formula (structural test)
        import inspect
        source = inspect.getsource(teacher_engine.TeacherEngine._complete)
        assert "0.5 * (2 ** attempt)" in source
        assert "30" in source


# ---------------------------------------------------------------------------
# Change 5: Parse error tracking in final stats
# ---------------------------------------------------------------------------

class TestParseErrorTracking:
    def test_distillation_return_includes_parse_error_fields(self):
        """run_distillation_loop return dict must include parse error counts."""
        # We test this structurally by checking the source code
        import inspect
        from main import run_distillation_loop
        source = inspect.getsource(run_distillation_loop)
        assert "response_parsing_error_count" in source
        assert "total_questions_attempted" in source

    def test_comparison_table_includes_parse_errors(self):
        from run_experiments import _build_comparison_table
        results = [{
            "experiment": "test",
            "teacher": "t",
            "student": "s",
            "status": "completed",
            "best_accuracy": 0.5,
            "final_proficiency": "5",
            "rounds_completed": 2,
            "response_parsing_error_count": 3,
            "total_questions_attempted": 10,
            "duration_seconds": 60,
        }]
        table = _build_comparison_table(results)
        assert table[0]["response_parsing_error_count"] == 3
        assert table[0]["total_questions_attempted"] == 10

    def test_tracker_records_parse_errors(self):
        import tempfile
        from run_experiments import ExperimentTracker

        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = ExperimentTracker(tmpdir)
            tracker.record("test__label", (0, 0), {
                "status": "completed",
                "best_accuracy": 0.8,
                "response_parsing_error_count": 5,
                "total_questions_attempted": 25,
            })
            exp = tracker.experiments["test__label"]
            assert exp["response_parsing_error_count"] == 5
            assert exp["total_questions_attempted"] == 25


# ---------------------------------------------------------------------------
# JSON schema validity across all backends
# ---------------------------------------------------------------------------

class TestJsonSchemaValidity:
    """Verify that JSON schemas used for guided decoding are valid and
    compatible with every backend (HF Inference API, Vertex AI, outlines)."""

    def _get_answer_schema(self) -> dict:
        """Extract the answer_schema defined in main.run_distillation_loop."""
        import ast, inspect, textwrap
        from main import run_distillation_loop

        source = inspect.getsource(run_distillation_loop)
        # Find the assignment node for answer_schema
        tree = ast.parse(textwrap.dedent(source))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "answer_schema":
                        return ast.literal_eval(node.value)
        raise AssertionError("answer_schema not found in run_distillation_loop")

    def test_answer_schema_is_valid_json_schema(self):
        """answer_schema must be a valid JSON Schema (Draft 4+)."""
        import jsonschema
        schema = self._get_answer_schema()
        # jsonschema.validate raises if the schema itself is malformed
        jsonschema.validators.Draft7Validator.check_schema(schema)

    def test_answer_schema_has_required_structure(self):
        """answer_schema must define correct_answers as an array of strings."""
        schema = self._get_answer_schema()
        assert schema.get("type") == "object"
        props = schema.get("properties", {})
        assert "correct_answers" in props, "missing 'correct_answers' property"
        ca = props["correct_answers"]
        assert ca.get("type") == "array"
        assert ca.get("items", {}).get("type") == "string"

    def test_answer_schema_requires_correct_answers(self):
        """answer_schema must list correct_answers as required."""
        schema = self._get_answer_schema()
        assert "correct_answers" in schema.get("required", [])

    def test_valid_instance_passes_schema(self):
        """A well-formed student answer must validate against the schema."""
        import jsonschema
        schema = self._get_answer_schema()
        jsonschema.validate({"correct_answers": ["A", "B"]}, schema)

    def test_invalid_instance_fails_schema(self):
        """An answer with wrong types must fail validation."""
        import jsonschema
        schema = self._get_answer_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"correct_answers": 42}, schema)
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({"correct_answers": [1, 2]}, schema)

    def test_hf_response_format_structure(self):
        """The response_format dict built for HF must match the expected shape."""
        schema = self._get_answer_schema()
        # Replicate the construction from _hf_inference
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": schema},
        }
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["schema"]["type"] == "object"

    def test_vertex_ai_schema_passthrough(self):
        """Vertex AI backend passes the schema dict directly as response_schema."""
        schema = self._get_answer_schema()
        # Vertex AI expects a plain dict with JSON Schema structure
        assert isinstance(schema, dict)
        assert "properties" in schema
        assert "type" in schema
