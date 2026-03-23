"""
test_fault_tolerance.py
=======================
Tests that the experiment orchestrator is resilient to failures in individual
experiments, preserving progress from successful runs.

Covers:
  - Single experiment failure does not block others in the same batch
  - Failed batches do not prevent subsequent batches from running
  - Completed experiment results are saved to disk
  - Partial round artifacts survive later-round failures
  - Teacher API transient errors are retried
  - Teacher returning garbage JSON is handled gracefully
  - Student CUDA OOM is caught and reported
  - Summary includes both failed and completed experiments
  - Tracker persists to disk and is loaded on rerun
  - Completed experiments are skipped on rerun
  - Failed experiments are automatically retried (max 2 times)
  - Artifact verification demotes experiments with missing checkpoints
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class _InlineExecutor:
    """Drop-in replacement for ProcessPoolExecutor that runs tasks
    in the current thread.  Mocks are not picklable so we cannot
    send them to real subprocesses."""

    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def submit(self, fn, *args, **kwargs):
        from concurrent.futures import Future
        fut: Future = Future()
        try:
            result = fn(*args, **kwargs)
            fut.set_result(result)
        except BaseException as exc:
            fut.set_exception(exc)
        return fut


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ok_result(label: str, output_dir: str = "/tmp/test") -> dict[str, Any]:
    """A mock successful experiment result."""
    return {
        "experiment": label,
        "output_dir": output_dir,
        "status": "completed",
        "teacher": "openai/gpt-4o",
        "student": "meta-llama/Llama-3.3-8B-Instruct",
        "best_accuracy": 0.75,
        "final_proficiency": "7",
        "rounds_completed": 3,
        "round_results": [],
        "duration_seconds": 10.0,
    }


def _fail_result(label: str, error: str = "boom", output_dir: str = "/tmp/test") -> dict[str, Any]:
    """A mock failed experiment result."""
    return {
        "experiment": label,
        "output_dir": output_dir,
        "status": "failed",
        "error": error,
        "duration_seconds": 1.0,
    }


def _make_base_config() -> dict[str, Any]:
    """Minimal base config for tests."""
    return {
        "teacher": {"provider": "openai", "model": "gpt-4o", "temperature": 0.7},
        "student": {"model_name_or_path": "meta-llama/Llama-3.3-8B-Instruct"},
        "training": {"use_4bit": True, "learning_rate": 2e-4, "num_train_epochs": 1,
                      "per_device_train_batch_size": 1, "gradient_accumulation_steps": 1,
                      "max_seq_length": 512},
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05},
        "data_source": {"task_definitions": "prompt_templates/malware_analysis_task.json",
                         "hybrid_analysis_dir": "hybrid-analysis", "reports_per_round": 2},
        "exam": {"num_questions": 2},
        "curriculum": {"num_examples": 5},
    }


# ---------------------------------------------------------------------------
# Test Group 1: Single experiment failure isolation within a batch
# ---------------------------------------------------------------------------

class TestBatchIsolation:
    """One experiment failing must not prevent others in the same batch."""

    def test_one_experiment_fails_others_succeed(self, tmp_path):
        """If 1 of 3 experiments raises, run_batch returns 3 results
        with 2 completed and 1 failed."""
        from run_experiments import run_batch, BATCHES

        call_count = {"n": 0}

        def mock_run_single(experiment_config, rounds, target_accuracy,
                            output_dir, truncate_input, experiment_label):
            call_count["n"] += 1
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            if call_count["n"] == 2:
                raise ConnectionError("Simulated network failure")
            return _ok_result(experiment_label, output_dir)

        with mock.patch("run_experiments._run_single_experiment", side_effect=mock_run_single), \
             mock.patch("run_experiments.ProcessPoolExecutor", _InlineExecutor):
            results = run_batch(
                batch_label="batch_1",
                pairs=BATCHES[0],
                base_config=_make_base_config(),
                rounds=3,
                target_accuracy=0.8,
                root_output_dir=str(tmp_path),
                max_parallel=1,
                truncate_input=True,
            )

        assert len(results) == 3
        statuses = [r["status"] for r in results]
        assert statuses.count("completed") == 2
        assert statuses.count("failed") == 1

    def test_all_experiments_fail_batch_still_returns(self, tmp_path):
        """Even if all 3 experiments fail, run_batch returns 3 failed results."""
        from run_experiments import run_batch, BATCHES

        def mock_run_single(**kwargs):
            raise RuntimeError("Everything is broken")

        with mock.patch("run_experiments._run_single_experiment", side_effect=mock_run_single), \
             mock.patch("run_experiments.ProcessPoolExecutor", _InlineExecutor):
            results = run_batch(
                batch_label="batch_1",
                pairs=BATCHES[0],
                base_config=_make_base_config(),
                rounds=1,
                target_accuracy=0.8,
                root_output_dir=str(tmp_path),
                max_parallel=1,
                truncate_input=True,
            )

        assert len(results) == 3
        assert all(r["status"] == "failed" for r in results)

    def test_batch_results_json_written(self, tmp_path):
        """batch_results.json is written even when some experiments fail."""
        from run_experiments import run_batch, BATCHES

        call_count = {"n": 0}

        def mock_run_single(**kwargs):
            call_count["n"] += 1
            Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
            if call_count["n"] == 1:
                raise ValueError("API cost cap reached")
            return _ok_result(kwargs["experiment_label"], kwargs["output_dir"])

        with mock.patch("run_experiments._run_single_experiment", side_effect=mock_run_single), \
             mock.patch("run_experiments.ProcessPoolExecutor", _InlineExecutor):
            run_batch(
                batch_label="batch_1",
                pairs=BATCHES[0],
                base_config=_make_base_config(),
                rounds=1,
                target_accuracy=0.8,
                root_output_dir=str(tmp_path),
                max_parallel=1,
                truncate_input=True,
            )

        batch_results_path = tmp_path / "batch_1" / "batch_results.json"
        assert batch_results_path.exists()
        data = json.loads(batch_results_path.read_text())
        assert len(data) == 3


# ---------------------------------------------------------------------------
# Test Group 2: Cross-batch isolation
# ---------------------------------------------------------------------------

class TestCrossBatchIsolation:
    """A failed batch must not prevent subsequent batches from running."""

    def test_failed_batch_does_not_block_later_batches(self, tmp_path):
        """All 3 batches run even if batch 1 completely fails."""
        from run_experiments import run_all_experiments, TEACHERS, STUDENTS, BATCHES

        batch_labels_called = []

        def mock_run_batch(batch_label, pairs, **kwargs):
            batch_labels_called.append(batch_label)
            batch_dir = Path(kwargs["root_output_dir"]) / batch_label
            batch_dir.mkdir(parents=True, exist_ok=True)

            results = []
            for ti, si in pairs:
                label = f"{TEACHERS[ti]['name']}__{STUDENTS[si]['name']}"
                exp_dir = str(batch_dir / label)
                Path(exp_dir).mkdir(parents=True, exist_ok=True)
                if batch_label == "batch_1":
                    results.append(_fail_result(label, "error", exp_dir))
                else:
                    # Create fake adapter so verification passes
                    (Path(exp_dir) / "experiment_result.json").write_text("{}")
                    (Path(exp_dir) / "round_1" / "adapter").mkdir(parents=True, exist_ok=True)
                    results.append(_ok_result(label, exp_dir))
            return results

        with mock.patch("run_experiments.run_batch", side_effect=mock_run_batch):
            results = run_all_experiments(
                base_config=_make_base_config(),
                rounds=1,
                target_accuracy=0.8,
                root_output_dir=str(tmp_path),
                max_parallel=1,
                truncate_input=True,
                max_retries=2,
            )

        # All 3 initial batches + retries for the 3 failed from batch 1
        assert "batch_1" in batch_labels_called
        assert "batch_2" in batch_labels_called
        assert "batch_3" in batch_labels_called

    def test_summary_includes_failed_and_completed(self, tmp_path):
        """experiment_summary.json records both successes and failures."""
        from run_experiments import run_all_experiments, TEACHERS, STUDENTS

        def mock_run_batch(batch_label, pairs, **kwargs):
            batch_dir = Path(kwargs["root_output_dir"]) / batch_label
            batch_dir.mkdir(parents=True, exist_ok=True)
            results = []
            for i, (ti, si) in enumerate(pairs):
                label = f"{TEACHERS[ti]['name']}__{STUDENTS[si]['name']}"
                exp_dir = str(batch_dir / label)
                Path(exp_dir).mkdir(parents=True, exist_ok=True)
                if i == 1:
                    results.append(_fail_result(label, "timeout", exp_dir))
                else:
                    (Path(exp_dir) / "experiment_result.json").write_text("{}")
                    (Path(exp_dir) / "round_1" / "adapter").mkdir(parents=True, exist_ok=True)
                    results.append(_ok_result(label, exp_dir))
            return results

        with mock.patch("run_experiments.run_batch", side_effect=mock_run_batch):
            run_all_experiments(
                base_config=_make_base_config(),
                rounds=1,
                target_accuracy=0.8,
                root_output_dir=str(tmp_path),
                max_parallel=1,
                truncate_input=True,
                max_retries=0,
            )

        summary_path = tmp_path / "experiment_summary.json"
        assert summary_path.exists()
        summary = json.loads(summary_path.read_text())
        assert summary["completed"] + summary["failed"] == summary["total_experiments"]


# ---------------------------------------------------------------------------
# Test Group 3: Progress preservation
# ---------------------------------------------------------------------------

class TestProgressPreservation:
    """Completed work must survive later failures."""

    def test_experiment_result_json_saved_on_success(self, tmp_path):
        """_run_single_experiment writes experiment_result.json to disk."""
        from run_experiments import _run_single_experiment

        exp_dir = str(tmp_path / "test_exp")

        with mock.patch("main.run_distillation_loop") as mock_loop:
            mock_loop.return_value = _ok_result("test", exp_dir)
            result = _run_single_experiment(
                experiment_config=_make_base_config(),
                rounds=1,
                target_accuracy=0.8,
                output_dir=exp_dir,
                truncate_input=True,
                experiment_label="test",
            )

        assert result["status"] == "completed"
        result_path = Path(exp_dir) / "experiment_result.json"
        assert result_path.exists()
        saved = json.loads(result_path.read_text())
        assert saved["status"] == "completed"

    def test_experiment_result_json_saved_on_failure(self, tmp_path):
        """_run_single_experiment writes experiment_result.json even on failure."""
        from run_experiments import _run_single_experiment

        exp_dir = str(tmp_path / "test_exp_fail")

        with mock.patch("main.run_distillation_loop", side_effect=RuntimeError("CUDA OOM")):
            result = _run_single_experiment(
                experiment_config=_make_base_config(),
                rounds=1,
                target_accuracy=0.8,
                output_dir=exp_dir,
                truncate_input=True,
                experiment_label="test_fail",
            )

        assert result["status"] == "failed"
        result_path = Path(exp_dir) / "experiment_result.json"
        assert result_path.exists()
        saved = json.loads(result_path.read_text())
        assert saved["status"] == "failed"
        assert "CUDA OOM" in saved["error"]

    def test_partial_round_artifacts_survive_later_failure(self, tmp_path):
        """If round 1 succeeds but round 2 crashes, round 1 artifacts persist."""
        exp_dir = tmp_path / "partial_exp"

        round_count = {"n": 0}

        def mock_generate_exam(*args, **kwargs):
            round_count["n"] += 1
            if round_count["n"] > 1:
                raise ConnectionError("API cost cap reached")
            return [{"question": {"question": "test?", "options": ["A. yes"]},
                      "answer": {"correct_options": ["A"]}}]

        def mock_evaluate(*args, **kwargs):
            return {
                "feedback": "ok", "proficiency": 5,
                "metrics": {"exact_match_accuracy": 0.5, "avg_jaccard": 0.5},
                "strengths": [], "weaknesses": [], "breakdowns": {},
                "dataset": [{"question": {"question": "q", "options": ["A. a"]},
                             "answer": {"correct_options": ["A"]}}],
            }

        config = _make_base_config()
        config["data_source"]["hybrid_analysis_dir"] = str(tmp_path / "fake_ha")
        ha_dir = tmp_path / "fake_ha" / "family1"
        ha_dir.mkdir(parents=True)
        (ha_dir / "sample1").write_text(json.dumps({"sha256": "abc", "verdict": "malicious"}))

        mock_teacher = mock.MagicMock()
        mock_teacher.generate_exam = mock.MagicMock(side_effect=mock_generate_exam)
        mock_teacher.evaluate_and_generate_curriculum = mock.MagicMock(side_effect=mock_evaluate)

        with mock.patch("teacher_engine.TeacherEngine", return_value=mock_teacher), \
             mock.patch("teacher_engine.load_template", return_value="TASK: %s"), \
             mock.patch("student_trainer.StudentTrainer.generate", return_value='{"correct_options": ["A"]}'), \
             mock.patch("student_trainer.StudentTrainer.train"), \
             mock.patch("student_trainer.StudentTrainer.save"), \
             mock.patch("student_trainer.StudentTrainer._load_model_and_tokenizer"):

            from main import run_distillation_loop
            try:
                run_distillation_loop(
                    config=config,
                    rounds=3,
                    target_accuracy=0.99,
                    output_dir=str(exp_dir),
                    truncate_input=False,
                )
            except ConnectionError:
                pass

        assert (exp_dir / "round_1" / "exam.json").exists()
        assert (exp_dir / "round_1" / "exam_results.json").exists()
        assert (exp_dir / "round_1" / "evaluation.json").exists()
        assert (exp_dir / "round_1" / "curriculum.json").exists()


# ---------------------------------------------------------------------------
# Test Group 4: Specific failure modes
# ---------------------------------------------------------------------------

class TestSpecificFailures:
    """Individual failure modes are caught and reported gracefully."""

    def test_teacher_api_network_error(self, tmp_path):
        from run_experiments import _run_single_experiment

        exp_dir = str(tmp_path / "net_err")

        with mock.patch("main.run_distillation_loop",
                         side_effect=ConnectionError("Connection refused")):
            result = _run_single_experiment(
                experiment_config=_make_base_config(),
                rounds=1, target_accuracy=0.8,
                output_dir=exp_dir, truncate_input=True,
                experiment_label="net_err",
            )

        assert result["status"] == "failed"
        assert "Connection refused" in result["error"]

    def test_teacher_returns_garbage_json(self):
        from teacher_engine import TeacherEngine

        engine = TeacherEngine.__new__(TeacherEngine)
        engine.provider = "openai"
        engine.model = "gpt-4o"
        engine.temperature = 0.7
        engine.templates_dir = "prompt_templates"

        with mock.patch.object(engine, "_complete", return_value="not json at all {{{"):
            result = engine.generate_exam(
                task_description="test", data_source="[]", num_questions=2,
            )

        assert result == []

    def test_teacher_retry_on_transient_error(self):
        from teacher_engine import TeacherEngine

        engine = TeacherEngine.__new__(TeacherEngine)
        engine.provider = "openai"
        engine.model = "gpt-4o"
        engine.temperature = 0.7

        call_count = {"n": 0}

        def flaky_provider(prompt):
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ConnectionError("rate limited")
            return '{"exam": []}'

        with mock.patch.object(engine, "_call_provider", side_effect=flaky_provider), \
             mock.patch("time.sleep"):
            result = engine._complete("test prompt", max_retries=3)

        assert result == '{"exam": []}'
        assert call_count["n"] == 3

    def test_teacher_retry_exhausted_raises(self):
        from teacher_engine import TeacherEngine

        engine = TeacherEngine.__new__(TeacherEngine)
        engine.provider = "openai"
        engine.model = "gpt-4o"
        engine.temperature = 0.7

        with mock.patch.object(engine, "_call_provider",
                                side_effect=ConnectionError("always fails")), \
             mock.patch("time.sleep"):
            with pytest.raises(ConnectionError, match="always fails"):
                engine._complete("test", max_retries=3)

    def test_student_cuda_oom(self, tmp_path):
        from run_experiments import _run_single_experiment

        exp_dir = str(tmp_path / "oom")

        with mock.patch("main.run_distillation_loop",
                         side_effect=RuntimeError("CUDA out of memory")):
            result = _run_single_experiment(
                experiment_config=_make_base_config(),
                rounds=1, target_accuracy=0.8,
                output_dir=exp_dir, truncate_input=True,
                experiment_label="oom_test",
            )

        assert result["status"] == "failed"
        assert "CUDA out of memory" in result["error"]

    def test_student_model_download_failure(self, tmp_path):
        from run_experiments import _run_single_experiment

        exp_dir = str(tmp_path / "dl_fail")

        with mock.patch("main.run_distillation_loop",
                         side_effect=OSError("404 model not found")):
            result = _run_single_experiment(
                experiment_config=_make_base_config(),
                rounds=1, target_accuracy=0.8,
                output_dir=exp_dir, truncate_input=True,
                experiment_label="dl_fail",
            )

        assert result["status"] == "failed"
        assert "404" in result["error"]


# ---------------------------------------------------------------------------
# Test Group 5: End-to-end summary integrity
# ---------------------------------------------------------------------------

class TestSummaryIntegrity:

    def test_comparison_table_sorted_by_accuracy(self, tmp_path):
        from run_experiments import _build_comparison_table

        results = [
            {**_ok_result("low"), "best_accuracy": 0.3},
            {**_ok_result("high"), "best_accuracy": 0.9},
            {**_ok_result("mid"), "best_accuracy": 0.6},
        ]

        table = _build_comparison_table(results)
        accuracies = [row["best_accuracy"] for row in table]
        assert accuracies == [0.9, 0.6, 0.3]


# ---------------------------------------------------------------------------
# Test Group 6: Tracker persistence and resume
# ---------------------------------------------------------------------------

class TestTrackerPersistence:
    """The tracker file (tracker.json) persists state across reruns."""

    def test_tracker_saved_to_disk(self, tmp_path):
        """ExperimentTracker.save() writes tracker.json."""
        from run_experiments import ExperimentTracker

        tracker = ExperimentTracker(str(tmp_path))
        tracker.record("exp_a", (0, 0), _ok_result("exp_a"))
        tracker.record("exp_b", (1, 1), _fail_result("exp_b", "error"))

        tracker_path = tmp_path / "tracker.json"
        assert tracker_path.exists()
        data = json.loads(tracker_path.read_text())
        assert "exp_a" in data["experiments"]
        assert "exp_b" in data["experiments"]
        assert data["summary"]["completed"] == 1
        assert data["summary"]["failed"] == 1

    def test_tracker_loaded_on_init(self, tmp_path):
        """A new ExperimentTracker reads existing tracker.json."""
        from run_experiments import ExperimentTracker

        # Create and populate a tracker
        tracker1 = ExperimentTracker(str(tmp_path))
        tracker1.record("exp_a", (0, 0), _ok_result("exp_a"))
        tracker1.record("exp_b", (1, 1), _fail_result("exp_b", "error"))

        # Create a second tracker from the same directory — should load state
        tracker2 = ExperimentTracker(str(tmp_path))
        assert len(tracker2.experiments) == 2
        assert tracker2.experiments["exp_a"]["status"] == "completed"
        assert tracker2.experiments["exp_b"]["status"] == "failed"
        assert tracker2.num_completed == 1
        assert tracker2.num_failed == 1

    def test_rerun_skips_completed_experiments(self, tmp_path):
        """On rerun, completed experiments are skipped; only failed ones run."""
        from run_experiments import (
            run_all_experiments, ExperimentTracker,
            TEACHERS, STUDENTS, BATCHES,
        )

        # Pre-populate tracker with all 9 experiments: 7 completed, 2 failed
        tracker = ExperimentTracker(str(tmp_path))
        failed_pairs = []
        for batch_pairs in BATCHES:
            for ti, si in batch_pairs:
                label = f"{TEACHERS[ti]['name']}__{STUDENTS[si]['name']}"
                exp_dir = str(tmp_path / "initial" / label)
                Path(exp_dir).mkdir(parents=True, exist_ok=True)
                if (ti, si) in [(0, 0), (2, 1)]:
                    # These 2 experiments failed
                    tracker.record(label, (ti, si), _fail_result(label, "error", exp_dir))
                    failed_pairs.append((ti, si))
                else:
                    # Create fake artifacts so verification passes
                    (Path(exp_dir) / "experiment_result.json").write_text("{}")
                    (Path(exp_dir) / "round_1" / "adapter").mkdir(parents=True, exist_ok=True)
                    tracker.record(label, (ti, si), _ok_result(label, exp_dir))

        assert tracker.num_completed == 7
        assert tracker.num_failed == 2

        # Now rerun — should only run the 2 failed experiments
        batch_calls = []

        def mock_run_batch(batch_label, pairs, **kwargs):
            batch_calls.append({"label": batch_label, "pairs": pairs})
            batch_dir = Path(kwargs["root_output_dir"]) / batch_label
            batch_dir.mkdir(parents=True, exist_ok=True)
            results = []
            for ti, si in pairs:
                label = f"{TEACHERS[ti]['name']}__{STUDENTS[si]['name']}"
                exp_dir = str(batch_dir / label)
                Path(exp_dir).mkdir(parents=True, exist_ok=True)
                (Path(exp_dir) / "experiment_result.json").write_text("{}")
                (Path(exp_dir) / "round_1" / "adapter").mkdir(parents=True, exist_ok=True)
                results.append(_ok_result(label, exp_dir))
            return results

        with mock.patch("run_experiments.run_batch", side_effect=mock_run_batch):
            run_all_experiments(
                base_config=_make_base_config(),
                rounds=1,
                target_accuracy=0.8,
                root_output_dir=str(tmp_path),
                max_parallel=1,
                truncate_input=True,
                max_retries=0,
            )

        # Only the batches containing the 2 failed experiments should run
        all_pairs_run = []
        for call in batch_calls:
            all_pairs_run.extend(call["pairs"])
        assert len(all_pairs_run) == 2
        assert set(map(tuple, all_pairs_run)) == {(0, 0), (2, 1)}


# ---------------------------------------------------------------------------
# Test Group 7: Retry logic
# ---------------------------------------------------------------------------

class TestRetryLogic:
    """Failed experiments are retried up to max_retries times."""

    def test_failed_experiments_are_retried(self, tmp_path):
        """Experiments that fail in initial batches are retried."""
        from run_experiments import run_all_experiments, TEACHERS, STUDENTS

        call_log = []

        def mock_run_batch(batch_label, pairs, **kwargs):
            call_log.append(batch_label)
            batch_dir = Path(kwargs["root_output_dir"]) / batch_label
            batch_dir.mkdir(parents=True, exist_ok=True)
            results = []
            for ti, si in pairs:
                label = f"{TEACHERS[ti]['name']}__{STUDENTS[si]['name']}"
                exp_dir = str(batch_dir / label)
                Path(exp_dir).mkdir(parents=True, exist_ok=True)
                # Make one experiment fail in initial batches, succeed in retries
                if label == "gpt4o__llama3.3_8b" and "retry" not in batch_label:
                    results.append(_fail_result(label, "transient error", exp_dir))
                else:
                    (Path(exp_dir) / "experiment_result.json").write_text("{}")
                    (Path(exp_dir) / "round_1" / "adapter").mkdir(parents=True, exist_ok=True)
                    results.append(_ok_result(label, exp_dir))
            return results

        with mock.patch("run_experiments.run_batch", side_effect=mock_run_batch):
            results = run_all_experiments(
                base_config=_make_base_config(),
                rounds=1,
                target_accuracy=0.8,
                root_output_dir=str(tmp_path),
                max_parallel=1,
                truncate_input=True,
                max_retries=2,
            )

        # Should have initial batches + a retry
        assert "batch_1" in call_log
        assert "batch_2" in call_log
        assert "batch_3" in call_log
        assert "retry_1" in call_log

        # After retry, the experiment should be completed
        tracker_path = tmp_path / "tracker.json"
        tracker_data = json.loads(tracker_path.read_text())
        assert tracker_data["experiments"]["gpt4o__llama3.3_8b"]["status"] == "completed"

    def test_retry_capped_at_max_retries(self, tmp_path):
        """Retries stop after max_retries even if experiments still fail."""
        from run_experiments import run_all_experiments, TEACHERS, STUDENTS

        call_log = []

        def mock_run_batch(batch_label, pairs, **kwargs):
            call_log.append(batch_label)
            batch_dir = Path(kwargs["root_output_dir"]) / batch_label
            batch_dir.mkdir(parents=True, exist_ok=True)
            results = []
            for ti, si in pairs:
                label = f"{TEACHERS[ti]['name']}__{STUDENTS[si]['name']}"
                exp_dir = str(batch_dir / label)
                Path(exp_dir).mkdir(parents=True, exist_ok=True)
                # This experiment always fails
                if label == "gpt4o__llama3.3_8b":
                    results.append(_fail_result(label, "permanent error", exp_dir))
                else:
                    (Path(exp_dir) / "experiment_result.json").write_text("{}")
                    (Path(exp_dir) / "round_1" / "adapter").mkdir(parents=True, exist_ok=True)
                    results.append(_ok_result(label, exp_dir))
            return results

        with mock.patch("run_experiments.run_batch", side_effect=mock_run_batch):
            run_all_experiments(
                base_config=_make_base_config(),
                rounds=1,
                target_accuracy=0.8,
                root_output_dir=str(tmp_path),
                max_parallel=1,
                truncate_input=True,
                max_retries=2,
            )

        # Should have 3 initial batches + exactly 2 retries (not 3+)
        retry_calls = [c for c in call_log if c.startswith("retry_")]
        assert len(retry_calls) == 2
        assert "retry_1" in call_log
        assert "retry_2" in call_log
        assert "retry_3" not in call_log

        # Summary should list the permanently failed experiment
        summary = json.loads((tmp_path / "experiment_summary.json").read_text())
        assert "gpt4o__llama3.3_8b" in summary["permanently_failed"]

    def test_no_retry_when_all_succeed(self, tmp_path):
        """No retries happen if all experiments pass."""
        from run_experiments import run_all_experiments, TEACHERS, STUDENTS

        call_log = []

        def mock_run_batch(batch_label, pairs, **kwargs):
            call_log.append(batch_label)
            batch_dir = Path(kwargs["root_output_dir"]) / batch_label
            batch_dir.mkdir(parents=True, exist_ok=True)
            results = []
            for ti, si in pairs:
                label = f"{TEACHERS[ti]['name']}__{STUDENTS[si]['name']}"
                exp_dir = str(batch_dir / label)
                Path(exp_dir).mkdir(parents=True, exist_ok=True)
                (Path(exp_dir) / "experiment_result.json").write_text("{}")
                (Path(exp_dir) / "round_1" / "adapter").mkdir(parents=True, exist_ok=True)
                results.append(_ok_result(label, exp_dir))
            return results

        with mock.patch("run_experiments.run_batch", side_effect=mock_run_batch):
            run_all_experiments(
                base_config=_make_base_config(),
                rounds=1,
                target_accuracy=0.8,
                root_output_dir=str(tmp_path),
                max_parallel=1,
                truncate_input=True,
                max_retries=2,
            )

        retry_calls = [c for c in call_log if c.startswith("retry_")]
        assert len(retry_calls) == 0


# ---------------------------------------------------------------------------
# Test Group 8: Artifact verification
# ---------------------------------------------------------------------------

class TestArtifactVerification:
    """Artifact verification catches experiments with missing checkpoints."""

    def test_missing_adapter_demotes_to_failed(self, tmp_path):
        """An experiment marked completed but missing its adapter is demoted."""
        from run_experiments import ExperimentTracker

        tracker = ExperimentTracker(str(tmp_path))

        # Record as completed but don't create any adapter files
        exp_dir = str(tmp_path / "no_adapter_exp")
        Path(exp_dir).mkdir(parents=True)
        (Path(exp_dir) / "experiment_result.json").write_text("{}")
        # Deliberately do NOT create adapter directory

        tracker.record("exp_a", (0, 0), _ok_result("exp_a", exp_dir))
        assert tracker.num_completed == 1

        demoted = tracker.verify_artifacts()
        assert "exp_a" in demoted
        assert tracker.experiments["exp_a"]["status"] == "failed"
        assert "no adapter" in tracker.experiments["exp_a"]["error"]

    def test_missing_result_json_demotes_to_failed(self, tmp_path):
        """An experiment missing experiment_result.json is demoted."""
        from run_experiments import ExperimentTracker

        tracker = ExperimentTracker(str(tmp_path))

        exp_dir = str(tmp_path / "no_result_exp")
        Path(exp_dir).mkdir(parents=True)
        # Create adapter but NOT experiment_result.json
        (Path(exp_dir) / "round_1" / "adapter").mkdir(parents=True)

        tracker.record("exp_b", (1, 1), _ok_result("exp_b", exp_dir))

        demoted = tracker.verify_artifacts()
        assert "exp_b" in demoted
        assert "experiment_result.json" in tracker.experiments["exp_b"]["error"]

    def test_valid_artifacts_pass_verification(self, tmp_path):
        """An experiment with all artifacts passes verification."""
        from run_experiments import ExperimentTracker

        tracker = ExperimentTracker(str(tmp_path))

        exp_dir = str(tmp_path / "good_exp")
        Path(exp_dir).mkdir(parents=True)
        (Path(exp_dir) / "experiment_result.json").write_text("{}")
        (Path(exp_dir) / "round_1" / "adapter").mkdir(parents=True)

        tracker.record("exp_c", (2, 2), _ok_result("exp_c", exp_dir))

        demoted = tracker.verify_artifacts()
        assert len(demoted) == 0
        assert tracker.experiments["exp_c"]["artifacts_verified"] is True

    def test_failed_experiments_not_verified(self, tmp_path):
        """Already-failed experiments are skipped during verification."""
        from run_experiments import ExperimentTracker

        tracker = ExperimentTracker(str(tmp_path))
        tracker.record("exp_d", (0, 1), _fail_result("exp_d", "original error"))

        demoted = tracker.verify_artifacts()
        assert len(demoted) == 0  # already failed, not re-checked
