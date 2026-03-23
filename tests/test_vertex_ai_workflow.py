"""
test_vertex_ai_workflow.py
==========================
Tests that the experiment orchestration correctly propagates Vertex AI
configuration through the full pipeline:

  - build_experiment_config() preserves student.backend and vertex_ai section
  - create_student_trainer() returns the correct trainer type
  - Artifact verification recognises vertex_ai_metadata.json
  - End-to-end orchestration passes Vertex AI config to every experiment
  - All 9 experiments in the 3x3 matrix receive consistent Vertex AI config
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vertex_ai_base_config() -> dict[str, Any]:
    """Base config that mirrors a real Vertex AI deployment."""
    return {
        "teacher": {"provider": "openai", "model": "gpt-4o", "temperature": 0.7},
        "student": {
            "model_name_or_path": "meta-llama/Llama-3.3-8B-Instruct",
            "backend": "vertex_ai",
        },
        "vertex_ai": {
            "project": "smala-test-project",
            "location": "us-central1",
            "staging_bucket": "gs://smala-test-bucket/smala",
        },
        "training": {
            "use_4bit": True,
            "learning_rate": 2e-4,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "max_seq_length": 512,
        },
        "lora": {"r": 8, "alpha": 16, "dropout": 0.05},
        "data_source": {
            "task_definitions": "prompt_templates/malware_analysis_task.json",
            "hybrid_analysis_dir": "hybrid-analysis",
            "reports_per_round": 2,
        },
        "exam": {"num_questions": 2},
        "curriculum": {"num_examples": 5},
    }


def _ok_result(label: str, output_dir: str = "/tmp/test") -> dict[str, Any]:
    return {
        "experiment": label,
        "output_dir": output_dir,
        "status": "completed",
        "best_accuracy": 0.75,
        "final_proficiency": "7",
        "rounds_completed": 3,
        "round_results": [],
        "duration_seconds": 10.0,
    }


class _InlineExecutor:
    """Drop-in for ProcessPoolExecutor that runs in-thread (mocks aren't picklable)."""

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
# Test Group 1: build_experiment_config preserves Vertex AI fields
# ---------------------------------------------------------------------------

class TestBuildExperimentConfig:
    """build_experiment_config must carry student.backend and vertex_ai through."""

    def test_preserves_student_backend(self):
        """student.backend from base config must survive the merge."""
        from run_experiments import build_experiment_config, TEACHERS, STUDENTS

        config = build_experiment_config(
            teacher=TEACHERS[0],
            student=STUDENTS[0],
            base_config=_make_vertex_ai_base_config(),
            output_dir="/tmp/test",
        )

        assert config["student"]["backend"] == "vertex_ai"

    def test_overrides_model_name_or_path(self):
        """student.model_name_or_path must reflect the specific student, not the base."""
        from run_experiments import build_experiment_config, TEACHERS, STUDENTS

        config = build_experiment_config(
            teacher=TEACHERS[0],
            student=STUDENTS[1],  # gemma3_4b
            base_config=_make_vertex_ai_base_config(),
            output_dir="/tmp/test",
        )

        assert config["student"]["model_name_or_path"] == STUDENTS[1]["model_name_or_path"]
        assert config["student"]["backend"] == "vertex_ai"

    def test_preserves_vertex_ai_section(self):
        """The vertex_ai config block must be passed through unchanged."""
        from run_experiments import build_experiment_config, TEACHERS, STUDENTS

        base = _make_vertex_ai_base_config()
        config = build_experiment_config(
            teacher=TEACHERS[0],
            student=STUDENTS[0],
            base_config=base,
            output_dir="/tmp/test",
        )

        assert config["vertex_ai"]["project"] == "smala-test-project"
        assert config["vertex_ai"]["location"] == "us-central1"
        assert config["vertex_ai"]["staging_bucket"] == "gs://smala-test-bucket/smala"

    def test_does_not_mutate_base_config(self):
        """build_experiment_config must deep-copy; original must stay unchanged."""
        from run_experiments import build_experiment_config, TEACHERS, STUDENTS

        base = _make_vertex_ai_base_config()
        original_student_model = base["student"]["model_name_or_path"]

        build_experiment_config(
            teacher=TEACHERS[0],
            student=STUDENTS[2],  # mistral_7b
            base_config=base,
            output_dir="/tmp/test",
        )

        assert base["student"]["model_name_or_path"] == original_student_model

    def test_local_backend_preserved_when_set(self):
        """When base config says backend=local, that must also survive."""
        from run_experiments import build_experiment_config, TEACHERS, STUDENTS

        base = _make_vertex_ai_base_config()
        base["student"]["backend"] = "local"

        config = build_experiment_config(
            teacher=TEACHERS[0],
            student=STUDENTS[0],
            base_config=base,
            output_dir="/tmp/test",
        )

        assert config["student"]["backend"] == "local"

    def test_all_nine_pairs_get_vertex_ai_backend(self):
        """Every pair in the 3x3 matrix must inherit backend from base config."""
        from run_experiments import (
            build_experiment_config, TEACHERS, STUDENTS, BATCHES,
        )

        base = _make_vertex_ai_base_config()

        for batch in BATCHES:
            for ti, si in batch:
                config = build_experiment_config(
                    teacher=TEACHERS[ti],
                    student=STUDENTS[si],
                    base_config=base,
                    output_dir=f"/tmp/test/{ti}_{si}",
                )
                assert config["student"]["backend"] == "vertex_ai", (
                    f"Teacher {TEACHERS[ti]['name']} -> Student {STUDENTS[si]['name']} "
                    f"lost backend=vertex_ai"
                )
                assert config["vertex_ai"]["project"] == "smala-test-project", (
                    f"Teacher {TEACHERS[ti]['name']} -> Student {STUDENTS[si]['name']} "
                    f"lost vertex_ai.project"
                )


# ---------------------------------------------------------------------------
# Test Group 2: create_student_trainer returns the correct type
# ---------------------------------------------------------------------------

class TestCreateStudentTrainer:
    """create_student_trainer must dispatch on student.backend."""

    def test_vertex_ai_backend_returns_vertex_trainer(self):
        """backend=vertex_ai must produce a VertexAIStudentTrainer."""
        from student_trainer import create_student_trainer, VertexAIStudentTrainer

        config = _make_vertex_ai_base_config()
        trainer = create_student_trainer(config)

        assert isinstance(trainer, VertexAIStudentTrainer)
        assert trainer.project == "smala-test-project"
        assert trainer.staging_bucket == "gs://smala-test-bucket/smala"

    def test_local_backend_returns_local_trainer(self):
        """backend=local must produce a (local) StudentTrainer."""
        from student_trainer import create_student_trainer, StudentTrainer

        config = _make_vertex_ai_base_config()
        config["student"]["backend"] = "local"
        trainer = create_student_trainer(config)

        assert isinstance(trainer, StudentTrainer)
        assert not isinstance(trainer, type(None))

    def test_missing_backend_defaults_to_local(self):
        """When backend is absent, default to local StudentTrainer."""
        from student_trainer import create_student_trainer, StudentTrainer

        config = _make_vertex_ai_base_config()
        del config["student"]["backend"]
        trainer = create_student_trainer(config)

        assert isinstance(trainer, StudentTrainer)

    def test_vertex_trainer_inherits_training_params(self):
        """VertexAIStudentTrainer must pick up learning_rate/epochs/lora from config."""
        from student_trainer import create_student_trainer, VertexAIStudentTrainer

        config = _make_vertex_ai_base_config()
        config["training"]["learning_rate"] = 1e-5
        config["training"]["num_train_epochs"] = 5
        config["lora"]["r"] = 32

        trainer = create_student_trainer(config)
        assert isinstance(trainer, VertexAIStudentTrainer)
        assert trainer.learning_rate == 1e-5
        assert trainer.num_train_epochs == 5
        assert trainer.lora_r == 32


# ---------------------------------------------------------------------------
# Test Group 3: Artifact verification with Vertex AI metadata
# ---------------------------------------------------------------------------

class TestVertexAIArtifactVerification:
    """Artifact verification must recognise vertex_ai_metadata.json as valid."""

    def test_vertex_ai_metadata_passes_verification(self, tmp_path):
        """An experiment with vertex_ai_metadata.json + result passes."""
        from run_experiments import ExperimentTracker

        tracker = ExperimentTracker(str(tmp_path))

        exp_dir = str(tmp_path / "vertex_exp")
        Path(exp_dir).mkdir(parents=True)
        (Path(exp_dir) / "experiment_result.json").write_text("{}")
        (Path(exp_dir) / "vertex_ai_metadata.json").write_text(
            json.dumps({"backend": "vertex_ai", "project": "test"})
        )

        tracker.record("vertex_exp", (0, 0), _ok_result("vertex_exp", exp_dir))

        demoted = tracker.verify_artifacts()
        assert len(demoted) == 0
        assert tracker.experiments["vertex_exp"]["artifacts_verified"] is True

    def test_vertex_ai_no_metadata_no_adapter_demoted(self, tmp_path):
        """Without vertex_ai_metadata.json or adapter dir, experiment is demoted."""
        from run_experiments import ExperimentTracker

        tracker = ExperimentTracker(str(tmp_path))

        exp_dir = str(tmp_path / "missing_exp")
        Path(exp_dir).mkdir(parents=True)
        (Path(exp_dir) / "experiment_result.json").write_text("{}")

        tracker.record("missing_exp", (0, 0), _ok_result("missing_exp", exp_dir))

        demoted = tracker.verify_artifacts()
        assert "missing_exp" in demoted


# ---------------------------------------------------------------------------
# Test Group 4: End-to-end orchestration with Vertex AI config
# ---------------------------------------------------------------------------

class TestOrchestrationVertexAI:
    """The full 3x3 orchestrator must propagate Vertex AI config to each experiment."""

    def test_experiment_configs_contain_vertex_ai(self, tmp_path):
        """Every config passed to _run_single_experiment must have backend=vertex_ai."""
        from run_experiments import run_batch, BATCHES, TEACHERS, STUDENTS

        captured_configs = []

        def mock_run_single(experiment_config, rounds, target_accuracy,
                            output_dir, truncate_input, experiment_label):
            captured_configs.append(experiment_config)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            return _ok_result(experiment_label, output_dir)

        with mock.patch("run_experiments._run_single_experiment", side_effect=mock_run_single), \
             mock.patch("run_experiments.ProcessPoolExecutor", _InlineExecutor):
            run_batch(
                batch_label="batch_1",
                pairs=BATCHES[0],
                base_config=_make_vertex_ai_base_config(),
                rounds=1,
                target_accuracy=0.8,
                root_output_dir=str(tmp_path),
                max_parallel=1,
                truncate_input=True,
            )

        assert len(captured_configs) == 3
        for i, cfg in enumerate(captured_configs):
            ti, si = BATCHES[0][i]
            assert cfg["student"]["backend"] == "vertex_ai", (
                f"Experiment {TEACHERS[ti]['name']}->{STUDENTS[si]['name']} "
                f"missing backend=vertex_ai"
            )
            assert cfg["vertex_ai"]["project"] == "smala-test-project", (
                f"Experiment {TEACHERS[ti]['name']}->{STUDENTS[si]['name']} "
                f"missing vertex_ai.project"
            )
            # Student model must match the specific student for this pair
            assert cfg["student"]["model_name_or_path"] == STUDENTS[si]["model_name_or_path"]

    def test_full_matrix_all_configs_correct(self, tmp_path):
        """All 9 experiments across 3 batches must get correct teacher+student+vertex config."""
        from run_experiments import (
            run_all_experiments, TEACHERS, STUDENTS, BATCHES,
        )

        captured_configs = {}

        def mock_run_batch(batch_label, pairs, **kwargs):
            from run_experiments import build_experiment_config

            batch_dir = Path(kwargs["root_output_dir"]) / batch_label
            batch_dir.mkdir(parents=True, exist_ok=True)
            results = []
            for ti, si in pairs:
                label = f"{TEACHERS[ti]['name']}__{STUDENTS[si]['name']}"
                exp_dir = str(batch_dir / label)
                Path(exp_dir).mkdir(parents=True, exist_ok=True)

                # Replicate what run_batch does internally
                cfg = build_experiment_config(
                    teacher=TEACHERS[ti],
                    student=STUDENTS[si],
                    base_config=kwargs.get("base_config", _make_vertex_ai_base_config()),
                    output_dir=exp_dir,
                )
                captured_configs[label] = cfg

                (Path(exp_dir) / "experiment_result.json").write_text("{}")
                (Path(exp_dir) / "vertex_ai_metadata.json").write_text("{}")
                results.append(_ok_result(label, exp_dir))
            return results

        # Patch run_batch but pass through base_config
        original_run_batch = None

        def patched_run_batch(batch_label, pairs, base_config, **kwargs):
            kwargs["base_config"] = base_config
            return mock_run_batch(batch_label, pairs, **kwargs)

        with mock.patch("run_experiments.run_batch", side_effect=patched_run_batch):
            run_all_experiments(
                base_config=_make_vertex_ai_base_config(),
                rounds=1,
                target_accuracy=0.8,
                root_output_dir=str(tmp_path),
                max_parallel=1,
                truncate_input=True,
                max_retries=0,
            )

        # All 9 combinations must have been captured
        assert len(captured_configs) == 9

        for label, cfg in captured_configs.items():
            assert cfg["student"]["backend"] == "vertex_ai", (
                f"{label}: missing backend=vertex_ai"
            )
            assert cfg["vertex_ai"]["project"] == "smala-test-project", (
                f"{label}: missing vertex_ai.project"
            )

    def test_distillation_loop_creates_vertex_trainer(self, tmp_path):
        """run_distillation_loop must create a VertexAIStudentTrainer when backend=vertex_ai."""
        from student_trainer import VertexAIStudentTrainer

        trainer_instances = []

        original_create = None

        def mock_create(config):
            from student_trainer import create_student_trainer as real_create
            # We can't call real_create because it would try to init Vertex AI,
            # so just record what type would be created and return a mock
            backend = config.get("student", {}).get("backend", "local")
            trainer_instances.append(backend)
            m = mock.MagicMock(spec=VertexAIStudentTrainer)
            m.generate.return_value = '{"correct_options": ["A"]}'
            return m

        config = _make_vertex_ai_base_config()
        config["data_source"]["hybrid_analysis_dir"] = str(tmp_path / "fake_ha")
        ha_dir = tmp_path / "fake_ha" / "family1"
        ha_dir.mkdir(parents=True)
        (ha_dir / "sample1").write_text(
            json.dumps({"sha256": "abc", "verdict": "malicious"})
        )

        mock_teacher = mock.MagicMock()
        mock_teacher.generate_exam.return_value = [
            {"question": {"question": "test?", "options": ["A. yes"]},
             "answer": {"correct_options": ["A"]}}
        ]
        mock_teacher.evaluate_and_generate_curriculum.return_value = {
            "feedback": "ok", "proficiency": 5,
            "metrics": {"exact_match_accuracy": 0.9, "avg_jaccard": 0.9},
            "strengths": [], "weaknesses": [], "breakdowns": {},
            "dataset": [{"question": {"question": "q", "options": ["A. a"]},
                         "answer": {"correct_options": ["A"]}}],
        }

        with mock.patch("teacher_engine.TeacherEngine", return_value=mock_teacher), \
             mock.patch("student_trainer.create_student_trainer", side_effect=mock_create), \
             mock.patch("teacher_engine.load_template", return_value="TASK: %s"):

            from main import run_distillation_loop

            run_distillation_loop(
                config=config,
                rounds=1,
                target_accuracy=0.99,
                output_dir=str(tmp_path / "output"),
                truncate_input=False,
            )

        assert len(trainer_instances) == 1
        assert trainer_instances[0] == "vertex_ai"


# ---------------------------------------------------------------------------
# Test Group 5: Existing test base config compatibility
# ---------------------------------------------------------------------------

class TestExistingTestConfigCompat:
    """Ensure the existing _make_base_config (without backend) still works
    and defaults to local backend."""

    def test_config_without_backend_defaults_local(self):
        """A config dict without student.backend must default to local."""
        from student_trainer import create_student_trainer, StudentTrainer

        config = {
            "student": {"model_name_or_path": "meta-llama/Llama-3.3-8B-Instruct"},
            "training": {"use_4bit": True, "learning_rate": 2e-4,
                         "num_train_epochs": 1},
            "lora": {"r": 8, "alpha": 16, "dropout": 0.05},
        }
        trainer = create_student_trainer(config)
        assert isinstance(trainer, StudentTrainer)

    def test_build_experiment_config_no_backend_in_base(self):
        """When base config has no student.backend, built config also lacks it
        (defaults to local in create_student_trainer)."""
        from run_experiments import build_experiment_config, TEACHERS, STUDENTS

        base = {
            "teacher": {"provider": "openai", "model": "gpt-4o", "temperature": 0.7},
            "student": {"model_name_or_path": "meta-llama/Llama-3.3-8B-Instruct"},
            "training": {"use_4bit": True},
            "lora": {"r": 8, "alpha": 16, "dropout": 0.05},
        }

        config = build_experiment_config(
            teacher=TEACHERS[0],
            student=STUDENTS[0],
            base_config=base,
            output_dir="/tmp/test",
        )

        # backend not in base, so it shouldn't appear in built config either
        # create_student_trainer will default to "local"
        assert config["student"].get("backend", "local") == "local"
