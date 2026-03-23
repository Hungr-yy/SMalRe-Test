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
import os
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
            student=STUDENTS[2],  # qwen2.5_7b
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
# Test Group 3: HF_TOKEN passthrough to Vertex AI training container
# ---------------------------------------------------------------------------

def _mock_gcp_modules():
    """Inject mock google.cloud modules into sys.modules so train() can import them."""
    mock_aiplatform = mock.MagicMock()
    mock_storage = mock.MagicMock()
    mock_google = mock.MagicMock()
    mock_google.cloud.aiplatform = mock_aiplatform
    mock_google.cloud.storage = mock_storage

    patches = {
        "google": mock_google,
        "google.cloud": mock_google.cloud,
        "google.cloud.aiplatform": mock_aiplatform,
        "google.cloud.storage": mock_storage,
    }
    return patches, mock_aiplatform


class TestHFTokenPassthrough:
    """HF_TOKEN must be passed to the Vertex AI training container for gated models."""

    def test_hf_token_passed_to_training_job(self):
        """When HF_TOKEN is set, it must appear in environment_variables."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="meta-llama/Llama-3.3-8B-Instruct",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )

        mock_job = mock.MagicMock()
        mock_job.run.return_value = mock.MagicMock(resource_name="projects/test/models/123")

        gcp_modules, mock_aiplatform = _mock_gcp_modules()
        mock_aiplatform.CustomTrainingJob.return_value = mock_job

        curriculum = [{"question": {"question": "q", "options": ["A. a"]},
                       "answer": {"correct_options": ["A"]}}]

        with mock.patch.dict("sys.modules", gcp_modules), \
             mock.patch.object(trainer, "_upload_to_gcs"), \
             mock.patch.object(trainer, "_download_from_gcs"), \
             mock.patch.dict("os.environ", {"HF_TOKEN": "hf_test_token_123"}):
            trainer.train(curriculum)

        call_kwargs = mock_job.run.call_args
        env_vars = call_kwargs.kwargs.get("environment_variables")
        assert env_vars is not None
        assert env_vars["HF_TOKEN"] == "hf_test_token_123"

    def test_no_hf_token_passes_none(self):
        """When HF_TOKEN is not set, environment_variables should be None."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )

        mock_job = mock.MagicMock()
        mock_job.run.return_value = mock.MagicMock(resource_name="projects/test/models/123")

        gcp_modules, mock_aiplatform = _mock_gcp_modules()
        mock_aiplatform.CustomTrainingJob.return_value = mock_job

        curriculum = [{"question": {"question": "q", "options": ["A. a"]},
                       "answer": {"correct_options": ["A"]}}]

        # Ensure HF_TOKEN is NOT in the environment
        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with mock.patch.dict("sys.modules", gcp_modules), \
             mock.patch.object(trainer, "_upload_to_gcs"), \
             mock.patch.object(trainer, "_download_from_gcs"), \
             mock.patch.dict("os.environ", env, clear=True):
            trainer.train(curriculum)

        call_kwargs = mock_job.run.call_args
        env_vars = call_kwargs.kwargs.get("environment_variables")
        assert env_vars is None

    def test_all_three_students_get_hf_token_when_set(self):
        """All 3 student models must receive HF_TOKEN in their training jobs."""
        from run_experiments import STUDENTS
        from student_trainer import VertexAIStudentTrainer

        for student in STUDENTS:
            trainer = VertexAIStudentTrainer(
                model_name_or_path=student["model_name_or_path"],
                project="test-project",
                location="us-central1",
                staging_bucket="gs://test-bucket/smala",
            )

            mock_job = mock.MagicMock()
            mock_job.run.return_value = mock.MagicMock(resource_name="projects/test/models/123")

            gcp_modules, mock_aiplatform = _mock_gcp_modules()
            mock_aiplatform.CustomTrainingJob.return_value = mock_job

            curriculum = [{"question": {"question": "q", "options": ["A. a"]},
                           "answer": {"correct_options": ["A"]}}]

            with mock.patch.dict("sys.modules", gcp_modules), \
                 mock.patch.object(trainer, "_upload_to_gcs"), \
                 mock.patch.object(trainer, "_download_from_gcs"), \
                 mock.patch.dict("os.environ", {"HF_TOKEN": "hf_test_token"}):
                trainer.train(curriculum)

            call_kwargs = mock_job.run.call_args
            env_vars = call_kwargs.kwargs.get("environment_variables")
            assert env_vars is not None, (
                f"{student['name']}: HF_TOKEN not passed to training job"
            )
            assert env_vars["HF_TOKEN"] == "hf_test_token", (
                f"{student['name']}: HF_TOKEN has wrong value"
            )

    def test_training_job_args_contain_correct_model_id(self):
        """The --model_id arg must match the student's HuggingFace model path."""
        from run_experiments import STUDENTS
        from student_trainer import VertexAIStudentTrainer

        for student in STUDENTS:
            trainer = VertexAIStudentTrainer(
                model_name_or_path=student["model_name_or_path"],
                project="test-project",
                location="us-central1",
                staging_bucket="gs://test-bucket/smala",
            )

            mock_job = mock.MagicMock()
            mock_job.run.return_value = mock.MagicMock(resource_name="projects/test/models/123")

            gcp_modules, mock_aiplatform = _mock_gcp_modules()
            mock_aiplatform.CustomTrainingJob.return_value = mock_job

            curriculum = [{"question": {"question": "q", "options": ["A. a"]},
                           "answer": {"correct_options": ["A"]}}]

            with mock.patch.dict("sys.modules", gcp_modules), \
                 mock.patch.object(trainer, "_upload_to_gcs"), \
                 mock.patch.object(trainer, "_download_from_gcs"), \
                 mock.patch.dict("os.environ", {"HF_TOKEN": "hf_test"}):
                trainer.train(curriculum)

            call_kwargs = mock_job.run.call_args
            args_list = call_kwargs.kwargs.get("args")
            model_id_arg = f"--model_id={student['model_name_or_path']}"
            assert model_id_arg in args_list, (
                f"{student['name']}: expected {model_id_arg} in training args, "
                f"got {args_list}"
            )


# ---------------------------------------------------------------------------
# Test Group 4: Round 1 inference fallback (Vertex AI base endpoint)
# ---------------------------------------------------------------------------

class TestRound1InferenceFallback:
    """Before any fine-tuning (Round 1), generate() must deploy and use
    a Vertex AI base model endpoint for ALL student models."""

    def test_generate_deploys_base_endpoint_when_no_tuned_endpoint(self):
        """When _tuned_endpoint is None, generate() must deploy a base endpoint."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="meta-llama/Llama-3.3-8B-Instruct",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )
        assert trainer._tuned_endpoint is None
        assert trainer._base_endpoint is None

        mock_endpoint = mock.MagicMock()
        mock_endpoint.predict.return_value = mock.MagicMock(
            predictions=[{"generated_text": '{"correct_options": ["A"]}'}]
        )

        with mock.patch.object(trainer, "_deploy_base_endpoint") as mock_deploy:
            def set_endpoint():
                trainer._base_endpoint = mock_endpoint
            mock_deploy.side_effect = set_endpoint

            result = trainer.generate("test prompt")

        mock_deploy.assert_called_once()
        assert result == '{"correct_options": ["A"]}'

    def test_base_endpoint_reused_across_calls(self):
        """Once deployed, the base endpoint must be reused, not redeployed."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )

        mock_endpoint = mock.MagicMock()
        mock_endpoint.predict.return_value = mock.MagicMock(
            predictions=[{"generated_text": '{"correct_options": ["B"]}'}]
        )
        # Pre-set the base endpoint (simulating it was already deployed)
        trainer._base_endpoint = mock_endpoint

        with mock.patch.object(trainer, "_deploy_base_endpoint") as mock_deploy:
            trainer.generate("prompt 1")
            trainer.generate("prompt 2")

        # Should NOT redeploy
        mock_deploy.assert_not_called()
        assert mock_endpoint.predict.call_count == 2

    def test_all_three_students_deploy_base_endpoint_on_round1(self):
        """All 3 student models must deploy a base endpoint on Round 1."""
        from run_experiments import STUDENTS
        from student_trainer import VertexAIStudentTrainer

        for student in STUDENTS:
            trainer = VertexAIStudentTrainer(
                model_name_or_path=student["model_name_or_path"],
                project="test-project",
                location="us-central1",
                staging_bucket="gs://test-bucket/smala",
            )

            mock_endpoint = mock.MagicMock()
            mock_endpoint.predict.return_value = mock.MagicMock(
                predictions=[{"generated_text": '{"correct_options": ["A"]}'}]
            )

            with mock.patch.object(trainer, "_deploy_base_endpoint") as mock_deploy:
                def set_endpoint():
                    trainer._base_endpoint = mock_endpoint
                mock_deploy.side_effect = set_endpoint

                result = trainer.generate("test prompt", max_new_tokens=128)

            mock_deploy.assert_called_once(), (
                f"{student['name']}: base endpoint not deployed on Round 1"
            )
            assert isinstance(result, str) and len(result) > 0, (
                f"{student['name']}: generate() returned empty on Round 1"
            )

    def test_generate_uses_tuned_endpoint_when_available(self):
        """After training, generate() must try the tuned endpoint first."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="google/gemma-3-4b-it",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )

        mock_endpoint = mock.MagicMock()
        mock_endpoint.predict.return_value = mock.MagicMock(
            predictions=[{"generated_text": '{"correct_options": ["C"]}'}]
        )
        mock_model = mock.MagicMock()
        mock_model.deploy.return_value = mock_endpoint
        trainer._tuned_endpoint = mock_model

        gcp_modules, mock_aiplatform = _mock_gcp_modules()

        with mock.patch.dict("sys.modules", gcp_modules), \
             mock.patch.object(trainer, "_deploy_base_endpoint") as mock_base_deploy:
            result = trainer.generate("test prompt")

        mock_model.deploy.assert_called_once()
        mock_base_deploy.assert_not_called()
        assert result == '{"correct_options": ["C"]}'

    def test_generate_falls_back_to_base_endpoint_when_tuned_fails(self):
        """If tuned endpoint raises, generate() must fall back to base endpoint."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="meta-llama/Llama-3.3-8B-Instruct",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )

        mock_model = mock.MagicMock()
        mock_model.deploy.side_effect = RuntimeError("endpoint unavailable")
        trainer._tuned_endpoint = mock_model

        mock_base = mock.MagicMock()
        mock_base.predict.return_value = mock.MagicMock(
            predictions=[{"generated_text": '{"correct_options": ["A"]}'}]
        )
        trainer._base_endpoint = mock_base

        gcp_modules, _ = _mock_gcp_modules()

        with mock.patch.dict("sys.modules", gcp_modules):
            result = trainer.generate("test prompt")

        mock_base.predict.assert_called_once()
        assert result == '{"correct_options": ["A"]}'

    def test_deploy_base_endpoint_passes_hf_token(self):
        """_deploy_base_endpoint must pass HF_TOKEN to the serving container."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="meta-llama/Llama-3.3-8B-Instruct",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )

        gcp_modules, mock_aiplatform = _mock_gcp_modules()
        mock_uploaded_model = mock.MagicMock()
        mock_aiplatform.Model.upload.return_value = mock_uploaded_model
        mock_uploaded_model.deploy.return_value = mock.MagicMock()

        with mock.patch.dict("sys.modules", gcp_modules), \
             mock.patch.dict("os.environ", {"HF_TOKEN": "hf_test_deploy"}):
            trainer._deploy_base_endpoint()

        upload_kwargs = mock_aiplatform.Model.upload.call_args.kwargs
        env_vars = upload_kwargs.get("serving_container_environment_variables", {})
        assert env_vars["MODEL_ID"] == "meta-llama/Llama-3.3-8B-Instruct"
        assert env_vars["HF_TOKEN"] == "hf_test_deploy"

    def test_cleanup_base_endpoint(self):
        """cleanup_base_endpoint must undeploy and delete the endpoint."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )

        mock_endpoint = mock.MagicMock()
        trainer._base_endpoint = mock_endpoint

        trainer.cleanup_base_endpoint()

        mock_endpoint.undeploy_all.assert_called_once()
        mock_endpoint.delete.assert_called_once()
        assert trainer._base_endpoint is None

    def test_cleanup_is_idempotent(self):
        """Calling cleanup when no endpoint exists must not raise."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )
        assert trainer._base_endpoint is None

        # Should not raise
        trainer.cleanup_base_endpoint()
        assert trainer._base_endpoint is None


# ---------------------------------------------------------------------------
# Test Group 5: Artifact verification with Vertex AI metadata
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
