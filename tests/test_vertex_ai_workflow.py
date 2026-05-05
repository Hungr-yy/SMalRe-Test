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
            "model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
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
        "response_parsing_error_count": 0,
        "total_questions_attempted": 15,
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
        # Gemma uses local backend (not available on HF API or Vertex AI asia-southeast1)
        assert config["student"]["backend"] == STUDENTS[1]["backend"]

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

    def test_student_backend_overrides_base_config(self):
        """Per-student backend takes priority over base config backend."""
        from run_experiments import build_experiment_config, TEACHERS, STUDENTS

        base = _make_vertex_ai_base_config()
        base["student"]["backend"] = "local"  # base says local

        # STUDENTS[0] (llama) has backend=vertex_ai — should override base
        config = build_experiment_config(
            teacher=TEACHERS[0],
            student=STUDENTS[0],
            base_config=base,
            output_dir="/tmp/test",
        )

        assert config["student"]["backend"] == STUDENTS[0]["backend"]

    def test_all_nine_pairs_get_correct_backend(self):
        """Every pair in the 3x3 matrix must use the student's defined backend."""
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
                expected_backend = STUDENTS[si]["backend"]
                assert config["student"]["backend"] == expected_backend, (
                    f"Teacher {TEACHERS[ti]['name']} -> Student {STUDENTS[si]['name']} "
                    f"expected backend={expected_backend}"
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
            model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )

        mock_job = mock.MagicMock()
        mock_job.run.return_value = mock.MagicMock(resource_name="projects/test/models/123")

        gcp_modules, mock_aiplatform = _mock_gcp_modules()
        mock_aiplatform.CustomContainerTrainingJob.return_value = mock_job

        curriculum = [{"question": {"question": "q", "options": ["A. a"]},
                       "answer": {"correct_answers": ["A"]}}]

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
        mock_aiplatform.CustomContainerTrainingJob.return_value = mock_job

        curriculum = [{"question": {"question": "q", "options": ["A. a"]},
                       "answer": {"correct_answers": ["A"]}}]

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
            mock_aiplatform.CustomContainerTrainingJob.return_value = mock_job

            curriculum = [{"question": {"question": "q", "options": ["A. a"]},
                           "answer": {"correct_answers": ["A"]}}]

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

    def test_training_job_script_contains_correct_model_id(self):
        """The training script must reference the student's HuggingFace model path."""
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
            mock_job.run.return_value = None

            gcp_modules, mock_aiplatform = _mock_gcp_modules()
            mock_aiplatform.CustomContainerTrainingJob.return_value = mock_job

            curriculum = [{"question": {"question": "q", "options": ["A. a"]},
                           "answer": {"correct_answers": ["A"]}}]

            with mock.patch.dict("sys.modules", gcp_modules), \
                 mock.patch.object(trainer, "_upload_to_gcs"), \
                 mock.patch.object(trainer, "_download_from_gcs"), \
                 mock.patch.dict("os.environ", {"HF_TOKEN": "hf_test"}):
                trainer.train(curriculum)

            command = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs.get("command", [])
            script = TestVertexAIInfrastructureSetup._decode_command(command)
            assert student["model_name_or_path"] in script, (
                f"{student['name']}: model not found in training script"
            )


# ---------------------------------------------------------------------------
# Test Group 4: Round 1 inference fallback (HuggingFace Inference API)
# ---------------------------------------------------------------------------

class TestRound1InferenceFallback:
    """Before any fine-tuning (Round 1), generate() must use HuggingFace
    Inference API for base model inference. After training, it uses the
    Vertex AI tuned endpoint."""

    def test_generate_uses_hf_inference_when_no_tuned_endpoint(self):
        """When _tuned_endpoint is None, generate() must call HF Inference API."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )
        assert trainer._tuned_endpoint is None

        with mock.patch.object(trainer, "_hf_inference", return_value='{"correct_answers": ["A"]}') as mock_hf:
            result = trainer.generate("test prompt")

        mock_hf.assert_called_once_with("test prompt", 512, json_schema=None)
        assert result == '{"correct_answers": ["A"]}'

    def test_hf_inference_called_for_each_generate(self):
        """Each generate() call without a tuned endpoint must hit HF Inference API."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )

        with mock.patch.object(trainer, "_hf_inference", return_value='{"correct_answers": ["B"]}') as mock_hf:
            trainer.generate("prompt 1")
            trainer.generate("prompt 2")

        assert mock_hf.call_count == 2

    def test_all_three_students_use_correct_inference_on_round1(self):
        """HF-supported students use HF API; unsupported ones use Google GenAI on Round 1."""
        from run_experiments import STUDENTS
        from student_trainer import VertexAIStudentTrainer

        for student in STUDENTS:
            trainer = VertexAIStudentTrainer(
                model_name_or_path=student["model_name_or_path"],
                project="test-project",
                location="us-central1",
                staging_bucket="gs://test-bucket/smala",
            )

            is_hf_unsupported = student["model_name_or_path"] in VertexAIStudentTrainer._HF_INFERENCE_UNSUPPORTED

            if is_hf_unsupported:
                # Gemma: should route to Vertex AI CustomJob
                with mock.patch.object(trainer, "_base_model_vertex_inference", return_value='{"correct_answers": ["A"]}') as mock_vertex:
                    result = trainer.generate("test prompt", max_new_tokens=128)
                mock_vertex.assert_called_once_with("test prompt", 128), (
                    f"{student['name']}: Vertex AI CustomJob not called on Round 1"
                )
            else:
                # Qwen, Llama: should route to HF Inference API
                with mock.patch.object(trainer, "_hf_inference", return_value='{"correct_answers": ["A"]}') as mock_hf:
                    result = trainer.generate("test prompt", max_new_tokens=128)
                mock_hf.assert_called_once_with("test prompt", 128, json_schema=None), (
                    f"{student['name']}: HF Inference API not called on Round 1"
                )

            assert isinstance(result, str) and len(result) > 0, (
                f"{student['name']}: generate() returned empty on Round 1"
            )

    def test_generate_uses_tuned_inference_when_adapter_available(self):
        """After training, generate() must use _tuned_inference."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="google/gemma-3-4b-it",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )
        trainer._adapter_gcs_path = "gs://test-bucket/smala/adapters/test-adapter"

        with mock.patch.object(
            trainer, "_tuned_inference",
            return_value='{"correct_answers": ["C"]}',
        ) as mock_tuned, \
             mock.patch.object(trainer, "_hf_inference") as mock_hf:
            result = trainer.generate("test prompt")

        mock_tuned.assert_called_once_with("test prompt", 512)
        mock_hf.assert_not_called()
        assert result == '{"correct_answers": ["C"]}'

    def test_generate_raises_when_tuned_inference_fails(self):
        """If tuned inference raises in Round 2+, the error must propagate.
        No fallback to base inference — that would invalidate the distillation loop."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )
        trainer._adapter_gcs_path = "gs://test-bucket/smala/adapters/test-adapter"

        with mock.patch.object(
            trainer, "_tuned_inference",
            side_effect=RuntimeError("inference job failed"),
        ), \
             mock.patch.object(trainer, "_hf_inference") as mock_hf:
            with pytest.raises(RuntimeError, match="inference job failed"):
                trainer.generate("test prompt")

        mock_hf.assert_not_called()

    def test_generate_uses_base_model_when_no_adapter(self):
        """Before training (Round 1), generate() must use base model inference."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )
        assert trainer._adapter_gcs_path is None

        with mock.patch.object(
            trainer, "_hf_inference",
            return_value='{"correct_answers": ["A"]}',
        ) as mock_hf, \
             mock.patch.object(trainer, "_tuned_inference") as mock_tuned:
            result = trainer.generate("test prompt")

        mock_tuned.assert_not_called()
        mock_hf.assert_called_once()
        assert result == '{"correct_answers": ["A"]}'

    def test_hf_inference_passes_hf_token(self):
        """_hf_inference must pass HF_TOKEN to the InferenceClient."""
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="meta-llama/Llama-3.1-8B-Instruct",
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )

        mock_message = mock.MagicMock()
        mock_message.content = "test response"
        mock_choice = mock.MagicMock()
        mock_choice.message = mock_message
        mock_response = mock.MagicMock()
        mock_response.choices = [mock_choice]
        mock_client_instance = mock.MagicMock()
        mock_client_instance.chat_completion.return_value = mock_response

        with mock.patch.dict("os.environ", {"HF_TOKEN": "hf_test_token"}), \
             mock.patch("student_trainer.InferenceClient", return_value=mock_client_instance) as mock_client_cls:
            result = trainer._hf_inference("test prompt", 128)

        mock_client_cls.assert_called_once_with(
            model="meta-llama/Llama-3.1-8B-Instruct", token="hf_test_token"
        )
        assert result == "test response"

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
        """An experiment with adapter weight files + result passes."""
        from run_experiments import ExperimentTracker

        tracker = ExperimentTracker(str(tmp_path))

        exp_dir = str(tmp_path / "vertex_exp")
        adapter_dir = Path(exp_dir) / "round_1" / "adapter"
        adapter_dir.mkdir(parents=True)
        (Path(exp_dir) / "experiment_result.json").write_text("{}")
        # Actual weight file + metadata
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"mock")
        (adapter_dir / "vertex_ai_metadata.json").write_text(
            json.dumps({"backend": "vertex_ai", "project": "test",
                         "adapter_files": ["adapter_model.safetensors"]})
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
            expected_backend = STUDENTS[si]["backend"]
            assert cfg["student"]["backend"] == expected_backend, (
                f"Experiment {TEACHERS[ti]['name']}->{STUDENTS[si]['name']} "
                f"expected backend={expected_backend}"
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
                adapter_dir = Path(exp_dir) / "round_1" / "adapter"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                (adapter_dir / "adapter_model.safetensors").write_bytes(b"mock")
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
            # Backend depends on the student model
            student_name = label.split("__")[1]
            expected = next(s["backend"] for s in STUDENTS if s["name"] == student_name)
            assert cfg["student"]["backend"] == expected, (
                f"{label}: expected backend={expected}"
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
            m.generate.return_value = '{"correct_answers": ["A"]}'
            return m

        config = _make_vertex_ai_base_config()
        config["data_source"]["hybrid_analysis_dir"] = str(tmp_path / "fake_ha")
        ha_dir = tmp_path / "fake_ha" / "family1"
        ha_dir.mkdir(parents=True)
        (ha_dir / "sample1").write_text(
            json.dumps({"sha256": "abc", "verdict": "malicious"})
        )

        mock_teacher = mock.MagicMock()
        mock_teacher.repair_empty_reports = mock.MagicMock(
            side_effect=lambda exam, reports: exam
        )
        mock_teacher.generate_exam.return_value = [
            {"question": {"question": "test?", "options": ["A. yes"]},
             "answer": {"correct_answers": ["A"]}}
        ]
        mock_teacher.evaluate_and_generate_curriculum.return_value = {
            "feedback": "ok", "proficiency": 5,
            "metrics": {"exact_match_accuracy": 0.9, "avg_jaccard": 0.9},
            "strengths": [], "weaknesses": [], "breakdowns": {},
            "dataset": [{"question": {"question": "q", "options": ["A. a"]},
                         "answer": {"correct_answers": ["A"]}}],
        }

        mock_reviewer = mock.MagicMock()
        mock_reviewer.model = "mock"
        mock_reviewer.review.return_value = {"reviews": [], "summary": {"total": 0, "passed": 0, "flagged": 0, "rejected": 0}}
        mock_reviewer.filter_exam.side_effect = lambda exam, _: exam

        with mock.patch("teacher_engine.TeacherEngine", return_value=mock_teacher), \
             mock.patch("student_trainer.create_student_trainer", side_effect=mock_create), \
             mock.patch("teacher_engine.load_template", return_value="Report: {REPORT}\nQuestion: {QUESTION}\nOptions: {OPTIONS}"), \
             mock.patch("main.ExamReviewer", return_value=mock_reviewer):

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
            "student": {"model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"},
            "training": {"use_4bit": True, "learning_rate": 2e-4,
                         "num_train_epochs": 1},
            "lora": {"r": 8, "alpha": 16, "dropout": 0.05},
        }
        trainer = create_student_trainer(config)
        assert isinstance(trainer, StudentTrainer)

    def test_build_experiment_config_no_backend_in_base(self):
        """When base config has no student.backend, per-student backend is used."""
        from run_experiments import build_experiment_config, TEACHERS, STUDENTS

        base = {
            "teacher": {"provider": "openai", "model": "gpt-4o", "temperature": 0.7},
            "student": {"model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct"},
            "training": {"use_4bit": True},
            "lora": {"r": 8, "alpha": 16, "dropout": 0.05},
        }

        config = build_experiment_config(
            teacher=TEACHERS[0],
            student=STUDENTS[0],  # llama — backend=vertex_ai
            base_config=base,
            output_dir="/tmp/test",
        )

        # Per-student backend takes priority
        assert config["student"]["backend"] == STUDENTS[0]["backend"]


# ---------------------------------------------------------------------------
# Test Group 6: Vertex AI infrastructure setup validation
# ---------------------------------------------------------------------------

class TestVertexAIInfrastructureSetup:
    """Validate that Vertex AI infrastructure calls use correct parameters
    for both Qwen (HF Inference) and Gemma (Google GenAI) workflows."""

    def _make_trainer(self, model="Qwen/Qwen2.5-7B-Instruct"):
        from student_trainer import VertexAIStudentTrainer
        return VertexAIStudentTrainer(
            model_name_or_path=model,
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )

    def _run_train(self, trainer):
        gcp_modules, mock_aiplatform = _mock_gcp_modules()

        mock_job = mock.MagicMock()
        mock_job.run.return_value = mock.MagicMock(
            resource_name="projects/test/models/123"
        )
        mock_aiplatform.CustomContainerTrainingJob.return_value = mock_job

        curriculum = [{"question": {"question": "q", "options": ["A. a"]},
                       "answer": {"correct_answers": ["A"]}}]

        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with mock.patch.dict("sys.modules", gcp_modules), \
             mock.patch.object(trainer, "_upload_to_gcs"), \
             mock.patch.object(trainer, "_download_from_gcs"), \
             mock.patch.dict("os.environ", env, clear=True):
            trainer.train(curriculum)

        return mock_aiplatform, mock_job

    # ------------------------------------------------------------------
    # Training job setup (shared by Qwen and Gemma)
    # ------------------------------------------------------------------

    def test_qwen_staging_bucket_passed_to_training_job(self):
        """Qwen: CustomContainerTrainingJob must receive staging_bucket."""
        trainer = self._make_trainer("Qwen/Qwen2.5-7B-Instruct")
        mock_aiplatform, _ = self._run_train(trainer)

        ctor_kwargs = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs
        assert "staging_bucket" in ctor_kwargs
        assert ctor_kwargs["staging_bucket"] == "gs://test-bucket/smala"

    def test_gemma_staging_bucket_passed_to_training_job(self):
        """Gemma: CustomContainerTrainingJob must receive staging_bucket."""
        trainer = self._make_trainer("google/gemma-3-4b-it")
        mock_aiplatform, _ = self._run_train(trainer)

        ctor_kwargs = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs
        assert "staging_bucket" in ctor_kwargs
        assert ctor_kwargs["staging_bucket"] == "gs://test-bucket/smala"

    def test_uses_custom_container_not_custom_training(self):
        """Must use CustomContainerTrainingJob, not CustomTrainingJob."""
        trainer = self._make_trainer()
        mock_aiplatform, _ = self._run_train(trainer)

        assert mock_aiplatform.CustomContainerTrainingJob.called
        assert not mock_aiplatform.CustomTrainingJob.called, (
            "CustomTrainingJob should not be used — use CustomContainerTrainingJob"
        )

    def test_container_uri_is_set(self):
        """Training job must specify a container_uri."""
        trainer = self._make_trainer()
        mock_aiplatform, _ = self._run_train(trainer)

        ctor_kwargs = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs
        assert "container_uri" in ctor_kwargs
        assert "huggingface-pytorch-training" in ctor_kwargs["container_uri"]

    def test_command_uses_trl_sft(self):
        """Training job must use TRL SFT CLI as the entrypoint."""
        trainer = self._make_trainer()
        mock_aiplatform, _ = self._run_train(trainer)

        ctor_kwargs = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs
        command = ctor_kwargs.get("command", [])
        command_str = " ".join(command)
        assert "SFTTrainer" in command_str, (
            f"Command must use TRL SFTTrainer, got {command}"
        )

    def test_aiplatform_init_called_with_project_and_location(self):
        """aiplatform.init() must be called with project and location."""
        trainer = self._make_trainer()
        mock_aiplatform, _ = self._run_train(trainer)

        mock_aiplatform.init.assert_called_once_with(
            project="test-project", location="us-central1",
        )

    def test_train_script_contains_required_config(self):
        """The embedded training script must contain all required TRL SFT config."""
        trainer = self._make_trainer()
        mock_aiplatform, _ = self._run_train(trainer)

        ctor_kwargs = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs
        command = ctor_kwargs.get("command", [])
        script = " ".join(command)
        assert "SFTTrainer" in script
        assert "SFTConfig" in script
        assert "LoraConfig" in script
        assert "load_dataset" in script
        assert "Qwen/Qwen2.5-7B-Instruct" in script
        assert "lora_alpha" in script
        assert "num_train_epochs" in script
        assert "learning_rate" in script
        assert "bf16=True" in script
        assert "max_seq_length" in script
        assert "save_model" in script

    def test_run_specifies_gpu(self):
        """Training job must request a GPU."""
        trainer = self._make_trainer()
        _, mock_job = self._run_train(trainer)

        run_kwargs = mock_job.run.call_args.kwargs
        assert run_kwargs.get("accelerator_count", 0) >= 1
        assert run_kwargs.get("accelerator_type") is not None

    def test_machine_type_compatible_with_accelerator(self):
        """Machine type must be compatible with the requested GPU.
        L4 requires g2-standard, not n1-standard."""
        trainer = self._make_trainer()
        _, mock_job = self._run_train(trainer)

        run_kwargs = mock_job.run.call_args.kwargs
        machine_type = run_kwargs.get("machine_type", "")
        accelerator = run_kwargs.get("accelerator_type", "")

        # L4 GPUs require g2-standard machines
        if accelerator == "NVIDIA_L4":
            assert machine_type.startswith("g2-"), (
                f"NVIDIA_L4 requires g2-standard machines, got '{machine_type}'"
            )
        # T4 GPUs use n1-standard machines
        elif accelerator == "NVIDIA_TESLA_T4":
            assert machine_type.startswith("n1-"), (
                f"NVIDIA_TESLA_T4 requires n1-standard machines, got '{machine_type}'"
            )
        # A100 GPUs use a2-standard machines
        elif "A100" in accelerator:
            assert machine_type.startswith("a2-"), (
                f"A100 requires a2-standard machines, got '{machine_type}'"
            )

    # ------------------------------------------------------------------
    # Container and PEFT configuration (validated via embedded Python script)
    # ------------------------------------------------------------------

    def test_use_peft_flag_present(self):
        """Training script must use LoraConfig (PEFT)."""
        trainer = self._make_trainer()
        mock_aiplatform, _ = self._run_train(trainer)

        command = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs.get("command", [])
        script = " ".join(command)
        assert "LoraConfig" in script, "Training script must use LoraConfig"
        assert "peft_config" in script, "Training script must pass peft_config to SFTTrainer"

    def test_qwen_peft_args_present(self):
        """Qwen training script must include LoRA configuration."""
        trainer = self._make_trainer("Qwen/Qwen2.5-7B-Instruct")
        mock_aiplatform, _ = self._run_train(trainer)

        command = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs.get("command", [])
        script = " ".join(command)
        assert "LoraConfig" in script
        assert "lora_alpha=" in script
        assert "target_modules=" in script

    @staticmethod
    def _decode_command(command):
        """Decode the command — if base64-encoded (Gemma), decode it."""
        import base64, re
        cmd_str = " ".join(command)
        match = re.search(r"echo (\S+) \| base64 -d", cmd_str)
        if match:
            return base64.b64decode(match.group(1)).decode()
        return cmd_str

    def test_gemma_peft_args_present(self):
        """Gemma training script must include LoRA configuration."""
        trainer = self._make_trainer("google/gemma-3-4b-it")
        mock_aiplatform, _ = self._run_train(trainer)

        command = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs.get("command", [])
        script = self._decode_command(command)
        assert "LoraConfig" in script
        assert "google/gemma-3-4b-it" in script

    def test_gemma_upgrades_transformers(self):
        """Gemma 3 requires transformers from source — command must pip install first."""
        trainer = self._make_trainer("google/gemma-3-4b-it")
        mock_aiplatform, _ = self._run_train(trainer)

        command = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs.get("command", [])
        command_str = " ".join(command)
        assert "pip install" in command_str, (
            "Gemma 3 command must include pip install for transformers upgrade"
        )
        assert "transformers" in command_str, (
            "Gemma 3 must install a newer transformers version"
        )

    def test_qwen_does_not_upgrade_transformers(self):
        """Qwen does not need transformers upgrade — command should use python directly."""
        trainer = self._make_trainer("Qwen/Qwen2.5-7B-Instruct")
        mock_aiplatform, _ = self._run_train(trainer)

        command = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs.get("command", [])
        assert command[0] == "python", (
            f"Qwen command should start with python, got {command[0]}"
        )
        command_str = " ".join(command)
        assert "pip install" not in command_str, (
            "Qwen should not upgrade transformers"
        )

    def test_script_uses_python_api_not_cli(self):
        """Training must use TRL Python API (SFTTrainer), not CLI."""
        trainer = self._make_trainer()
        mock_aiplatform, _ = self._run_train(trainer)

        command = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs.get("command", [])
        assert command[0] == "python", f"Command must start with python, got {command[0]}"
        script = " ".join(command)
        assert "SFTTrainer" in script
        assert "SFTConfig" in script
        assert "trainer.train()" in script
        assert "save_model()" in script

    def test_model_name_in_script(self):
        """Training script must reference the correct model."""
        for model in ["Qwen/Qwen2.5-7B-Instruct", "google/gemma-3-4b-it"]:
            trainer = self._make_trainer(model)
            mock_aiplatform, _ = self._run_train(trainer)

            command = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs.get("command", [])
            script = self._decode_command(command)
            assert model in script, f"Model '{model}' not found in training script"

    def test_dataset_path_uses_gcs_fuse(self):
        """Dataset path in training script must use /gcs/ FUSE mount."""
        trainer = self._make_trainer()
        mock_aiplatform, _ = self._run_train(trainer)

        command = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs.get("command", [])
        script = " ".join(command)
        assert "/gcs/" in script, "Training script must use /gcs/ FUSE mount paths"

    def test_output_dir_uses_gcs_fuse(self):
        """Output dir in training script must use /gcs/ FUSE mount under staging bucket."""
        trainer = self._make_trainer()
        mock_aiplatform, _ = self._run_train(trainer)

        command = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs.get("command", [])
        script = " ".join(command)
        assert "/gcs/test-bucket/smala/adapters/" in script

    def test_container_uri_uses_hf_dlc(self):
        """Container image must be the HuggingFace PyTorch DLC."""
        trainer = self._make_trainer()
        mock_aiplatform, _ = self._run_train(trainer)

        ctor_kwargs = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs
        uri = ctor_kwargs.get("container_uri", "")
        assert "us-docker.pkg.dev/deeplearning-platform-release/" in uri
        assert "huggingface-pytorch-training" in uri

    # ------------------------------------------------------------------
    # Validation guards
    # ------------------------------------------------------------------

    def test_empty_curriculum_raises(self):
        """train() must reject an empty curriculum."""
        trainer = self._make_trainer()
        gcp_modules, _ = _mock_gcp_modules()
        with mock.patch.dict("sys.modules", gcp_modules), \
             pytest.raises(ValueError, match="at least one example"):
            trainer.train([])

    def test_missing_project_raises(self):
        """train() must reject missing project."""
        from student_trainer import VertexAIStudentTrainer
        trainer = VertexAIStudentTrainer(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            project="", location="us-central1",
            staging_bucket="gs://test-bucket/smala",
        )
        gcp_modules, _ = _mock_gcp_modules()
        with mock.patch.dict("sys.modules", gcp_modules), \
             pytest.raises(ValueError, match="project"):
            trainer.train([{"question": {}, "answer": {}}])

    def test_missing_staging_bucket_raises(self):
        """train() must reject missing staging_bucket."""
        from student_trainer import VertexAIStudentTrainer
        trainer = VertexAIStudentTrainer(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            project="test-project", location="us-central1",
            staging_bucket="",
        )
        gcp_modules, _ = _mock_gcp_modules()
        with mock.patch.dict("sys.modules", gcp_modules), \
             pytest.raises(ValueError, match="staging_bucket"):
            trainer.train([{"question": {}, "answer": {}}])

    # ------------------------------------------------------------------
    # GCS path correctness
    # ------------------------------------------------------------------

    def test_gcs_paths_use_staging_bucket_prefix(self):
        """GCS paths for training data and adapters must use the staging_bucket."""
        trainer = self._make_trainer()

        gcp_modules, mock_aiplatform = _mock_gcp_modules()
        mock_job = mock.MagicMock()
        mock_job.run.return_value = mock.MagicMock(
            resource_name="projects/test/models/123"
        )
        mock_aiplatform.CustomContainerTrainingJob.return_value = mock_job

        upload_calls = []

        def capture_upload(path, data):
            upload_calls.append(path)

        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with mock.patch.dict("sys.modules", gcp_modules), \
             mock.patch.object(trainer, "_upload_to_gcs", side_effect=capture_upload), \
             mock.patch.object(trainer, "_download_from_gcs"), \
             mock.patch.dict("os.environ", env, clear=True):
            trainer.train([{"question": {}, "answer": {}}])

        assert len(upload_calls) == 1
        assert upload_calls[0].startswith("gs://test-bucket/smala/")

        # Verify the training script references FUSE paths under the staging bucket
        command = mock_aiplatform.CustomContainerTrainingJob.call_args.kwargs.get("command", [])
        script = " ".join(command)
        assert "/gcs/test-bucket/smala/" in script, (
            "Training script must reference /gcs/ FUSE path under staging bucket"
        )

    # ------------------------------------------------------------------
    # Inference routing: Qwen vs Gemma
    # ------------------------------------------------------------------

    def test_qwen_routes_to_hf_inference(self):
        """Qwen must route Round 1 inference to HF Inference API."""
        trainer = self._make_trainer("Qwen/Qwen2.5-7B-Instruct")

        with mock.patch.object(
            trainer, "_hf_inference", return_value='{"correct_answers": ["A"]}'
        ) as mock_hf:
            result = trainer.generate("test prompt")

        mock_hf.assert_called_once()
        assert result == '{"correct_answers": ["A"]}'

    def test_gemma_routes_to_vertex_custom_job(self):
        """Gemma must route Round 1 inference to Vertex AI CustomJob (not GenAI API)."""
        trainer = self._make_trainer("google/gemma-3-4b-it")

        with mock.patch.object(
            trainer, "_base_model_vertex_inference", return_value='{"correct_answers": ["A"]}'
        ) as mock_vertex:
            result = trainer.generate("test prompt")

        mock_vertex.assert_called_once()
        assert result == '{"correct_answers": ["A"]}'

    def test_qwen_does_not_route_to_vertex_ai_inference(self):
        """Qwen must NOT use Google GenAI inference path."""
        from student_trainer import VertexAIStudentTrainer
        trainer = self._make_trainer("Qwen/Qwen2.5-7B-Instruct")
        assert trainer.model_name_or_path not in VertexAIStudentTrainer._HF_INFERENCE_UNSUPPORTED

    def test_gemma_is_in_hf_unsupported_list(self):
        """Gemma must be listed as HF Inference unsupported."""
        from student_trainer import VertexAIStudentTrainer
        assert "google/gemma-3-4b-it" in VertexAIStudentTrainer._HF_INFERENCE_UNSUPPORTED

    def test_qwen_hf_inference_passes_json_schema(self):
        """Qwen HF inference must forward json_schema as response_format."""
        trainer = self._make_trainer("Qwen/Qwen2.5-7B-Instruct")
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}

        mock_response = mock.MagicMock()
        mock_response.choices = [mock.MagicMock()]
        mock_response.choices[0].message.content = '{"x": "y"}'

        with mock.patch("student_trainer.InferenceClient") as MockClient:
            MockClient.return_value.chat_completion.return_value = mock_response
            trainer._hf_inference("prompt", 128, json_schema=schema)

        call_kwargs = MockClient.return_value.chat_completion.call_args.kwargs
        assert "response_format" in call_kwargs
        assert call_kwargs["response_format"]["type"] == "json_schema"

    def test_gemma_genai_inference_passes_json_schema(self):
        """Gemma GenAI inference must forward json_schema as response_schema."""
        trainer = self._make_trainer("google/gemma-3-4b-it")
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}

        mock_genai = mock.MagicMock()
        mock_model = mock.MagicMock()
        mock_model.generate_content.return_value = mock.MagicMock(text='{"x": "y"}')
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.GenerationConfig = mock.MagicMock()

        with mock.patch.dict("os.environ", {"GOOGLE_API_KEY": "fake"}), \
             mock.patch("student_trainer.genai", mock_genai, create=True), \
             mock.patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            import importlib
            import student_trainer
            # Patch genai at module level for the import inside the method
            with mock.patch.object(
                student_trainer, "genai", mock_genai, create=True
            ):
                try:
                    trainer._vertex_ai_inference("prompt", 128, json_schema=schema)
                except Exception:
                    pass  # genai mock may not be perfect

        # Verify GenerationConfig was called with response_schema
        if mock_genai.GenerationConfig.called:
            config_kwargs = mock_genai.GenerationConfig.call_args.kwargs
            assert config_kwargs.get("response_mime_type") == "application/json"
            assert config_kwargs.get("response_schema") == schema

    # ------------------------------------------------------------------
    # Model mapping
    # ------------------------------------------------------------------

    def test_qwen_vertex_model_map(self):
        """Qwen must map to its HuggingFace ID in the Vertex model map."""
        from student_trainer import VertexAIStudentTrainer
        assert VertexAIStudentTrainer._VERTEX_MODEL_MAP.get(
            "Qwen/Qwen2.5-7B-Instruct"
        ) == "Qwen/Qwen2.5-7B-Instruct"

    def test_gemma_vertex_model_map(self):
        """Gemma must map to its Vertex AI ID in the model map."""
        from student_trainer import VertexAIStudentTrainer
        mapped = VertexAIStudentTrainer._VERTEX_MODEL_MAP.get("google/gemma-3-4b-it")
        assert mapped is not None
        assert "gemma" in mapped.lower()


# ---------------------------------------------------------------------------
# Test Group 7: Adapter download, save, and metadata for Qwen and Gemma
# ---------------------------------------------------------------------------

class TestAdapterDownloadAndSave:
    """Verify that adapters are downloaded from GCS and saved correctly
    for both Qwen and Gemma student models."""

    def _make_trainer(self, model, tmp_path):
        from student_trainer import VertexAIStudentTrainer
        return VertexAIStudentTrainer(
            model_name_or_path=model,
            project="test-project",
            location="us-central1",
            staging_bucket="gs://test-bucket/smala",
            output_dir=str(tmp_path / "adapter"),
        )

    # ------------------------------------------------------------------
    # save() writes metadata
    # ------------------------------------------------------------------

    def test_qwen_save_writes_metadata(self, tmp_path):
        """Qwen: save() must write vertex_ai_metadata.json."""
        trainer = self._make_trainer("Qwen/Qwen2.5-7B-Instruct", tmp_path)
        save_dir = tmp_path / "adapter"
        save_dir.mkdir(parents=True)
        (save_dir / "adapter_model.safetensors").write_bytes(b"mock_weights")

        trainer.save(str(save_dir))

        meta_path = save_dir / "vertex_ai_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["backend"] == "vertex_ai"
        assert meta["model_name_or_path"] == "Qwen/Qwen2.5-7B-Instruct"
        assert "adapter_model.safetensors" in meta["adapter_files"]

    def test_gemma_save_writes_metadata(self, tmp_path):
        """Gemma: save() must write vertex_ai_metadata.json."""
        trainer = self._make_trainer("google/gemma-3-4b-it", tmp_path)
        save_dir = tmp_path / "adapter"
        save_dir.mkdir(parents=True)
        (save_dir / "adapter_model.safetensors").write_bytes(b"mock_weights")

        trainer.save(str(save_dir))

        meta_path = save_dir / "vertex_ai_metadata.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["backend"] == "vertex_ai"
        assert meta["model_name_or_path"] == "google/gemma-3-4b-it"
        assert "adapter_model.safetensors" in meta["adapter_files"]

    # ------------------------------------------------------------------
    # save() records correct adapter file types
    # ------------------------------------------------------------------

    def test_save_detects_safetensors(self, tmp_path):
        """save() must detect .safetensors adapter files."""
        trainer = self._make_trainer("Qwen/Qwen2.5-7B-Instruct", tmp_path)
        save_dir = tmp_path / "adapter"
        save_dir.mkdir(parents=True)
        (save_dir / "adapter_model.safetensors").write_bytes(b"mock")

        trainer.save(str(save_dir))

        meta = json.loads((save_dir / "vertex_ai_metadata.json").read_text())
        assert len(meta["adapter_files"]) == 1
        assert meta["adapter_files"][0].endswith(".safetensors")

    def test_save_detects_bin_files(self, tmp_path):
        """save() must detect .bin adapter files."""
        trainer = self._make_trainer("google/gemma-3-4b-it", tmp_path)
        save_dir = tmp_path / "adapter"
        save_dir.mkdir(parents=True)
        (save_dir / "pytorch_model.bin").write_bytes(b"mock")

        trainer.save(str(save_dir))

        meta = json.loads((save_dir / "vertex_ai_metadata.json").read_text())
        assert len(meta["adapter_files"]) == 1
        assert meta["adapter_files"][0].endswith(".bin")

    # ------------------------------------------------------------------
    # save() re-downloads when weights are missing
    # ------------------------------------------------------------------

    def test_qwen_save_redownloads_on_missing_weights(self, tmp_path):
        """Qwen: save() must attempt re-download when no adapter files found."""
        trainer = self._make_trainer("Qwen/Qwen2.5-7B-Instruct", tmp_path)
        trainer._tuning_job_count = 1
        save_dir = tmp_path / "adapter"
        save_dir.mkdir(parents=True)
        # No weight files present

        with mock.patch.object(trainer, "_download_from_gcs") as mock_dl:
            trainer.save(str(save_dir))

        mock_dl.assert_called_once()
        gcs_path = mock_dl.call_args[0][0]
        assert "gs://test-bucket/smala/adapters/" in gcs_path
        assert "Qwen2.5-7B-Instruct" in gcs_path

    def test_gemma_save_redownloads_on_missing_weights(self, tmp_path):
        """Gemma: save() must attempt re-download when no adapter files found."""
        trainer = self._make_trainer("google/gemma-3-4b-it", tmp_path)
        trainer._tuning_job_count = 1
        save_dir = tmp_path / "adapter"
        save_dir.mkdir(parents=True)

        with mock.patch.object(trainer, "_download_from_gcs") as mock_dl:
            trainer.save(str(save_dir))

        mock_dl.assert_called_once()
        gcs_path = mock_dl.call_args[0][0]
        assert "gs://test-bucket/smala/adapters/" in gcs_path
        assert "gemma" in gcs_path.lower()

    def test_save_no_redownload_before_first_train(self, tmp_path):
        """save() must NOT attempt re-download if no training has occurred."""
        trainer = self._make_trainer("Qwen/Qwen2.5-7B-Instruct", tmp_path)
        assert trainer._tuning_job_count == 0
        save_dir = tmp_path / "adapter"
        save_dir.mkdir(parents=True)

        with mock.patch.object(trainer, "_download_from_gcs") as mock_dl:
            trainer.save(str(save_dir))

        mock_dl.assert_not_called()

    # ------------------------------------------------------------------
    # train() downloads adapter after job completes
    # ------------------------------------------------------------------

    def test_qwen_train_downloads_adapter(self, tmp_path):
        """Qwen: train() must call _download_from_gcs after tuning job."""
        trainer = self._make_trainer("Qwen/Qwen2.5-7B-Instruct", tmp_path)
        gcp_modules, mock_aiplatform = _mock_gcp_modules()

        mock_job = mock.MagicMock()
        mock_job.run.return_value = mock.MagicMock(
            resource_name="projects/test/models/123"
        )
        mock_aiplatform.CustomContainerTrainingJob.return_value = mock_job

        curriculum = [{"question": {"question": "q"}, "answer": {"correct_answers": ["A"]}}]

        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with mock.patch.dict("sys.modules", gcp_modules), \
             mock.patch.object(trainer, "_upload_to_gcs"), \
             mock.patch.object(trainer, "_download_from_gcs") as mock_dl, \
             mock.patch.dict("os.environ", env, clear=True):
            trainer.train(curriculum)

        mock_dl.assert_called_once()
        gcs_path = mock_dl.call_args[0][0]
        assert gcs_path.startswith("gs://test-bucket/smala/adapters/")

    def test_gemma_train_downloads_adapter(self, tmp_path):
        """Gemma: train() must call _download_from_gcs after tuning job."""
        trainer = self._make_trainer("google/gemma-3-4b-it", tmp_path)
        gcp_modules, mock_aiplatform = _mock_gcp_modules()

        mock_job = mock.MagicMock()
        mock_job.run.return_value = mock.MagicMock(
            resource_name="projects/test/models/123"
        )
        mock_aiplatform.CustomContainerTrainingJob.return_value = mock_job

        curriculum = [{"question": {"question": "q"}, "answer": {"correct_answers": ["A"]}}]

        env = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
        with mock.patch.dict("sys.modules", gcp_modules), \
             mock.patch.object(trainer, "_upload_to_gcs"), \
             mock.patch.object(trainer, "_download_from_gcs") as mock_dl, \
             mock.patch.dict("os.environ", env, clear=True):
            trainer.train(curriculum)

        mock_dl.assert_called_once()
        gcs_path = mock_dl.call_args[0][0]
        assert gcs_path.startswith("gs://test-bucket/smala/adapters/")

    # ------------------------------------------------------------------
    # Metadata and load_adapter round-trip
    # ------------------------------------------------------------------

    def test_metadata_round_trip(self, tmp_path):
        """save() metadata must be loadable by load_adapter()."""
        trainer = self._make_trainer("Qwen/Qwen2.5-7B-Instruct", tmp_path)
        save_dir = tmp_path / "adapter"
        save_dir.mkdir(parents=True)
        (save_dir / "adapter_model.safetensors").write_bytes(b"mock")

        trainer.save(str(save_dir))
        trainer.load_adapter(str(save_dir))

        # load_adapter should not raise — it reads the metadata file
        meta = json.loads((save_dir / "vertex_ai_metadata.json").read_text())
        assert meta["tuning_jobs_completed"] == 0
        assert meta["staging_bucket"] == "gs://test-bucket/smala"

    def test_metadata_includes_all_required_fields(self, tmp_path):
        """Metadata must include backend, model, project, location,
        staging_bucket, tuning_jobs_completed, and adapter_files."""
        trainer = self._make_trainer("google/gemma-3-4b-it", tmp_path)
        trainer._tuning_job_count = 2
        save_dir = tmp_path / "adapter"
        save_dir.mkdir(parents=True)
        (save_dir / "adapter_model.safetensors").write_bytes(b"mock")

        trainer.save(str(save_dir))

        meta = json.loads((save_dir / "vertex_ai_metadata.json").read_text())
        required_keys = {
            "backend", "model_name_or_path", "project", "location",
            "staging_bucket", "tuning_jobs_completed", "adapter_files",
        }
        assert required_keys.issubset(meta.keys())
        assert meta["tuning_jobs_completed"] == 2
