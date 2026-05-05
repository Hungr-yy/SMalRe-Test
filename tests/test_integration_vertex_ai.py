"""
Integration tests for Vertex AI adapter workflow.

These tests run against real GCP infrastructure and are gated behind
the RUN_INTEGRATION_TESTS environment variable.  They will not run
in CI unless explicitly enabled.

Required environment variables:
    RUN_INTEGRATION_TESTS=1
    VERTEX_AI_PROJECT=<your-gcp-project>
    VERTEX_AI_LOCATION=<region, e.g. asia-southeast1>
    VERTEX_AI_STAGING_BUCKET=<gs://your-bucket/smala>
    GOOGLE_APPLICATION_CREDENTIALS=<path-to-service-account-key>
    HF_TOKEN=<huggingface-token>  (for gated models like Qwen)

Usage:
    RUN_INTEGRATION_TESTS=1 python -m pytest tests/test_integration_vertex_ai.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from project root so VERTEX_AI_*, HF_TOKEN etc. are available
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Skip entire module unless integration tests are enabled
pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_INTEGRATION_TESTS", "") != "1",
    reason="Integration tests disabled (set RUN_INTEGRATION_TESTS=1 to enable)",
)


def _get_config():
    """Build config from environment variables."""
    project = os.environ.get("VERTEX_AI_PROJECT", "")
    location = os.environ.get("VERTEX_AI_LOCATION", "asia-southeast1")
    staging_bucket = os.environ.get("VERTEX_AI_STAGING_BUCKET", "")

    if not project or not staging_bucket:
        pytest.skip(
            "VERTEX_AI_PROJECT and VERTEX_AI_STAGING_BUCKET must be set"
        )

    return project, location, staging_bucket


# ---------------------------------------------------------------------------
# Test Group 1: GCS upload/download round-trip
# ---------------------------------------------------------------------------

class TestGCSRoundTrip:
    """Verify that upload and download to/from GCS works for both models."""

    def test_qwen_gcs_round_trip(self):
        """Upload curriculum to GCS, download it back, verify contents."""
        project, location, staging_bucket = _get_config()
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
            project=project, location=location,
            staging_bucket=staging_bucket,
        )

        test_id = uuid.uuid4().hex[:8]
        test_data = json.dumps({"test": True, "model": "qwen", "id": test_id})
        gcs_path = f"{staging_bucket}/integration_test/{test_id}/test_file.json"

        # Upload
        trainer._upload_to_gcs(gcs_path, test_data)

        # Download
        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = f"{staging_bucket}/integration_test/{test_id}"
            trainer._download_from_gcs(prefix, tmpdir)

            downloaded = Path(tmpdir) / "test_file.json"
            assert downloaded.exists(), f"Downloaded file not found at {downloaded}"
            content = json.loads(downloaded.read_text())
            assert content["id"] == test_id
            assert content["model"] == "qwen"

        # Cleanup
        _cleanup_gcs(staging_bucket, f"integration_test/{test_id}")

    def test_gemma_gcs_round_trip(self):
        """Upload curriculum to GCS, download it back, verify contents."""
        project, location, staging_bucket = _get_config()
        from student_trainer import VertexAIStudentTrainer

        trainer = VertexAIStudentTrainer(
            model_name_or_path="google/gemma-3-4b-it",
            project=project, location=location,
            staging_bucket=staging_bucket,
        )

        test_id = uuid.uuid4().hex[:8]
        test_data = json.dumps({"test": True, "model": "gemma", "id": test_id})
        gcs_path = f"{staging_bucket}/integration_test/{test_id}/test_file.json"

        trainer._upload_to_gcs(gcs_path, test_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            prefix = f"{staging_bucket}/integration_test/{test_id}"
            trainer._download_from_gcs(prefix, tmpdir)

            downloaded = Path(tmpdir) / "test_file.json"
            assert downloaded.exists()
            content = json.loads(downloaded.read_text())
            assert content["id"] == test_id
            assert content["model"] == "gemma"

        _cleanup_gcs(staging_bucket, f"integration_test/{test_id}")


# ---------------------------------------------------------------------------
# Test Group 2: Save and metadata after simulated adapter download
# ---------------------------------------------------------------------------

class TestSaveWithRealGCS:
    """Simulate the exact flow from main.py:

        student.output_dir = adapter_path
        student.train(accumulated_curriculum)   # uploads, trains, downloads
        student.save(adapter_path)              # verifies weights, writes metadata

    We skip the actual GPU training job but replicate everything else:
    upload fake adapter weights to the exact GCS path train() would use,
    then call save() which re-downloads when local weights are missing.
    """

    def test_qwen_train_save_flow(self):
        """Qwen: simulate train() GCS output → save() re-download → verify."""
        self._run_save_test("Qwen/Qwen2.5-7B-Instruct")

    def test_gemma_train_save_flow(self):
        """Gemma: simulate train() GCS output → save() re-download → verify."""
        self._run_save_test("google/gemma-3-4b-it")

    def _run_save_test(self, model_name: str):
        project, location, staging_bucket = _get_config()
        from student_trainer import VertexAIStudentTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mimic main.py: adapter_path = round_dir / "adapter"
            adapter_path = str(Path(tmpdir) / "round_1" / "adapter")

            trainer = VertexAIStudentTrainer(
                model_name_or_path=model_name,
                project=project, location=location,
                staging_bucket=staging_bucket,
                output_dir=adapter_path,
            )

            # Simulate train() having run: set tuning_job_count so save()
            # knows the GCS path to re-download from
            trainer._tuning_job_count = 1

            # Build the exact GCS path that train() would use
            vertex_model_id = trainer._VERTEX_MODEL_MAP.get(
                model_name, model_name
            )
            display_name = (
                f"smala-{vertex_model_id.split('/')[-1]}-"
                f"round{trainer._tuning_job_count}"
            )
            adapter_gcs_path = f"{staging_bucket}/adapters/{display_name}"

            # Upload fake adapter files to the exact path train() would
            # write to — simulating what the training container produces
            _upload_bytes_to_gcs(
                f"{adapter_gcs_path}/adapter_model.safetensors",
                b"\x00" * 1024,
            )
            trainer._upload_to_gcs(
                f"{adapter_gcs_path}/adapter_config.json",
                json.dumps({
                    "base_model_name_or_path": model_name,
                    "peft_type": "LORA",
                    "r": trainer.lora_r,
                }),
            )

            # Now run save() on an empty local dir — this is what happens
            # if train()'s download failed or was interrupted.  save()
            # should detect missing weights and re-download from GCS.
            trainer.save(adapter_path)

            # Verify: adapter files downloaded by save()'s re-download
            adapter_files = (
                list(Path(adapter_path).glob("*.safetensors"))
                + list(Path(adapter_path).glob("*.bin"))
            )
            assert len(adapter_files) >= 1, (
                f"No adapter weight files found in {adapter_path} — "
                f"save() re-download from {adapter_gcs_path} failed"
            )

            # Verify: metadata written correctly
            meta_path = Path(adapter_path) / "vertex_ai_metadata.json"
            assert meta_path.exists(), "vertex_ai_metadata.json not created"

            meta = json.loads(meta_path.read_text())
            assert meta["backend"] == "vertex_ai"
            assert meta["model_name_or_path"] == model_name
            assert meta["project"] == project
            assert meta["staging_bucket"] == staging_bucket
            assert meta["tuning_jobs_completed"] == 1
            assert "adapter_model.safetensors" in meta["adapter_files"]

            # Verify: adapter_config.json also downloaded
            assert (Path(adapter_path) / "adapter_config.json").exists(), (
                "adapter_config.json not downloaded"
            )

        # Cleanup GCS
        _cleanup_gcs(staging_bucket, f"adapters/{display_name}")


# ---------------------------------------------------------------------------
# Test Group 3: Full training job (expensive — runs a real GPU job)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("RUN_FULL_TRAINING_TEST", "") != "1",
    reason="Full training test disabled (set RUN_FULL_TRAINING_TEST=1 to enable)",
)
class TestFullTrainingJob:
    """Skip to Step 4 of the distillation loop: deploy a real Vertex AI
    training job with the HuggingFace DLC + TRL SFT, fine-tune with LoRA,
    download the adapter, and verify everything works end-to-end.

    This tests the exact same code path as main.py lines 731-737:
        student.output_dir = adapter_path
        student.train(accumulated_curriculum)
        student.save(adapter_path)

    WARNING: Creates real GCP resources and incurs GPU costs (~$1-2 per test).
    Gated behind RUN_FULL_TRAINING_TEST=1.

    Usage:
        RUN_INTEGRATION_TESTS=1 RUN_FULL_TRAINING_TEST=1 \
        python -m pytest tests/test_integration_vertex_ai.py::TestFullTrainingJob -v -s
    """

    # Minimal malware analysis curriculum (same format the teacher produces)
    _CURRICULUM = [
        {
            "question": {
                "question": "What persistence mechanism does this malware use?",
                "options": [
                    "A. Registry run key modification",
                    "B. Scheduled task creation",
                    "C. DLL side-loading",
                    "D. Boot record modification",
                ],
                "detonation_report": {
                    "sha256": "abc123def456",
                    "verdict": "malicious",
                    "threat_score": 95,
                    "signatures": [
                        {"description": "Modifies HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run"},
                        {"description": "Creates file in AppData\\Roaming"},
                    ],
                },
            },
            "answer": {"correct_answers": ["A"]},
        },
        {
            "question": {
                "question": "What network behavior is observed in the detonation report?",
                "options": [
                    "A. DNS tunneling via TXT records",
                    "B. HTTP POST to C2 server",
                    "C. No network activity observed",
                    "D. SMTP exfiltration",
                ],
                "detonation_report": {
                    "sha256": "def456abc789",
                    "verdict": "malicious",
                    "threat_score": 80,
                    "signatures": [
                        {"description": "HTTP POST request to 185.x.x.x:8080"},
                        {"description": "Sends system information to remote server"},
                    ],
                },
            },
            "answer": {"correct_answers": ["B"]},
        },
        {
            "question": {
                "question": "Which evasion techniques does the malware employ?",
                "options": [
                    "A. Process injection via WriteProcessMemory",
                    "B. Anti-debugging checks",
                    "C. UPX packing",
                    "D. Reflective DLL loading",
                ],
                "detonation_report": {
                    "sha256": "789ghi012jkl",
                    "verdict": "malicious",
                    "threat_score": 90,
                    "signatures": [
                        {"description": "Calls IsDebuggerPresent"},
                        {"description": "PE file is UPX compressed"},
                        {"description": "Injects code into svchost.exe"},
                    ],
                },
            },
            "answer": {"correct_answers": ["A", "B", "C"]},
        },
    ]

    def test_qwen_full_train_and_save(self):
        """Qwen 2.5 7B: Step 4 — deploy training job, fine-tune, download adapter."""
        self._run_full_training("Qwen/Qwen2.5-7B-Instruct")

    def test_gemma_full_train_and_save(self):
        """Gemma 3 4B: Step 4 — deploy training job, fine-tune, download adapter."""
        self._run_full_training("google/gemma-3-4b-it")

    def _run_full_training(self, model_name: str):
        project, location, staging_bucket = _get_config()
        from student_trainer import VertexAIStudentTrainer, _curriculum_to_jsonl

        # Use a unique test prefix to avoid clashing with real experiment artifacts
        test_id = uuid.uuid4().hex[:8]
        test_staging_bucket = f"{staging_bucket}/integration_test/{test_id}"

        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_path = str(Path(tmpdir) / "round_1" / "adapter")

            trainer = VertexAIStudentTrainer(
                model_name_or_path=model_name,
                project=project, location=location,
                staging_bucket=test_staging_bucket,
                output_dir=adapter_path,
                num_train_epochs=1,    # minimal to save cost
                lora_r=8,              # small adapter
                max_seq_length=512,    # short to save memory
            )

            # --- Verify JSONL format before sending to Vertex AI ---
            jsonl = _curriculum_to_jsonl(self._CURRICULUM)
            lines = jsonl.strip().split("\n")
            assert len(lines) == len(self._CURRICULUM), (
                f"Expected {len(self._CURRICULUM)} JSONL lines, got {len(lines)}"
            )
            for i, line in enumerate(lines):
                parsed = json.loads(line)
                assert "messages" in parsed, (
                    f"JSONL line {i} missing 'messages' key: {list(parsed.keys())}"
                )
                messages = parsed["messages"]
                assert len(messages) == 2, (
                    f"JSONL line {i}: expected 2 messages (user+assistant), got {len(messages)}"
                )
                assert messages[0]["role"] == "user"
                assert messages[1]["role"] == "assistant"

            # --- Run real training (Step 4) ---
            print(f"\n{'='*60}")
            print(f"Starting Vertex AI training: {model_name}")
            print(f"Project: {project}, Location: {location}")
            print(f"This will take 10-30 minutes and cost ~$1-2")
            print(f"{'='*60}\n")

            trainer.train(self._CURRICULUM)

            # --- Verify adapter was downloaded ---
            adapter_files = (
                list(Path(adapter_path).glob("**/*.safetensors"))
                + list(Path(adapter_path).glob("**/*.bin"))
            )
            assert len(adapter_files) >= 1, (
                f"No adapter weight files found in {adapter_path} after training. "
                f"Contents: {list(Path(adapter_path).rglob('*'))}"
            )
            print(f"Adapter files: {[f.name for f in adapter_files]}")

            # --- Verify save() writes metadata ---
            trainer.save(adapter_path)

            meta_path = Path(adapter_path) / "vertex_ai_metadata.json"
            assert meta_path.exists(), "vertex_ai_metadata.json not created"

            meta = json.loads(meta_path.read_text())
            assert meta["backend"] == "vertex_ai"
            assert meta["model_name_or_path"] == model_name
            assert meta["project"] == project
            assert meta["staging_bucket"] == test_staging_bucket
            assert meta["tuning_jobs_completed"] == 1
            assert len(meta["adapter_files"]) >= 1

            # --- Verify adapter config was produced ---
            adapter_configs = list(Path(adapter_path).glob("**/adapter_config.json"))
            if adapter_configs:
                config = json.loads(adapter_configs[0].read_text())
                print(f"Adapter config: {json.dumps(config, indent=2)}")
                # Verify LoRA config matches what we requested
                if "r" in config:
                    assert config["r"] == 8, f"Expected lora_r=8, got {config['r']}"
                if "peft_type" in config:
                    assert config["peft_type"] == "LORA"

            print(f"\nTraining succeeded for {model_name}")
            print(f"Adapter saved to: {adapter_path}")
            print(f"Metadata: {json.dumps(meta, indent=2)}")

        # Cleanup test artifacts from GCS
        _cleanup_gcs(staging_bucket, f"integration_test/{test_id}")


# ---------------------------------------------------------------------------
# Test Group 4: Tuned inference (Round 2+ — base model + LoRA adapter)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("RUN_FULL_TRAINING_TEST", "") != "1",
    reason="Tuned inference test disabled (set RUN_FULL_TRAINING_TEST=1)",
)
class TestTunedInference:
    """Test the Round 2+ inference path: spin up a Vertex AI CustomJob
    that loads the base model + LoRA adapter and runs inference.

    This is the exact flow used by generate() after training:
        1. Train in Round 1 → adapter saved to GCS
        2. Round 2 calls generate() → routes to _tuned_inference()
        3. CustomJob loads base model + adapter, runs prompt, uploads result
        4. Result downloaded from GCS

    Requires a real adapter on GCS — runs train() first to produce one,
    then tests inference with it.

    WARNING: Creates real GCP resources and incurs GPU costs.

    Usage:
        RUN_INTEGRATION_TESTS=1 RUN_FULL_TRAINING_TEST=1 \
        python -m pytest tests/test_integration_vertex_ai.py::TestTunedInference -v -s
    """

    # Same minimal curriculum as TestFullTrainingJob
    _CURRICULUM = [
        {
            "question": {
                "question": "What persistence mechanism does this malware use?",
                "options": [
                    "A. Registry run key modification",
                    "B. Scheduled task creation",
                    "C. DLL side-loading",
                ],
                "detonation_report": {
                    "sha256": "abc123", "verdict": "malicious",
                    "signatures": [
                        {"description": "Modifies HKCU\\Software\\Microsoft\\Windows\\CurrentVersion\\Run"},
                    ],
                },
            },
            "answer": {"correct_answers": ["A"]},
        },
        {
            "question": {
                "question": "What network behavior is observed?",
                "options": [
                    "A. DNS tunneling",
                    "B. HTTP POST to C2 server",
                    "C. No network activity",
                ],
                "detonation_report": {
                    "sha256": "def456", "verdict": "malicious",
                    "signatures": [
                        {"description": "HTTP POST request to 185.x.x.x:8080"},
                    ],
                },
            },
            "answer": {"correct_answers": ["B"]},
        },
    ]

    def test_qwen_tuned_inference(self):
        """Qwen: train → batched tuned inference (same as Round 2 Step 2)."""
        self._run_tuned_inference_test("Qwen/Qwen2.5-7B-Instruct", use_qlora=True)

    def test_gemma_base_inference(self):
        """Gemma: base model inference via Vertex AI CustomJob (Round 1)."""
        self._run_base_inference_test("google/gemma-3-4b-it")

    def test_gemma_tuned_inference(self):
        """Gemma: train → batched tuned inference (same as Round 2 Step 2)."""
        self._run_tuned_inference_test("google/gemma-3-4b-it", use_qlora=False)

    def _run_base_inference_test(self, model_name: str):
        """Test Round 1 base model inference via Vertex AI CustomJob."""
        project, location, staging_bucket = _get_config()
        from student_trainer import VertexAIStudentTrainer

        test_id = uuid.uuid4().hex[:8]
        test_staging_bucket = f"{staging_bucket}/integration_test/{test_id}"

        trainer = VertexAIStudentTrainer(
            model_name_or_path=model_name,
            project=project, location=location,
            staging_bucket=test_staging_bucket,
        )

        assert trainer._adapter_gcs_path is None, "Should have no adapter for Round 1"

        print(f"\n{'='*60}")
        print(f"Base model inference: {model_name} ({len(self._TEST_PROMPTS)} prompts)")
        print(f"{'='*60}\n")

        results = trainer.generate_batch(self._TEST_PROMPTS, max_new_tokens=128)

        assert len(results) == len(self._TEST_PROMPTS)
        for i, result in enumerate(results):
            assert result is not None, f"Prompt {i} returned None"
            assert len(result) > 0, f"Prompt {i} returned empty string"
            print(f"Prompt {i}: {result[:100]}...")

        print(f"\nBase inference test passed for {model_name}")

        _cleanup_gcs(staging_bucket, f"integration_test/{test_id}")

    def _run_tuned_inference_test(self, model_name: str, use_qlora: bool = False):
        project, location, staging_bucket = _get_config()
        from student_trainer import VertexAIStudentTrainer

        test_id = uuid.uuid4().hex[:8]
        test_staging_bucket = f"{staging_bucket}/integration_test/{test_id}"

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / f"test_exp_{test_id}"
            adapter_path = str(exp_dir / "round_1" / "adapter")

            trainer = VertexAIStudentTrainer(
                model_name_or_path=model_name,
                project=project, location=location,
                staging_bucket=test_staging_bucket,
                output_dir=adapter_path,
                num_train_epochs=1,
                lora_r=8,
                max_seq_length=512,
                use_qlora_inference=use_qlora,
            )

            # --- Step 1: Train to produce an adapter ---
            print(f"\n{'='*60}")
            print(f"Step 1: Training {model_name} to produce adapter")
            print(f"{'='*60}\n")

            trainer.train(self._CURRICULUM)

            assert trainer._adapter_gcs_path is not None, (
                "Training did not set _adapter_gcs_path"
            )
            print(f"Adapter at: {trainer._adapter_gcs_path}")

            # --- Step 2: Batched tuned inference (same as Round 2 Step 2) ---
            # Multiple prompts in one CustomJob, same as main.py's
            # generate_batch() call with 18 exam questions.
            print(f"\n{'='*60}")
            print(f"Step 2: Batched tuned inference ({len(self._TEST_PROMPTS)} prompts)")
            print(f"{'='*60}\n")

            results = trainer.generate_batch(
                self._TEST_PROMPTS, max_new_tokens=128,
            )

            assert len(results) == len(self._TEST_PROMPTS), (
                f"Expected {len(self._TEST_PROMPTS)} results, got {len(results)}"
            )
            for i, result in enumerate(results):
                assert result is not None, f"Prompt {i} returned None"
                assert len(result) > 0, f"Prompt {i} returned empty string"
                print(f"Prompt {i}: {result[:100]}...")

            # --- Step 3: Verify generate() routes to batched inference ---
            print(f"\n{'='*60}")
            print(f"Step 3: Verifying single generate() also works")
            print(f"{'='*60}\n")

            single_result = trainer.generate(
                self._TEST_PROMPTS[0], max_new_tokens=128,
            )
            assert single_result is not None
            assert len(single_result) > 0
            print(f"Single generate() result: {single_result[:100]}...")

            print(f"\nTuned inference test passed for {model_name}")

        # Cleanup
        _cleanup_gcs(staging_bucket, f"integration_test/{test_id}")

    # Test prompts matching the real TASK_PROMPT format
    _TEST_PROMPTS = [
        (
            'Given this detonation report: {"sha256": "test1", "verdict": "malicious", '
            '"signatures": [{"description": "Creates scheduled task for persistence"}]}.\n\n'
            'Answer the following multi-choice question. '
            'Select ALL correct answers — one or more options may be correct.\n\n'
            'Question: What persistence mechanism is used?\n\n'
            'Options:\nA. Registry run key\nB. Scheduled task\nC. DLL injection\n\n'
            'Respond in JSON: {"correct_answers": ["A", "B", ...]}'
        ),
        (
            'Given this detonation report: {"sha256": "test2", "verdict": "malicious", '
            '"signatures": [{"description": "HTTP POST to 185.x.x.x:8080"}, '
            '{"description": "Sends system info to remote server"}]}.\n\n'
            'Answer the following multi-choice question. '
            'Select ALL correct answers — one or more options may be correct.\n\n'
            'Question: What network behavior is observed?\n\n'
            'Options:\nA. DNS tunneling\nB. HTTP C2 callback\nC. No network activity\n\n'
            'Respond in JSON: {"correct_answers": ["A", "B", ...]}'
        ),
        (
            'Given this detonation report: {"sha256": "test3", "verdict": "malicious", '
            '"signatures": [{"description": "Calls IsDebuggerPresent"}, '
            '{"description": "PE file is UPX compressed"}]}.\n\n'
            'Answer the following multi-choice question. '
            'Select ALL correct answers — one or more options may be correct.\n\n'
            'Question: Which evasion techniques are used?\n\n'
            'Options:\nA. Anti-debugging\nB. UPX packing\nC. Process hollowing\nD. Reflective DLL loading\n\n'
            'Respond in JSON: {"correct_answers": ["A", "B", ...]}'
        ),
    ]


# ---------------------------------------------------------------------------
# Test Group 5: VRAM stress test with largest reports (signature cap validation)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("RUN_FULL_TRAINING_TEST", "") != "1",
    reason="VRAM stress test disabled (set RUN_FULL_TRAINING_TEST=1)",
)
class TestVRAMStressLargestReports:
    """Validate that the signature cap keeps all reports within L4 VRAM limits.

    Loads the 5 largest real detonation reports (capped at the proposed
    signature limit), builds real TASK_PROMPT prompts, and runs batched
    inference on a single Vertex AI CustomJob. If all 5 pass without OOM,
    the cap is validated for production use.

    WARNING: Requires a trained adapter on GCS. Runs train() first.
    Costs ~$2-3 (training + inference).

    Usage:
        RUN_INTEGRATION_TESTS=1 RUN_FULL_TRAINING_TEST=1 \
        python -m pytest tests/test_integration_vertex_ai.py::TestVRAMStressLargestReports -v -s
    """

    # Proposed signature cap to validate
    SIGNATURE_CAP = 39

    # Minimal curriculum for training (same as other tests)
    _CURRICULUM = [
        {
            "question": {
                "question": "What persistence mechanism does this malware use?",
                "options": ["A. Registry run key", "B. Scheduled task", "C. DLL injection"],
                "detonation_report": {"sha256": "abc", "verdict": "malicious",
                    "signatures": [{"description": "Modifies Run key"}]},
            },
            "answer": {"correct_answers": ["A"]},
        },
        {
            "question": {
                "question": "What network behavior is observed?",
                "options": ["A. DNS tunneling", "B. HTTP C2 callback", "C. No activity"],
                "detonation_report": {"sha256": "def", "verdict": "malicious",
                    "signatures": [{"description": "HTTP POST to C2"}]},
            },
            "answer": {"correct_answers": ["B"]},
        },
    ]

    def test_largest_reports_fit_in_vram(self):
        """The 5 largest reports (sig-capped) must all pass inference without OOM."""
        project, location, staging_bucket = _get_config()
        from main import load_hybrid_analysis_reports
        from teacher_engine import load_template
        from student_trainer import VertexAIStudentTrainer

        test_id = uuid.uuid4().hex[:8]
        test_staging_bucket = f"{staging_bucket}/integration_test/{test_id}"

        # Load real reports, cap signatures, pick 5 largest
        reports = load_hybrid_analysis_reports("hybrid-analysis", truncate=True)
        for r in reports:
            sigs = r.get("signatures", [])
            if len(sigs) > self.SIGNATURE_CAP:
                r["signatures"] = sigs[:self.SIGNATURE_CAP]

        reports_by_size = sorted(reports, key=lambda r: len(json.dumps(r)), reverse=True)
        top_5 = reports_by_size[:5]

        print(f"\n{'='*60}")
        print(f"5 largest reports (signatures capped at {self.SIGNATURE_CAP}):")
        for i, r in enumerate(top_5):
            size = len(json.dumps(r))
            sigs = len(r.get("signatures", []))
            print(f"  {i+1}. {size:,} chars, {sigs} sigs, family={r.get('_family','?')}")
        print(f"{'='*60}\n")

        # Build prompts using real TASK_PROMPT template
        template = load_template("TASK_PROMPT")
        prompts = []
        for r in top_5:
            report_json = json.dumps(r)
            prompt = (template
                .replace("{REPORT}", report_json)
                .replace("{QUESTION}", "What are the key behaviors observed in this detonation report?")
                .replace("{OPTIONS}",
                    "A. Process injection\n"
                    "B. Registry modification\n"
                    "C. Network communication\n"
                    "D. File encryption\n"
                    "E. Anti-debugging"))
            prompts.append(prompt)
            print(f"Prompt length: {len(prompt):,} chars (~{len(prompt)//4:,} tokens)")

        with tempfile.TemporaryDirectory() as tmpdir:
            exp_dir = Path(tmpdir) / f"vram_test_{test_id}"
            adapter_path = str(exp_dir / "round_1" / "adapter")

            trainer = VertexAIStudentTrainer(
                model_name_or_path="Qwen/Qwen2.5-7B-Instruct",
                project=project, location=location,
                staging_bucket=test_staging_bucket,
                output_dir=adapter_path,
                num_train_epochs=1,
                lora_r=8,
                max_seq_length=512,
            )

            # Train to produce adapter
            print(f"\nTraining to produce adapter...")
            trainer.train(self._CURRICULUM)
            assert trainer._adapter_gcs_path is not None

            # Run batched inference with the 5 largest reports
            print(f"\nRunning batched inference with 5 largest reports...")
            results = trainer.generate_batch(prompts, max_new_tokens=128)

            # Verify all 5 succeeded
            assert len(results) == 5, f"Expected 5 results, got {len(results)}"
            for i, result in enumerate(results):
                assert result is not None, f"Report {i+1} returned None"
                assert len(result) > 0, f"Report {i+1} returned empty string"
                size = len(json.dumps(top_5[i]))
                print(f"Report {i+1} ({size:,} chars): {result[:80]}...")

            print(f"\nVRAM stress test PASSED — all 5 largest reports fit with cap {self.SIGNATURE_CAP}")

        _cleanup_gcs(staging_bucket, f"integration_test/{test_id}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _upload_bytes_to_gcs(gcs_uri: str, data: bytes) -> None:
    """Upload bytes to a GCS URI."""
    from google.cloud import storage

    parts = gcs_uri.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    blob_path = parts[1] if len(parts) > 1 else ""

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(data)


def _cleanup_gcs(staging_bucket: str, prefix: str) -> None:
    """Delete all blobs under a GCS prefix (test cleanup)."""
    from google.cloud import storage

    parts = staging_bucket.replace("gs://", "").split("/", 1)
    bucket_name = parts[0]
    full_prefix = f"{parts[1]}/{prefix}" if len(parts) > 1 else prefix

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blobs = list(bucket.list_blobs(prefix=full_prefix))
        for blob in blobs:
            blob.delete()
    except Exception:
        pass  # best-effort cleanup
