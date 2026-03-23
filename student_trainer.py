"""
student_trainer.py
==================
Fine-tunes a Small Language Model (SLM) on a curriculum produced by the teacher.

Supports two backends:
  - ``"local"``    – PyTorch + PEFT LoRA/QLoRA on a local GPU
  - ``"vertex_ai"`` – Google Vertex AI open-model tuning (cloud-based)

Both backends expose the same interface: ``train()``, ``generate()``, ``save()``.
The backend is selected via ``student.backend`` in the config YAML.

Typical usage
-------------
>>> from student_trainer import create_student_trainer
>>> trainer = create_student_trainer(config)
>>> trainer.train(curriculum_dataset)
>>> trainer.save("outputs/student_adapter")
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: str) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _build_dataset(examples: list[dict[str, Any]]):
    """Convert curriculum examples to a HuggingFace Dataset.

    Supports two formats:
      1. **EXAM_EVALUATION format** (question/answer dicts from the teacher's
         remediation dataset) – converted using the TASK_PROMPT template.
      2. **Legacy format** (instruction/input/output dicts) – kept for
         backwards compatibility.
    """
    from datasets import Dataset  # type: ignore

    records = []
    for ex in examples:
        if "question" in ex and "answer" in ex:
            text = _format_qa_example(ex)
        else:
            text = _format_instruction_example(ex)
        records.append({"text": text})
    return Dataset.from_list(records)


def _format_qa_example(ex: dict[str, Any]) -> str:
    """Format a question/answer pair as training text."""
    question_data = ex["question"]
    answer_data = ex["answer"]

    input_json = json.dumps(question_data, indent=2)
    response_json = json.dumps(answer_data)

    return (
        f"You are given a Hybrid Analysis malware detonation report "
        f"and a multiple-choice question.\n\n"
        f"Your job is to select ALL correct answer options and ONLY "
        f"the correct answer options.\n"
        f"Base your answer only on evidence grounded in the detonation "
        f"report and the question.\n\n"
        f"[INPUT_JSON]\n{input_json}\n\n"
        f"### Response:\n{response_json}"
    )


def _format_instruction_example(ex: dict[str, Any]) -> str:
    """Format a legacy instruction/input/output example as training text."""
    instruction = ex.get("instruction", "")
    context = ex.get("input", "")
    response = ex.get("output", "")
    if context:
        return (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{context}\n\n"
            f"### Response:\n{response}"
        )
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"### Response:\n{response}"
    )


def _curriculum_to_jsonl(examples: list[dict[str, Any]]) -> str:
    """Convert curriculum examples to JSONL format for Vertex AI tuning."""
    lines = []
    for ex in examples:
        if "question" in ex and "answer" in ex:
            text = _format_qa_example(ex)
        else:
            text = _format_instruction_example(ex)
        # Vertex AI expects {"input_text": ..., "output_text": ...} or
        # {"text_input": ..., "output": ...} depending on the model.
        # Use the generic text format for open model tuning.
        lines.append(json.dumps({"input_text": text}))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_student_trainer(config: dict[str, Any]) -> "StudentTrainer":
    """Create the appropriate student trainer based on config.

    Parameters
    ----------
    config:
        Full config dict.  ``config["student"]["backend"]`` selects the
        backend (``"local"`` or ``"vertex_ai"``).  Defaults to ``"local"``.
    """
    student_cfg = config.get("student", {})
    backend = student_cfg.get("backend", "local")

    if backend == "vertex_ai":
        return VertexAIStudentTrainer.from_dict(config)
    return StudentTrainer.from_dict(config)


# ---------------------------------------------------------------------------
# StudentTrainer (local backend)
# ---------------------------------------------------------------------------

class StudentTrainer:
    """Fine-tunes an SLM locally with LoRA or QLoRA.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model id or local path for the base student model.
    output_dir:
        Where to save the trained LoRA adapter.
    use_4bit:
        Load the base model in 4-bit (QLoRA) to reduce VRAM.
    lora_r:
        LoRA rank.
    lora_alpha:
        LoRA scaling factor.
    lora_dropout:
        Dropout applied to LoRA layers.
    learning_rate:
        AdamW learning rate.
    num_train_epochs:
        Number of fine-tuning epochs.
    per_device_train_batch_size:
        Batch size per GPU.
    gradient_accumulation_steps:
        Steps to accumulate before an optimizer update.
    max_seq_length:
        Maximum token length for training examples.
    """

    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str = "outputs/student_adapter",
        use_4bit: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        learning_rate: float = 2e-4,
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        max_seq_length: int = 2048,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.output_dir = output_dir
        self.use_4bit = use_4bit
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.per_device_train_batch_size = per_device_train_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_seq_length = max_seq_length

        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str) -> "StudentTrainer":
        """Construct a :class:`StudentTrainer` from a YAML config file."""
        cfg = _load_yaml(config_path)
        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "StudentTrainer":
        """Construct a :class:`StudentTrainer` from a config dict."""
        student_cfg = cfg.get("student", {})
        training_cfg = cfg.get("training", {})
        lora_cfg = cfg.get("lora", {})

        return cls(
            model_name_or_path=student_cfg.get("model_name_or_path", "meta-llama/Llama-3.3-8B-Instruct"),
            output_dir=cfg.get("output_dir", "outputs/student_adapter"),
            use_4bit=training_cfg.get("use_4bit", True),
            lora_r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            learning_rate=training_cfg.get("learning_rate", 2e-4),
            num_train_epochs=training_cfg.get("num_train_epochs", 3),
            per_device_train_batch_size=training_cfg.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 4),
            max_seq_length=training_cfg.get("max_seq_length", 2048),
        )

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model_and_tokenizer(self):
        """Lazy-load the base model and tokenizer (with optional 4-bit quantisation)."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        logger.info("Loading tokenizer from %s", self.model_name_or_path)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
        )
        pad_token_added = False
        if self._tokenizer.pad_token is None:
            self._tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            pad_token_added = True

        logger.info(
            "Loading model from %s (4-bit=%s)", self.model_name_or_path, self.use_4bit
        )
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            self._model = prepare_model_for_kbit_training(self._model)
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name_or_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        if pad_token_added:
            self._model.resize_token_embeddings(len(self._tokenizer))

        lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self._model = get_peft_model(self._model, lora_config)
        self._model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, curriculum: list[dict[str, Any]]) -> None:
        """Fine-tune the student SLM on *curriculum*."""
        from trl import SFTTrainer  # type: ignore
        from transformers import TrainingArguments

        if not curriculum:
            raise ValueError("curriculum must contain at least one example")

        if self._model is None:
            self._load_model_and_tokenizer()

        dataset = _build_dataset(curriculum)
        logger.info("Training on %d examples", len(dataset))

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            fp16=False,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=self._model,
            train_dataset=dataset,
            tokenizer=self._tokenizer,
            args=training_args,
            max_seq_length=self.max_seq_length,
            dataset_text_field="text",
        )
        trainer.train()
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Run inference with the (possibly fine-tuned) student model."""
        import torch

        if self._model is None:
            self._load_model_and_tokenizer()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )
        decoded = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return decoded[len(prompt):].strip()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        """Save the LoRA adapter weights."""
        save_path = path or self.output_dir
        os.makedirs(save_path, exist_ok=True)
        if self._model is not None:
            self._model.save_pretrained(save_path)
            logger.info("Adapter saved to %s", save_path)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(save_path)

    def load_adapter(self, adapter_path: str) -> None:
        """Load a previously saved LoRA adapter on top of the base model."""
        from peft import PeftModel  # type: ignore

        if self._model is None:
            self._load_model_and_tokenizer()
        self._model = PeftModel.from_pretrained(self._model, adapter_path)
        logger.info("Adapter loaded from %s", adapter_path)


# ---------------------------------------------------------------------------
# VertexAIStudentTrainer (cloud backend)
# ---------------------------------------------------------------------------

class VertexAIStudentTrainer:
    """Fine-tunes an SLM via Google Vertex AI open-model tuning.

    Workflow per training round:
      1. Convert curriculum to JSONL
      2. Upload to GCS staging bucket
      3. Launch a Vertex AI supervised tuning job (LoRA / PEFT_ADAPTER)
      4. Poll until the job completes
      5. Download the adapter weights from GCS to local output_dir

    For inference, uses the Vertex AI prediction endpoint of the
    tuned model, falling back to the base model if no tuned version exists.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace model id used as the base for tuning.
    project:
        GCP project ID.
    location:
        GCP region (e.g. ``"us-central1"``).
    staging_bucket:
        GCS bucket URI for staging training data and adapters
        (e.g. ``"gs://my-bucket/smala"``).
    output_dir:
        Local directory for downloading adapter checkpoints.
    """

    # Maps HuggingFace model IDs to Vertex AI model identifiers
    _VERTEX_MODEL_MAP = {
        "meta-llama/Llama-3.3-8B-Instruct": "meta/llama-3.3-8b-instruct",
        "google/gemma-3-4b-it": "google/gemma-3-4b-it",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    }

    def __init__(
        self,
        model_name_or_path: str,
        project: str,
        location: str,
        staging_bucket: str,
        output_dir: str = "outputs/student_adapter",
        learning_rate: float = 2e-4,
        num_train_epochs: int = 3,
        lora_r: int = 16,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.project = project
        self.location = location
        self.staging_bucket = staging_bucket.rstrip("/")
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.lora_r = lora_r

        self._tuned_endpoint = None
        self._tuning_job_count = 0

        # Vertex AI endpoint for base model inference (Round 1, before fine-tuning).
        # Lazy-deployed on first generate() call when no tuned endpoint exists.
        self._base_endpoint = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VertexAIStudentTrainer":
        """Construct from a config dict."""
        student_cfg = cfg.get("student", {})
        vertex_cfg = cfg.get("vertex_ai", {})
        training_cfg = cfg.get("training", {})
        lora_cfg = cfg.get("lora", {})

        return cls(
            model_name_or_path=student_cfg.get("model_name_or_path", "meta-llama/Llama-3.3-8B-Instruct"),
            project=vertex_cfg.get("project", "") or os.environ.get("VERTEX_AI_PROJECT", ""),
            location=vertex_cfg.get("location", "") or os.environ.get("VERTEX_AI_LOCATION", "asia-southeast1"),
            staging_bucket=vertex_cfg.get("staging_bucket", "") or os.environ.get("VERTEX_AI_STAGING_BUCKET", ""),
            output_dir=cfg.get("output_dir", "outputs/student_adapter"),
            learning_rate=training_cfg.get("learning_rate", 2e-4),
            num_train_epochs=training_cfg.get("num_train_epochs", 3),
            lora_r=lora_cfg.get("r", 16),
        )

    # ------------------------------------------------------------------
    # Training (Vertex AI)
    # ------------------------------------------------------------------

    def train(self, curriculum: list[dict[str, Any]]) -> None:
        """Fine-tune via Vertex AI open-model tuning.

        1. Converts curriculum to JSONL and uploads to GCS.
        2. Creates a Vertex AI supervised tuning job with PEFT_ADAPTER mode.
        3. Polls until the job completes.
        4. Downloads the adapter to :attr:`output_dir`.
        """
        from google.cloud import aiplatform, storage

        if not curriculum:
            raise ValueError("curriculum must contain at least one example")

        if not self.project or not self.staging_bucket:
            raise ValueError(
                "vertex_ai.project and vertex_ai.staging_bucket must be set "
                "in the config for Vertex AI backend"
            )

        aiplatform.init(project=self.project, location=self.location)
        self._tuning_job_count += 1

        # --- Step 1: Upload training data to GCS ---
        jsonl_data = _curriculum_to_jsonl(curriculum)
        gcs_train_path = (
            f"{self.staging_bucket}/training_data/"
            f"curriculum_{self._tuning_job_count}.jsonl"
        )
        self._upload_to_gcs(gcs_train_path, jsonl_data)
        logger.info("Uploaded %d training examples to %s", len(curriculum), gcs_train_path)

        # --- Step 2: Create tuning job ---
        vertex_model_id = self._VERTEX_MODEL_MAP.get(
            self.model_name_or_path, self.model_name_or_path
        )
        display_name = (
            f"smala-{vertex_model_id.split('/')[-1]}-"
            f"round{self._tuning_job_count}"
        )

        logger.info(
            "Creating Vertex AI tuning job: model=%s, epochs=%d, lora_rank=%d",
            vertex_model_id, self.num_train_epochs, self.lora_r,
        )

        tuning_job = aiplatform.CustomTrainingJob(
            display_name=display_name,
            container_uri=(
                "us-docker.pkg.dev/vertex-ai/"
                "vertex-vision-model-garden-dockers/"
                "pytorch-peft-train:latest"
            ),
            model_serving_container_image_uri=(
                "us-docker.pkg.dev/vertex-ai/"
                "vertex-vision-model-garden-dockers/"
                "pytorch-peft-serve:latest"
            ),
        )

        # Build environment variables for the training container.
        # HF_TOKEN is required for gated models (e.g. Llama 3.3 8B Instruct).
        env_vars = {}
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env_vars["HF_TOKEN"] = hf_token

        # Launch the training job
        model = tuning_job.run(
            args=[
                f"--model_id={self.model_name_or_path}",
                f"--dataset_path={gcs_train_path}",
                f"--output_dir={self.staging_bucket}/adapters/{display_name}",
                f"--lora_rank={self.lora_r}",
                f"--epochs={self.num_train_epochs}",
                f"--learning_rate={self.learning_rate}",
            ],
            replica_count=1,
            machine_type="n1-standard-8",
            accelerator_type="NVIDIA_L4",
            accelerator_count=1,
            model_display_name=display_name,
            environment_variables=env_vars if env_vars else None,
        )

        logger.info("Tuning job complete: %s", model.resource_name)

        # --- Step 3: Download adapter to local directory ---
        adapter_gcs_path = f"{self.staging_bucket}/adapters/{display_name}"
        local_adapter_path = str(Path(self.output_dir) / "adapter")
        self._download_from_gcs(adapter_gcs_path, local_adapter_path)
        logger.info("Adapter downloaded to %s", local_adapter_path)

        # Store the model resource for inference
        self._tuned_endpoint = model

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _deploy_base_endpoint(self) -> None:
        """Deploy the base HuggingFace model to a Vertex AI endpoint.

        Used for inference before any fine-tuning has completed (Round 1).
        The endpoint is kept alive and reused across rounds until a tuned
        endpoint becomes available.  Works for all model families (Llama,
        Gemma, Qwen) since it uses the same PEFT serving container that
        the training pipeline produces.
        """
        from google.cloud import aiplatform

        aiplatform.init(project=self.project, location=self.location)

        display_name = (
            f"smala-base-{self.model_name_or_path.split('/')[-1]}"
        )

        logger.info(
            "Deploying base model to Vertex AI endpoint: %s", self.model_name_or_path
        )

        # Build environment variables for the serving container
        env_vars = {"MODEL_ID": self.model_name_or_path}
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env_vars["HF_TOKEN"] = hf_token

        # Upload model with the serving container
        model = aiplatform.Model.upload(
            display_name=display_name,
            serving_container_image_uri=(
                "us-docker.pkg.dev/vertex-ai/"
                "vertex-vision-model-garden-dockers/"
                "pytorch-peft-serve:latest"
            ),
            serving_container_environment_variables=env_vars,
        )

        # Deploy to an endpoint
        self._base_endpoint = model.deploy(
            machine_type="g2-standard-8",
            accelerator_type="NVIDIA_L4",
            accelerator_count=1,
            deploy_request_timeout=1800,
        )
        logger.info("Base model endpoint deployed: %s", self._base_endpoint.resource_name)

    def _predict_endpoint(self, endpoint, prompt: str, max_new_tokens: int) -> str:
        """Run a prediction against a Vertex AI endpoint."""
        response = endpoint.predict(
            instances=[{"prompt": prompt, "max_tokens": max_new_tokens}]
        )
        return response.predictions[0].get("generated_text", "")

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Run inference with the student model.

        Tries the Vertex AI tuned endpoint first (available after at least
        one training round).  Falls back to a Vertex AI-hosted base model
        endpoint — this is deployed on-demand on Round 1 before any
        fine-tuning has occurred and works for all model families (Llama,
        Gemma, Qwen).  No local GPU required.
        """
        # Try the tuned endpoint first (available after training)
        if self._tuned_endpoint is not None:
            try:
                from google.cloud import aiplatform

                aiplatform.init(project=self.project, location=self.location)
                endpoint = self._tuned_endpoint.deploy(
                    machine_type="n1-standard-4",
                    accelerator_type="NVIDIA_L4",
                    accelerator_count=1,
                )
                result = self._predict_endpoint(endpoint, prompt, max_new_tokens)
                endpoint.undeploy_all()
                return result
            except Exception as exc:
                logger.warning(
                    "Tuned endpoint inference failed, falling back to base endpoint: %s",
                    exc,
                )

        # Fallback: deploy and use the base model on Vertex AI.
        # Lazy-deployed on first call, reused across subsequent rounds.
        if self._base_endpoint is None:
            self._deploy_base_endpoint()

        return self._predict_endpoint(self._base_endpoint, prompt, max_new_tokens)

    def cleanup_base_endpoint(self) -> None:
        """Undeploy and clean up the base model endpoint.

        Called after training produces a tuned endpoint, or at experiment end.
        """
        if self._base_endpoint is not None:
            try:
                self._base_endpoint.undeploy_all()
                self._base_endpoint.delete()
                logger.info("Base model endpoint cleaned up.")
            except Exception as exc:
                logger.warning("Failed to clean up base endpoint: %s", exc)
            self._base_endpoint = None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | None = None) -> None:
        """Save adapter metadata locally.

        The actual adapter weights are already downloaded from GCS
        during :meth:`train`.  This writes a metadata file linking
        the local path to the GCS source.
        """
        save_path = path or self.output_dir
        os.makedirs(save_path, exist_ok=True)

        metadata = {
            "backend": "vertex_ai",
            "model_name_or_path": self.model_name_or_path,
            "project": self.project,
            "location": self.location,
            "staging_bucket": self.staging_bucket,
            "tuning_jobs_completed": self._tuning_job_count,
        }
        with open(os.path.join(save_path, "vertex_ai_metadata.json"), "w") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("Vertex AI metadata saved to %s", save_path)

    def load_adapter(self, adapter_path: str) -> None:
        """Load adapter metadata (Vertex AI adapters are managed in the cloud)."""
        meta_path = os.path.join(adapter_path, "vertex_ai_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                metadata = json.load(fh)
            logger.info("Loaded Vertex AI metadata from %s", meta_path)

    # ------------------------------------------------------------------
    # GCS helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _upload_to_gcs(gcs_uri: str, data: str) -> None:
        """Upload a string to a GCS URI (gs://bucket/path)."""
        from google.cloud import storage

        # Parse gs://bucket/path
        parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        blob_path = parts[1] if len(parts) > 1 else ""

        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.upload_from_string(data)

    @staticmethod
    def _download_from_gcs(gcs_prefix: str, local_dir: str) -> None:
        """Download all files under a GCS prefix to a local directory."""
        from google.cloud import storage

        parts = gcs_prefix.replace("gs://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        os.makedirs(local_dir, exist_ok=True)
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        blobs = bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            # Preserve relative path structure
            relative_path = blob.name[len(prefix):].lstrip("/")
            if not relative_path:
                continue
            local_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            logger.debug("Downloaded %s → %s", blob.name, local_path)
