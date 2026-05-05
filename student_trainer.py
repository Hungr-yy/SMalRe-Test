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
from huggingface_hub import InferenceClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Retry helper (CyberSOCEval-aligned exponential backoff)
# ---------------------------------------------------------------------------

def _retry_with_backoff(
    fn,
    *args,
    max_retries: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 30.0,
    **kwargs,
):
    """Call *fn* with exponential backoff on transient failures.

    Delay schedule: ``base_delay * 2^attempt``, capped at *max_delay*.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = min(base_delay * (2 ** attempt), max_delay)
                logger.warning(
                    "Inference call failed (attempt %d/%d): %s — retrying in %.1fs.",
                    attempt + 1, max_retries + 1, exc, wait,
                )
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


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
    """Format a question/answer pair as training text.

    Mirrors the TASK_PROMPT template (aligned with CyberSOCEval) so that
    the SLM sees the same format at training and inference time.
    The detonation report, question, and options are separated into
    distinct readable sections rather than buried in a JSON blob.
    """
    question_data = ex["question"]
    answer_data = ex["answer"]

    report_json = json.dumps(question_data.get("detonation_report", {}), indent=2)
    question_text = question_data.get("question", "")
    options_text = "\n".join(question_data.get("options", []))
    response_json = json.dumps(answer_data)

    return (
        f"Given this detonation report: {report_json}.\n\n"
        f"Answer the following multi-choice question. "
        f"Select ALL correct answers — one or more options may be correct.\n\n"
        f"Question: {question_text}\n\n"
        f"Options:\n{options_text}\n\n"
        f"You need to return the list of correct answers. "
        f"Respond in a json with the following structure:\n"
        f'{{\n    "correct_answers": string[] '
        f"// The list of the letters corresponding to the correct answers, "
        f"just the letters\n}}\n\n"
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
    """Convert curriculum examples to JSONL format for TRL SFT training.

    TRL SFT trainer expects chat-style messages or a single "text" field.
    We use the chat messages format for better instruction following:
    {"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}
    """
    lines = []
    for ex in examples:
        if "question" in ex and "answer" in ex:
            question_data = ex["question"]
            answer_data = ex["answer"]

            report_json = json.dumps(question_data.get("detonation_report", {}), indent=2)
            question_text = question_data.get("question", "")
            options_text = "\n".join(question_data.get("options", []))
            response_json = json.dumps(answer_data)

            user_content = (
                f"Given this detonation report: {report_json}.\n\n"
                f"Answer the following multi-choice question. "
                f"Select ALL correct answers — one or more options may be correct.\n\n"
                f"Question: {question_text}\n\n"
                f"Options:\n{options_text}\n\n"
                f"Respond in JSON: {{\"correct_answers\": [\"A\", \"B\", ...]}}"
            )
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response_json},
            ]
        else:
            instruction = ex.get("instruction", "")
            context = ex.get("input", "")
            response = ex.get("output", "")
            user_content = f"{instruction}\n\n{context}" if context else instruction
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": response},
            ]
        lines.append(json.dumps({"messages": messages}))
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
            model_name_or_path=student_cfg.get("model_name_or_path", "meta-llama/Llama-3.1-8B-Instruct"),
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

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        json_schema: dict | None = None,
    ) -> str:
        """Run inference with the (possibly fine-tuned) student model.

        Parameters
        ----------
        json_schema:
            Optional JSON schema for guided decoding.  When provided and the
            ``outlines`` library is available, token generation is constrained
            to produce valid JSON matching the schema.
        """
        import torch

        if self._model is None:
            self._load_model_and_tokenizer()

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }

        # Guided decoding via outlines (if available and schema provided)
        if json_schema is not None:
            try:
                from outlines.integrations.transformers import JSONPrefixAllowedTokens
                prefix_allowed = JSONPrefixAllowedTokens(
                    schema=json_schema,
                    tokenizer_or_pipe=self._tokenizer,
                )
                generate_kwargs["prefix_allowed_tokens_fn"] = prefix_allowed
                logger.debug("Guided decoding enabled (outlines).")
            except ImportError:
                logger.debug("outlines not installed; skipping guided decoding.")
            except Exception as exc:
                logger.debug("Guided decoding setup failed: %s; skipping.", exc)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **generate_kwargs)
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
        "meta-llama/Llama-3.1-8B-Instruct": "meta/llama-3.1-8b-instruct",
        "google/gemma-3-4b-it": "gemma-3-4b-it",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen/Qwen2.5-7B-Instruct",
    }

    # Models that need a newer transformers version than what's in the
    # HF DLC container (4.48). Gemma 3 requires 4.49+.
    _NEEDS_TRANSFORMERS_UPGRADE = {
        "google/gemma-3-4b-it",
    }

    # Models that are NOT available on the HF serverless Inference API
    # and must use Vertex AI for base model inference (Round 1).
    _HF_INFERENCE_UNSUPPORTED = {
        "google/gemma-3-4b-it",
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
        max_seq_length: int = 2048,
        use_qlora_inference: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.project = project
        self.location = location
        self.staging_bucket = staging_bucket.rstrip("/")
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.lora_r = lora_r
        self._max_seq_length = max_seq_length

        self.use_qlora_inference = use_qlora_inference

        self._tuned_endpoint = None
        self._tuning_job_count = 0
        self._adapter_gcs_path: str | None = None

        # Base endpoint for models that need Vertex AI for Round 1 inference
        # (e.g. Gemma, which is not on the HF serverless Inference API).
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
            model_name_or_path=student_cfg.get("model_name_or_path", "meta-llama/Llama-3.1-8B-Instruct"),
            project=vertex_cfg.get("project", "") or os.environ.get("VERTEX_AI_PROJECT", ""),
            location=vertex_cfg.get("location", "") or os.environ.get("VERTEX_AI_LOCATION", "asia-southeast1"),
            staging_bucket=vertex_cfg.get("staging_bucket", "") or os.environ.get("VERTEX_AI_STAGING_BUCKET", ""),
            output_dir=cfg.get("output_dir", "outputs/student_adapter"),
            learning_rate=training_cfg.get("learning_rate", 2e-4),
            num_train_epochs=training_cfg.get("num_train_epochs", 3),
            lora_r=lora_cfg.get("r", 16),
            max_seq_length=training_cfg.get("max_seq_length", 2048),
            use_qlora_inference=student_cfg.get("use_qlora_inference", False),
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
        # Include output_dir basename to namespace per-experiment
        # (e.g. "gpt5.1__qwen2.5_7b" from the experiment label)
        experiment_tag = Path(self.output_dir).parent.parent.name or "default"
        display_name = (
            f"smala-{experiment_tag}-{vertex_model_id.split('/')[-1]}-"
            f"round{self._tuning_job_count}"
        )

        logger.info(
            "Creating Vertex AI tuning job: model=%s, epochs=%d, lora_rank=%d",
            vertex_model_id, self.num_train_epochs, self.lora_r,
        )

        # HuggingFace PyTorch DLC with TRL, PEFT, and Transformers
        _HF_TRAINING_CONTAINER = (
            "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/"
            "huggingface-pytorch-training-cu121.2-3.transformers.4-48"
            ".ubuntu2204.py311"
        )

        adapter_output_dir = f"{self.staging_bucket}/adapters/{display_name}"

        # Convert gs:// URIs to /gcs/ FUSE mount paths for the container.
        # Vertex AI automatically mounts GCS at /gcs/ inside the container.
        def _gcs_to_fuse(uri: str) -> str:
            return uri.replace("gs://", "/gcs/", 1) if uri.startswith("gs://") else uri

        fuse_train_path = _gcs_to_fuse(gcs_train_path)
        fuse_output_dir = _gcs_to_fuse(adapter_output_dir)

        # TRL CLI doesn't support local file paths via --dataset_name.
        # Use a Python one-liner that loads the JSONL via datasets and
        # runs SFT with TRL's Python API.
        train_script = (
            "import torch; "
            "from datasets import load_dataset; "
            "from transformers import AutoModelForCausalLM; "
            "from trl import SFTConfig, SFTTrainer; "
            "from peft import LoraConfig; "
            f"dataset = load_dataset('json', data_files='{fuse_train_path}', split='train'); "
            f"model = AutoModelForCausalLM.from_pretrained("
            f"'{self.model_name_or_path}', "
            "torch_dtype=torch.bfloat16, "
            + ("attn_implementation='eager', "
               if self.model_name_or_path in self._NEEDS_TRANSFORMERS_UPGRADE
               else "")
            + "device_map='auto'); "
            f"peft_config = LoraConfig(r={self.lora_r}, lora_alpha={self.lora_r * 2}, "
            f"lora_dropout=0.05, target_modules='all-linear', task_type='CAUSAL_LM'); "
            f"sft_config = SFTConfig("
            f"output_dir='{fuse_output_dir}', "
            f"num_train_epochs={self.num_train_epochs}, "
            f"learning_rate={self.learning_rate}, "
            f"per_device_train_batch_size=1, "
            f"gradient_accumulation_steps=8, "
            f"max_seq_length={self._max_seq_length}, "
            f"bf16=True, "
            f"gradient_checkpointing=True, "
            f"logging_steps=10, "
            f"save_strategy='epoch', "
            f"report_to='none'); "
            f"trainer = SFTTrainer("
            f"model=model, "
            f"args=sft_config, "
            f"train_dataset=dataset, "
            f"peft_config=peft_config); "
            f"trainer.train(); "
            f"trainer.save_model()"
        )

        # Gemma 3 needs transformers 4.49+; prepend pip install if needed
        # Gemma 3 needs transformers 4.51.3+; write script to file to avoid
        # shell escaping issues with multiline Python in sh -c
        if self.model_name_or_path in self._NEEDS_TRANSFORMERS_UPGRADE:
            import base64
            encoded = base64.b64encode(train_script.encode()).decode()
            train_cmd = [
                "sh", "-c",
                "pip install -q 'transformers==4.51.3' flash-attn --no-build-isolation && "
                f"echo {encoded} | base64 -d > /tmp/_train.py && "
                "python /tmp/_train.py",
            ]
        else:
            train_cmd = ["python", "-c", train_script]

        tuning_job = aiplatform.CustomContainerTrainingJob(
            display_name=display_name,
            container_uri=_HF_TRAINING_CONTAINER,
            command=train_cmd,
            staging_bucket=self.staging_bucket,
        )

        # Build environment variables for the training container.
        # HF_TOKEN is required for gated models (e.g. Qwen 2.5 7B).
        env_vars = {}
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env_vars["HF_TOKEN"] = hf_token

        # Launch the training job
        model = tuning_job.run(
            replica_count=1,
            machine_type="g2-standard-12",
            accelerator_type="NVIDIA_L4",
            accelerator_count=1,
            environment_variables=env_vars if env_vars else None,
        )

        logger.info("Tuning job complete: %s", display_name)

        # --- Step 3: Download adapter to local directory ---
        adapter_gcs_path = f"{self.staging_bucket}/adapters/{display_name}"
        self._download_from_gcs(adapter_gcs_path, self.output_dir)
        logger.info("Adapter downloaded to %s", self.output_dir)

        # Store the GCS adapter path for inference in subsequent rounds
        self._adapter_gcs_path = adapter_gcs_path

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _hf_inference(
        self,
        prompt: str,
        max_new_tokens: int,
        json_schema: dict | None = None,
    ) -> str:
        """Run inference via HuggingFace Inference API (serverless).

        Used for base model inference before any fine-tuning (Round 1).
        Much faster than deploying a Vertex AI endpoint — no cold start,
        no GPU provisioning.  Same model weights, same results.
        """
        hf_token = os.environ.get("HF_TOKEN", "") or None
        client = InferenceClient(model=self.model_name_or_path, token=hf_token)

        logger.info(
            "Running HF Inference API for base model: %s", self.model_name_or_path
        )

        extra_kwargs: dict[str, Any] = {}
        if json_schema is not None:
            extra_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": json_schema},
            }

        response = client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.7,
            **extra_kwargs,
        )
        return response.choices[0].message.content

    # Rate limiting for Google GenAI API (tokens per minute)
    _GENAI_TPM_LIMIT = 15_000  # input tokens per minute
    _genai_token_log: list[tuple[float, int]] = []  # (timestamp, token_count)

    def _wait_for_genai_quota(self, estimated_tokens: int) -> None:
        """Sleep if sending *estimated_tokens* would exceed the per-minute limit."""
        now = time.time()
        window_start = now - 60

        # Prune entries older than 1 minute
        self._genai_token_log = [
            (ts, tc) for ts, tc in self._genai_token_log if ts > window_start
        ]

        tokens_in_window = sum(tc for _, tc in self._genai_token_log)

        if tokens_in_window + estimated_tokens > self._GENAI_TPM_LIMIT:
            # Wait until the oldest entry expires from the window
            if self._genai_token_log:
                oldest_ts = self._genai_token_log[0][0]
                wait_seconds = oldest_ts - window_start + 1
            else:
                wait_seconds = 60
            logger.info(
                "GenAI rate limit: %d + %d > %d TPM. Waiting %.0fs.",
                tokens_in_window, estimated_tokens, self._GENAI_TPM_LIMIT, wait_seconds,
            )
            time.sleep(wait_seconds)

    def _vertex_ai_inference(
        self,
        prompt: str,
        max_new_tokens: int,
        json_schema: dict | None = None,
    ) -> str:
        """Run inference via Google GenAI API for base model.

        Used for models not available on the HF serverless Inference API
        (e.g. Gemma).  Calls the model through Google's generative AI API
        which hosts Gemma natively — no endpoint deployment needed.

        Includes rate limiting to stay within the per-minute token quota.
        """
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        genai.configure(api_key=api_key)

        vertex_model_id = self._VERTEX_MODEL_MAP.get(
            self.model_name_or_path, self.model_name_or_path
        )

        # Estimate input tokens (~4 chars per token)
        estimated_tokens = len(prompt) // 4
        self._wait_for_genai_quota(estimated_tokens)

        logger.info(
            "Running Google GenAI inference for base model: %s (~%d tokens)",
            vertex_model_id, estimated_tokens,
        )

        gen_config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_new_tokens,
            "temperature": 0.7,
        }
        if json_schema is not None:
            gen_config_kwargs["response_mime_type"] = "application/json"
            gen_config_kwargs["response_schema"] = json_schema

        model = genai.GenerativeModel(vertex_model_id)
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(**gen_config_kwargs),
        )

        # Log actual usage for rate tracking
        self._genai_token_log.append((time.time(), estimated_tokens))

        return response.text

    def _predict_endpoint(self, endpoint, prompt: str, max_new_tokens: int) -> str:
        """Run a prediction against a Vertex AI endpoint."""
        response = endpoint.predict(
            instances=[{"prompt": prompt, "max_tokens": max_new_tokens}]
        )
        return response.predictions[0].get("generated_text", "")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        json_schema: dict | None = None,
    ) -> str:
        """Run inference with the student model.

        After training (Rounds 2+), uses a Vertex AI CustomJob to load
        the base model + LoRA adapter and run inference on a GPU.
        For base model inference (Round 1), uses the HuggingFace Inference
        API for supported models, or Google GenAI for unsupported ones.

        Parameters
        ----------
        json_schema:
            Optional JSON schema for guided decoding (passed to HF/GenAI).
        """
        # Rounds 2+: use the fine-tuned adapter via a Vertex AI inference job.
        # If this fails, raise immediately — do NOT fall back to base
        # inference, as that would invalidate the distillation loop.
        if self._adapter_gcs_path is not None:
            return self._tuned_inference(prompt, max_new_tokens)

        # Round 1: base model inference
        # For models not on HF Inference API (e.g. Gemma), use a Vertex AI
        # CustomJob — same as tuned inference but without the adapter.
        if self.model_name_or_path in self._HF_INFERENCE_UNSUPPORTED:
            return self._base_model_vertex_inference(prompt, max_new_tokens)

        return _retry_with_backoff(
            self._hf_inference, prompt, max_new_tokens,
            json_schema=json_schema,
        )

    def _tuned_inference(self, prompt: str, max_new_tokens: int) -> str:
        """Run tuned inference for a single prompt.

        Delegates to generate_batch() with a single-item list.
        """
        results = self.generate_batch([prompt], max_new_tokens=max_new_tokens)
        return results[0]

    def _base_model_vertex_inference(self, prompt: str, max_new_tokens: int) -> str:
        """Run base model inference for a single prompt via Vertex AI CustomJob.

        Used for models not available on HF Inference API (e.g. Gemma).
        Delegates to _batched_vertex_inference without an adapter.
        """
        results = self._batched_vertex_inference(
            [prompt], max_new_tokens, adapter_path=None,
        )
        return results[0]

    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 128,
    ) -> list[str]:
        """Run batched inference with base model + LoRA adapter.

        Spins up ONE Vertex AI CustomJob that loads the model + adapter
        once, processes all prompts sequentially, and uploads results
        to GCS as a single JSON file.

        For Round 1 (no adapter), falls back to per-prompt base inference.

        Parameters
        ----------
        prompts:
            List of prompt strings to process.
        max_new_tokens:
            Maximum tokens to generate per prompt.

        Returns
        -------
        list[str]
            Generated responses, one per prompt (same order).
        """
        if not prompts:
            return []

        # Round 1: no adapter
        if self._adapter_gcs_path is None:
            if self.model_name_or_path in self._HF_INFERENCE_UNSUPPORTED:
                # Gemma etc: use batched Vertex AI CustomJob (no adapter)
                return self._batched_vertex_inference(
                    prompts, max_new_tokens, adapter_path=None,
                )
            # HF-supported models: per-prompt API calls
            results = []
            for prompt in prompts:
                results.append(self.generate(prompt, max_new_tokens=max_new_tokens))
            return results

        # Rounds 2+: batched tuned inference via a single CustomJob.
        # If this fails, raise immediately — do NOT fall back to base
        # inference, as that would invalidate the distillation loop.
        return self._batched_vertex_inference(
            prompts, max_new_tokens, adapter_path=self._adapter_gcs_path,
        )

    def _batched_vertex_inference(
        self,
        prompts: list[str],
        max_new_tokens: int,
        adapter_path: str | None = None,
    ) -> list[str]:
        """Run all prompts in a single Vertex AI CustomJob.

        Works for both base model (adapter_path=None) and tuned model
        (adapter_path=GCS path to adapter). One GPU container handles
        all prompts sequentially — no per-prompt job overhead.

        1. Upload prompts as JSONL to GCS
        2. Spin up one GPU container with base model (+ adapter if given)
        3. Process all prompts sequentially
        4. Upload results as JSONL to GCS
        5. Download and return results
        """
        from google.cloud import aiplatform, storage
        import uuid

        aiplatform.init(project=self.project, location=self.location)

        _HF_TRAINING_CONTAINER = (
            "us-docker.pkg.dev/deeplearning-platform-release/gcr.io/"
            "huggingface-pytorch-training-cu121.2-3.transformers.4-48"
            ".ubuntu2204.py311"
        )

        inference_id = uuid.uuid4().hex[:8]

        # Upload prompts to GCS as JSONL
        prompts_jsonl = "\n".join(
            json.dumps({"index": i, "prompt": p})
            for i, p in enumerate(prompts)
        )
        prompts_gcs_path = f"{self.staging_bucket}/inference/{inference_id}_prompts.jsonl"
        self._upload_to_gcs(prompts_gcs_path, prompts_jsonl)

        results_gcs_path = f"{self.staging_bucket}/inference/{inference_id}_results.jsonl"

        def _gcs_to_fuse(uri: str) -> str:
            return uri.replace("gs://", "/gcs/", 1) if uri.startswith("gs://") else uri

        fuse_prompts_path = _gcs_to_fuse(prompts_gcs_path)

        if adapter_path:
            fuse_adapter_path = _gcs_to_fuse(adapter_path)
            logger.info(
                "Batched Vertex AI inference: %d prompts, adapter=%s",
                len(prompts), adapter_path,
            )
        else:
            logger.info(
                "Batched Vertex AI inference: %d prompts, base model=%s",
                len(prompts), self.model_name_or_path,
            )

        # Build the model loading section of the script
        model_load_script = (
            f"tokenizer = AutoTokenizer.from_pretrained('{self.model_name_or_path}')\n"
        )

        # Gemma 3 needs flash_attention_2 — SDPA and eager both fail on PyTorch 2.3
        attn_arg = (
            "    attn_implementation='flash_attention_2',\n"
            if self.model_name_or_path in self._NEEDS_TRANSFORMERS_UPGRADE
            else ""
        )

        if self.use_qlora_inference:
            model_load_script += (
                "from transformers import BitsAndBytesConfig\n"
                "bnb_config = BitsAndBytesConfig(\n"
                "    load_in_4bit=True,\n"
                "    bnb_4bit_quant_type='nf4',\n"
                "    bnb_4bit_compute_dtype=torch.bfloat16,\n"
                ")\n"
                f"model = AutoModelForCausalLM.from_pretrained(\n"
                f"    '{self.model_name_or_path}',\n"
                "    quantization_config=bnb_config,\n"
                "    torch_dtype=torch.bfloat16,\n"
                + attn_arg +
                "    device_map='auto',\n"
                ")\n"
                "print('Base model loaded (4-bit QLoRA)')\n"
            )
        else:
            model_load_script += (
                f"model = AutoModelForCausalLM.from_pretrained(\n"
                f"    '{self.model_name_or_path}',\n"
                "    torch_dtype=torch.bfloat16,\n"
                + attn_arg +
                "    device_map='auto',\n"
                ")\n"
                "print('Base model loaded (bf16)')\n"
            )

        if adapter_path:
            model_load_script += (
                f"from peft import PeftModel\n"
                f"model = PeftModel.from_pretrained(model, '{fuse_adapter_path}')\n"
                f"print('Adapter loaded: {fuse_adapter_path}')\n"
            )

        model_load_script += "model.eval()\n"

        inference_script = (
            "import json, torch\n"
            "from google.cloud import storage\n"
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n"
            "\n"
            + model_load_script
            + "\n"
            f"with open('{fuse_prompts_path}') as f:\n"
            "    prompts = [json.loads(line) for line in f]\n"
            "\n"
            "results = []\n"
            "for item in prompts:\n"
            "    idx = item['index']\n"
            "    prompt = item['prompt']\n"
            "    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)\n"
            "    with torch.no_grad():\n"
            f"        outputs = model.generate(**inputs, max_new_tokens={max_new_tokens}, do_sample=False)\n"
            "    result = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)\n"
            "    results.append(json.dumps({'index': idx, 'result': result}))\n"
            "    del inputs, outputs\n"
            "    torch.cuda.empty_cache()\n"
            "    print(f'Processed prompt {idx + 1}/{len(prompts)}')\n"
            "\n"
            "output = '\\n'.join(results)\n"
            f"local_path = '/tmp/{inference_id}_results.jsonl'\n"
            "with open(local_path, 'w') as f:\n"
            "    f.write(output)\n"
            "\n"
            "client = storage.Client()\n"
            f"parts = '{results_gcs_path}'.replace('gs://', '').split('/', 1)\n"
            "client.bucket(parts[0]).blob(parts[1]).upload_from_filename(local_path)\n"
            "print(f'Uploaded {len(results)} results')\n"
        )

        env_vars = {}
        hf_token = os.environ.get("HF_TOKEN", "")
        if hf_token:
            env_vars["HF_TOKEN"] = hf_token

        # Gemma 3 needs transformers 4.51.3+ and flash-attn (eager/SDPA
        # both fail on PyTorch 2.3). Write script to file to avoid
        # shell escaping issues with multiline Python in sh -c.
        if self.model_name_or_path in self._NEEDS_TRANSFORMERS_UPGRADE:
            import base64
            encoded = base64.b64encode(inference_script.encode()).decode()
            infer_cmd = [
                "sh", "-c",
                "pip install -q 'transformers==4.51.3' flash-attn --no-build-isolation && "
                f"echo {encoded} | base64 -d > /tmp/_infer.py && "
                "python /tmp/_infer.py",
            ]
        else:
            infer_cmd = ["python", "-c", inference_script]

        job = aiplatform.CustomContainerTrainingJob(
            display_name=f"smala-inference-{inference_id}",
            container_uri=_HF_TRAINING_CONTAINER,
            command=infer_cmd,
            staging_bucket=self.staging_bucket,
        )

        job.run(
            replica_count=1,
            machine_type="g2-standard-12",
            accelerator_type="NVIDIA_L4",
            accelerator_count=1,
            environment_variables=env_vars if env_vars else None,
        )

        # Download results from GCS
        parts = results_gcs_path.replace("gs://", "").split("/", 1)
        client = storage.Client()
        bucket = client.bucket(parts[0])
        blob = bucket.blob(parts[1])
        results_text = blob.download_as_text()

        # Parse results and reorder by index
        result_map: dict[int, str] = {}
        for line in results_text.strip().split("\n"):
            if line:
                item = json.loads(line)
                result_map[item["index"]] = item["result"]

        ordered_results = [
            result_map.get(i, "") for i in range(len(prompts))
        ]

        # Cleanup GCS artifacts
        try:
            bucket.blob(parts[1]).delete()
            prompts_parts = prompts_gcs_path.replace("gs://", "").split("/", 1)
            bucket.blob(prompts_parts[1]).delete()
        except Exception:
            pass

        logger.info(
            "Batched tuned inference complete: %d/%d results",
            len(result_map), len(prompts),
        )
        return ordered_results

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
        """Save adapter metadata and verify adapter weights exist locally.

        The actual adapter weights are downloaded from GCS during
        :meth:`train`.  This writes a metadata file and verifies
        the weights are present on disk.
        """
        save_path = path or self.output_dir
        os.makedirs(save_path, exist_ok=True)

        # Check that adapter weight files were downloaded from GCS
        adapter_files = list(Path(save_path).glob("*.safetensors")) + \
                        list(Path(save_path).glob("*.bin"))
        if adapter_files:
            logger.info(
                "Adapter weights verified: %d file(s) in %s",
                len(adapter_files), save_path,
            )
        else:
            logger.warning(
                "No adapter weight files (.safetensors/.bin) found in %s — "
                "GCS download may have failed. Attempting re-download.",
                save_path,
            )
            # Re-attempt download from GCS
            if self._tuning_job_count > 0:
                try:
                    vertex_model_id = self._VERTEX_MODEL_MAP.get(
                        self.model_name_or_path, self.model_name_or_path
                    )
                    experiment_tag = Path(self.output_dir).parent.parent.name or "default"
                    display_name = (
                        f"smala-{experiment_tag}-{vertex_model_id.split('/')[-1]}-"
                        f"round{self._tuning_job_count}"
                    )
                    adapter_gcs_path = f"{self.staging_bucket}/adapters/{display_name}"
                    self._download_from_gcs(adapter_gcs_path, save_path)
                    logger.info("Re-download complete to %s", save_path)
                except Exception as exc:
                    logger.error("Re-download failed: %s", exc)

        metadata = {
            "backend": "vertex_ai",
            "model_name_or_path": self.model_name_or_path,
            "project": self.project,
            "location": self.location,
            "staging_bucket": self.staging_bucket,
            "tuning_jobs_completed": self._tuning_job_count,
            "adapter_files": [f.name for f in adapter_files],
            "qlora_inference": self.use_qlora_inference,
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
            if not relative_path or relative_path.endswith("/"):
                continue  # skip directory markers
            local_path = os.path.join(local_dir, relative_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            try:
                blob.download_to_filename(local_path)
                logger.debug("Downloaded %s → %s", blob.name, local_path)
            except Exception as exc:
                # Skip transient/deleted blobs (e.g. GCS lists stale entries)
                logger.warning("Skipping %s: %s", blob.name, exc)
