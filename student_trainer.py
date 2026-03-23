"""
student_trainer.py
==================
Fine-tunes a Small Language Model (SLM) on a curriculum produced by the teacher
using Parameter-Efficient Fine-Tuning (PEFT) – specifically LoRA or QLoRA –
to minimise GPU memory requirements.

Typical usage
-------------
>>> from student_trainer import StudentTrainer
>>> trainer = StudentTrainer.from_config("configs/model_config.yaml")
>>> trainer.train(curriculum_dataset)
>>> trainer.save("outputs/student_adapter")
"""

from __future__ import annotations

import logging
import os
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
            # EXAM_EVALUATION format: question/answer pairs
            text = _format_qa_example(ex)
        else:
            # Legacy instruction/input/output format
            text = _format_instruction_example(ex)
        records.append({"text": text})
    return Dataset.from_list(records)


def _format_qa_example(ex: dict[str, Any]) -> str:
    """Format a question/answer pair as training text.

    Uses the TASK_PROMPT structure so the student learns to answer
    in the same format used during exam evaluation.
    """
    import json

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


# ---------------------------------------------------------------------------
# StudentTrainer
# ---------------------------------------------------------------------------

class StudentTrainer:
    """Fine-tunes an SLM with LoRA or QLoRA.

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
        student_cfg = cfg.get("student", {})
        training_cfg = cfg.get("training", {})
        lora_cfg = cfg.get("lora", {})

        return cls(
            model_name_or_path=student_cfg.get("model_name_or_path", "meta-llama/Meta-Llama-3-8B"),
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
        """Fine-tune the student SLM on *curriculum*.

        Parameters
        ----------
        curriculum:
            List of ``{"instruction", "input", "output"}`` dicts as produced
            by :meth:`teacher_engine.TeacherEngine.generate_curriculum`.
        """
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
        """Run inference with the (possibly fine-tuned) student model.

        Parameters
        ----------
        prompt:
            Input text / instruction.
        max_new_tokens:
            Maximum tokens to generate.

        Returns
        -------
        str
            The model's response text.
        """
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
        """Save the LoRA adapter weights.

        Parameters
        ----------
        path:
            Directory to save to; defaults to :attr:`output_dir`.
        """
        save_path = path or self.output_dir
        os.makedirs(save_path, exist_ok=True)
        if self._model is not None:
            self._model.save_pretrained(save_path)
            logger.info("Adapter saved to %s", save_path)
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(save_path)

    def load_adapter(self, adapter_path: str) -> None:
        """Load a previously saved LoRA adapter on top of the base model.

        Parameters
        ----------
        adapter_path:
            Directory containing the PEFT adapter files.
        """
        from peft import PeftModel  # type: ignore

        if self._model is None:
            self._load_model_and_tokenizer()
        self._model = PeftModel.from_pretrained(self._model, adapter_path)
        logger.info("Adapter loaded from %s", adapter_path)
