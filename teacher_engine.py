"""
teacher_engine.py
=================
Interacts with a high-capability "teacher" LLM (OpenAI GPT-4o, Google Gemini,
or Alibaba Qwen) to drive the NVIDIA-inspired data flywheel distillation loop:

  1. Exam Generation   – create exam questions from malware reports, informed by
                          prior proficiency and feedback (EXAM_PROMPT template)
  2. Evaluation + Curriculum – score student answers, diagnose weaknesses, and
                                generate targeted remediation examples in one pass
                                (EXAM_EVALUATION template)

Templates are loaded from the ``prompt_templates/`` directory rather than being
hardcoded, enabling iterative prompt engineering without code changes.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------
SUPPORTED_PROVIDERS = ("openai", "google", "anthropic", "qwen", "together")


def _build_openai_client(api_key: str):
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=api_key)


def _build_google_client(api_key: str):
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=api_key)
    return genai


def _build_anthropic_client(api_key: str):
    from anthropic import Anthropic  # type: ignore

    return Anthropic(api_key=api_key)


def _build_together_client(api_key: str):
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")


# ---------------------------------------------------------------------------
# Template loading
# ---------------------------------------------------------------------------

def load_template(name: str, templates_dir: str = "prompt_templates") -> str:
    """Load a prompt template file by name.

    Template files are Python string assignments (e.g., ``EXAM_PROMPT = \"\"\"...\"\"\"``).
    The string value is extracted safely using :mod:`ast`.
    """
    path = Path(templates_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"Template not found: {path}")

    text = path.read_text()

    # Extract the string value from a Python assignment
    try:
        tree = ast.parse(text)
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Assign):
                value = node.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    return value.value
    except SyntaxError:
        pass

    # Fallback: return raw file content
    return text


# ---------------------------------------------------------------------------
# TeacherEngine
# ---------------------------------------------------------------------------

class TeacherEngine:
    """Wraps a teacher LLM and exposes exam / evaluation / curriculum methods.

    Uses prompt templates from ``prompt_templates/`` to construct prompts,
    enabling the NVIDIA-inspired data flywheel approach where proficiency
    and feedback from prior rounds inform subsequent exam generation.

    Parameters
    ----------
    provider:
        One of ``"openai"``, ``"google"``, ``"qwen"``, or ``"together"``.
    model:
        Model identifier, e.g. ``"gpt-4o"``, ``"gemini-2.5-pro"``, ``"qwen-max"``.
    api_key:
        API key for the chosen provider.  Defaults to the corresponding
        ``*_API_KEY`` environment variable when not supplied.
    temperature:
        Sampling temperature used for all completions.
    templates_dir:
        Directory containing prompt template files.
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: str | None = None,
        temperature: float = 0.7,
        templates_dir: str = "prompt_templates",
    ) -> None:
        provider = provider.lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider}'. "
                f"Choose from: {SUPPORTED_PROVIDERS}"
            )
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.templates_dir = templates_dir

        if api_key is None:
            env_map = {
                "openai": "OPENAI_API_KEY",
                "google": "GOOGLE_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "qwen": "QWEN_API_KEY",
                "together": "TOGETHER_API_KEY",
            }
            api_key = os.environ.get(env_map[provider], "")

        if provider == "openai":
            self._client = _build_openai_client(api_key)
        elif provider == "google":
            self._client = _build_google_client(api_key)
            self._google_model = self._client.GenerativeModel(model)
        elif provider == "anthropic":
            self._client = _build_anthropic_client(api_key)
        elif provider in ("qwen", "together"):
            self._client = _build_together_client(api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # ------------------------------------------------------------------
    # Internal completion helper
    # ------------------------------------------------------------------

    def _complete(self, prompt: str, max_retries: int = 3) -> str:
        """Return a text completion for *prompt* using the teacher LLM.

        Retries up to *max_retries* times on transient errors (network
        failures, rate limits, server errors) with exponential backoff.
        """
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                return self._call_provider(prompt)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = 2 ** attempt  # 2s, 4s, 8s
                    logger.warning(
                        "Teacher API call failed (attempt %d/%d): %s  — retrying in %ds",
                        attempt, max_retries, exc, wait,
                    )
                    time.sleep(wait)

        raise last_exc  # type: ignore[misc]

    def _call_provider(self, prompt: str) -> str:
        """Dispatch a single completion call to the configured provider."""
        if self.provider == "google":
            response = self._google_model.generate_content(prompt)
            return response.text

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=8192,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text

        # OpenAI-compatible path (openai / together / qwen)
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Step 1 – Exam Generation (template-driven)
    # ------------------------------------------------------------------

    def generate_exam(
        self,
        task_description: str,
        data_source: str,
        proficiency: str = "N/A",
        feedback: str = "N/A",
        num_questions: int = 5,
    ) -> list[dict[str, Any]]:
        """Generate exam questions using the EXAM_PROMPT template.

        The template incorporates prior proficiency and feedback so that
        the teacher adjusts difficulty and focus areas each round (the
        NVIDIA data flywheel approach).

        Parameters
        ----------
        task_description:
            Description of the malware analysis task (from malware_analysis_task.json).
        data_source:
            JSON string of filtered Hybrid Analysis detonation reports for the
            teacher to ground questions in.
        proficiency:
            Proficiency score from the previous round (1-10 or "N/A").
        feedback:
            Teacher feedback from the previous round identifying weaknesses.
        num_questions:
            How many exam questions to generate.

        Returns
        -------
        list[dict]
            A list of exam item dicts, each with ``question`` and ``answer`` keys.
        """
        template = load_template("EXAM_PROMPT", self.templates_dir)
        prompt = template % (
            task_description,
            data_source,
            proficiency,
            feedback,
            num_questions,
        )

        raw = self._complete(prompt)
        parsed = self._parse_json(raw, default={"exam": []})

        # The template returns {"exam": [...]}, extract the list
        if isinstance(parsed, dict):
            return parsed.get("exam", [])
        return parsed

    # ------------------------------------------------------------------
    # Steps 3+4 – Evaluation + Curriculum Generation (combined)
    # ------------------------------------------------------------------

    def evaluate_and_generate_curriculum(
        self,
        task_description: str,
        exam_results: list[dict[str, Any]],
        data_source: str,
        num_examples: int = 100,
    ) -> dict[str, Any]:
        """Evaluate student answers and generate a remediation curriculum.

        Uses the EXAM_EVALUATION template which combines scoring, weakness
        diagnosis, and targeted training data generation in a single teacher
        call.  This is the core of the NVIDIA data flywheel: the teacher
        both evaluates and produces the next round's training curriculum.

        Parameters
        ----------
        task_description:
            Description of the malware analysis task.
        exam_results:
            List of exam result dicts, each containing ``question``,
            ``answer`` (gold), and ``model_answer`` (student prediction).
        data_source:
            JSON string of Hybrid Analysis reports for grounding new examples.
        num_examples:
            Number of remediation training examples to generate.

        Returns
        -------
        dict
            Keys: ``feedback``, ``proficiency``, ``metrics``, ``strengths``,
            ``weaknesses``, ``breakdowns``, ``dataset``.
        """
        template = load_template("EXAM_EVALUATION", self.templates_dir)
        prompt = template % (
            task_description,
            json.dumps(exam_results, indent=2),
            data_source,
            num_examples,
        )

        raw = self._complete(prompt)
        return self._parse_json(raw, default={
            "feedback": "",
            "proficiency": 1,
            "metrics": {},
            "strengths": [],
            "weaknesses": [],
            "breakdowns": {},
            "dataset": [],
        })

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str, default: Any) -> Any:
        """Attempt to extract and parse JSON from a raw LLM response."""
        # Strip markdown fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(
                line for line in lines if not line.startswith("```")
            ).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Could not parse LLM response as JSON; returning default.")
            logger.debug("Raw response: %s", raw)
            return default
