"""
teacher_engine.py
=================
Interacts with a high-capability "teacher" LLM (OpenAI GPT-4o, Google Gemini,
or Mistral) to drive the NVIDIA-inspired data flywheel distillation loop:

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
SUPPORTED_PROVIDERS = ("openai", "google", "anthropic", "mistral", "together")


def _build_openai_client(api_key: str):
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=api_key)


def _build_google_client(api_key: str):
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=api_key)
    return genai


def _build_anthropic_client(api_key: str):
    from anthropic import Anthropic  # type: ignore

    # Disable SDK built-in retry — we handle retries ourselves using
    # the Retry-After header from 429 responses.
    return Anthropic(api_key=api_key, max_retries=0)


def _build_mistral_client(api_key: str):
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=api_key, base_url="https://api.mistral.ai/v1")


def _build_together_client(api_key: str):
    from openai import OpenAI  # type: ignore

    return OpenAI(api_key=api_key, base_url="https://api.together.xyz/v1")


# ---------------------------------------------------------------------------
# Rate limit helpers
# ---------------------------------------------------------------------------

def _extract_retry_after(exc: Exception) -> float | None:
    """Extract Retry-After seconds from an Anthropic RateLimitError.

    Returns the number of seconds to wait, or None if the header is
    missing or the exception is not a rate limit error.
    """
    response = getattr(exc, "response", None)
    if response is None:
        return None
    headers = getattr(response, "headers", {})
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after is not None:
        try:
            return float(retry_after)
        except (ValueError, TypeError):
            pass
    return None


def _is_rate_limit(exc: Exception) -> bool:
    """Check if an exception is a rate limit error."""
    exc_str = str(exc).lower()
    return "rate_limit" in exc_str or "429" in exc_str or "rate limit" in exc_str


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
        One of ``"openai"``, ``"google"``, ``"anthropic"``, ``"mistral"``, or ``"together"``.
    model:
        Model identifier, e.g. ``"gpt-4o"``, ``"gemini-2.5-pro"``, ``"mistral-large-latest"``.
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
                "mistral": "MISTRAL_API_KEY",
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
        elif provider == "mistral":
            self._client = _build_mistral_client(api_key)
        elif provider == "together":
            self._client = _build_together_client(api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    # ------------------------------------------------------------------
    # Internal completion helper
    # ------------------------------------------------------------------

    def _complete(self, prompt: str, max_retries: int = 5) -> str:
        """Return a text completion for *prompt* using the teacher LLM.

        Retries up to *max_retries* times on transient errors (network
        failures, rate limits, server errors).  Rate limit errors (429)
        use longer waits (60s) since they require waiting for the
        per-minute token quota to reset.
        """
        last_exc: Exception | None = None

        for attempt in range(1, max_retries + 1):
            try:
                return self._call_provider(prompt)
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = _extract_retry_after(exc)
                    if wait is not None:
                        logger.warning(
                            "Teacher API rate limited (attempt %d/%d): "
                            "retry-after %ds.",
                            attempt, max_retries, wait,
                        )
                    elif _is_rate_limit(exc):
                        wait = 60
                        logger.warning(
                            "Teacher API rate limited (attempt %d/%d): %s "
                            "— waiting %ds.",
                            attempt, max_retries, type(exc).__name__, wait,
                        )
                    else:
                        wait = min(0.5 * (2 ** attempt), 30)
                        logger.warning(
                            "Teacher API call failed (attempt %d/%d): %s "
                            "— retrying in %ds.",
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

        # OpenAI-compatible path (openai / mistral / together)
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Step 1 – Exam Generation (template-driven)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Few-shot example loading
    # ------------------------------------------------------------------

    _few_shot_cache: list[dict[str, Any]] | None = None

    @classmethod
    def _load_few_shot_questions(
        cls,
        questions_path: str = "hybrid-analysis/questions.json",
    ) -> list[dict[str, Any]]:
        """Load and cache benchmark questions for few-shot prompting."""
        if cls._few_shot_cache is not None:
            return cls._few_shot_cache
        path = Path(questions_path)
        if not path.exists():
            logger.warning("Few-shot questions not found at %s; skipping.", path)
            cls._few_shot_cache = []
            return cls._few_shot_cache
        with open(path) as fh:
            cls._few_shot_cache = json.load(fh)
        logger.info("Loaded %d few-shot examples from %s", len(cls._few_shot_cache), path)
        return cls._few_shot_cache

    @staticmethod
    def _sample_few_shot(
        questions: list[dict[str, Any]],
        attack_family: str,
        n: int = 3,
    ) -> str:
        """Pick *n* diverse few-shot examples, preferring the same attack family.

        Returns a formatted string ready for template injection.
        """
        import random as _random

        if not questions:
            return "(No few-shot examples available.)"

        # Prefer same attack family, fill with others for diversity
        same_family = [q for q in questions if q.get("attack") == attack_family]
        other = [q for q in questions if q.get("attack") != attack_family]

        selected: list[dict[str, Any]] = []
        # At least 1 from same family if available, rest from other for diversity
        if same_family:
            selected.extend(_random.sample(same_family, min(1, len(same_family))))
        remaining = n - len(selected)
        if remaining > 0 and other:
            selected.extend(_random.sample(other, min(remaining, len(other))))
        # Top up from same family if other didn't have enough
        remaining = n - len(selected)
        if remaining > 0 and same_family:
            pool = [q for q in same_family if q not in selected]
            selected.extend(_random.sample(pool, min(remaining, len(pool))))

        # Format as numbered examples
        parts: list[str] = []
        for idx, q in enumerate(selected, 1):
            options_str = "\n    ".join(q.get("options", []))
            parts.append(
                f"Example {idx} ({q.get('attack', '?')}, {q.get('difficulty', '?')}, "
                f"{q.get('topic', '?')}):\n"
                f"  Q: {q.get('question', '')}\n"
                f"    {options_str}\n"
                f"  A: {json.dumps(q.get('correct_answers', []))}"
            )
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Step 1 – Exam Generation (template-driven)
    # ------------------------------------------------------------------

    def generate_exam(
        self,
        task_description: str,
        reports: list[dict[str, Any]],
        proficiency: str = "N/A",
        feedback: str = "N/A",
    ) -> list[dict[str, Any]]:
        """Generate exam questions — one question per detonation report.

        Follows the CyberSOCEval one-report-per-question paradigm: each
        report is sent individually to the teacher, which generates exactly
        one question grounded in that report.  The template incorporates
        prior proficiency and feedback (NVIDIA data flywheel).

        Few-shot examples from ``questions.json`` are sampled per report
        to calibrate question style and difficulty.

        Parameters
        ----------
        task_description:
            Description of the malware analysis task (from malware_analysis_task.json).
        reports:
            List of filtered Hybrid Analysis report dicts.  One question
            will be generated per report.
        proficiency:
            Proficiency score from the previous round (1-10 or "N/A").
        feedback:
            Teacher feedback from the previous round identifying weaknesses.

        Returns
        -------
        list[dict]
            A list of exam item dicts, each with ``question`` and ``answer`` keys.
        """
        template = load_template("EXAM_PROMPT", self.templates_dir)
        few_shot_pool = self._load_few_shot_questions()
        exam: list[dict[str, Any]] = []

        for i, report in enumerate(reports):
            report_json = json.dumps(report)
            attack_family = report.get("_family", "unknown")
            few_shot_text = self._sample_few_shot(few_shot_pool, attack_family)
            prompt = template % (
                task_description,
                report_json,
                proficiency,
                feedback,
                few_shot_text,
            )

            # Retry up to 2 extra times if the teacher returns unparseable JSON
            parsed = None
            for attempt in range(3):
                raw = self._complete(prompt)
                parsed = self._parse_json(raw, default={}, expected_fields={
                    "question": dict,
                    "answer": dict,
                }, schema_example=self._EXAM_SCHEMA_EXAMPLE)

                if isinstance(parsed, dict) and "question" in parsed:
                    break

                if attempt < 2:
                    logger.warning(
                        "Teacher returned invalid question for report %d "
                        "(attempt %d/3); retrying. Raw response: %.500s",
                        i + 1, attempt + 1, raw,
                    )

            # The template returns {"question": {...}, "answer": {...}}
            if isinstance(parsed, dict) and "question" in parsed:
                # Always inject the real detonation report — never trust
                # the teacher to echo it back correctly.
                if isinstance(parsed["question"], dict):
                    parsed["question"]["detonation_report"] = report
                exam.append(parsed)
                logger.debug("Generated question %d/%d", i + 1, len(reports))
            else:
                logger.error(
                    "Teacher failed to generate valid question for report %d "
                    "after 3 attempts; skipping. Last raw response: %.500s",
                    i + 1, raw if raw else "(empty)",
                )

        return exam

    # ------------------------------------------------------------------
    # Step 1c – Repair empty detonation reports
    # ------------------------------------------------------------------

    def repair_empty_reports(
        self,
        exam: list[dict[str, Any]],
        reports: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Fill in empty detonation reports while keeping question and answer.

        If a generated exam item has an empty or missing detonation_report,
        the original sampled report is injected.  If the original report is
        also unavailable, the teacher is asked to generate a plausible
        detonation report that supports the existing question and answer.

        Parameters
        ----------
        exam:
            List of exam item dicts from generate_exam().
        reports:
            The originally sampled reports (same order as exam generation).

        Returns
        -------
        list[dict]
            The exam with detonation reports filled in.
        """
        repaired = 0
        for i, item in enumerate(exam):
            if not isinstance(item, dict):
                continue
            question_data = item.get("question", {})
            if not isinstance(question_data, dict):
                continue

            det_report = question_data.get("detonation_report")
            if det_report and isinstance(det_report, dict) and len(det_report) > 1:
                continue  # report looks populated

            # Try to inject the original sampled report
            if i < len(reports) and reports[i]:
                question_data["detonation_report"] = reports[i]
                repaired += 1
                logger.info(
                    "Injected original report into question %d (empty detonation_report).", i,
                )
                continue

            # Last resort: ask the teacher to generate a supporting report
            q_text = question_data.get("question", "")
            options = question_data.get("options", [])
            answer = item.get("answer", {})
            attack = question_data.get("attack", "unknown")

            prompt = (
                "A malware analysis exam question was generated but its "
                "detonation_report field is empty. Generate a realistic "
                "Hybrid Analysis sandbox detonation report (JSON object) "
                "that would support the following question and answer.\n\n"
                f"Attack family: {attack}\n"
                f"Question: {q_text}\n"
                f"Options: {json.dumps(options)}\n"
                f"Correct answers: {json.dumps(answer.get('correct_answers', []))}\n\n"
                "Output ONLY the detonation report as a JSON object. "
                "No markdown, no explanation."
            )
            try:
                raw = self._complete(prompt)
                from json_repair import extract_json
                parsed = extract_json(raw, default={})
                if isinstance(parsed, dict) and parsed:
                    question_data["detonation_report"] = parsed
                    repaired += 1
                    logger.info(
                        "Generated detonation report for question %d via teacher.", i,
                    )
                else:
                    logger.warning(
                        "Failed to generate detonation report for question %d.", i,
                    )
            except Exception as exc:
                logger.warning(
                    "Error generating detonation report for question %d: %s", i, exc,
                )

        if repaired:
            logger.info("Repaired %d empty detonation reports.", repaired)
        return exam

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

        Split into two phases:
          1. **Evaluation** — score the exam, diagnose weaknesses, assign
             proficiency (single call via EXAM_EVALUATION template).
          2. **Curriculum generation** — generate training examples in
             batches of ``_CURRICULUM_BATCH_SIZE`` until ``num_examples``
             are produced (via CURRICULUM_GENERATION template).

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
        # --- Phase 1: Evaluation ---
        eval_template = load_template("EXAM_EVALUATION", self.templates_dir)

        # Strip full detonation reports to stay within context limits.
        slim_results: list[dict[str, Any]] = []
        for item in exam_results:
            slim = dict(item)
            q = slim.get("question")
            if isinstance(q, dict) and "detonation_report" in q:
                q = dict(q)
                q.pop("detonation_report", None)
                slim["question"] = q
            slim_results.append(slim)

        eval_prompt = eval_template % (
            task_description,
            json.dumps(slim_results, indent=2),
            data_source,
        )

        raw = self._complete(eval_prompt)
        result = self._parse_json(
            raw,
            default={
                "feedback": "",
                "proficiency": 1,
                "metrics": {},
                "strengths": [],
                "weaknesses": [],
                "breakdowns": {},
            },
            expected_fields={
                "feedback": str,
                "proficiency": int,
                "metrics": dict,
                "strengths": list,
                "weaknesses": list,
                "breakdowns": dict,
            },
            schema_example=self._EVAL_SCHEMA_EXAMPLE,
        )

        # Validate and repair critical fields that must be the correct type
        result = self._repair_evaluation_fields(result)

        # --- Phase 2: Curriculum generation in batches ---
        dataset = self._generate_curriculum_batched(
            task_description=task_description,
            evaluation=result,
            data_source=data_source,
            num_examples=num_examples,
        )
        result["dataset"] = dataset

        return result

    # Expected types for evaluation fields — used by _repair_evaluation_fields
    _EVAL_FIELD_TYPES: dict[str, type] = {
        "feedback": str,
        "proficiency": int,
        "metrics": dict,
        "strengths": list,
        "weaknesses": list,
        "breakdowns": dict,
    }

    def _repair_evaluation_fields(
        self,
        result: dict[str, Any],
        max_repair_attempts: int = 3,
    ) -> dict[str, Any]:
        """Validate and repair individual evaluation fields.

        If a field has the wrong type (e.g. weaknesses is a dict instead
        of a list), sends the malformed field back to the teacher LLM
        for repair. Falls back to the default value after max attempts.
        """
        from json_repair import extract_json

        defaults = {
            "feedback": "",
            "proficiency": 1,
            "metrics": {},
            "strengths": [],
            "weaknesses": [],
            "breakdowns": {},
        }

        for field, expected_type in self._EVAL_FIELD_TYPES.items():
            value = result.get(field)
            if isinstance(value, expected_type):
                continue

            logger.warning(
                "Evaluation field '%s' is %s instead of %s; attempting repair.",
                field, type(value).__name__, expected_type.__name__,
            )

            # Try LLM repair: ask the teacher to convert the field
            repaired = False
            for attempt in range(1, max_repair_attempts + 1):
                repair_prompt = (
                    f"Your previous evaluation response had a malformed '{field}' field.\n\n"
                    f"The '{field}' field was:\n"
                    f"{json.dumps(value, indent=2)}\n\n"
                    f"Convert it into a valid JSON {expected_type.__name__}.\n"
                    f"For reference, '{field}' should be: {json.dumps(defaults[field])}\n\n"
                    f"Output ONLY the corrected '{field}' value as JSON. "
                    f"No markdown, no explanation."
                )
                try:
                    raw = self._complete(repair_prompt)
                    parsed = extract_json(raw, default=defaults[field])
                    if isinstance(parsed, expected_type):
                        result[field] = parsed
                        logger.info(
                            "Repaired '%s' on attempt %d/%d.",
                            field, attempt, max_repair_attempts,
                        )
                        repaired = True
                        break
                except Exception as exc:
                    logger.warning(
                        "Repair attempt %d/%d for '%s' failed: %s",
                        attempt, max_repair_attempts, field, exc,
                    )

            if not repaired:
                raise RuntimeError(
                    f"Evaluation field '{field}' is {type(value).__name__} "
                    f"instead of {expected_type.__name__}. "
                    f"All {max_repair_attempts} repair attempts failed. "
                    f"Aborting experiment."
                )

        return result

    # Batch size for curriculum generation — each batch is a separate
    # teacher call to stay within output token limits.
    _CURRICULUM_BATCH_SIZE = 20

    def _generate_curriculum_batched(
        self,
        task_description: str,
        evaluation: dict[str, Any],
        data_source: str,
        num_examples: int,
    ) -> list[dict[str, Any]]:
        """Generate curriculum examples in batches.

        Passes the full evaluation context (feedback, proficiency,
        weaknesses, breakdowns, strengths) to each batch so the teacher
        can target remediation examples to the student's specific gaps.
        """
        template = load_template("CURRICULUM_GENERATION", self.templates_dir)

        feedback = evaluation.get("feedback", "")
        proficiency = evaluation.get("proficiency", 1)
        weaknesses_text = json.dumps(evaluation.get("weaknesses", []), indent=2)
        breakdowns_text = json.dumps(evaluation.get("breakdowns", {}), indent=2)
        strengths_text = json.dumps(evaluation.get("strengths", []), indent=2)

        # Schema example for the repair loop
        curriculum_schema_example = (
            '[{"question": {"sha256": "...", "attack": "infostealers", '
            '"topic": "...", "difficulty": "medium", "detonation_report": {}, '
            '"question": "...", "options": ["A. ...", "B. ..."]}, '
            '"answer": {"correct_answers": ["A"]}}]'
        )

        dataset: list[dict[str, Any]] = []
        initial_batches = (num_examples + self._CURRICULUM_BATCH_SIZE - 1) // self._CURRICULUM_BATCH_SIZE
        consecutive_failures = 0
        batch_num = 0

        # Phase 1: initial batches (up to 5 for 100 examples)
        for _ in range(initial_batches):
            remaining = num_examples - len(dataset)
            if remaining <= 0:
                break
            batch_num += 1
            batch_size = min(remaining, self._CURRICULUM_BATCH_SIZE)

            valid = self._run_curriculum_batch(
                template, task_description, feedback, proficiency,
                weaknesses_text, breakdowns_text, strengths_text,
                data_source, batch_size, curriculum_schema_example,
            )

            if valid:
                dataset.extend(valid)
                consecutive_failures = 0
                logger.info(
                    "Curriculum batch %d: %d valid (%d/%d total)",
                    batch_num, len(valid), len(dataset), num_examples,
                )
            else:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    logger.error("3 consecutive failed batches; stopping.")
                    break

        # Phase 2: deficit-filling batches if under target
        deficit = num_examples - len(dataset)
        if deficit > 0:
            deficit_batches = (deficit + self._CURRICULUM_BATCH_SIZE - 1) // self._CURRICULUM_BATCH_SIZE
            logger.info(
                "Curriculum deficit: %d examples short. "
                "Generating %d additional batches.",
                deficit, deficit_batches,
            )
            consecutive_failures = 0

            for _ in range(deficit_batches):
                remaining = num_examples - len(dataset)
                if remaining <= 0:
                    break
                batch_num += 1
                batch_size = min(remaining, self._CURRICULUM_BATCH_SIZE)

                valid = self._run_curriculum_batch(
                    template, task_description, feedback, proficiency,
                    weaknesses_text, breakdowns_text, strengths_text,
                    data_source, batch_size, curriculum_schema_example,
                )

                if valid:
                    dataset.extend(valid)
                    consecutive_failures = 0
                    logger.info(
                        "Deficit batch %d: %d valid (%d/%d total)",
                        batch_num, len(valid), len(dataset), num_examples,
                    )
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        logger.error(
                            "3 consecutive failed deficit batches; stopping."
                        )
                        break

        logger.info(
            "Curriculum generation complete: %d/%d examples",
            len(dataset), num_examples,
        )
        return dataset

    def _run_curriculum_batch(
        self,
        template: str,
        task_description: str,
        feedback: str,
        proficiency: int,
        weaknesses_text: str,
        breakdowns_text: str,
        strengths_text: str,
        data_source: str,
        batch_size: int,
        schema_example: str,
    ) -> list[dict[str, Any]]:
        """Run a single curriculum generation batch with all guardrails.

        Returns validated items or an empty list on failure.
        """
        prompt = template % (
            task_description,
            feedback,
            proficiency,
            weaknesses_text,
            breakdowns_text,
            strengths_text,
            data_source,
            batch_size,
        )

        batch = self._generate_and_repair_curriculum_batch(
            prompt, schema_example,
        )

        # Unwrap dict wrapper if teacher returned {"dataset": [...]}
        if isinstance(batch, dict):
            for key, val in batch.items():
                if isinstance(val, list):
                    logger.info(
                        "Curriculum batch returned dict; unwrapping '%s' key.", key,
                    )
                    batch = val
                    break
            else:
                logger.warning(
                    "Curriculum batch returned dict with no list values."
                )
                return []

        if isinstance(batch, list):
            return self._validate_curriculum_items(batch)

        logger.warning(
            "Curriculum batch returned %s instead of list.",
            type(batch).__name__,
        )
        return []

    def _generate_and_repair_curriculum_batch(
        self,
        prompt: str,
        schema_example: str,
        max_repair_attempts: int = 3,
    ) -> Any:
        """Generate a curriculum batch with LLM repair loop.

        1. Call the teacher to generate the batch.
        2. Parse with extract_json.
        3. If the result is not a list, send the malformed output back
           to the teacher with the expected format for repair.
        4. Repeat up to max_repair_attempts.
        """
        from json_repair import extract_json

        raw = self._complete(prompt)
        batch = extract_json(raw, default=[])

        # If it's already a valid list, return immediately
        if isinstance(batch, list) and batch:
            return batch

        # Repair loop: send malformed output back to the teacher
        current = batch
        for attempt in range(1, max_repair_attempts + 1):
            logger.warning(
                "Curriculum repair attempt %d/%d: expected list but got %s.",
                attempt, max_repair_attempts, type(current).__name__,
            )

            repair_prompt = (
                "Your previous response was not a valid JSON list of training examples.\n\n"
                "Here is your previous output:\n"
                f"{json.dumps(current, indent=2) if current else raw[:2000]}\n\n"
                "Reformat it into a JSON list matching this example:\n"
                f"{schema_example}\n\n"
                "Output ONLY the corrected JSON list. No markdown, no explanation."
            )

            try:
                repaired_raw = self._complete(repair_prompt)
                repaired = extract_json(repaired_raw, default=[])
                if isinstance(repaired, list) and repaired:
                    logger.info(
                        "Curriculum repair attempt %d/%d succeeded (%d items).",
                        attempt, max_repair_attempts, len(repaired),
                    )
                    return repaired
                current = repaired
            except Exception as exc:
                logger.warning(
                    "Curriculum repair attempt %d/%d failed: %s",
                    attempt, max_repair_attempts, exc,
                )

        logger.error("All %d curriculum repair attempts failed.", max_repair_attempts)
        return current

    @staticmethod
    def _validate_curriculum_items(
        batch: list[Any],
    ) -> list[dict[str, Any]]:
        """Filter curriculum items to only those with valid structure."""
        valid: list[dict[str, Any]] = []
        for item in batch:
            if not isinstance(item, dict):
                continue
            if "question" not in item or "answer" not in item:
                continue
            q = item["question"]
            a = item["answer"]
            if not isinstance(q, dict) or not isinstance(a, dict):
                continue
            if "correct_answers" not in a:
                continue
            if not q.get("question"):
                continue
            if not q.get("options"):
                continue
            valid.append(item)
        return valid

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    # JSON examples for repair prompts — mirrors the templates in prompt_templates/
    _EXAM_SCHEMA_EXAMPLE = """{
  "question": {
    "sha256": "abc123...",
    "attack": "infostealers",
    "topic": "persistence mechanisms",
    "difficulty": "medium",
    "detonation_report": { "...": "..." },
    "question": "Which registry keys does the malware modify for persistence?",
    "options": ["A. HKCU\\\\Software\\\\Microsoft\\\\Windows\\\\CurrentVersion\\\\Run", "B. HKLM\\\\SYSTEM\\\\CurrentControlSet\\\\Services", "C. HKCU\\\\Environment"]
  },
  "answer": {
    "correct_answers": ["A", "C"]
  }
}"""

    _EVAL_SCHEMA_EXAMPLE = """{
  "feedback": "The model struggles with ...",
  "proficiency": 3,
  "metrics": {
    "num_questions": 5,
    "num_exact_matches": 2,
    "exact_match_accuracy": 0.4,
    "avg_jaccard": 0.55,
    "parse_error_rate": 0.0,
    "macro_option_precision": 0.5,
    "macro_option_recall": 0.6,
    "avg_predicted_answer_count": 1.8,
    "avg_gold_answer_count": 2.0,
    "answer_count_bias": "under-selecting"
  },
  "strengths": ["identifies network IOCs"],
  "weaknesses": [{"area_type": "topic", "area_name": "persistence", "description": "...", "severity": "high", "evidence_question_indices": [0, 2]}],
  "breakdowns": {"by_attack": {}, "by_difficulty": {}, "by_topic": {}},
  "dataset": [{"question": {"sha256": "...", "attack": "...", "topic": "...", "difficulty": "...", "detonation_report": {}, "question": "...", "options": ["A. ...", "B. ..."]}, "answer": {"correct_answers": ["A"]}}]
}"""

    def _parse_json(
        self,
        raw: str,
        default: Any,
        expected_fields: dict[str, type] | None = None,
        schema_example: str | None = None,
    ) -> Any:
        """Extract and parse JSON from a raw LLM response.

        Uses a multi-strategy approach (direct parse → fence stripping →
        brace extraction → repair → field-level reconstruction) so that
        prose-wrapped or mildly malformed JSON is still recovered.

        If *default* is a dict and the result is not, a repair loop runs
        up to 3 times: teacher LLM reformat → programmatic repair.  Only
        returns the default after all 3 attempts fail.

        Parameters
        ----------
        schema_example:
            A concrete JSON example of the expected output, shown to the
            teacher during repair so it knows the exact structure needed.
        """
        from json_repair import extract_json

        result = extract_json(raw, default=default, expected_fields=expected_fields)

        if isinstance(default, dict) and not isinstance(result, dict):
            result = self._repair_loop(
                result, default, expected_fields, schema_example=schema_example,
            )

        # Final fallback chain if repair loop could not produce a dict
        if isinstance(default, dict) and not isinstance(result, dict):
            # List-wrap: treat as dataset/content entries inside the default
            if isinstance(result, list) and result:
                list_key = self._guess_list_field(default)
                if list_key:
                    logger.warning(
                        "Last-resort list-wrap: placing %d items into '%s'.",
                        len(result), list_key,
                    )
                    patched = dict(default)
                    patched[list_key] = result
                    return patched

            # Nothing worked — abort the experiment
            raise RuntimeError(
                "JSON repair exhausted all strategies. "
                f"1) extract_json returned {type(result).__name__} instead of dict. "
                "2) Teacher LLM repair loop (3 attempts) failed. "
                "3) List-wrap fallback not applicable. "
                "Aborting experiment."
            )

        return result

    @staticmethod
    def _guess_list_field(default: dict) -> str | None:
        """Return the first key in *default* whose value is a list, or None."""
        for key, val in default.items():
            if isinstance(val, list):
                return key
        return None

    def _repair_loop(
        self,
        malformed: Any,
        default: dict,
        expected_fields: dict[str, type] | None = None,
        max_attempts: int = 3,
        schema_example: str | None = None,
    ) -> dict:
        """Try to convert a non-dict LLM response into the expected object.

        Each attempt:
          1. Send the malformed output to the teacher LLM for reformatting,
             including a concrete JSON example of the expected structure.
          2. Run programmatic repair (extract_json) on the teacher's response.
          3. If the result is a valid dict, return it.

        After *max_attempts* failures, logs an error and returns the original
        malformed input so callers can apply their own fallback logic.
        """
        from json_repair import extract_json

        if schema_example is None:
            schema_example = json.dumps(
                {k: type(v).__name__ for k, v in default.items()} if default else {},
                indent=2,
            )

        current = malformed
        for attempt in range(1, max_attempts + 1):
            logger.warning(
                "Repair attempt %d/%d: expected JSON object but got %s.",
                attempt, max_attempts, type(current).__name__,
            )

            repair_prompt = (
                "Your previous response was not a valid JSON object.\n\n"
                "Here is your previous output:\n"
                f"{json.dumps(current, indent=2)}\n\n"
                "Reformat it into a single JSON object matching this example:\n"
                f"{schema_example}\n\n"
                "Output ONLY the corrected JSON object. No markdown, no explanation."
            )

            try:
                repaired_raw = self._complete(repair_prompt)
                repaired = extract_json(
                    repaired_raw, default=default, expected_fields=expected_fields,
                )
                if isinstance(repaired, dict):
                    logger.info(
                        "Repair attempt %d/%d succeeded.", attempt, max_attempts,
                    )
                    return repaired
                # Feed the still-wrong output into the next attempt
                current = repaired
            except Exception as exc:
                logger.warning(
                    "Repair attempt %d/%d failed: %s", attempt, max_attempts, exc,
                )

        logger.error(
            "All %d repair attempts failed.", max_attempts,
        )
        return malformed
