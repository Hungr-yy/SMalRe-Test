"""
exam_reviewer.py
================
Reviews AI-generated exam questions before they are used to evaluate
student models.  Uses Claude as a professional malware analyst to:

  1. Detect plagiarism against the benchmark question bank (questions.json)
  2. Validate factual correctness of questions and answer keys
  3. Check distractor quality
  4. Ensure CyberSOCEval-standard clarity and formatting

Sits between exam generation and student evaluation in the distillation
loop.  Rejected questions are removed; flagged questions are kept but
logged for manual inspection.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ExamReviewer:
    """Reviews generated exam questions using Claude as a malware analyst.

    Parameters
    ----------
    questions_path:
        Path to the benchmark ``questions.json`` file.
    templates_dir:
        Directory containing prompt templates.
    model:
        Anthropic model ID to use for review.
    """

    def __init__(
        self,
        questions_path: str = "hybrid-analysis/questions.json",
        templates_dir: str = "prompt_templates",
        model: str = "claude-sonnet-4-6",
    ) -> None:
        self.model = model
        self.templates_dir = templates_dir
        self._benchmark_questions = self._load_benchmark(questions_path)
        self._template = self._load_template(templates_dir)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @staticmethod
    def _load_benchmark(path: str) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            logger.warning("Benchmark questions not found at %s", p)
            return []
        with open(p) as fh:
            questions = json.load(fh)
        logger.info("Loaded %d benchmark questions for review", len(questions))
        return questions

    @staticmethod
    def _load_template(templates_dir: str) -> str:
        from teacher_engine import load_template
        return load_template("EXAM_REVIEW", templates_dir)

    def _get_client(self):
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        return Anthropic(api_key=api_key, max_retries=0)

    # ------------------------------------------------------------------
    # Completion
    # ------------------------------------------------------------------

    def _complete(self, prompt: str, max_retries: int = 5) -> str:
        """Call Claude for review with retry on transient errors."""
        import time

        from teacher_engine import _extract_retry_after, _is_rate_limit

        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                client = self._get_client()
                response = client.messages.create(
                    model=self.model,
                    max_tokens=8192,
                    temperature=0.0,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    wait = _extract_retry_after(exc)
                    if wait is not None:
                        logger.warning(
                            "Reviewer rate limited (attempt %d/%d): "
                            "retry-after %ds.",
                            attempt, max_retries, wait,
                        )
                    elif _is_rate_limit(exc):
                        wait = 60
                        logger.warning(
                            "Reviewer rate limited (attempt %d/%d): %s "
                            "— waiting %ds.",
                            attempt, max_retries, exc, wait,
                        )
                    else:
                        wait = min(0.5 * (2 ** attempt), 30)
                        logger.warning(
                            "Reviewer API call failed (attempt %d/%d): %s "
                            "— retrying in %ds.",
                            attempt, max_retries, exc, wait,
                        )
                    time.sleep(wait)

        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Review
    # ------------------------------------------------------------------

    def review(self, exam: list[dict[str, Any]]) -> dict[str, Any]:
        """Review a generated exam and return verdicts per question.

        Parameters
        ----------
        exam:
            List of exam item dicts (``{"question": {...}, "answer": {...}}``).

        Returns
        -------
        dict
            Review results with ``reviews`` list and ``summary``.
        """
        if not exam:
            return {"reviews": [], "summary": _empty_summary()}

        # Build benchmark reference — include all questions for plagiarism check
        benchmark_text = self._format_benchmark()
        exam_text = json.dumps(exam, indent=2)

        # Use str.replace instead of % formatting to avoid issues with
        # % characters in benchmark questions or exam JSON
        prompt = self._template
        prompt = prompt.replace("%s", benchmark_text, 1)
        prompt = prompt.replace("%s", exam_text, 1)

        raw = self._complete(prompt)
        result = self._parse_review(raw)

        # Safeguard: if reasoning contains reject-level language but verdict
        # is "flag", escalate to "reject" to avoid keeping broken questions.
        _reject_signals = [
            "factually incorrect", "incorrect answer", "answer key is wrong",
            "unanswerable", "not supported by", "REJECT",
        ]
        for r in result["reviews"]:
            if r.get("verdict") == "flag":
                reasoning = r.get("reasoning", "").lower()
                if any(signal.lower() in reasoning for signal in _reject_signals):
                    logger.warning(
                        "Escalating question %d from 'flag' to 'reject' — "
                        "reasoning contains reject-level issues: %s",
                        r.get("question_index", -1),
                        r.get("reasoning", "")[:200],
                    )
                    r["verdict"] = "reject"

        passed = sum(1 for r in result["reviews"] if r.get("verdict") == "pass")
        flagged = sum(1 for r in result["reviews"] if r.get("verdict") == "flag")
        rejected = sum(1 for r in result["reviews"] if r.get("verdict") == "reject")

        result["summary"] = {
            "total": len(result["reviews"]),
            "passed": passed,
            "flagged": flagged,
            "rejected": rejected,
        }

        logger.info(
            "Exam review: %d passed, %d flagged, %d rejected (out of %d)",
            passed, flagged, rejected, len(exam),
        )

        return result

    def filter_exam(
        self, exam: list[dict[str, Any]], review_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Remove rejected questions from the exam.

        Flagged questions are kept but logged.  Returns the filtered exam.
        """
        reviews = review_result.get("reviews", [])

        # Build a set of rejected indices
        rejected_indices: set[int] = set()
        for r in reviews:
            idx = r.get("question_index", -1)
            verdict = r.get("verdict", "pass")
            if verdict == "reject":
                rejected_indices.add(idx)
                logger.warning(
                    "Rejected question %d: %s",
                    idx, r.get("reasoning", "no reason given"),
                )
            elif verdict == "flag":
                logger.info(
                    "Flagged question %d (kept): %s",
                    idx, r.get("reasoning", ""),
                )

        filtered = [
            item for i, item in enumerate(exam)
            if i not in rejected_indices
        ]

        if rejected_indices:
            logger.info(
                "Filtered exam: %d → %d questions (%d rejected)",
                len(exam), len(filtered), len(rejected_indices),
            )

        return filtered

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_benchmark(self) -> str:
        """Format benchmark questions for the review prompt.

        Includes all questions so Claude can do a thorough plagiarism check.
        To manage token budget, only includes question text, options, answers,
        and metadata — not full detonation reports.
        """
        parts: list[str] = []
        for i, q in enumerate(self._benchmark_questions):
            options_str = "\n    ".join(q.get("options", []))
            parts.append(
                f"[{i}] ({q.get('attack', '?')}, {q.get('difficulty', '?')}, "
                f"{q.get('topic', '?')})\n"
                f"  Q: {q.get('question', '')}\n"
                f"    {options_str}\n"
                f"  A: {json.dumps(q.get('correct_answers', []))}"
            )
        return "\n\n".join(parts)

    def _parse_review(self, raw: str) -> dict[str, Any]:
        """Parse the review response from Claude."""
        from json_repair import extract_json

        result = extract_json(raw, default={"reviews": []}, expected_fields={
            "reviews": list,
        })

        if not isinstance(result, dict):
            logger.warning("Review response was not a dict; returning empty reviews.")
            return {"reviews": [], "summary": _empty_summary()}

        # Ensure reviews is a list
        if not isinstance(result.get("reviews"), list):
            result["reviews"] = []

        return result


def _empty_summary() -> dict[str, int]:
    return {"total": 0, "passed": 0, "flagged": 0, "rejected": 0}
