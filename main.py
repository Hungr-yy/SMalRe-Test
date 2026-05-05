"""
main.py
=======
Orchestrates the NVIDIA-inspired iterative teacher-student knowledge
distillation loop (data flywheel) for malware analysis:

  1. Generate an exam grounded in Hybrid Analysis reports  (teacher, EXAM_PROMPT)
  2. Student attempts the exam                             (student, TASK_PROMPT)
  3. Teacher evaluates + generates remediation curriculum   (teacher, EXAM_EVALUATION)
  4. Fine-tune the student on accumulated curriculum        (LoRA / QLoRA)
  5. Repeat, passing proficiency & feedback into the next round

Key NVIDIA data flywheel principles implemented:
  - Proficiency tracking (1-10 scale) across rounds
  - Teacher feedback informs next round's exam generation
  - Curriculum accumulation (new examples combine with prior rounds)
  - Progressive difficulty based on student proficiency

Usage
-----
    # Single experiment (CLI)
    python main.py --config configs/model_config.yaml [options]

    # 3x3 experiment matrix (see run_experiments.py)
    python run_experiments.py --config configs/model_config.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from exam_reviewer import ExamReviewer

load_dotenv()  # Load API keys from .env file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict[str, Any]:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Data source loading
# ---------------------------------------------------------------------------

def load_task_description(task_path: str) -> str:
    """Load the task description from malware_analysis_task.json."""
    with open(task_path) as fh:
        task_data = json.load(fh)
    return json.dumps(task_data, indent=2)


def load_hybrid_analysis_reports(
    data_dir: str,
    truncate: bool = True,
) -> list[dict[str, Any]]:
    """Load all Hybrid Analysis JSON reports from the data directory.

    Parameters
    ----------
    data_dir:
        Path to the hybrid-analysis directory containing malware family subdirs.
    truncate:
        Apply DataFilter to reduce report size for context efficiency.

    Returns
    -------
    list[dict]
        List of (possibly filtered) report dicts, each augmented with a
        ``_family`` key indicating the malware family subdirectory.
    """
    from data_filter import DataFilter

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.warning("Data directory not found: %s", data_dir)
        return []

    filt = DataFilter(mode="essential") if truncate else DataFilter(mode="none")
    reports: list[dict[str, Any]] = []

    # Walk family subdirectories
    for family_dir in sorted(data_path.iterdir()):
        if not family_dir.is_dir():
            continue
        family_name = family_dir.name
        for report_file in sorted(family_dir.iterdir()):
            if not report_file.is_file():
                continue
            try:
                with open(report_file) as fh:
                    raw = json.load(fh)
                filtered = filt.filter(raw)
                filtered["_family"] = family_name
                reports.append(filtered)
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("Skipping %s: %s", report_file, exc)

    logger.info("Loaded %d reports from %s", len(reports), data_dir)
    return reports


def sample_reports(
    reports: list[dict[str, Any]],
    num_reports: int = 5,
) -> list[dict[str, Any]]:
    """Sample a diverse set of individual reports for one-report-per-question.

    Follows the CyberSOCEval paradigm: each question is grounded in exactly
    one detonation report.  Returns ``num_reports`` individual report dicts,
    sampled evenly across malware families, with at most one report per
    malware variant (``vx_family``) within each family to avoid duplicate
    questions.

    Reports are pre-filtered by :class:`data_filter.DataFilter` using the
    CyberSOCEval truncation (hash removal, description trimming, MITRE
    condensing), guaranteeing every report is under ~20k tokens and fits
    within all teacher context windows.
    """
    if not reports:
        return []

    # Group by family
    by_family: dict[str, list[dict[str, Any]]] = {}
    for r in reports:
        family = r.get("_family", "unknown")
        by_family.setdefault(family, []).append(r)

    per_family = max(1, num_reports // len(by_family))
    sampled: list[dict[str, Any]] = []

    for family_reports in by_family.values():
        # Sub-group by vx_family (malware variant) to ensure diversity
        by_variant: dict[str, list[dict[str, Any]]] = {}
        for r in family_reports:
            variant = r.get("vx_family", "") or "unknown"
            by_variant.setdefault(variant, []).append(r)

        # Pick 1 report per variant, shuffled, up to per_family
        variant_picks: list[dict[str, Any]] = []
        variant_keys = list(by_variant.keys())
        random.shuffle(variant_keys)
        for vk in variant_keys:
            variant_picks.append(random.choice(by_variant[vk]))
            if len(variant_picks) >= per_family:
                break

        sampled.extend(variant_picks)

    # Fill to num_reports if needed, still respecting variant diversity
    sampled_ids = {id(r) for r in sampled}
    remaining = [r for r in reports if id(r) not in sampled_ids]
    if len(sampled) < num_reports and remaining:
        # Prefer variants not yet represented
        used_variants: set[str] = {r.get("vx_family", "") for r in sampled}
        unseen = [r for r in remaining if r.get("vx_family", "") not in used_variants]
        pool = unseen if unseen else remaining
        extra = min(num_reports - len(sampled), len(pool))
        sampled.extend(random.sample(pool, extra))

    # Final fill if still short (all variants exhausted)
    if len(sampled) < num_reports:
        still_remaining = [r for r in reports if id(r) not in {id(s) for s in sampled}]
        if still_remaining:
            extra = min(num_reports - len(sampled), len(still_remaining))
            sampled.extend(random.sample(still_remaining, extra))

    selected = sampled[:num_reports]
    logger.debug(
        "Sampled %d reports across %d families (%d unique variants)",
        len(selected), len(by_family),
        len({r.get("vx_family", "") for r in selected}),
    )
    return selected


# ---------------------------------------------------------------------------
# Distillation loop
# ---------------------------------------------------------------------------

def run_distillation_loop(
    config: dict[str, Any],
    rounds: int,
    target_accuracy: float,
    output_dir: str,
    truncate_input: bool,
) -> dict[str, Any]:
    """Run the iterative distillation loop and return results.

    Parameters
    ----------
    config:
        Configuration dict.  Must contain ``teacher`` and ``student`` sections.
        The ``student`` section must include ``model_name_or_path``.
        Optionally include ``_config_path`` for YAML-based StudentTrainer init,
        or set student config directly for programmatic use.
    rounds:
        Maximum distillation rounds.
    target_accuracy:
        Stop early when this accuracy is reached.
    output_dir:
        Directory to save adapters, curricula, and results.
    truncate_input:
        Apply DataFilter truncation to reports.

    Returns
    -------
    dict
        Results dict with keys: ``best_accuracy``, ``final_proficiency``,
        ``rounds_completed``, ``round_results`` (list of per-round metrics).
    """
    from teacher_engine import TeacherEngine, load_template
    from student_trainer import create_student_trainer, StudentTrainer

    # ------------------------------------------------------------------
    # Initialise teacher
    # ------------------------------------------------------------------
    teacher_cfg = config.get("teacher", {})
    teacher = TeacherEngine(
        provider=teacher_cfg.get("provider", "openai"),
        model=teacher_cfg.get("model", "gpt-4o"),
        api_key=teacher_cfg.get("api_key") or None,
        temperature=teacher_cfg.get("temperature", 0.7),
    )
    teacher_name = f"{teacher_cfg.get('provider')}/{teacher_cfg.get('model')}"
    logger.info("Teacher: %s", teacher_name)

    # ------------------------------------------------------------------
    # Initialise student (local or Vertex AI backend)
    # ------------------------------------------------------------------
    if "_config_path" in config:
        cfg_from_file = load_config(config["_config_path"])
        cfg_from_file["output_dir"] = output_dir
        student = create_student_trainer(cfg_from_file)
    else:
        # Programmatic construction (used by run_experiments.py)
        config["output_dir"] = output_dir
        student = create_student_trainer(config)

    student_name = config.get("student", {}).get("model_name_or_path", "unknown")
    logger.info("Student: %s", student_name)

    # ------------------------------------------------------------------
    # Load task description and data source
    # ------------------------------------------------------------------
    data_cfg = config.get("data_source", {})
    task_path = data_cfg.get(
        "task_definitions", "prompt_templates/malware_analysis_task.json"
    )
    data_dir = data_cfg.get("hybrid_analysis_dir", "hybrid-analysis")

    task_description = load_task_description(task_path)
    logger.info("Loaded task definitions from %s", task_path)

    all_reports = load_hybrid_analysis_reports(data_dir, truncate=truncate_input)

    # Initialise exam reviewer (uses Claude as a malware analyst)
    questions_path = str(Path(data_dir) / "questions.json")
    reviewer = ExamReviewer(questions_path=questions_path)
    logger.info("Exam reviewer initialised (model=%s)", reviewer.model)

    # Load TASK_PROMPT template for student exam attempts
    task_prompt_template = load_template("TASK_PROMPT")

    # JSON schema for guided decoding (CyberSOCEval-aligned)
    answer_schema = {
        "type": "object",
        "properties": {
            "correct_answers": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["correct_answers"],
    }

    # ------------------------------------------------------------------
    # State tracked across rounds (data flywheel)
    # ------------------------------------------------------------------
    proficiency: str = "N/A"
    feedback: str = "N/A"
    accumulated_curriculum: list[dict[str, Any]] = []
    best_accuracy = 0.0
    total_parse_errors = 0
    total_questions_attempted = 0
    round_results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Iterative loop
    # ------------------------------------------------------------------
    for round_num in range(1, rounds + 1):
        logger.info("=" * 60)
        logger.info("[%s → %s] Round %d / %d",
                     teacher_name, student_name, round_num, rounds)
        logger.info("=" * 60)

        round_dir = Path(output_dir) / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # Sample one report per question (CyberSOCEval paradigm)
        num_q = config.get("exam", {}).get("num_questions", 5)
        round_reports = sample_reports(all_reports, num_reports=num_q)

        # Step 1 – Generate exam: one question per report
        logger.info("[Step 1] Generating %d questions (1 per report, proficiency=%s) …",
                     num_q, proficiency)
        exam = teacher.generate_exam(
            task_description=task_description,
            reports=round_reports,
            proficiency=proficiency,
            feedback=feedback,
        )
        if not exam:
            # Retry exam generation once before skipping
            logger.warning("Teacher returned no exam questions; retrying …")
            exam = teacher.generate_exam(
                task_description=task_description,
                reports=round_reports,
                proficiency=proficiency,
                feedback=feedback,
            )
            if not exam:
                logger.error("Teacher returned no exam questions after retry; skipping round.")
                continue

        # Step 1a – Repair empty detonation reports programmatically
        exam = teacher.repair_empty_reports(exam, round_reports)

        with open(round_dir / "exam.json", "w") as fh:
            json.dump(exam, fh, indent=2)
        logger.info("Generated %d exam questions", len(exam))

        # Step 1b – Review-regenerate loop
        # First pass reviews all questions.  Subsequent passes only review
        # regenerated replacements.  Repeat until all pass or max attempts.
        max_review_attempts = 5
        all_review_results: list[dict[str, Any]] = []
        indices_to_review: list[int] | None = None  # None = review all

        for review_attempt in range(1, max_review_attempts + 1):
            # Build the subset to review
            if indices_to_review is None:
                # First pass: review everything
                questions_to_review = exam
                logger.info(
                    "[Step 1b] Review attempt %d/%d (all %d questions) …",
                    review_attempt, max_review_attempts, len(exam),
                )
            else:
                # Subsequent passes: only review regenerated questions
                questions_to_review = [exam[i] for i in indices_to_review if i < len(exam)]
                logger.info(
                    "[Step 1b] Review attempt %d/%d (%d regenerated questions) …",
                    review_attempt, max_review_attempts, len(questions_to_review),
                )

            if not questions_to_review:
                logger.info("No questions to review; all passed.")
                break

            review_result = reviewer.review(questions_to_review)
            all_review_results.append(review_result)

            reviews = review_result.get("reviews", [])

            # Map review indices back to exam indices
            if indices_to_review is not None:
                index_map = [i for i in indices_to_review if i < len(exam)]
            else:
                index_map = list(range(len(exam)))

            # Collect rejected exam indices and their reasons
            rejected: dict[int, str] = {}
            for r in reviews:
                review_idx = r.get("question_index", -1)
                verdict = r.get("verdict", "pass")
                reasoning = r.get("reasoning", "no reason given")
                # Map back to exam index
                if 0 <= review_idx < len(index_map):
                    exam_idx = index_map[review_idx]
                else:
                    continue
                if verdict == "reject":
                    rejected[exam_idx] = reasoning
                    logger.warning("Rejected question %d: %s", exam_idx, reasoning)
                elif verdict == "flag":
                    logger.info("Flagged question %d (kept): %s", exam_idx, reasoning)

            if not rejected:
                logger.info("All %d questions passed review.", len(exam))
                break

            # Build rejection summary to feed back to the teacher
            rejection_feedback_parts: list[str] = []
            for idx in sorted(rejected):
                q_text = ""
                if 0 <= idx < len(exam):
                    q_text = (
                        exam[idx].get("question", {}).get("question", "")[:120]
                    )
                rejection_feedback_parts.append(
                    f"- Question {idx} rejected: {rejected[idx]}"
                    + (f' (question: "{q_text}…")' if q_text else "")
                )
            rejection_summary = (
                "The following questions were rejected by the exam reviewer. "
                "Do NOT repeat the same issues.\n"
                + "\n".join(rejection_feedback_parts)
            )

            # Augment feedback with rejection reasons for regeneration
            regen_feedback = (
                f"{feedback}\n\n[REVIEWER REJECTION FEEDBACK]\n{rejection_summary}"
                if feedback and feedback != "N/A"
                else rejection_summary
            )

            # Regenerate: if all rejected, regenerate the full exam;
            # otherwise regenerate only the rejected questions.
            if len(rejected) >= len(exam):
                logger.info(
                    "All %d questions rejected; regenerating full exam …",
                    len(exam),
                )
                exam = teacher.generate_exam(
                    task_description=task_description,
                    reports=round_reports,
                    proficiency=proficiency,
                    feedback=regen_feedback,
                )
                if not exam:
                    logger.warning("Teacher returned no questions on regeneration.")
                    exam = []
                # Next pass must review all since entire exam is new
                indices_to_review = None
            else:
                logger.info(
                    "%d/%d questions rejected; regenerating failed ones …",
                    len(rejected), len(exam),
                )
                regenerated_indices: list[int] = []
                for idx in sorted(rejected):
                    if idx < 0 or idx >= len(exam):
                        continue
                    # Get the report for this question
                    report_data = (
                        exam[idx].get("question", {})
                        .get("detonation_report", {})
                    )
                    if not report_data and idx < len(round_reports):
                        report_data = round_reports[idx]
                    report_with_family = dict(report_data)
                    if "_family" not in report_with_family and idx < len(round_reports):
                        report_with_family["_family"] = round_reports[idx].get(
                            "_family", "unknown"
                        )

                    replacements = teacher.generate_exam(
                        task_description=task_description,
                        reports=[report_with_family],
                        proficiency=proficiency,
                        feedback=regen_feedback,
                    )
                    if replacements:
                        exam[idx] = replacements[0]
                        regenerated_indices.append(idx)
                        logger.info("Regenerated question %d", idx)
                    else:
                        logger.warning(
                            "Failed to regenerate question %d; keeping original.", idx
                        )
                # Next pass only reviews the regenerated replacements
                indices_to_review = regenerated_indices
        else:
            # Exhausted all review attempts — drop any still-rejected questions
            logger.warning(
                "Review loop exhausted %d attempts.", max_review_attempts,
            )
            # Only review the last batch of regenerated questions
            if indices_to_review is not None and indices_to_review:
                final_subset = [exam[i] for i in indices_to_review if i < len(exam)]
                final_review = reviewer.review(final_subset)
                all_review_results.append(final_review)
                # Map back and filter
                final_rejected: set[int] = set()
                for r in final_review.get("reviews", []):
                    review_idx = r.get("question_index", -1)
                    if r.get("verdict") == "reject" and 0 <= review_idx < len(indices_to_review):
                        final_rejected.add(indices_to_review[review_idx])
                exam = [q for i, q in enumerate(exam) if i not in final_rejected]
            else:
                final_review = reviewer.review(exam)
                all_review_results.append(final_review)
                exam = reviewer.filter_exam(exam, final_review)
            logger.warning(
                "%d questions remain after final filter.", len(exam),
            )

        # Save review history
        with open(round_dir / "exam_review.json", "w") as fh:
            json.dump(all_review_results, fh, indent=2)

        if not exam:
            logger.error("No exam questions survived review; skipping round.")
            continue
        logger.info("Exam ready: %d questions", len(exam))

        # Step 2 – Student attempts exam (using TASK_PROMPT)
        logger.info("[Step 2] Student attempting exam …")

        # Build all prompts first, then run batched inference
        normalised_items: list[dict[str, Any]] = []
        prompts: list[str] = []
        for item in exam:
            # Normalise item to a dict
            if isinstance(item, str):
                item = {"question": {"question": item, "options": [],
                                     "detonation_report": {}},
                        "answer": {"correct_answers": None}}
            elif not isinstance(item, dict):
                item = {"question": {"question": str(item), "options": [],
                                     "detonation_report": {}},
                        "answer": {"correct_answers": None}}

            question_data = item.get("question", item)
            if isinstance(question_data, str):
                question_data = {"question": question_data, "options": [],
                                 "detonation_report": {}}
            elif not isinstance(question_data, dict):
                question_data = {"question": str(question_data), "options": [],
                                 "detonation_report": {}}

            # Extract fields for the prompt (CyberSOCEval-aligned)
            report_json = json.dumps(question_data.get("detonation_report", {}))
            question_text = question_data.get("question", "")
            options = question_data.get("options", [])
            if not isinstance(options, list):
                options = []
            options_text = "\n".join(str(o) for o in options)
            prompt = (task_prompt_template
                          .replace("{REPORT}", report_json)
                          .replace("{QUESTION}", str(question_text))
                          .replace("{OPTIONS}", options_text))

            normalised_items.append({"item": item, "question_data": question_data})
            prompts.append(prompt)

        # Batched inference: one CustomJob for all prompts in Rounds 2+,
        # or per-prompt API calls in Round 1.
        if hasattr(student, "generate_batch"):
            raw_responses = student.generate_batch(prompts, max_new_tokens=128)
        else:
            raw_responses = [
                student.generate(p, max_new_tokens=128, json_schema=answer_schema)
                for p in prompts
            ]

        # Parse all responses
        exam_results: list[dict[str, Any]] = []
        for norm, raw_response in zip(normalised_items, raw_responses):
            item = norm["item"]
            question_data = norm["question_data"]

            model_answer = _parse_model_answer(raw_response)

            answer_data = item.get("answer", {})
            if not isinstance(answer_data, dict):
                answer_data = {"correct_answers": list(answer_data) if isinstance(answer_data, list) else []}

            exam_results.append({
                "question": question_data,
                "answer": answer_data,
                "model_answer": model_answer,
                "parse_status": model_answer.pop("_parse_status", "ok"),
            })
            logger.debug(
                "Q: %s  →  %s",
                str(question_data.get("question", ""))[:60],
                model_answer,
            )

        with open(round_dir / "exam_results.json", "w") as fh:
            json.dump(exam_results, fh, indent=2)

        # Client-side metrics (deterministic, CyberSOCEval-aligned)
        client_metrics = _compute_exam_metrics(exam_results)
        total_parse_errors += client_metrics["parse_error_count"]
        total_questions_attempted += client_metrics["total_questions"]
        logger.info(
            "[Client metrics] Accuracy: %.2f  |  Jaccard: %.2f  |  "
            "Parse errors: %d/%d (%.0f%%)",
            client_metrics["exact_match_accuracy"],
            client_metrics["avg_jaccard"],
            client_metrics["parse_error_count"],
            client_metrics["total_questions"],
            client_metrics["parse_error_rate"] * 100,
        )

        # Step 3+4 – Teacher evaluates and generates curriculum
        logger.info("[Step 3] Teacher evaluating + generating curriculum …")
        num_examples = config.get("curriculum", {}).get("num_examples", 100)
        # Pass round reports as data_source for remediation example generation
        data_source_for_eval = json.dumps(round_reports)
        evaluation = teacher.evaluate_and_generate_curriculum(
            task_description=task_description,
            exam_results=exam_results,
            data_source=data_source_for_eval,
            num_examples=num_examples,
        )

        # Retry evaluation if it returned no dataset (the critical output)
        eval_dataset = evaluation.get("dataset", []) if isinstance(evaluation, dict) else []
        if not eval_dataset:
            logger.warning("Teacher returned no curriculum dataset; retrying evaluation …")
            evaluation = teacher.evaluate_and_generate_curriculum(
                task_description=task_description,
                exam_results=exam_results,
                data_source=data_source_for_eval,
                num_examples=num_examples,
            )

        with open(round_dir / "evaluation.json", "w") as fh:
            json.dump(evaluation, fh, indent=2)

        # Extract metrics — guard against malformed evaluation responses
        if not isinstance(evaluation, dict):
            logger.error(
                "MALFORMED EVALUATION (Round %d): Teacher returned %s instead of "
                "a JSON object. This means accuracy, proficiency, and feedback "
                "are all unknown for this round. The student will NOT be trained "
                "this round. Check the teacher prompt and raw response in "
                "%s/evaluation.json. Defaulting to: accuracy=0.0, proficiency=1, "
                "empty curriculum.",
                round_num, type(evaluation).__name__, round_dir,
            )
            evaluation = {}

        metrics = evaluation.get("metrics", {})
        if not isinstance(metrics, dict):
            logger.error(
                "MALFORMED METRICS (Round %d): 'metrics' field is %s instead of "
                "a JSON object. Accuracy and Jaccard scores cannot be extracted — "
                "both default to 0.0. The student's actual performance this round "
                "is unknown. Check %s/evaluation.json for the raw teacher response.",
                round_num, type(metrics).__name__, round_dir,
            )
            metrics = {}

        new_proficiency = evaluation.get("proficiency", 1)
        new_feedback = evaluation.get("feedback", "")
        if not isinstance(new_proficiency, (int, float)):
            raise RuntimeError(
                f"MALFORMED PROFICIENCY (Round {round_num}): 'proficiency' is "
                f"{type(new_proficiency).__name__} instead of int. "
                f"Check {round_dir}/evaluation.json. Aborting experiment."
            )
        if not isinstance(new_feedback, str):
            raise RuntimeError(
                f"MALFORMED FEEDBACK (Round {round_num}): 'feedback' is "
                f"{type(new_feedback).__name__} instead of str. "
                f"Check {round_dir}/evaluation.json. Aborting experiment."
            )
        new_dataset = evaluation.get("dataset", [])
        if not isinstance(new_dataset, list):
            raise RuntimeError(
                f"MALFORMED DATASET (Round {round_num}): 'dataset' is "
                f"{type(new_dataset).__name__} instead of list. "
                f"Check {round_dir}/evaluation.json. Aborting experiment."
            )

        # Use client-side metrics as authoritative (deterministic);
        # fall back to teacher metrics only if client computation returned 0
        accuracy = client_metrics["exact_match_accuracy"] or metrics.get("exact_match_accuracy", 0.0)
        avg_jaccard = client_metrics["avg_jaccard"] or metrics.get("avg_jaccard", 0.0)

        logger.info(
            "Proficiency: %s  |  Accuracy: %.2f  |  Jaccard: %.2f  |  "
            "Parse errors: %d",
            new_proficiency, accuracy, avg_jaccard,
            client_metrics["parse_error_count"],
        )

        weaknesses = evaluation.get("weaknesses", [])
        if not isinstance(weaknesses, list):
            raise RuntimeError(
                f"MALFORMED WEAKNESSES (Round {round_num}): 'weaknesses' field is "
                f"{type(weaknesses).__name__} instead of a list after all repair "
                f"attempts. Check {round_dir}/evaluation.json. Aborting experiment."
            )
        if weaknesses:
            weak_areas = [
                w.get("area_name", "unknown") if isinstance(w, dict) else str(w)
                for w in weaknesses[:5]
            ]
            logger.info("Top weaknesses: %s", weak_areas)

        # Record round results (includes CyberSOCEval-aligned metrics)
        round_results.append({
            "round": round_num,
            "proficiency": new_proficiency,
            "accuracy": accuracy,
            "avg_jaccard": avg_jaccard,
            "parse_error_count": client_metrics["parse_error_count"],
            "parse_error_rate": client_metrics["parse_error_rate"],
            "total_questions": client_metrics["total_questions"],
            "num_new_examples": len(new_dataset),
            "num_accumulated_examples": len(accumulated_curriculum) + len(new_dataset),
            "weaknesses": [
                w.get("area_name", "") if isinstance(w, dict) else str(w)
                for w in weaknesses
            ],
        })

        # Update state for next round (data flywheel)
        proficiency = str(new_proficiency)
        feedback = new_feedback

        # Accumulate curriculum across rounds
        accumulated_curriculum.extend(new_dataset)
        logger.info(
            "Curriculum: %d new examples, %d accumulated total",
            len(new_dataset), len(accumulated_curriculum),
        )

        with open(round_dir / "curriculum.json", "w") as fh:
            json.dump(new_dataset, fh, indent=2)

        if not accumulated_curriculum:
            logger.warning("No curriculum examples; skipping fine-tuning this round.")
            continue

        # Step 5 – Fine-tune student on accumulated curriculum
        logger.info("[Step 4] Fine-tuning student on %d accumulated examples …",
                     len(accumulated_curriculum))
        adapter_path = str(round_dir / "adapter")
        student.output_dir = adapter_path
        student.train(accumulated_curriculum)
        student.save(adapter_path)

        # Clean up base model endpoint now that a tuned endpoint exists
        if hasattr(student, "cleanup_base_endpoint"):
            student.cleanup_base_endpoint()

        # Track best performance
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_path = Path(output_dir) / "best_adapter"
            if best_path.exists() or best_path.is_symlink():
                best_path.unlink()
            best_path.symlink_to(Path(adapter_path).resolve())
            logger.info("New best accuracy %.4f; adapter at %s", best_accuracy, best_path)

        if accuracy >= target_accuracy:
            logger.info(
                "Target accuracy %.2f reached (%.4f). Stopping.",
                target_accuracy, accuracy,
            )
            break

    logger.info("Distillation loop complete. Best accuracy: %.4f", best_accuracy)

    # Clean up any remaining base model endpoint
    if hasattr(student, "cleanup_base_endpoint"):
        student.cleanup_base_endpoint()

    # Save final accumulated curriculum
    with open(Path(output_dir) / "accumulated_curriculum.json", "w") as fh:
        json.dump(accumulated_curriculum, fh, indent=2)

    # Return results for experiment orchestrator
    return {
        "teacher": teacher_name,
        "student": student_name,
        "best_accuracy": best_accuracy,
        "final_proficiency": proficiency,
        "rounds_completed": len(round_results),
        "response_parsing_error_count": total_parse_errors,
        "total_questions_attempted": total_questions_attempted,
        "qlora_inference": getattr(student, "use_qlora_inference", False),
        "round_results": round_results,
    }


# ---------------------------------------------------------------------------
# Client-side metrics (CyberSOCEval-aligned)
# ---------------------------------------------------------------------------

def _jaccard_similarity(pred: list[str], gold: list[str]) -> float:
    """Compute Jaccard similarity between two label sets.

    Returns 1.0 when both sets are empty (no answers expected or given).
    """
    pred_set, gold_set = set(pred), set(gold)
    intersection = len(pred_set & gold_set)
    union = len(pred_set | gold_set)
    return float(intersection) / union if union > 0 else 1.0


def _compute_exam_metrics(exam_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute deterministic client-side metrics from exam results.

    Returns a dict with:
      - ``exact_match_count`` / ``exact_match_accuracy``
      - ``avg_jaccard``
      - ``parse_error_count`` / ``parse_error_rate``
      - ``total_questions``
    """
    total = len(exam_results)
    if total == 0:
        return {
            "exact_match_count": 0, "exact_match_accuracy": 0.0,
            "avg_jaccard": 0.0,
            "parse_error_count": 0, "parse_error_rate": 0.0,
            "total_questions": 0,
        }

    exact_matches = 0
    jaccard_sum = 0.0
    parse_errors = 0

    for entry in exam_results:
        status = entry.get("parse_status", "ok")
        if status == "error":
            parse_errors += 1
            # Parse failures get 0.0 Jaccard (CyberSOCEval convention)
            continue

        pred = entry.get("model_answer", {}).get("correct_answers", [])
        gold_answer = entry.get("answer", {})
        gold = gold_answer.get("correct_answers") if isinstance(gold_answer, dict) else None

        # Skip scoring when gold answers are unknown (None)
        if gold is None:
            continue

        if not isinstance(pred, list):
            pred = []
        if not isinstance(gold, list):
            gold = []

        j = _jaccard_similarity(pred, gold)
        jaccard_sum += j
        if j == 1.0:
            exact_matches += 1

    scoreable = total - parse_errors
    return {
        "exact_match_count": exact_matches,
        "exact_match_accuracy": exact_matches / scoreable if scoreable > 0 else 0.0,
        "avg_jaccard": jaccard_sum / scoreable if scoreable > 0 else 0.0,
        "parse_error_count": parse_errors,
        "parse_error_rate": parse_errors / total if total > 0 else 0.0,
        "total_questions": total,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_model_answer(raw: str) -> dict[str, Any]:
    """Parse the student model's raw response into a structured answer dict.

    Aligned with CyberSOCEval: expects JSON ``{"correct_answers": ["A", "C"]}``.
    Uses robust JSON extraction, with regex letter extraction as fallback.

    Returns a dict with ``correct_answers`` and ``_parse_status``:
      - ``"ok"``       – JSON parsed successfully
      - ``"fallback"`` – regex letter extraction used
      - ``"error"``    – no answer could be extracted (parse failure)
    """
    import re
    from json_repair import extract_json

    # Primary path: extract JSON (matches CyberSOCEval's extract_json approach)
    parsed = extract_json(raw, default=None, expected_fields={
        "correct_answers": list,
    })
    if isinstance(parsed, dict) and "correct_answers" in parsed:
        parsed["_parse_status"] = "ok"
        return parsed

    # Fallback: extract option labels from free-form text
    labels = re.findall(r"\b([A-J])\b", raw.upper())
    if labels:
        return {"correct_answers": sorted(set(labels)), "_parse_status": "fallback"}

    return {"correct_answers": [], "_parse_status": "error"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SMalA – iterative teacher-student distillation for malware analysis"
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to model_config.yaml",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Maximum number of distillation rounds",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=2.0,
        dest="target_accuracy",
        help="Stop early when benchmark accuracy exceeds this value (default: 2.0 = run all rounds)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        dest="output_dir",
        help="Directory for adapters, curricula, and results",
    )
    parser.add_argument(
        "--truncate-input",
        action="store_true",
        dest="truncate_input",
        help="Apply DataFilter truncation to report context",
    )
    parser.add_argument(
        "--qlora-inference",
        action="store_true",
        dest="qlora_inference",
        help="Enable QLoRA (4-bit) for student inference. "
             "Required for 7B+ models on L4 GPU with full reports.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not os.path.exists(args.config):
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    config = load_config(args.config)
    config["_config_path"] = args.config  # pass path for StudentTrainer.from_config

    if args.qlora_inference:
        config.setdefault("student", {})["use_qlora_inference"] = True

    run_distillation_loop(
        config=config,
        rounds=args.rounds,
        target_accuracy=args.target_accuracy,
        output_dir=args.output_dir,
        truncate_input=args.truncate_input,
    )


if __name__ == "__main__":
    main()
