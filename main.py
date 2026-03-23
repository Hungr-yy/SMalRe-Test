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
    python main.py --config configs/model_config.yaml [options]

    Options:
      --config PATH            Path to model_config.yaml (default: configs/model_config.yaml)
      --rounds INT             Maximum distillation rounds (default: 5)
      --target-accuracy FLOAT  Stop when student accuracy exceeds this threshold (default: 0.80)
      --output-dir PATH        Where to save adapters and results (default: outputs/)
      --truncate-input         Apply DataFilter truncation to reports
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


def sample_data_source(
    reports: list[dict[str, Any]],
    num_reports: int = 10,
) -> str:
    """Sample a diverse batch of reports and return as a JSON string.

    Samples evenly across malware families for diversity.
    """
    if not reports:
        return "[]"

    # Group by family
    by_family: dict[str, list[dict[str, Any]]] = {}
    for r in reports:
        family = r.get("_family", "unknown")
        by_family.setdefault(family, []).append(r)

    # Sample evenly across families
    per_family = max(1, num_reports // len(by_family))
    sampled: list[dict[str, Any]] = []
    for family_reports in by_family.values():
        sampled.extend(random.sample(family_reports, min(per_family, len(family_reports))))

    # If we need more to reach num_reports, sample from remainder
    remaining = [r for r in reports if r not in sampled]
    if len(sampled) < num_reports and remaining:
        extra = min(num_reports - len(sampled), len(remaining))
        sampled.extend(random.sample(remaining, extra))

    return json.dumps(sampled[:num_reports], indent=2)


# ---------------------------------------------------------------------------
# Distillation loop
# ---------------------------------------------------------------------------

def run_distillation_loop(
    config: dict[str, Any],
    rounds: int,
    target_accuracy: float,
    output_dir: str,
    truncate_input: bool,
) -> None:
    from teacher_engine import TeacherEngine, load_template
    from student_trainer import StudentTrainer

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
    logger.info(
        "Teacher: %s / %s", teacher_cfg.get("provider"), teacher_cfg.get("model")
    )

    # ------------------------------------------------------------------
    # Initialise student
    # ------------------------------------------------------------------
    student = StudentTrainer.from_config(config["_config_path"])
    logger.info("Student: %s", config.get("student", {}).get("model_name_or_path"))

    # ------------------------------------------------------------------
    # Load task description and data source
    # ------------------------------------------------------------------
    data_cfg = config.get("data_source", {})
    task_path = data_cfg.get(
        "task_definitions", "prompt_templates/malware_analysis_task.json"
    )
    data_dir = data_cfg.get("hybrid_analysis_dir", "hybrid-analysis")
    reports_per_round = data_cfg.get("reports_per_round", 10)

    task_description = load_task_description(task_path)
    logger.info("Loaded task definitions from %s", task_path)

    all_reports = load_hybrid_analysis_reports(data_dir, truncate=truncate_input)

    # Load TASK_PROMPT template for student exam attempts
    task_prompt_template = load_template("TASK_PROMPT")

    # ------------------------------------------------------------------
    # State tracked across rounds (data flywheel)
    # ------------------------------------------------------------------
    proficiency: str = "N/A"
    feedback: str = "N/A"
    accumulated_curriculum: list[dict[str, Any]] = []
    best_accuracy = 0.0

    # ------------------------------------------------------------------
    # Iterative loop
    # ------------------------------------------------------------------
    for round_num in range(1, rounds + 1):
        logger.info("=" * 60)
        logger.info("Distillation round %d / %d", round_num, rounds)
        logger.info("=" * 60)

        round_dir = Path(output_dir) / f"round_{round_num}"
        round_dir.mkdir(parents=True, exist_ok=True)

        # Sample diverse reports for this round
        data_source = sample_data_source(all_reports, num_reports=reports_per_round)

        # Step 1 – Generate exam (informed by prior proficiency & feedback)
        logger.info("[Step 1] Generating exam (proficiency=%s) …", proficiency)
        num_q = config.get("exam", {}).get("num_questions", 5)
        exam = teacher.generate_exam(
            task_description=task_description,
            data_source=data_source,
            proficiency=proficiency,
            feedback=feedback,
            num_questions=num_q,
        )
        if not exam:
            logger.error("Teacher returned no exam questions; skipping round.")
            continue

        with open(round_dir / "exam.json", "w") as fh:
            json.dump(exam, fh, indent=2)
        logger.info("Generated %d exam questions", len(exam))

        # Step 2 – Student attempts exam (using TASK_PROMPT)
        logger.info("[Step 2] Student attempting exam …")
        exam_results: list[dict[str, Any]] = []
        for item in exam:
            question_data = item.get("question", item)
            input_json = json.dumps(question_data)
            prompt = task_prompt_template % input_json

            raw_response = student.generate(prompt, max_new_tokens=128)

            # Parse student's answer
            model_answer = _parse_model_answer(raw_response)

            exam_results.append({
                "question": question_data,
                "answer": item.get("answer", {}),
                "model_answer": model_answer,
            })
            logger.debug(
                "Q: %s  →  %s",
                question_data.get("question", "")[:60],
                model_answer,
            )

        with open(round_dir / "exam_results.json", "w") as fh:
            json.dump(exam_results, fh, indent=2)

        # Step 3+4 – Teacher evaluates and generates curriculum
        logger.info("[Step 3] Teacher evaluating + generating curriculum …")
        num_examples = config.get("curriculum", {}).get("num_examples", 100)
        evaluation = teacher.evaluate_and_generate_curriculum(
            task_description=task_description,
            exam_results=exam_results,
            data_source=data_source,
            num_examples=num_examples,
        )

        with open(round_dir / "evaluation.json", "w") as fh:
            json.dump(evaluation, fh, indent=2)

        # Extract metrics
        metrics = evaluation.get("metrics", {})
        new_proficiency = evaluation.get("proficiency", 1)
        new_feedback = evaluation.get("feedback", "")
        new_dataset = evaluation.get("dataset", [])
        accuracy = metrics.get("exact_match_accuracy", 0.0)
        avg_jaccard = metrics.get("avg_jaccard", 0.0)

        logger.info(
            "Proficiency: %s  |  Accuracy: %.2f  |  Jaccard: %.2f",
            new_proficiency, accuracy, avg_jaccard,
        )

        weaknesses = evaluation.get("weaknesses", [])
        if weaknesses:
            weak_areas = [w.get("area_name", "unknown") for w in weaknesses[:5]]
            logger.info("Top weaknesses: %s", weak_areas)

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

    # Save final accumulated curriculum
    with open(Path(output_dir) / "accumulated_curriculum.json", "w") as fh:
        json.dump(accumulated_curriculum, fh, indent=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_model_answer(raw: str) -> dict[str, Any]:
    """Parse the student model's raw response into a structured answer dict."""
    import re

    # Try to parse as JSON first
    try:
        text = raw.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(l for l in lines if not l.startswith("```")).strip()
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "correct_options" in parsed:
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: extract option labels from free-form text
    labels = re.findall(r"\b([A-J])\b", raw.upper())
    return {"correct_options": sorted(set(labels))}


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
        default=0.80,
        dest="target_accuracy",
        help="Stop early when benchmark accuracy exceeds this value",
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not os.path.exists(args.config):
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    config = load_config(args.config)
    config["_config_path"] = args.config  # pass path for StudentTrainer.from_config

    run_distillation_loop(
        config=config,
        rounds=args.rounds,
        target_accuracy=args.target_accuracy,
        output_dir=args.output_dir,
        truncate_input=args.truncate_input,
    )


if __name__ == "__main__":
    main()
