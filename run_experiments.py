"""
run_experiments.py
==================
Orchestrates the 3×3 teacher-student experiment matrix across 3 sequential
batches, running up to 3 experiments in parallel per batch.

Experiment matrix (Latin square):
    Batch 1: T1→S1, T2→S2, T3→S3
    Batch 2: T1→S2, T2→S3, T3→S1  (all models reset)
    Batch 3: T1→S3, T2→S1, T3→S2  (all models reset)

Teachers:
    T1 = GPT-4o          (OpenAI)
    T2 = Gemini 2.5 Pro  (Google)
    T3 = Claude Sonnet 4.6 (Anthropic)

Students:
    S1 = Llama 3.1 8B Instruct       (Meta)
    S2 = Gemma 3 4B                   (Google)
    S3 = Qwen 2.5 7B Instruct         (Alibaba / Qwen)

Each experiment saves its adapters and results under:
    outputs/<batch_N>/<teacher_short>_<student_short>/

After all 9 experiments complete (including retries of failures), a summary
comparison table is saved to:
    outputs/experiment_summary.json

Fault tolerance:
    - Failed experiments are automatically retried (up to --max-retries times)
    - Successful experiments are verified (adapter + result files exist on disk)
    - A persistent tracker (tracker.json) records pass/fail status throughout
    - Progress from completed experiments is never lost during retries

Usage
-----
    python run_experiments.py --config configs/model_config.yaml [options]

    Options:
      --config PATH            Base config for training/LoRA hyperparameters
      --rounds INT             Distillation rounds per experiment (default: 5)
      --target-accuracy FLOAT  Early-stop threshold per experiment (default: 0.80)
      --output-dir PATH        Root output directory (default: outputs)
      --max-parallel INT       Max parallel experiments per batch (default: 3)
      --max-retries INT        Max retry rounds for failed experiments (default: 2)
      --truncate-input         Apply DataFilter truncation to reports
      --teachers NAME [NAME ...] Select teachers (default: all)
      --students NAME [NAME ...] Select students (default: all)

    Examples:
      # All 3 teachers → Qwen only
      python run_experiments.py --teachers gpt4o gemini2.5pro claude_sonnet4.6 --students qwen2.5_7b

      # Gemini + Claude → Llama + Gemma
      python run_experiments.py --teachers gemini2.5pro claude_sonnet4.6 --students llama3.1_8b gemma3_4b
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()  # Load API keys from .env file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("experiments")


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

TEACHERS = [
    {
        "name": "gpt4o",
        "provider": "openai",
        "model": "gpt-4o",
    },
    {
        "name": "gemini2.5pro",
        "provider": "google",
        "model": "gemini-2.5-pro",
    },
    {
        "name": "claude_sonnet4.6",
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
    },
]

STUDENTS = [
    {
        "name": "llama3.1_8b",
        "model_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
    },
    {
        "name": "gemma3_4b",
        "model_name_or_path": "google/gemma-3-4b-it",
    },
    {
        "name": "qwen2.5_7b",
        "model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
    },
]

# Name-to-index lookup for CLI filtering
_TEACHER_NAMES = {t["name"]: i for i, t in enumerate(TEACHERS)}
_STUDENT_NAMES = {s["name"]: i for i, s in enumerate(STUDENTS)}


def _build_batches(
    teacher_indices: list[int],
    student_indices: list[int],
    max_parallel: int = 3,
) -> list[list[tuple[int, int]]]:
    """Build batches of (teacher_idx, student_idx) pairs.

    All selected teacher×student combinations are generated, then chunked
    into batches of ``max_parallel`` experiments each.
    """
    pairs = [
        (ti, si)
        for ti in teacher_indices
        for si in student_indices
    ]
    # Chunk into batches
    return [pairs[i:i + max_parallel] for i in range(0, len(pairs), max_parallel)]


# Default: full 3×3 Latin square (used by tests and when no --teachers/--students given)
BATCHES = _build_batches(list(range(len(TEACHERS))), list(range(len(STUDENTS))))


# ---------------------------------------------------------------------------
# Experiment tracker
# ---------------------------------------------------------------------------

class ExperimentTracker:
    """Tracks pass/fail status of all experiments and their artifacts.

    Persists to ``tracker.json`` in the output directory so that state
    survives orchestrator restarts.
    """

    def __init__(self, root_output_dir: str) -> None:
        self.root_output_dir = Path(root_output_dir)
        self.tracker_path = self.root_output_dir / "tracker.json"
        # Key: "Ti__Sj" label, Value: result dict
        self.experiments: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load existing tracker state from disk if available."""
        if self.tracker_path.exists():
            try:
                with open(self.tracker_path) as fh:
                    data = json.load(fh)
                self.experiments = data.get("experiments", {})
                logger.info(
                    "Loaded tracker: %d experiments (%d completed, %d failed)",
                    len(self.experiments),
                    self.num_completed,
                    self.num_failed,
                )
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not load tracker.json; starting fresh.")

    def save(self) -> None:
        """Persist tracker state to disk."""
        self.root_output_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "experiments": self.experiments,
            "summary": {
                "total": len(self.experiments),
                "completed": self.num_completed,
                "failed": self.num_failed,
                "verified": self.num_verified,
            },
        }
        with open(self.tracker_path, "w") as fh:
            json.dump(data, fh, indent=2)

    def record(self, label: str, pair: tuple[int, int], result: dict[str, Any]) -> None:
        """Record the result of a single experiment."""
        self.experiments[label] = {
            "teacher_idx": pair[0],
            "student_idx": pair[1],
            "teacher_name": TEACHERS[pair[0]]["name"],
            "student_name": STUDENTS[pair[1]]["name"],
            "status": result.get("status", "unknown"),
            "output_dir": result.get("output_dir", ""),
            "best_accuracy": result.get("best_accuracy", 0.0),
            "final_proficiency": result.get("final_proficiency", "N/A"),
            "rounds_completed": result.get("rounds_completed", 0),
            "duration_seconds": result.get("duration_seconds", 0),
            "error": result.get("error"),
            "artifacts_verified": False,
            "result": result,
        }
        self.save()

    def record_batch(
        self,
        pairs: list[tuple[int, int]],
        results: list[dict[str, Any]],
    ) -> None:
        """Record results from a batch, matching results to pairs by label."""
        # Build a label→pair map
        label_to_pair = {}
        for ti, si in pairs:
            label = f"{TEACHERS[ti]['name']}__{STUDENTS[si]['name']}"
            label_to_pair[label] = (ti, si)

        for result in results:
            label = result.get("experiment", "")
            pair = label_to_pair.get(label)
            if pair is not None:
                self.record(label, pair, result)

    def verify_artifacts(self) -> list[str]:
        """Verify that completed experiments have saved their artifacts.

        Checks for:
          - ``experiment_result.json`` in the experiment output directory
          - ``best_adapter/`` directory (or symlink) exists

        Returns a list of labels that failed verification (demoted to failed).
        """
        demoted: list[str] = []

        for label, exp in self.experiments.items():
            if exp["status"] != "completed":
                continue

            output_dir = Path(exp.get("output_dir", ""))
            issues = []

            # Check experiment_result.json
            result_path = output_dir / "experiment_result.json"
            if not result_path.exists():
                issues.append("missing experiment_result.json")

            # Check adapter checkpoint exists (local or Vertex AI)
            best_adapter = output_dir / "best_adapter"
            vertex_meta = output_dir / "vertex_ai_metadata.json"
            if not best_adapter.exists() and not vertex_meta.exists():
                # Also check if any round adapter exists
                has_any_adapter = any(
                    (output_dir / f"round_{i}" / "adapter").exists()
                    for i in range(1, 20)
                )
                if not has_any_adapter:
                    issues.append("no adapter checkpoint found")

            if issues:
                logger.warning(
                    "Experiment %s FAILED verification: %s — demoting to failed",
                    label, "; ".join(issues),
                )
                exp["status"] = "failed"
                exp["error"] = f"Artifact verification failed: {'; '.join(issues)}"
                exp["artifacts_verified"] = False
                demoted.append(label)
            else:
                exp["artifacts_verified"] = True
                logger.info("Experiment %s verified OK", label)

        self.save()
        return demoted

    def get_failed_pairs(self) -> list[tuple[int, int]]:
        """Return the (teacher_idx, student_idx) pairs for all failed experiments."""
        failed = []
        for exp in self.experiments.values():
            if exp["status"] != "completed":
                failed.append((exp["teacher_idx"], exp["student_idx"]))
        return failed

    def get_failed_labels(self) -> list[str]:
        """Return labels of all failed experiments."""
        return [label for label, exp in self.experiments.items()
                if exp["status"] != "completed"]

    @property
    def num_completed(self) -> int:
        return sum(1 for e in self.experiments.values() if e["status"] == "completed")

    @property
    def num_failed(self) -> int:
        return sum(1 for e in self.experiments.values() if e["status"] != "completed")

    @property
    def num_verified(self) -> int:
        return sum(1 for e in self.experiments.values() if e.get("artifacts_verified"))

    def all_results(self) -> list[dict[str, Any]]:
        """Return the full result dicts for all experiments."""
        return [exp.get("result", exp) for exp in self.experiments.values()]


# ---------------------------------------------------------------------------
# Experiment config builder
# ---------------------------------------------------------------------------

def build_experiment_config(
    teacher: dict[str, str],
    student: dict[str, str],
    base_config: dict[str, Any],
    output_dir: str,
) -> dict[str, Any]:
    """Build a complete config dict for a single experiment.

    Merges the teacher/student definitions with the base config
    (training hyperparameters, LoRA settings, data source paths).
    """
    config = copy.deepcopy(base_config)

    # Override teacher
    config["teacher"] = {
        "provider": teacher["provider"],
        "model": teacher["model"],
        "api_key": "",  # read from environment
        "temperature": base_config.get("teacher", {}).get("temperature", 0.7),
    }

    # Override student – merge with base config to preserve backend and other fields
    config["student"] = {
        **config.get("student", {}),
        "model_name_or_path": student["model_name_or_path"],
    }

    # Remove _config_path to force programmatic StudentTrainer construction
    config.pop("_config_path", None)

    config["output_dir"] = output_dir

    return config


# ---------------------------------------------------------------------------
# Single experiment runner (called in subprocess)
# ---------------------------------------------------------------------------

def _run_single_experiment(
    experiment_config: dict[str, Any],
    rounds: int,
    target_accuracy: float,
    output_dir: str,
    truncate_input: bool,
    experiment_label: str,
) -> dict[str, Any]:
    """Run one distillation loop.  Designed to be called via ProcessPoolExecutor."""
    # Re-configure logging for the subprocess
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [{experiment_label}] %(name)s: %(message)s",
        force=True,
    )
    sub_logger = logging.getLogger("experiments")
    sub_logger.info("Starting experiment: %s", experiment_label)

    start_time = time.time()

    try:
        from main import run_distillation_loop

        results = run_distillation_loop(
            config=experiment_config,
            rounds=rounds,
            target_accuracy=target_accuracy,
            output_dir=output_dir,
            truncate_input=truncate_input,
        )
        results["experiment"] = experiment_label
        results["output_dir"] = output_dir
        results["status"] = "completed"
        results["duration_seconds"] = round(time.time() - start_time, 1)

    except BaseException as exc:
        sub_logger.error("Experiment %s failed: %s", experiment_label, exc, exc_info=True)
        results = {
            "experiment": experiment_label,
            "output_dir": output_dir,
            "status": "failed",
            "error": str(exc),
            "duration_seconds": round(time.time() - start_time, 1),
        }

    # Persist result to disk so it survives orchestrator crashes
    try:
        result_path = Path(output_dir) / "experiment_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as fh:
            json.dump(results, fh, indent=2)
    except OSError:
        sub_logger.warning("Could not save experiment_result.json for %s", experiment_label)

    sub_logger.info(
        "Experiment %s finished in %.1fs — status: %s",
        experiment_label, results["duration_seconds"], results["status"],
    )
    return results


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    batch_label: str,
    pairs: list[tuple[int, int]],
    base_config: dict[str, Any],
    rounds: int,
    target_accuracy: float,
    root_output_dir: str,
    max_parallel: int,
    truncate_input: bool,
) -> list[dict[str, Any]]:
    """Run a batch of experiments (up to ``max_parallel`` at a time).

    Parameters
    ----------
    batch_label:
        Label for this batch (used for output directory naming, e.g.
        ``"batch_1"`` or ``"retry_1"``).
    pairs:
        List of ``(teacher_index, student_index)`` tuples.
    base_config:
        Base config with training/LoRA/data_source settings.
    rounds:
        Distillation rounds per experiment.
    target_accuracy:
        Early-stop threshold.
    root_output_dir:
        Root output directory.
    max_parallel:
        Maximum concurrent experiments.
    truncate_input:
        Whether to truncate report context.

    Returns
    -------
    list[dict]
        Results from each experiment in this batch.
    """
    batch_dir = Path(root_output_dir) / batch_label
    batch_dir.mkdir(parents=True, exist_ok=True)

    experiments = []
    for ti, si in pairs:
        teacher = TEACHERS[ti]
        student = STUDENTS[si]
        label = f"{teacher['name']}__{student['name']}"
        exp_output_dir = str(batch_dir / label)

        config = build_experiment_config(
            teacher=teacher,
            student=student,
            base_config=base_config,
            output_dir=exp_output_dir,
        )
        experiments.append((config, label, exp_output_dir))

    logger.info(
        "%s: running %d experiments (max_parallel=%d)",
        batch_label, len(experiments), max_parallel,
    )
    for _, label, _ in experiments:
        logger.info("  • %s", label)

    batch_results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for config, label, exp_output_dir in experiments:
            future = executor.submit(
                _run_single_experiment,
                experiment_config=config,
                rounds=rounds,
                target_accuracy=target_accuracy,
                output_dir=exp_output_dir,
                truncate_input=truncate_input,
                experiment_label=label,
            )
            futures[future] = label

        for future in as_completed(futures):
            label = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                logger.error("Experiment %s raised: %s", label, exc)
                result = {
                    "experiment": label,
                    "status": "failed",
                    "error": str(exc),
                }
            batch_results.append(result)

    # Save batch results
    with open(batch_dir / "batch_results.json", "w") as fh:
        json.dump(batch_results, fh, indent=2)

    return batch_results


# ---------------------------------------------------------------------------
# Full orchestration
# ---------------------------------------------------------------------------

def run_all_experiments(
    base_config: dict[str, Any],
    rounds: int,
    target_accuracy: float,
    root_output_dir: str,
    max_parallel: int,
    truncate_input: bool,
    max_retries: int = 2,
) -> list[dict[str, Any]]:
    """Run all 9 experiments across 3 sequential batches, then retry failures.

    Flow:
      1. Run 3 initial batches (Latin square pairing)
      2. Verify artifacts of all completed experiments
      3. Collect failed experiments
      4. Retry failed experiments (up to ``max_retries`` rounds)
      5. Produce final summary

    Between batches, all student models are reset (fresh base weights,
    no accumulated curriculum) so each experiment starts clean.

    Returns
    -------
    list[dict]
        All experiment results (including retries).
    """
    tracker = ExperimentTracker(root_output_dir)
    total_start = time.time()

    # ------------------------------------------------------------------
    # Phase 1: Run the 3 initial batches
    # ------------------------------------------------------------------
    for batch_num, pairs in enumerate(BATCHES, start=1):
        # Skip pairs that already completed (from a previous run)
        remaining_pairs = [
            p for p in pairs
            if f"{TEACHERS[p[0]]['name']}__{STUDENTS[p[1]]['name']}"
            not in tracker.experiments
            or tracker.experiments[
                f"{TEACHERS[p[0]]['name']}__{STUDENTS[p[1]]['name']}"
            ].get("status") != "completed"
        ]

        if not remaining_pairs:
            logger.info("Batch %d: all experiments already completed, skipping.", batch_num)
            continue

        logger.info("=" * 70)
        logger.info("BATCH %d / %d", batch_num, len(BATCHES))
        logger.info("=" * 70)

        batch_results = run_batch(
            batch_label=f"batch_{batch_num}",
            pairs=remaining_pairs,
            base_config=base_config,
            rounds=rounds,
            target_accuracy=target_accuracy,
            root_output_dir=root_output_dir,
            max_parallel=max_parallel,
            truncate_input=truncate_input,
        )
        tracker.record_batch(remaining_pairs, batch_results)

        completed = sum(1 for r in batch_results if r.get("status") == "completed")
        logger.info(
            "Batch %d complete. %d/%d experiments succeeded.",
            batch_num, completed, len(batch_results),
        )

        if batch_num < len(BATCHES):
            logger.info("Resetting all models for next batch …")

    # ------------------------------------------------------------------
    # Phase 2: Verify artifacts of completed experiments
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("VERIFYING ARTIFACTS")
    logger.info("=" * 70)
    demoted = tracker.verify_artifacts()
    if demoted:
        logger.warning(
            "%d experiments demoted to failed after artifact verification: %s",
            len(demoted), demoted,
        )

    logger.info(
        "Tracker: %d completed, %d failed, %d verified",
        tracker.num_completed, tracker.num_failed, tracker.num_verified,
    )

    # ------------------------------------------------------------------
    # Phase 3: Retry failed experiments
    # ------------------------------------------------------------------
    for retry_num in range(1, max_retries + 1):
        failed_pairs = tracker.get_failed_pairs()
        if not failed_pairs:
            logger.info("All experiments succeeded. No retries needed.")
            break

        failed_labels = tracker.get_failed_labels()
        logger.info("=" * 70)
        logger.info("RETRY %d / %d  (%d failed experiments)", retry_num, max_retries, len(failed_pairs))
        logger.info("Retrying: %s", failed_labels)
        logger.info("=" * 70)

        retry_results = run_batch(
            batch_label=f"retry_{retry_num}",
            pairs=failed_pairs,
            base_config=base_config,
            rounds=rounds,
            target_accuracy=target_accuracy,
            root_output_dir=root_output_dir,
            max_parallel=max_parallel,
            truncate_input=truncate_input,
        )
        tracker.record_batch(failed_pairs, retry_results)

        # Verify retried experiments
        demoted = tracker.verify_artifacts()
        if demoted:
            logger.warning("Retry %d: %d experiments still failing verification", retry_num, len(demoted))

        completed = sum(1 for r in retry_results if r.get("status") == "completed")
        logger.info(
            "Retry %d complete. %d/%d experiments recovered.",
            retry_num, completed, len(retry_results),
        )

    # ------------------------------------------------------------------
    # Phase 4: Final summary
    # ------------------------------------------------------------------
    total_duration = time.time() - total_start
    all_results = tracker.all_results()

    # Log final status of any still-failed experiments
    still_failed = tracker.get_failed_labels()
    if still_failed:
        logger.error(
            "PERMANENTLY FAILED experiments (after %d retries): %s",
            max_retries, still_failed,
        )
    else:
        logger.info("All 9 experiments completed and verified successfully.")

    summary = {
        "total_experiments": len(tracker.experiments),
        "completed": tracker.num_completed,
        "failed": tracker.num_failed,
        "verified": tracker.num_verified,
        "retries_used": max(0, min(max_retries, sum(
            1 for i in range(1, max_retries + 1)
            if (Path(root_output_dir) / f"retry_{i}").exists()
        ))),
        "permanently_failed": still_failed,
        "total_duration_seconds": round(total_duration, 1),
        "experiments": all_results,
        "comparison_table": _build_comparison_table(all_results),
    }

    summary_path = Path(root_output_dir) / "experiment_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Summary saved to %s", summary_path)

    _print_comparison_table(summary["comparison_table"])

    return all_results


def _build_comparison_table(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a flat comparison table from experiment results."""
    table = []
    for r in results:
        entry = {
            "experiment": r.get("experiment", "unknown"),
            "teacher": r.get("teacher", "unknown"),
            "student": r.get("student", "unknown"),
            "status": r.get("status", "unknown"),
            "best_accuracy": r.get("best_accuracy", 0.0),
            "final_proficiency": r.get("final_proficiency", "N/A"),
            "rounds_completed": r.get("rounds_completed", 0),
            "duration_seconds": r.get("duration_seconds", 0),
        }
        table.append(entry)

    # Sort by best_accuracy descending
    table.sort(key=lambda x: x.get("best_accuracy", 0), reverse=True)
    return table


def _print_comparison_table(table: list[dict[str, Any]]) -> None:
    """Print a formatted comparison table to the log."""
    logger.info("")
    logger.info("=" * 90)
    logger.info("EXPERIMENT RESULTS COMPARISON")
    logger.info("=" * 90)
    logger.info(
        "%-40s  %-8s  %-10s  %-6s  %-8s",
        "Experiment", "Status", "Accuracy", "Prof.", "Rounds",
    )
    logger.info("-" * 90)
    for row in table:
        logger.info(
            "%-40s  %-8s  %-10.4f  %-6s  %-8d",
            row["experiment"],
            row["status"],
            row.get("best_accuracy", 0.0),
            row.get("final_proficiency", "N/A"),
            row.get("rounds_completed", 0),
        )
    logger.info("=" * 90)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SMalA – 3×3 teacher-student experiment matrix orchestrator.\n"
            "Runs 9 experiments across 3 sequential batches with parallel "
            "execution within each batch.  Failed experiments are automatically "
            "retried."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Base config for training/LoRA hyperparameters (default: configs/model_config.yaml)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=5,
        help="Distillation rounds per experiment (default: 5)",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.80,
        dest="target_accuracy",
        help="Early-stop threshold per experiment (default: 0.80)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        dest="output_dir",
        help="Root output directory (default: outputs)",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=3,
        dest="max_parallel",
        help="Max parallel experiments per batch (default: 3, set to 1 for single GPU)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        dest="max_retries",
        help="Max retry rounds for failed experiments (default: 2)",
    )
    parser.add_argument(
        "--truncate-input",
        action="store_true",
        dest="truncate_input",
        help="Apply DataFilter truncation to reports",
    )
    parser.add_argument(
        "--teachers",
        nargs="+",
        metavar="NAME",
        help=(
            "Teacher names to run (default: all). "
            f"Choices: {', '.join(_TEACHER_NAMES)}"
        ),
    )
    parser.add_argument(
        "--students",
        nargs="+",
        metavar="NAME",
        help=(
            "Student names to run (default: all). "
            f"Choices: {', '.join(_STUDENT_NAMES)}"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not os.path.exists(args.config):
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)

    # Resolve selected teachers/students (default: all)
    if args.teachers:
        teacher_indices = []
        for name in args.teachers:
            if name not in _TEACHER_NAMES:
                logger.error("Unknown teacher: %s. Choices: %s", name, list(_TEACHER_NAMES))
                sys.exit(1)
            teacher_indices.append(_TEACHER_NAMES[name])
    else:
        teacher_indices = list(range(len(TEACHERS)))

    if args.students:
        student_indices = []
        for name in args.students:
            if name not in _STUDENT_NAMES:
                logger.error("Unknown student: %s. Choices: %s", name, list(_STUDENT_NAMES))
                sys.exit(1)
            student_indices.append(_STUDENT_NAMES[name])
    else:
        student_indices = list(range(len(STUDENTS)))

    # Build batches dynamically from selected models
    global BATCHES
    BATCHES = _build_batches(teacher_indices, student_indices, args.max_parallel)

    selected_teachers = [TEACHERS[i]["name"] for i in teacher_indices]
    selected_students = [STUDENTS[i]["name"] for i in student_indices]
    num_experiments = len(teacher_indices) * len(student_indices)

    from main import load_config

    base_config = load_config(args.config)

    logger.info("SMalA Experiment Matrix")
    logger.info("Teachers:    %s", selected_teachers)
    logger.info("Students:    %s", selected_students)
    logger.info("Experiments: %d (%d batches)", num_experiments, len(BATCHES))
    logger.info("Rounds:      %d per experiment", args.rounds)
    logger.info("Parallel:    %d per batch", args.max_parallel)
    logger.info("Retries:     %d max", args.max_retries)
    logger.info("")

    run_all_experiments(
        base_config=base_config,
        rounds=args.rounds,
        target_accuracy=args.target_accuracy,
        root_output_dir=args.output_dir,
        max_parallel=args.max_parallel,
        truncate_input=args.truncate_input,
        max_retries=args.max_retries,
    )


if __name__ == "__main__":
    main()
