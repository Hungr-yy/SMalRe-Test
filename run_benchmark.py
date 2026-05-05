"""
run_benchmark.py
================
Run the CyberSOCEval-style malware analysis benchmark against
fine-tuned adapters from the distillation experiments.

Evaluates each adapter on the full questions.json (609 MCQs) using
Jaccard similarity scoring, matching the CyberSOCEval methodology.

Usage:
    # Evaluate all 3 Qwen Round 5 adapters + base model
    python run_benchmark.py \
        --adapters-dir outputs_qwen/qwen2.5_7b/batch_1 \
        --round 5 \
        --questions hybrid-analysis/questions.json \
        --reports-dir hybrid-analysis \
        --output-dir benchmark_results \
        --truncate-input \
        --qlora-inference \
        --include-base

    # Evaluate a single adapter
    python run_benchmark.py \
        --adapter-path outputs_qwen/qwen2.5_7b/batch_1/claude_sonnet4.6__qwen2.5_7b_qlora/round_5/adapter \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --questions hybrid-analysis/questions.json \
        --reports-dir hybrid-analysis \
        --output-dir benchmark_results \
        --qlora-inference
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# CyberSOCEval evaluation logic (from malware_analysis.py)
# ---------------------------------------------------------------------------

def jaccard_similarity(pred: list[str], gold: list[str]) -> float:
    """Jaccard similarity between two label sets."""
    pred_set, gold_set = set(pred), set(gold)
    intersection = len(pred_set & gold_set)
    union = len(pred_set | gold_set)
    return float(intersection) / union if union > 0 else 0.0


def extract_json_answer(response: str) -> dict | None:
    """Extract JSON answer from model response.

    Tries: json_repair → regex fallback → Claude LLM repair.
    """
    from json_repair import extract_json

    # Strategy 1: programmatic JSON extraction
    result = extract_json(response, default=None, expected_fields={
        "correct_answers": list,
    })
    if isinstance(result, dict) and "correct_answers" in result:
        return result

    # Strategy 2: regex letter extraction
    labels = re.findall(r"\b([A-J])\b", response.upper())
    if labels:
        return {"correct_answers": sorted(set(labels))}

    # Strategy 3: Claude LLM repair
    repaired = _llm_repair_answer(response)
    if repaired:
        return repaired

    return None


def _llm_repair_answer(response: str) -> dict | None:
    """Ask Claude to extract the answer from a malformed response."""
    try:
        from anthropic import Anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None

        client = Anthropic(api_key=api_key, max_retries=0)
        repair_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            temperature=0.0,
            messages=[{
                "role": "user",
                "content": (
                    "The following is a malformed response from a language model "
                    "that was asked to answer a multiple-choice question. "
                    "Extract the answer letters (A, B, C, etc.) and return "
                    'ONLY valid JSON: {"correct_answers": ["A", "B"]}\n\n'
                    f"Response:\n{response[:1000]}"
                ),
            }],
        )
        from json_repair import extract_json
        parsed = extract_json(
            repair_response.content[0].text,
            default=None,
            expected_fields={"correct_answers": list},
        )
        if isinstance(parsed, dict) and "correct_answers" in parsed:
            logger.info("LLM repair extracted answer from malformed response")
            return parsed
    except Exception as exc:
        logger.debug("LLM repair failed: %s", exc)
    return None


def load_questions(questions_path: str) -> list[dict[str, Any]]:
    """Load benchmark questions."""
    with open(questions_path) as f:
        questions = json.load(f)
    logger.info("Loaded %d benchmark questions from %s", len(questions), questions_path)
    return questions


def load_report(reports_dir: str, sha256: str, attack: str) -> dict[str, Any]:
    """Load a detonation report by sha256 and attack family."""
    # Try attack family subdirectory
    report_path = Path(reports_dir) / attack / sha256
    if not report_path.exists():
        # Try all subdirectories
        for subdir in Path(reports_dir).iterdir():
            if subdir.is_dir():
                candidate = subdir / sha256
                if candidate.exists():
                    report_path = candidate
                    break
    if not report_path.exists():
        return {}
    with open(report_path) as f:
        return json.load(f)


def truncate_report(report: dict[str, Any]) -> dict[str, Any]:
    """Apply CyberSOCEval truncation to a report."""
    from data_filter import DataFilter
    filt = DataFilter(mode="essential")
    return filt.filter(report)


def build_prompt(report: dict, question: str, options: list[str]) -> str:
    """Build a prompt matching the TASK_PROMPT template."""
    report_json = json.dumps(report)
    options_text = "\n".join(str(o) for o in options)
    return (
        f"Given this detonation report: {report_json}.\n\n"
        f"Answer the following multi-choice question. "
        f"Select ALL correct answers — one or more options may be correct.\n\n"
        f"Question: {question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"You need to return the list of correct answers. "
        f"Respond in a json with the following structure:\n"
        f'{{\n    "correct_answers": string[] '
        f"// The list of the letters corresponding to the correct answers, "
        f"just the letters\n}}\n\n"
        f"### Response:\n"
    )


# ---------------------------------------------------------------------------
# Inference via Vertex AI CustomJob (reuses student_trainer logic)
# ---------------------------------------------------------------------------

def run_inference_batch(
    prompts: list[str],
    base_model: str,
    adapter_path: str | None,
    staging_bucket: str,
    project: str,
    location: str,
    use_qlora: bool = False,
) -> list[str]:
    """Run batched inference using a Vertex AI CustomJob."""
    from student_trainer import VertexAIStudentTrainer

    trainer = VertexAIStudentTrainer(
        model_name_or_path=base_model,
        project=project,
        location=location,
        staging_bucket=staging_bucket,
        use_qlora_inference=use_qlora,
    )

    if adapter_path:
        trainer._adapter_gcs_path = adapter_path

    return trainer.generate_batch(prompts, max_new_tokens=128)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    questions: list[dict],
    reports_dir: str,
    base_model: str,
    adapter_gcs_path: str | None,
    staging_bucket: str,
    project: str,
    location: str,
    use_qlora: bool,
    truncate: bool,
    label: str,
    output_dir: str = "benchmark_results",
) -> dict[str, Any]:
    """Run the full benchmark and return results."""

    # Build prompts
    prompts = []
    valid_questions = []
    for q in questions:
        sha256 = q.get("sha256", "")
        attack = q.get("attack", "")
        report = load_report(reports_dir, sha256, attack)
        if not report:
            logger.warning("Report not found for sha256=%s, attack=%s", sha256, attack)
            continue
        if truncate:
            report = truncate_report(report)

        prompt = build_prompt(
            report,
            q.get("question", ""),
            q.get("options", []),
        )
        prompts.append(prompt)
        valid_questions.append(q)

    logger.info(
        "Running benchmark: %s — %d questions, adapter=%s",
        label, len(prompts), adapter_gcs_path or "base model",
    )

    # Check for existing progress (resume from crash)
    progress_path = Path(output_dir) / f".progress_{label}.jsonl" if output_dir else None
    completed_indices: set[int] = set()
    cached_responses: dict[int, str] = {}
    if progress_path and progress_path.exists():
        for line in progress_path.read_text().strip().split("\n"):
            if line:
                entry = json.loads(line)
                completed_indices.add(entry["index"])
                cached_responses[entry["index"]] = entry["response"]
        logger.info("Resuming: %d/%d prompts already completed", len(completed_indices), len(prompts))

    # Find prompts that still need inference
    remaining_indices = [i for i in range(len(prompts)) if i not in completed_indices]

    if remaining_indices:
        remaining_prompts = [prompts[i] for i in remaining_indices]
        logger.info(
            "Running inference: %d prompts in 1 batch (%d cached), adapter=%s",
            len(remaining_prompts), len(completed_indices),
            adapter_gcs_path or "base model",
        )

        responses = run_inference_batch(
            remaining_prompts, base_model, adapter_gcs_path,
            staging_bucket, project, location, use_qlora,
        )

        # Save progress
        if progress_path:
            with open(progress_path, "a") as pf:
                for idx, resp in zip(remaining_indices, responses):
                    cached_responses[idx] = resp
                    pf.write(json.dumps({"index": idx, "response": resp}) + "\n")
            logger.info("Progress saved: %d/%d total responses", len(cached_responses), len(prompts))
    else:
        logger.info("All %d prompts already cached — skipping inference", len(prompts))

    # Reconstruct ordered responses
    all_responses = [cached_responses.get(i, "") for i in range(len(prompts))]

    # Evaluate
    results = []
    correct = 0
    incorrect = 0
    parse_errors = 0
    total_jaccard = 0.0

    stats_by_topic = {}
    stats_by_difficulty = {}
    stats_by_attack = {}

    for q, response in zip(valid_questions, all_responses):
        gold = q.get("correct_answers", [])
        parsed = extract_json_answer(response)

        if parsed:
            pred = parsed.get("correct_answers", [])
            score = jaccard_similarity(pred, gold)
            total_jaccard += score
            if score == 1.0:
                correct += 1
                status = "correct"
            else:
                incorrect += 1
                status = "incorrect"
        else:
            pred = []
            score = 0.0
            parse_errors += 1
            status = "parse_error"

        result = {
            "sha256": q.get("sha256"),
            "attack": q.get("attack"),
            "topic": q.get("topic"),
            "difficulty": q.get("difficulty"),
            "question": q.get("question"),
            "gold_answers": gold,
            "model_answers": pred,
            "jaccard": score,
            "status": status,
            "raw_response": response[:500],
        }
        results.append(result)

        # Aggregate stats
        for key, stats_dict in [
            (q.get("topic", "unknown"), stats_by_topic),
            (q.get("difficulty", "unknown"), stats_by_difficulty),
            (q.get("attack", "unknown"), stats_by_attack),
        ]:
            if key not in stats_dict:
                stats_dict[key] = {"correct": 0, "incorrect": 0, "parse_error": 0, "total_jaccard": 0.0, "count": 0}
            stats_dict[key]["count"] += 1
            stats_dict[key][status] += 1 if status != "incorrect" else 0
            if status == "incorrect":
                stats_dict[key]["incorrect"] += 1
            stats_dict[key]["total_jaccard"] += score

    total = correct + incorrect + parse_errors
    accuracy = correct / total if total > 0 else 0.0
    avg_jaccard = total_jaccard / (correct + incorrect) if (correct + incorrect) > 0 else 0.0

    # Compute per-category stats
    for stats_dict in [stats_by_topic, stats_by_difficulty, stats_by_attack]:
        for key, s in stats_dict.items():
            n = s["count"]
            s["accuracy"] = s["correct"] / n if n > 0 else 0.0
            s["avg_jaccard"] = s["total_jaccard"] / n if n > 0 else 0.0

    summary = {
        "label": label,
        "base_model": base_model,
        "adapter": adapter_gcs_path or "none (base model)",
        "qlora": use_qlora,
        "total_questions": total,
        "correct": correct,
        "incorrect": incorrect,
        "parse_errors": parse_errors,
        "accuracy": accuracy,
        "avg_jaccard": avg_jaccard,
        "by_topic": stats_by_topic,
        "by_difficulty": stats_by_difficulty,
        "by_attack": stats_by_attack,
        "results": results,
    }

    logger.info(
        "Benchmark complete: %s — accuracy=%.4f, jaccard=%.4f, "
        "correct=%d, incorrect=%d, parse_errors=%d",
        label, accuracy, avg_jaccard, correct, incorrect, parse_errors,
    )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def find_adapters(adapters_dir: str, round_num: int) -> list[dict]:
    """Find all adapters for a given round in the experiment directory."""
    adapters = []
    for exp_dir in sorted(Path(adapters_dir).iterdir()):
        if not exp_dir.is_dir():
            continue
        adapter_dir = exp_dir / f"round_{round_num}" / "adapter"
        meta_path = adapter_dir / "vertex_ai_metadata.json"
        if meta_path.exists():
            meta = json.load(open(meta_path))
            adapters.append({
                "label": exp_dir.name,
                "adapter_dir": str(adapter_dir),
                "base_model": meta.get("model_name_or_path", "unknown"),
                "staging_bucket": meta.get("staging_bucket", ""),
                "qlora": meta.get("qlora_inference", False),
                "gcs_path": f"{meta.get('staging_bucket', '')}/adapters/"
                            f"smala-{exp_dir.name}-{meta.get('model_name_or_path', '').split('/')[-1]}-"
                            f"round{round_num}",
            })
    return adapters


def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run CyberSOCEval benchmark on fine-tuned adapters")

    # Adapter selection (choose one)
    parser.add_argument("--adapters-dir", help="Directory containing experiment results (finds all adapters)")
    parser.add_argument("--round", type=int, default=5, help="Which round's adapter to evaluate (default: 5)")
    parser.add_argument("--adapter-path", help="Path to a single adapter's GCS location")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct", help="Base model name")

    # Benchmark data
    parser.add_argument("--questions", default="hybrid-analysis/questions.json", help="Path to questions.json")
    parser.add_argument("--reports-dir", default="hybrid-analysis", help="Path to detonation reports directory")

    # Output
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory for results")

    # Inference config
    parser.add_argument("--truncate-input", action="store_true", help="Apply DataFilter truncation")
    parser.add_argument("--qlora-inference", action="store_true", help="Use QLoRA 4-bit inference")
    parser.add_argument("--include-base", action="store_true", help="Also benchmark the base model (no adapter) for comparison")
    parser.add_argument("--project", default=os.environ.get("VERTEX_AI_PROJECT", ""), help="GCP project")
    parser.add_argument("--location", default=os.environ.get("VERTEX_AI_LOCATION", "asia-southeast1"), help="GCP region")
    parser.add_argument("--staging-bucket", default=os.environ.get("VERTEX_AI_STAGING_BUCKET", ""), help="GCS staging bucket")

    args = parser.parse_args()

    questions = load_questions(args.questions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []

    # Base model evaluation (no adapter) for comparison
    if args.include_base:
        base_model_name = args.base_model
        if args.adapters_dir:
            # Infer base model from first adapter's metadata
            adapters_preview = find_adapters(args.adapters_dir, args.round)
            if adapters_preview:
                base_model_name = adapters_preview[0]["base_model"]

        logger.info("=" * 60)
        logger.info("Benchmarking BASE MODEL: %s (no adapter)", base_model_name)
        logger.info("=" * 60)

        base_summary = run_benchmark(
            questions=questions,
            reports_dir=args.reports_dir,
            base_model=base_model_name,
            adapter_gcs_path=None,
            staging_bucket=args.staging_bucket,
            project=args.project,
            location=args.location,
            use_qlora=args.qlora_inference,
            truncate=args.truncate_input,
            label=f"base_model_{base_model_name.split('/')[-1]}",
            output_dir=args.output_dir,
        )
        result_path = output_dir / f"base_model_{base_model_name.split('/')[-1]}.json"
        with open(result_path, "w") as f:
            json.dump(base_summary, f, indent=2)
        logger.info("Saved: %s", result_path)
        all_summaries.append(base_summary)

    if args.adapters_dir:
        # Find all adapters for the given round
        adapters = find_adapters(args.adapters_dir, args.round)
        if not adapters:
            logger.error("No adapters found in %s for round %d", args.adapters_dir, args.round)
            sys.exit(1)

        logger.info("Found %d adapters for round %d:", len(adapters), args.round)
        for a in adapters:
            logger.info("  %s: %s", a["label"], a["gcs_path"])

        for adapter in adapters:
            summary = run_benchmark(
                questions=questions,
                reports_dir=args.reports_dir,
                base_model=adapter["base_model"],
                adapter_gcs_path=adapter["gcs_path"],
                staging_bucket=adapter.get("staging_bucket") or args.staging_bucket,
                project=args.project,
                location=args.location,
                use_qlora=adapter.get("qlora") or args.qlora_inference,
                truncate=args.truncate_input,
                label=f"{adapter['label']}_round{args.round}",
                output_dir=args.output_dir,
            )
            # Save individual result
            result_path = output_dir / f"{adapter['label']}_round{args.round}.json"
            with open(result_path, "w") as f:
                json.dump(summary, f, indent=2)
            logger.info("Saved: %s", result_path)
            all_summaries.append(summary)

    elif args.adapter_path:
        # Single adapter — extract label from GCS path or local path
        # GCS: gs://smala/adapters/smala-gpt5.1__qwen2.5_7b_qlora-Qwen2.5-7B-Instruct-round1
        # We want: gpt5.1__qwen2.5_7b_qlora_round1
        adapter_name = args.adapter_path.rstrip("/").split("/")[-1]  # last segment
        # Parse: smala-<experiment_label>-<model>-round<N>
        parts = adapter_name.split("-")
        if parts[0] == "smala" and len(parts) >= 3:
            # Find round number at the end
            round_part = parts[-1]  # e.g. "round1"
            # Find model name (could contain hyphens)
            # The experiment label is between "smala-" and the model name
            # Use the known model maps to find where the model starts
            adapter_label = "-".join(parts[1:])  # drop "smala-" prefix
            # Replace the model-round suffix with just _roundN
            for model_suffix in ["Qwen2.5-7B-Instruct", "gemma-3-4b-it", "Llama-3.1-8B-Instruct"]:
                if model_suffix in adapter_label:
                    exp_part = adapter_label.split(f"-{model_suffix}")[0]
                    adapter_label = f"{exp_part}_{round_part}"
                    break
        else:
            adapter_label = adapter_name

        summary = run_benchmark(
            questions=questions,
            reports_dir=args.reports_dir,
            base_model=args.base_model,
            adapter_gcs_path=args.adapter_path,
            staging_bucket=args.staging_bucket,
            project=args.project,
            location=args.location,
            use_qlora=args.qlora_inference,
            truncate=args.truncate_input,
            label=adapter_label,
            output_dir=args.output_dir,
        )
        result_path = output_dir / f"{adapter_label}_benchmark.json"
        with open(result_path, "w") as f:
            json.dump(summary, f, indent=2)
        all_summaries.append(summary)

    else:
        logger.error("Provide either --adapters-dir or --adapter-path")
        sys.exit(1)

    # Print comparison table
    base_accuracy = None
    for s in all_summaries:
        if "base_model" in s["label"]:
            base_accuracy = s["accuracy"]
            break

    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS (CyberSOCEval — 609 questions)")
    print("=" * 100)
    print(f"{'Label':<45} {'Accuracy':>8} {'Jaccard':>8} {'Correct':>8} {'Parse Err':>9} {'vs Base':>8}")
    print("-" * 100)
    for s in all_summaries:
        delta = ""
        if base_accuracy is not None and "base_model" not in s["label"]:
            diff = s["accuracy"] - base_accuracy
            delta = f"{diff:+.4f}"
        print(
            f"{s['label']:<45} "
            f"{s['accuracy']:>8.4f} "
            f"{s['avg_jaccard']:>8.4f} "
            f"{s['correct']:>8} "
            f"{s['parse_errors']:>9} "
            f"{delta:>8}"
        )
    print("=" * 100)

    # Save combined summary
    summary_path = output_dir / "benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "adapters_evaluated": len(all_summaries),
            "questions_per_adapter": len(questions),
            "results": [{k: v for k, v in s.items() if k != "results"} for s in all_summaries],
        }, f, indent=2)
    logger.info("Summary saved: %s", summary_path)


if __name__ == "__main__":
    main()
