# SMalA — Small Language Model Malware Analysis

**Fine-tuning Small Language Models (SLMs) to perform malware analysis via teacher-student knowledge distillation.**

---

## Overview

SMalA implements an automated, iterative **teacher-student knowledge distillation** pipeline that trains a lightweight SLM to reverse engineer malware. A powerful "teacher" LLM (e.g., GPT-4o, Gemini 2.5 Pro, or Claude Sonnet 4.6) generates exams, evaluates student responses, and produces a tailored synthetic curriculum that is used to fine-tune a compact "student" SLM (e.g., Llama 3.3 8B Instruct, Gemma 3 4B, or Qwen2.5-Coder-3B-Instruct).

```
Teacher LLM (GPT-4o / Gemini 2.5 Pro / Claude Sonnet 4.6)
       │
       ▼  1. Generate exam from malware detonation report
       │
Student SLM (Llama 3.3 8B Instruct / Gemma 3 4B / Qwen2.5-Coder-3B-Instruct)
       │
       ▼  2. Student attempts exam
       │
Teacher LLM
       │
       ▼  3. Evaluate answers, identify weaknesses
       │
Teacher LLM
       │
       ▼  4. Generate targeted curriculum (10k–100k examples)
       │
Student SLM fine-tuning (LoRA / QLoRA via PEFT)
       │
       ▼  5. Repeat until benchmark target is reached
```

---

## Repository Structure

```
SMalA/
├── main.py                          # Single-experiment distillation loop
├── run_experiments.py               # 3×3 experiment matrix orchestrator
├── teacher_engine.py                # Teacher LLM: exam, evaluation, curriculum
├── student_trainer.py               # Student SLM: LoRA/QLoRA fine-tuning
├── report_parser.py                 # Malware detonation report ingestion
├── data_filter.py                   # Context-window-aware report filtering
├── eval_suite.py                    # Accuracy + Jaccard benchmarking
├── requirements.txt
├── configs/
│   └── model_config.yaml            # Teacher/student model parameters
├── prompt_templates/
│   ├── EXAM_PROMPT                  # Teacher exam generation template
│   ├── TASK_PROMPT                  # Student exam attempt template
│   ├── EXAM_EVALUATION              # Evaluation + curriculum generation template
│   └── malware_analysis_task.json   # Structured task definitions
└── hybrid-analysis/                 # CyberSOCEval benchmark datasets
    ├── infostealers/                # 44 infostealer detonation reports
    ├── killers/                     # 30 EDR/AV killer reports
    ├── ransomware/                  # 34 ransomware reports
    ├── remcos/                      # 32 Remcos RAT reports
    ├── um_unhooking/                # 29 UM unhooking reports
    └── questions.json               # 100+ benchmark questions
```

---

## Selected Model Families

| Role    | Provider  | Model                                    |
|---------|-----------|------------------------------------------|
| Teacher | OpenAI    | GPT-4o                                   |
| Teacher | Google    | Gemini 2.5 Pro                           |
| Teacher | Anthropic | Claude Sonnet 4.6                        |
| Student | Meta      | Llama 3.3 8B Instruct                    |
| Student | Google    | Gemma 3 4B                               |
| Student | Alibaba   | Qwen2.5-Coder-3B-Instruct                |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API keys

```bash
cp .env.example .env
# Edit .env and fill in your keys:
#   OPENAI_API_KEY=sk-...
#   GOOGLE_API_KEY=AIza...
#   ANTHROPIC_API_KEY=sk-ant-...
```

The `.env` file is in `.gitignore` and will not be committed.

### 3. Run a single experiment

```bash
python main.py --config configs/model_config.yaml --truncate-input
```

This runs one teacher-student pair as defined in `model_config.yaml`. The orchestrator will:
1. Generate an exam from malware detonation reports
2. Have the student SLM attempt the exam
3. Evaluate answers and identify weaknesses
4. Produce a targeted synthetic curriculum
5. Fine-tune the student with LoRA/QLoRA
6. Repeat until the benchmark target accuracy is reached

### 4. Run the full 3×3 experiment matrix

```bash
python run_experiments.py --config configs/model_config.yaml --truncate-input
```

This runs all 9 teacher-student combinations across 3 sequential batches (3 experiments in parallel per batch). Between batches, all models are reset. Failed experiments are automatically retried (up to 2 times).

| Batch | Experiment 1       | Experiment 2            | Experiment 3             |
|-------|--------------------|-------------------------|--------------------------|
| 1     | GPT-4o → Llama 3.3 | Gemini 2.5 Pro → Gemma 3 | Claude Sonnet 4.6 → Qwen2.5 |
| 2     | GPT-4o → Gemma 3   | Gemini 2.5 Pro → Qwen2.5 | Claude Sonnet 4.6 → Llama 3.3 |
| 3     | GPT-4o → Qwen2.5   | Gemini 2.5 Pro → Llama 3.3 | Claude Sonnet 4.6 → Gemma 3 |

Each experiment saves adapters and results to `outputs/batch_N/<teacher>__<student>/`. A summary comparison table is saved to `outputs/experiment_summary.json`.

**Useful flags:**

```bash
# Single GPU (run experiments sequentially within each batch)
--max-parallel 1

# Fewer rounds for a quick test run
--rounds 2

# Custom output directory
--output-dir my_results/
```

**If the experiment gets interrupted**, just run the same command again. The tracker (`outputs/tracker.json`) remembers which experiments completed, and only failed/unfinished ones will rerun.

---

## Benchmarking (CyberSOCEval / Malware Analysis)

SMalA is aligned with the **CyberSOCEval** malware analysis benchmark. The `eval_suite.py` module reports:

- **Accuracy** — exact-match on multiple-choice questions
- **Average Jaccard Score** — partial-credit metric for overlapping answer sets

Malware families covered: EDR/AV Killers, Ransomware, RATs (Remcos), Infostealers, UM Unhooking.

To run against the CyberSOCEval suite directly:

```bash
python3 -m CybersecurityBenchmarks.benchmark.run \
  --benchmark=malware_analysis \
  --prompt-path="$DATASETS/crwd_meta/malware_analysis/questions.json" \
  --response-path="$DATASETS/crwd_meta/malware_analysis/responses.json" \
  --judge-response-path="$DATASETS/crwd_meta/malware_analysis/judge_responses.json" \
  --stat-path="$DATASETS/crwd_meta/malware_analysis/stats.json" \
  --llm-under-test=<PROVIDER>::<MODEL>::<API_KEY> \
  --truncate-input
```

---

## Context Window Management

Hybrid Analysis sandbox reports often exceed 128k tokens. `data_filter.py` implements pre-filtering that retains only essential fields (`total_processes`, `mitre_attcks`, `signatures`) with negligible impact on benchmark performance.

---

## Acknowledgements and Disclaimers

SmalA is a project originally made for Nanyang Technological University's Final Year Project Module SC4079. Special thanks to Gu Wenbo and Professor Liu Yang for their guidance and support in academic matters.

SmalA has not been implemented or tested in a production environment. Please use it at your own risk. None of the project contributors assume any responsibility for any damages through this project.

## License

MIT — see [LICENSE](LICENSE).
