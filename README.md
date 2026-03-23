# SMalA — Small Language Model Malware Analysis

**Fine-tuning Small Language Models (SLMs) to perform malware analysis via teacher-student knowledge distillation.**

---

## Overview

SMalA implements an automated, iterative **teacher-student knowledge distillation** pipeline that trains a lightweight SLM to reverse engineer malware. A powerful "teacher" LLM (e.g., GPT-4o, Gemini 2.5 Pro, or Claude Sonnet 4.6) generates exams, evaluates student responses, and produces a tailored synthetic curriculum that is used to fine-tune a compact "student" SLM (e.g., Llama 3.3 8B Instruct, Gemma 3 4B, or Qwen 2.5 7B Instruct).

```
Teacher LLM (GPT-4o / Gemini 2.5 Pro / Claude Sonnet 4.6)
       │
       ▼  1. Generate exam from malware detonation report
       │
Student SLM (Llama 3.3 8B Instruct / Gemma 3 4B / Qwen 2.5 7B Instruct)
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
├── tests/
│   ├── test_fault_tolerance.py      # Orchestrator resilience tests
│   └── test_vertex_ai_workflow.py   # Vertex AI config propagation tests
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
| Student | Qwen      | Qwen 2.5 7B Instruct                |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up API keys (teacher LLMs)

```bash
cp .env.example .env
```

Edit `.env` and fill in all required keys:

```
# Teacher LLM API keys (all 3 required for the 3×3 experiment matrix)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
ANTHROPIC_API_KEY=sk-ant-...

# Student SLM access (required for gated models: Llama 3.3, Gemma 3)
# Accept each model's license on its HuggingFace page first.
# Not needed for Qwen 2.5 7B Instruct (open access).
HF_TOKEN=hf_...

# Vertex AI (required when student.backend = "vertex_ai")
VERTEX_AI_PROJECT=smala-experiment
VERTEX_AI_LOCATION=us-central1
VERTEX_AI_STAGING_BUCKET=gs://smala-experiment-data/smala
```

The `.env` file is in `.gitignore` and will not be committed.

> **Note:** The `VERTEX_AI_*` environment variables serve as fallbacks when the corresponding fields in `configs/model_config.yaml` are left empty. You can set them in either place.

### 3. Set up Google Cloud (Vertex AI for student fine-tuning)

The student models are fine-tuned on Google Vertex AI. You need:

```bash
# a) Install the gcloud CLI (if not already installed)
# https://cloud.google.com/sdk/docs/install

# b) Authenticate
gcloud auth login
gcloud auth application-default login

# c) Create a GCP project (or use an existing one)
gcloud projects create smala-experiment --name="SMalA Experiment"
gcloud config set project smala-experiment

# d) Enable billing (required for GPU instances)
# Link a billing account at: https://console.cloud.google.com/billing

# e) Enable required APIs
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com

# f) Request GPU quota (new projects often have 0 GPU quota by default)
# Go to: https://console.cloud.google.com/iam-admin/quotas
# Filter for "NVIDIA L4" in your region (asia-southeast1) and request quota >= 1

# g) Create a GCS bucket for training data and adapter storage
gcloud storage buckets create gs://smala-experiment-data --location=asia-southeast1
```

> **Important:** New GCP projects often have **zero GPU quota** by default. You must request NVIDIA L4 quota in your region before running experiments, or the training/inference jobs will fail. Quota requests are typically approved within minutes for small amounts.

Then either set the `VERTEX_AI_*` variables in your `.env` (see step 2), or update `configs/model_config.yaml` directly:

```yaml
student:
  backend: vertex_ai

vertex_ai:
  project: smala-experiment
  location: asia-southeast1
  staging_bucket: gs://smala-experiment-data/smala
```

### 4. Run a single experiment

```bash
python main.py --config configs/model_config.yaml --truncate-input
```

This runs one teacher-student pair as defined in `model_config.yaml`. The orchestrator will:
1. Generate exam questions (one per detonation report, CyberSOCEval style)
2. Have the student SLM attempt each question
3. Evaluate answers and identify weaknesses
4. Produce a targeted synthetic curriculum
5. Fine-tune the student via Vertex AI (LoRA)
6. Repeat until the benchmark target accuracy is reached

### 5. Run the full 3×3 experiment matrix

```bash
python run_experiments.py --config configs/model_config.yaml --truncate-input
```

This runs all 9 teacher-student combinations across 3 sequential batches (3 experiments in parallel per batch). Between batches, all models are reset. Failed experiments are automatically retried (up to 2 times).

| Batch | Experiment 1       | Experiment 2            | Experiment 3             |
|-------|--------------------|-------------------------|--------------------------|
| 1     | GPT-4o → Llama 3.3 | Gemini 2.5 Pro → Gemma 3 | Claude Sonnet 4.6 → Qwen 2.5 |
| 2     | GPT-4o → Gemma 3   | Gemini 2.5 Pro → Qwen 2.5 | Claude Sonnet 4.6 → Llama 3.3 |
| 3     | GPT-4o → Qwen 2.5  | Gemini 2.5 Pro → Llama 3.3 | Claude Sonnet 4.6 → Gemma 3 |

Each experiment saves adapters and results to `outputs/batch_N/<teacher>__<student>/`. A summary comparison table is saved to `outputs/experiment_summary.json`.

**Useful flags:**

```bash
# Run experiments sequentially within each batch (default: 3 parallel)
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

## Student Fine-Tuning Backends

SMalA supports two backends for student model fine-tuning:

| Backend | Config value | Infrastructure | Best for |
|---------|-------------|----------------|----------|
| **Local** | `backend: local` | Local GPU with PyTorch + PEFT | Development, single experiments |
| **Vertex AI** | `backend: vertex_ai` | Google Cloud (GCS + Vertex AI) | Full experiment matrix, no local GPU |

To use Vertex AI, set in `configs/model_config.yaml`:

```yaml
student:
  backend: vertex_ai

vertex_ai:
  project: your-gcp-project-id
  location: asia-southeast1
  staging_bucket: gs://your-bucket/smala
```

Then authenticate: `gcloud auth application-default login`

### Vertex AI cost notes

The Vertex AI backend uses GPU-accelerated cloud instances. Key cost drivers:

| Resource | When | Machine | Approx. cost |
|----------|------|---------|-------------|
| **Base model endpoint** | Round 1 inference (before any fine-tuning) | `g2-standard-8` + L4 GPU | ~$1–2/hr |
| **Fine-tuning job** | Each training round | `n1-standard-8` + L4 GPU | ~$1–2/hr |
| **GCS storage** | Training data + adapter checkpoints | Standard storage | Negligible |

The base model endpoint is deployed once per experiment, reused for all Round 1 inference calls, then automatically cleaned up after the first training round completes. Fine-tuning jobs run for the duration of training only.

For the full 3×3 matrix (9 experiments, up to 5 rounds each), expect roughly **$30–80 in Vertex AI compute costs** depending on how many rounds run before early stopping.

---

## Tests

The test suite validates orchestration, fault tolerance, and Vertex AI config propagation without requiring API keys or GPU access.

```bash
# Run all tests
python -m pytest tests/ -v

# Run only Vertex AI workflow tests
python -m pytest tests/test_vertex_ai_workflow.py -v

# Run only fault tolerance tests
python -m pytest tests/test_fault_tolerance.py -v
```

| Test file | Tests | Covers |
|-----------|-------|--------|
| `test_fault_tolerance.py` | 25 | Batch isolation, cross-batch isolation, progress preservation, retry logic, artifact verification, tracker persistence |
| `test_vertex_ai_workflow.py` | 29 | Config propagation (`student.backend`, `vertex_ai` section), trainer type dispatch, HF_TOKEN passthrough to training/serving containers, Round 1 base endpoint deployment and inference for all models, endpoint cleanup, artifact verification with Vertex AI metadata, end-to-end orchestration across all 9 experiments |

---

## Context Window Management

Hybrid Analysis sandbox reports often exceed 128k tokens. `data_filter.py` implements the CyberSOCEval truncation strategy (hash removal, signature description trimming to 50 chars, MITRE ATT&CK condensing to tactic/technique/attck_id only), reducing every report to under 20k tokens with negligible impact on model performance.

Following CyberSOCEval's one-report-per-question paradigm, each exam question is grounded in exactly one detonation report. This guarantees every teacher API call fits within all context windows (GPT-4o 128k, Claude 200k, Gemini 1M) without dropping any reports.

---

## Acknowledgements and Disclaimers

SmalA is a project originally made for Nanyang Technological University's Final Year Project Module SC4079. Special thanks to Gu Wenbo and Professor Liu Yang for their guidance and support in academic matters.

SmalA has not been implemented or tested in a production environment. Please use it at your own risk. None of the project contributors assume any responsibility for any damages through this project.

## License

MIT — see [LICENSE](LICENSE).
