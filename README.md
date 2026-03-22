# SMalA — Small Language Model Malware Analysis

**Fine-tuning Small Language Models (SLMs) to perform malware analysis via teacher-student knowledge distillation.**

---

## Overview

SMalA implements an automated, iterative **teacher-student knowledge distillation** pipeline that trains a lightweight SLM to reverse engineer malware. A powerful "teacher" LLM (e.g., GPT-4o, Gemini 2.5 Pro, or Qwen-Max) generates exams, evaluates student responses, and produces a tailored synthetic curriculum that is used to fine-tune a compact "student" SLM (e.g., Llama 3 8B, Phi-3, Gemma 7B, or DeepSeek-R1-Distill-Qwen-7B). You can read more about the

```
Teacher LLM (GPT-4o / Gemini 2.5 Pro / Qwen-Max)
       │
       ▼  1. Generate exam from malware detonation report
       │
Student SLM (Llama 3 8B / Phi-3 / Gemma 7B)
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
SMalRe/
├── main.py                          # Iterative distillation orchestrator
├── teacher_engine.py                # Teacher LLM: exam, evaluation, curriculum
├── student_trainer.py               # Student SLM: LoRA/QLoRA fine-tuning
├── report_parser.py                 # Malware detonation report ingestion
├── data_filter.py                   # Context-window-aware report filtering
├── eval_suite.py                    # Accuracy + Jaccard benchmarking
├── requirements.txt
├── configs/
│   └── model_config.yaml            # Teacher/student model parameters
├── prompt_templates/
│   ├── malware_analysis_task.json   # Structured teacher prompts
│   └── reverse_engineering_curriculum.txt
└── benchmark_data/
    └── README.md                    # CyberSOCEval-compatible dataset guide
```

---

## Supported Model Families

| Role    | Provider  | Recommended Model                        |
|---------|-----------|------------------------------------------|
| Teacher | OpenAI    | GPT-4o, gpt-o3                           |
| Teacher | Google    | Gemini 2.5 Pro, Gemini 1.5 Pro           |
| Teacher | Alibaba   | Qwen-Max                                 |
| Student | Meta      | Llama 3 8B                               |
| Student | Microsoft | Phi-3                                    |
| Student | Google    | Gemma 7B                                 |
| Student | DeepSeek  | DeepSeek-R1-Distill-Qwen-7B              |

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# PDF/report processing system dependency
sudo apt-get install poppler-utils
```

### 2. Configure models

Edit `configs/model_config.yaml` to set your teacher and student model, API keys, and training hyperparameters.

### 3. Run the distillation loop

```bash
python main.py --config configs/model_config.yaml
```

The orchestrator will:
1. Generate an exam from malware detonation reports
2. Have the student SLM attempt the exam
3. Evaluate answers and identify weaknesses
4. Produce a targeted synthetic curriculum
5. Fine-tune the student with LoRA/QLoRA
6. Repeat until the benchmark target accuracy is reached

### 4. Benchmark your SLM

```bash
python eval_suite.py \
  --model-path ./outputs/student_adapter \
  --dataset benchmark_data/malware_analysis_questions.json \
  --output benchmark_data/results.json
```

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

```
## Context Window Management

Hybrid Analysis sandbox reports often exceed 128k tokens. `data_filter.py` implements pre-filtering that retains only essential fields (`total_processes`, `mitre_attcks`, `signatures`) with negligible impact on benchmark performance.

---
```

## Acknowledgements and Disclaimers

SmalA is a project originally made for Nanyang Technological University's Final Year Project Module SC4079. Special thanks to Gu Wenbo and Professor Liu Yang for their guidance and support in academic matters.

SmalA has not been implemented or tested in a production environment. Please use it at your own risk. None of the project contributors assume any responsibility for any damages through this project.

## License

MIT — see [LICENSE](LICENSE).
