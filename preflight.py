"""
preflight.py
============
Pre-flight checks to validate that model infrastructure is ready before
running a full experiment.

Checks:
  1. HuggingFace Inference API — can we reach the student model?
  2. Vertex AI connectivity — can we authenticate and access the project?
  3. GCS bucket — does the staging bucket exist and is it writable?
  4. Teacher API — can we reach the teacher LLM?

Usage
-----
    # Check all models in config
    python preflight.py --config configs/model_config.yaml

    # Check a specific student model
    python preflight.py --student Qwen/Qwen2.5-7B-Instruct

    # Check a specific teacher
    python preflight.py --teacher anthropic/claude-sonnet-4-6

    # Skip Vertex AI checks (e.g. if using local backend)
    python preflight.py --config configs/model_config.yaml --skip-vertex
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import yaml
from dotenv import load_dotenv

load_dotenv()

# ANSI colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def _pass(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {RED}✗{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}!{RESET} {msg}")


# ---------------------------------------------------------------------------
# Check 1: HuggingFace Inference API
# ---------------------------------------------------------------------------

# Models that use Google GenAI API instead of HF Inference API
_GOOGLE_GENAI_MODELS = {
    "google/gemma-3-4b-it",
}

# Maps HuggingFace model IDs to Google GenAI model IDs
_GENAI_MODEL_MAP = {
    "google/gemma-3-4b-it": "gemma-3-4b-it",
}


def check_student_inference(model_id: str, project: str, location: str) -> bool:
    """Check that the student model is reachable for inference.

    Routes to Google GenAI API or HF Inference API depending on model.
    """
    if model_id in _GOOGLE_GENAI_MODELS:
        return _check_google_genai_inference(model_id)
    return _check_hf_inference(model_id)


def _check_hf_inference(model_id: str) -> bool:
    """Send a short test prompt to the HF Inference API."""
    print(f"\n[1/4] Student inference (HF API): {model_id}")
    try:
        from huggingface_hub import InferenceClient

        hf_token = os.environ.get("HF_TOKEN", "") or None
        client = InferenceClient(model=model_id, token=hf_token)

        start = time.time()
        response = client.chat_completion(
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        elapsed = time.time() - start
        text = response.choices[0].message.content

        if text and len(text) > 0:
            _pass(f"Model responded in {elapsed:.1f}s: {text[:50]!r}")
            return True
        else:
            _fail("Model returned empty response")
            return False
    except Exception as exc:
        _fail(f"{exc}")
        return False


def _check_google_genai_inference(model_id: str) -> bool:
    """Test Google GenAI API inference for models not on HF Inference API."""
    genai_model_id = _GENAI_MODEL_MAP.get(model_id, model_id)
    print(f"\n[1/4] Student inference (Google GenAI API): {genai_model_id}")

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        _fail("GOOGLE_API_KEY not set in environment")
        return False

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(genai_model_id)

        start = time.time()
        response = model.generate_content(
            "Say OK",
            generation_config={"max_output_tokens": 5},
        )
        elapsed = time.time() - start
        text = response.text

        if text and len(text) > 0:
            _pass(f"Model responded in {elapsed:.1f}s: {text[:50]!r}")
            return True
        else:
            _fail("Model returned empty response")
            return False
    except Exception as exc:
        _fail(f"{exc}")
        return False


# ---------------------------------------------------------------------------
# Check 2: Vertex AI connectivity
# ---------------------------------------------------------------------------

def check_vertex_ai(project: str, location: str) -> bool:
    """Verify Vertex AI authentication and project access."""
    print(f"\n[2/4] Vertex AI connectivity: {project} ({location})")
    try:
        from google.cloud import aiplatform

        aiplatform.init(project=project, location=location)

        # List endpoints as a connectivity test (returns empty list if none)
        endpoints = aiplatform.Endpoint.list()
        _pass(f"Authenticated. {len(endpoints)} existing endpoint(s) found.")
        return True
    except Exception as exc:
        _fail(f"{exc}")
        return False


# ---------------------------------------------------------------------------
# Check 3: GCS bucket
# ---------------------------------------------------------------------------

def check_gcs_bucket(staging_bucket: str) -> bool:
    """Verify the GCS staging bucket exists and is accessible."""
    print(f"\n[3/4] GCS staging bucket: {staging_bucket}")
    if not staging_bucket:
        _warn("No staging bucket configured (only needed for Vertex AI backend)")
        return True
    try:
        from google.cloud import storage

        # Parse bucket name from gs://bucket-name/optional/prefix
        path = staging_bucket.replace("gs://", "")
        bucket_name = path.split("/")[0]

        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        _pass(f"Bucket '{bucket.name}' exists and is accessible")
        return True
    except Exception as exc:
        _fail(f"{exc}")
        return False


# ---------------------------------------------------------------------------
# Check 4: Teacher API
# ---------------------------------------------------------------------------

def check_teacher_api(provider: str, model: str) -> bool:
    """Send a minimal request to the teacher LLM API."""
    print(f"\n[4/4] Teacher API: {provider}/{model}")

    key_map = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    env_var = key_map.get(provider)
    if env_var and not os.environ.get(env_var):
        _fail(f"{env_var} not set in environment")
        return False

    try:
        if provider == "openai":
            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Say OK"}],
                max_tokens=3,
            )
            _pass(f"Response: {resp.choices[0].message.content!r}")
            return True

        elif provider == "google":
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            gmodel = genai.GenerativeModel(model)
            resp = gmodel.generate_content("Say OK")
            _pass(f"Response: {resp.text[:50]!r}")
            return True

        elif provider == "anthropic":
            from anthropic import Anthropic
            client = Anthropic()
            resp = client.messages.create(
                model=model,
                max_tokens=3,
                messages=[{"role": "user", "content": "Say OK"}],
            )
            _pass(f"Response: {resp.content[0].text!r}")
            return True

        else:
            _warn(f"No preflight check implemented for provider: {provider}")
            return True

    except Exception as exc:
        _fail(f"{exc}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SMalA pre-flight checks — validate infrastructure before running experiments",
    )
    parser.add_argument(
        "--config",
        default="configs/model_config.yaml",
        help="Path to model_config.yaml",
    )
    parser.add_argument(
        "--student",
        help="HuggingFace model ID to test (overrides config)",
    )
    parser.add_argument(
        "--teacher",
        help="Teacher as 'provider/model' (overrides config), e.g. anthropic/claude-sonnet-4-6",
    )
    parser.add_argument(
        "--skip-vertex",
        action="store_true",
        dest="skip_vertex",
        help="Skip Vertex AI and GCS checks",
    )
    args = parser.parse_args()

    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config) as fh:
            config = yaml.safe_load(fh) or {}

    # Resolve student model
    student_model = args.student or config.get("student", {}).get(
        "model_name_or_path", "Qwen/Qwen2.5-7B-Instruct"
    )

    # Resolve teacher
    if args.teacher:
        parts = args.teacher.split("/", 1)
        teacher_provider = parts[0]
        teacher_model = parts[1] if len(parts) > 1 else parts[0]
    else:
        teacher_cfg = config.get("teacher", {})
        teacher_provider = teacher_cfg.get("provider", "openai")
        teacher_model = teacher_cfg.get("model", "gpt-4o")

    # Resolve Vertex AI config
    vertex_cfg = config.get("vertex_ai", {})
    project = vertex_cfg.get("project", "") or os.environ.get("VERTEX_AI_PROJECT", "")
    location = vertex_cfg.get("location", "") or os.environ.get("VERTEX_AI_LOCATION", "asia-southeast1")
    staging_bucket = vertex_cfg.get("staging_bucket", "") or os.environ.get("VERTEX_AI_STAGING_BUCKET", "")
    backend = config.get("student", {}).get("backend", "vertex_ai")

    # Run checks
    print("=" * 50)
    print("SMalA Pre-flight Checks")
    print("=" * 50)

    results = []

    results.append(check_student_inference(student_model, project, location))

    if not args.skip_vertex and backend == "vertex_ai":
        results.append(check_vertex_ai(project, location))
        results.append(check_gcs_bucket(staging_bucket))
    else:
        print(f"\n[2/4] Vertex AI connectivity: SKIPPED")
        print(f"\n[3/4] GCS staging bucket: SKIPPED")

    results.append(check_teacher_api(teacher_provider, teacher_model))

    # Summary
    passed = sum(results)
    total = len(results)
    print(f"\n{'=' * 50}")
    if all(results):
        print(f"{GREEN}All {total} checks passed. Ready to run experiments.{RESET}")
    else:
        print(f"{RED}{total - passed}/{total} checks failed.{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
