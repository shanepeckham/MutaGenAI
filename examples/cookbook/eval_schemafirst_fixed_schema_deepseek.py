#!/usr/bin/env python3
"""Evaluate the schema-first prompt on a fixed-schema subset using Ollama deepseek-v3.2:cloud.

Loads 50 HIGH-tier evaluation cases that share the same schema structure
(NGOJobListing extraction) from .scrapegraph_cache/fixed_schema_70.json.
Runs the schema-first prompt template against deepseek-v3.2:cloud via Ollama
and saves results for comparison with GPT-4.1 and llama3.2 baselines.

Usage:
    uv run python examples/cookbook/eval_schemafirst_fixed_schema_deepseek.py
"""
from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_evolution_scrapegraph import (
    ScrapeGraphCase,
    score_schema_conformance,
    _classify_failure_bucket,
)

from MutaGenAI import LLMBackend, LLMClient, PromptEvolverConfig, FailureBucket


# ─────────────────────────────────────────────────────────────────────────
# Schema-first prompt — role + schema + minimal rules
# ─────────────────────────────────────────────────────────────────────────

SCHEMA_FIRST_PROMPT = textwrap.dedent("""\
You are a structured data extraction agent.

Extract information from the content below into a JSON object
matching this schema:

{output_schema}

Task: {user_prompt}

Content:
{web_content}

Rules:
- Output ONLY the JSON object — no markdown fences, no explanation.
- Every key in the schema must appear in the output.
- If a value is absent from the content, use null.
- Do not invent data not present in the source.
""")


# ─────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────

def load_fixed_schema_cases(split: str = "eval") -> list[ScrapeGraphCase]:
    """Load eval or train cases from the fixed-schema dataset."""
    cache = Path(".scrapegraph_cache/fixed_schema_70.json")
    if not cache.exists():
        raise FileNotFoundError(
            f"Fixed-schema dataset not found at {cache}. "
            "Run the dataset preparation script first."
        )
    with open(cache) as f:
        data = json.load(f)

    key = "eval_cases" if split == "eval" else "train_cases"
    raw = data[key]

    cases = []
    for i, r in enumerate(raw):
        cases.append(ScrapeGraphCase(
            case_id=f"fixed_{split}_{i}",
            prompt=r["prompt"],
            schema=r["schema"],
            content=r["content"],
            expected_response=r.get("response", ""),
            response_is_valid=r.get("response_is_valid", False),
            complexity_score=r.get("schema_complexity_score", 0.0),
            complexity_tier="HIGH",
        ))
    return cases


# ─────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────

def evaluate_prompt(
    prompt_template: str,
    cases: list[ScrapeGraphCase],
    client: LLMClient,
    temperature: float = 0.1,
    top_p: float = 0.95,
) -> dict:
    """Evaluate a prompt template and return aggregate results."""
    total_score = 0.0
    conform_count = 0
    buckets: dict[str, int] = {}
    per_case: list[dict] = []

    for case in cases:
        sys_prompt = (
            prompt_template
            .replace("{user_prompt}", case.prompt)
            .replace("{output_schema}", case.schema_str)
            .replace("{web_content}", case.content[:2000])
        )

        response = client.complete(
            system_prompt=sys_prompt,
            user_message="Extract the data as JSON:",
            temperature=temperature,
            top_p=top_p,
        )

        if response is None:
            per_case.append({"case_id": case.case_id, "score": 0.0, "bucket": "no_output"})
            buckets["no_output"] = buckets.get("no_output", 0) + 1
            continue

        score, detail = score_schema_conformance(response, case)
        total_score += score

        if detail["valid_json"]:
            conform_count += 1

        bucket = _classify_failure_bucket(score, detail)
        if bucket is None:
            bucket_name = "pass"
        else:
            bucket_name = bucket.value
            buckets[bucket_name] = buckets.get(bucket_name, 0) + 1

        per_case.append({"case_id": case.case_id, "score": round(score, 2), "bucket": bucket_name})

    n = len(cases) if cases else 1
    return {
        "score": round(total_score / n * 100, 2),
        "conformance": round(conform_count / n * 100, 2),
        "failure_buckets": buckets,
        "per_case": per_case,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  MutaGenAI — Fixed-Schema Baseline (NGOJobListing)           ║
    ║  50 eval cases · deepseek-v3.2:cloud via Ollama              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    cfg = PromptEvolverConfig(
        backend=LLMBackend.OLLAMA,
        ollama_model="deepseek-v3.2:cloud",
        timeout=120.0,
        max_tokens=4096,
    )
    client = LLMClient(cfg)

    if not client.is_available():
        print("  ✗ Ollama not available. Check that 'ollama serve' is running.")
        return

    print("  Backend: Ollama (deepseek-v3.2:cloud)")

    print("\n  Loading fixed-schema dataset (eval split)...")
    eval_cases = load_fixed_schema_cases("eval")
    print(f"  Eval: {len(eval_cases)} cases")

    print("\n  Schema-first prompt:")
    print("  " + "─" * 50)
    for line in SCHEMA_FIRST_PROMPT.strip().split("\n"):
        print(f"  │ {line}")
    print("  " + "─" * 50)

    print("\n  Running evaluation...")
    t0 = time.perf_counter()
    results = evaluate_prompt(SCHEMA_FIRST_PROMPT, eval_cases, client)
    wall_time = time.perf_counter() - t0

    print(f"\n  Results:")
    print(f"    Score:       {results['score']:.1f}%")
    print(f"    Conformance: {results['conformance']:.1f}%")
    print(f"    Buckets:     {results['failure_buckets']}")
    print(f"    Wall time:   {wall_time:.1f}s")

    log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": "scrapegraph_100k",
        "approach": "schema_first_static",
        "dataset": "fixed_schema_70 (eval split)",
        "schema_family": "NGOJobListing (jobs structural family)",
        "backend": "ollama",
        "model": "deepseek-v3.2:cloud",
        "prompt_template": SCHEMA_FIRST_PROMPT,
        "temperature": 0.1,
        "top_p": 0.95,
        "n_eval_cases": len(eval_cases),
        "score": results["score"],
        "conformance": results["conformance"],
        "failure_buckets": results["failure_buckets"],
        "wall_time": round(wall_time, 1),
        "per_case": results["per_case"],
    }

    out_path = Path("scrapegraph_fixed_schema_baseline_deepseek.json")
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
