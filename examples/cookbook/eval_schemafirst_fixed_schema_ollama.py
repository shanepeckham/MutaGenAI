#!/usr/bin/env python3
"""Evaluate the schema-first prompt on a fixed-schema subset using Ollama llama3.2.

Loads 50 HIGH-tier evaluation cases that share the same schema structure
(NGOJobListing extraction) from .scrapegraph_cache/fixed_schema_70.json.
Runs the schema-first prompt template against local Ollama llama3.2 (3B)
and saves results for comparison with the Azure GPT-4.1 baseline and
the evolutionary experiment.

Usage:
    uv run python examples/cookbook/eval_schemafirst_fixed_schema_ollama.py
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
            buckets["no_output"] = buckets.get("no_output", 0) + 1
            per_case.append({"case_id": case.case_id, "score": 0.0, "bucket": "no_output"})
            continue

        score, detail = score_schema_conformance(response, case)
        total_score += score

        if detail["valid_json"]:
            conform_count += 1

        bucket = _classify_failure_bucket(score, detail)
        bname = bucket.value if bucket is not None else "pass"
        if bucket is not None:
            buckets[bname] = buckets.get(bname, 0) + 1

        per_case.append({
            "case_id": case.case_id,
            "score": round(score, 4),
            "bucket": bname,
        })

    n = len(cases) if cases else 1
    return {
        "score": total_score / n * 100.0,
        "conformance": conform_count / n * 100.0,
        "failure_buckets": buckets,
        "n_cases": len(cases),
        "per_case": per_case,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Schema-First Baseline — Fixed Schema (NGOJobListing)        ║
    ║  50 eval cases · No evolution · Ollama llama3.2 (3B)         ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # ── Ollama llama3.2 backend ─────────────────────────────────────
    cfg = PromptEvolverConfig(
        backend=LLMBackend.OLLAMA,
        ollama_model="llama3.2",
        timeout=120.0,
        max_tokens=4096,
    )
    client = LLMClient(cfg)

    if not client.is_available():
        print("  ✗ Ollama not available. Check that 'ollama serve' is running.")
        return

    print(f"  Backend: Ollama (llama3.2, 3B)")

    # ── Load fixed-schema evaluation cases ──────────────────────────
    print("\n  Loading fixed-schema evaluation cases...")
    eval_cases = load_fixed_schema_cases("eval")
    print(f"  Loaded {len(eval_cases)} eval cases (all HIGH tier, same schema structure)")

    # ── Show the prompt ─────────────────────────────────────────────
    print("\n  Schema-first prompt:")
    print("  " + "─" * 50)
    for line in SCHEMA_FIRST_PROMPT.strip().split("\n"):
        print(f"  │ {line}")
    print("  " + "─" * 50)

    # ── Evaluate ────────────────────────────────────────────────────
    print("\n  Evaluating on 50 cases...")
    t0 = time.perf_counter()
    results = evaluate_prompt(SCHEMA_FIRST_PROMPT, eval_cases, client)
    wall_time = time.perf_counter() - t0

    print(f"\n  Results:")
    print(f"    Score:       {results['score']:.1f}%")
    print(f"    Conformance: {results['conformance']:.1f}%")
    print(f"    Wall time:   {wall_time:.1f}s")
    if results["failure_buckets"]:
        print(f"    Buckets:     {results['failure_buckets']}")

    # ── Save ────────────────────────────────────────────────────────
    log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": "scrapegraph_100k",
        "approach": "schema_first_static",
        "dataset": "fixed_schema_70 (eval split)",
        "schema_family": "NGOJobListing (jobs structural family)",
        "backend": "ollama",
        "model": "llama3.2",
        "prompt_template": SCHEMA_FIRST_PROMPT,
        "temperature": 0.1,
        "top_p": 0.95,
        "n_eval_cases": results["n_cases"],
        "score": round(results["score"], 2),
        "conformance": round(results["conformance"], 2),
        "failure_buckets": results["failure_buckets"],
        "wall_time": round(wall_time, 1),
        "per_case": results["per_case"],
    }

    out_path = Path("scrapegraph_fixed_schema_baseline_ollama.json")
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
