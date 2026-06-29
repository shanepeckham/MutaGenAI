#!/usr/bin/env python3
"""Evaluate the schema-first prompt approach on ScrapeGraphAI.

Applies the philosophy from docs/schemafirst-approach.html:
  - Let the schema carry intent via field names
  - Keep the prompt to: role + schema + minimal hard rules
  - No verbose instructions — the schema IS the spec

Runs on Azure GPT-4.1 and saves results alongside the evolutionary
experiment logs for direct comparison.

Usage:
    uv run python examples/cookbook/eval_schemafirst_baseline.py --max-samples 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import time
from pathlib import Path

# Add parent to path so we can import the scrapegraph helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_evolution_scrapegraph import (
    COMPLEXITY_TIERS,
    ScrapeGraphCase,
    load_scrapegraph_dataset,
    score_schema_conformance,
    _classify_failure_bucket,
)

from MutaGenAI import LLMBackend, LLMClient, PromptEvolverConfig, FailureBucket


# ─────────────────────────────────────────────────────────────────────────
# Schema-first prompt — derived from the methodology document
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


def evaluate_prompt(
    prompt_template: str,
    cases: list[ScrapeGraphCase],
    client: LLMClient,
    temperature: float = 0.1,
    top_p: float = 0.95,
) -> dict:
    """Evaluate a prompt template on a set of cases.

    Returns a dict with score, conformance, and failure bucket counts.
    """
    total_score = 0.0
    conform_count = 0
    buckets: dict[str, int] = {}

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
            continue

        score, detail = score_schema_conformance(response, case)
        total_score += score

        if detail["valid_json"]:
            conform_count += 1

        bucket = _classify_failure_bucket(score, detail)
        if bucket is not None:
            bname = bucket.value
            buckets[bname] = buckets.get(bname, 0) + 1

    n = len(cases) if cases else 1
    return {
        "score": total_score / n * 100.0,
        "conformance": conform_count / n * 100.0,
        "failure_buckets": buckets,
        "n_cases": len(cases),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Schema-first prompt baseline evaluation",
    )
    parser.add_argument(
        "--max-samples", type=int, default=15,
        help="Max cases per complexity tier (default: 15)",
    )
    parser.add_argument(
        "--max-download", type=int, default=500,
        help="Max rows to download from HuggingFace (default: 500)",
    )
    args = parser.parse_args()

    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  Schema-First Prompt Baseline — ScrapeGraphAI                ║
    ║  "Let the schema carry intent" vs evolutionary search        ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # ── Azure GPT-4.1 backend ───────────────────────────────────────
    cfg = PromptEvolverConfig(
        backend=LLMBackend.AZURE_OPENAI,
        azure_use_rbac=True,
        timeout=60.0,
    )
    client = LLMClient(cfg)

    if not client.is_available():
        print("  ✗ Azure OpenAI not available. Check AZURE_OPENAI_ENDPOINT.")
        return

    print(f"  Backend: Azure OpenAI ({os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'N/A')})")
    print(f"  Endpoint: {os.environ.get('AZURE_OPENAI_ENDPOINT', 'N/A')[:60]}...")

    # ── Load data ────────────────────────────────────────────────────
    print("\n  Loading ScrapeGraphAI benchmark data...")
    by_tier = load_scrapegraph_dataset(
        max_samples=args.max_samples,
        max_download=args.max_download,
    )

    for tier_name in COMPLEXITY_TIERS:
        cases = by_tier.get(tier_name, [])
        print(f"    {tier_name:<10}: {len(cases):3d} cases")

    if not by_tier:
        print("  No data loaded — exiting.")
        return

    # ── Show the prompt ──────────────────────────────────────────────
    print("\n  Schema-first prompt template:")
    print("  " + "─" * 50)
    for line in SCHEMA_FIRST_PROMPT.strip().split("\n"):
        print(f"  │ {line}")
    print("  " + "─" * 50)

    # ── Evaluate per tier ────────────────────────────────────────────
    results: dict[str, dict] = {}
    t0 = time.perf_counter()

    for tier_name in COMPLEXITY_TIERS:
        cases = by_tier.get(tier_name, [])
        if not cases:
            continue

        print(f"\n  Evaluating {tier_name} tier ({len(cases)} cases)...")
        result = evaluate_prompt(SCHEMA_FIRST_PROMPT, cases, client)
        results[tier_name] = result
        print(
            f"    Score: {result['score']:.1f}%  |  "
            f"Conformance: {result['conformance']:.1f}%  |  "
            f"Buckets: {result['failure_buckets']}"
        )

    wall_time = time.perf_counter() - t0

    # ── Comparison table ─────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("  Comparison: Schema-First Prompt vs Evolutionary Search (Azure GPT-4.1)")
    print(f"{'=' * 80}")

    # Load evolutionary results for comparison
    evo_path = Path("scrapegraph_experiment_log_azure_gpt41.json")
    evo_results: dict[str, dict] = {}
    if evo_path.exists():
        with open(evo_path) as f:
            evo_data = json.load(f)
        for exp in evo_data.get("experiments", []):
            if exp.get("algorithm") == "deep":
                evo_results[exp["tier"]] = {
                    "baseline": exp["baseline_score"],
                    "evolved": exp["evolved_score"],
                }

    print(
        f"  {'Tier':<8} {'Default':>10} {'Schema-1st':>12} "
        f"{'Evolved':>10} {'S1 vs Evo':>12}"
    )
    print(f"  {'─' * 56}")

    for tier_name in COMPLEXITY_TIERS:
        sf_score = results.get(tier_name, {}).get("score", 0.0)
        evo = evo_results.get(tier_name, {})
        evo_base = evo.get("baseline", 0.0)
        evo_best = evo.get("evolved", 0.0)
        delta = sf_score - evo_best

        print(
            f"  {tier_name:<8} {evo_base:>9.1f}% {sf_score:>11.1f}% "
            f"{evo_best:>9.1f}% {delta:>+11.1f}%"
        )

    print(f"\n  Wall time: {wall_time:.1f}s")

    # ── Save results ─────────────────────────────────────────────────
    log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": "scrapegraph_100k",
        "approach": "schema_first_handcrafted",
        "backend": "azure_openai",
        "model": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1"),
        "prompt_template": SCHEMA_FIRST_PROMPT,
        "wall_time": wall_time,
        "results": {
            tier: {
                "score": r["score"],
                "conformance": r["conformance"],
                "failure_buckets": r["failure_buckets"],
                "n_cases": r["n_cases"],
            }
            for tier, r in results.items()
        },
    }

    out_path = Path("scrapegraph_schemafirst_baseline.json")
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
