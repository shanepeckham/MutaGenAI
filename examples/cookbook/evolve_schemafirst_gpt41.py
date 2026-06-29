#!/usr/bin/env python3
"""Evolve the schema-first prompt on ScrapeGraphAI with Azure GPT-4.1.

Starts evolution from the single schema-first prompt template (derived
from docs/schemafirst-approach.html) instead of the 4 generic seeds.
Uses the same dataset, scoring, mutation operators, and algorithm
config ("deep") as the original experiments.

This lets us answer: can evolutionary search improve a well-crafted
schema-first prompt, and how does the result compare to evolution
starting from generic seeds?

Usage:
    uv run python examples/cookbook/evolve_schemafirst_gpt41.py --max-samples 50
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import textwrap
import time
from dataclasses import field
from pathlib import Path
from typing import Any

import numpy as np

# Add parent to path for scrapegraph helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_evolution_scrapegraph import (
    ALGORITHM_CONFIGS,
    COMPLEXITY_TIERS,
    ScrapeGraphCase,
    ScrapeGraphExperiment,
    load_scrapegraph_dataset,
    score_schema_conformance,
    evaluate_baseline,
    show_prompt_evolution,
    show_results_table,
    _classify_failure_bucket,
    _mutate_template,
    _crossover_templates,
    _score_prop_select,
)

from MutaGenAI import (
    ErrorProfile,
    FailureBucket,
    LLMBackend,
    LLMClient,
    ProblemType,
    PromptCandidate,
    PromptEvolverConfig,
    SelectionMethod,
    get_failure_bucket_mutations,
)


# ─────────────────────────────────────────────────────────────────────────
# Schema-first seed — the ONLY seed for this experiment
# ─────────────────────────────────────────────────────────────────────────

SCHEMA_FIRST_SEED = textwrap.dedent("""\
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
# Evolution engine (mirrors run_scrapegraph_evolution but with single seed)
# ─────────────────────────────────────────────────────────────────────────


def run_schemafirst_evolution(
    cases: list[ScrapeGraphCase],
    client: LLMClient,
    config: PromptEvolverConfig,
    algorithm_name: str = "deep",
    seed: int = 123,
    verbose: bool = True,
) -> ScrapeGraphExperiment:
    """Run prompt evolution starting from the schema-first seed only."""
    rng = np.random.default_rng(seed)
    tier = cases[0].complexity_tier if cases else "unknown"
    problem_type = config.problem_type or ProblemType.GENERATION
    error_profile = ErrorProfile()

    # ── Evaluate a candidate ─────────────────────────────────────────
    def evaluate(
        candidate: PromptCandidate,
        eval_cases: list[ScrapeGraphCase],
        track_buckets: bool = True,
    ) -> tuple[float, float]:
        """Returns (overall_score_pct, schema_conformance_pct)."""
        total = 0.0
        conform_total = 0.0
        for case in eval_cases:
            sys_prompt = (
                candidate.template
                .replace("{user_prompt}", case.prompt)
                .replace("{output_schema}", case.schema_str)
                .replace("{web_content}", case.content[:2000])
            )

            response = client.complete(
                system_prompt=sys_prompt,
                user_message="Extract the data as JSON:",
                temperature=candidate.temperature,
                top_p=candidate.top_p,
            )
            if response is None:
                total += float(rng.uniform(0, 0.05))
                if track_buckets:
                    error_profile.record_bucket(FailureBucket.NO_OUTPUT)
                continue

            score, detail = score_schema_conformance(response, case)
            total += score
            if detail["valid_json"]:
                conform_total += 1

            if track_buckets:
                bucket = _classify_failure_bucket(score, detail)
                if bucket is not None:
                    error_profile.record_bucket(bucket)

        n = len(eval_cases) if eval_cases else 1
        return (
            total / n * 100.0,
            conform_total / n * 100.0,
        )

    # ── Subsample for evaluation ─────────────────────────────────────
    if config.eval_sample_size and config.eval_sample_size < len(cases):
        eval_indices = rng.choice(
            len(cases), size=config.eval_sample_size, replace=False,
        )
        eval_cases = [cases[int(i)] for i in eval_indices]
    else:
        eval_cases = cases

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  ScrapeGraph Tier: {tier}  |  Algorithm: {algorithm_name}")
        print(
            f"  Cases: {len(eval_cases)}  |  Backend: {config.backend.value}"
        )
        print(f"  Seed: schema-first prompt (single seed)")
        print(f"{'=' * 60}")

    t0 = time.perf_counter()
    prompt_trace: list[dict[str, Any]] = []

    # ── Init islands with schema-first seed only ─────────────────────
    islands: list[list[PromptCandidate]] = [
        [] for _ in range(config.num_islands)
    ]

    best_conformance = 0.0

    # Place the schema-first seed on each island with different
    # temperature/top_p to give CMA-ES style diversity
    for isl_id in range(config.num_islands):
        cand = PromptCandidate(
            template=SCHEMA_FIRST_SEED,
            temperature=float(rng.uniform(*config.temperature_range)),
            top_p=float(rng.uniform(*config.top_p_range)),
            generation=0,
        )
        score, conformance = evaluate(cand, eval_cases)
        cand.score = score
        islands[isl_id].append(cand)

        if score >= best_conformance:
            best_conformance = conformance

        prompt_trace.append({
            "generation": 0,
            "score": round(score, 1),
            "template_hash": cand.hash,
            "template_preview": cand.template[:120].replace("\n", " "),
        })

    baseline_best = max(
        (c for isl in islands for c in isl), key=lambda c: c.score,
    )
    baseline_score = baseline_best.score

    if verbose:
        print(f"  Schema-first baseline: {baseline_score:.1f}%")

    # ── Evolution loop ───────────────────────────────────────────────
    best_overall = copy.deepcopy(baseline_best)
    history: list[tuple[int, float]] = [(0, baseline_score)]

    for gen in range(1, config.iterations + 1):
        bucket_mutations = get_failure_bucket_mutations(
            error_profile, problem_type,
        )

        for isl_id in range(config.num_islands):
            island = islands[isl_id]
            if not island:
                continue

            new_cands: list[PromptCandidate] = []
            for _ in range(config.population_size):
                parent_a = _score_prop_select(island, rng)

                if rng.random() < config.crossover_rate and len(island) > 1:
                    parent_b = _score_prop_select(island, rng)
                    child_tmpl = _crossover_templates(
                        parent_a.template, parent_b.template, rng,
                    )
                else:
                    child_tmpl = parent_a.template

                if rng.random() < config.mutation_rate:
                    child_tmpl = _mutate_template(
                        child_tmpl, rng, config.mutation_rate,
                        extra_mutations=bucket_mutations or None,
                    )

                temp = parent_a.temperature + float(rng.normal(0, 0.1))
                temp = float(np.clip(temp, *config.temperature_range))
                top_p = parent_a.top_p + float(rng.normal(0, 0.05))
                top_p = float(np.clip(top_p, *config.top_p_range))

                child = PromptCandidate(
                    template=child_tmpl,
                    temperature=temp,
                    top_p=top_p,
                    generation=gen,
                )

                score, conformance = evaluate(child, eval_cases)
                child.score = score
                new_cands.append(child)

                if score > best_overall.score:
                    best_conformance = conformance

            combined = island + new_cands
            combined.sort(key=lambda c: c.score, reverse=True)
            islands[isl_id] = combined[: config.elite_size]

        error_profile.decay(0.8)

        # Migration
        if gen % 3 == 0 and config.num_islands > 1:
            for src in range(config.num_islands):
                if not islands[src]:
                    continue
                best_src = max(islands[src], key=lambda c: c.score)
                dest = (src + 1) % config.num_islands
                migrant = PromptCandidate(
                    template=best_src.template,
                    temperature=best_src.temperature,
                    top_p=best_src.top_p,
                    generation=best_src.generation,
                    score=best_src.score,
                )
                islands[dest].append(migrant)

        gen_best = max(
            (c for isl in islands for c in isl), key=lambda c: c.score,
        )
        if gen_best.score > best_overall.score:
            best_overall = copy.deepcopy(gen_best)

        history.append((gen, best_overall.score))

        prompt_trace.append({
            "generation": gen,
            "score": round(best_overall.score, 1),
            "template_hash": best_overall.hash,
            "template_preview": best_overall.template[:120].replace("\n", " "),
        })

        if verbose:
            print(
                f"  Gen {gen:2d}/{config.iterations}  "
                f"best={best_overall.score:5.1f}%  "
                f"temp={best_overall.temperature:.3f}  "
                f"top_p={best_overall.top_p:.3f}"
            )
            if bucket_mutations:
                top_buckets = error_profile.worst_buckets(3)
                bucket_str = ", ".join(
                    f"{b}({c})" for b, c in top_buckets
                )
                print(f"         buckets: {bucket_str}")

    wall_time = time.perf_counter() - t0
    final_buckets = dict(error_profile.failure_buckets)

    return ScrapeGraphExperiment(
        tier=tier,
        algorithm=algorithm_name,
        backend=config.backend.value,
        n_cases=len(eval_cases),
        baseline_score=baseline_score,
        evolved_score=best_overall.score,
        best_prompt_template=best_overall.template,
        best_temperature=best_overall.temperature,
        best_top_p=best_overall.top_p,
        iterations=config.iterations,
        wall_time=wall_time,
        history=history,
        prompt_evolution=prompt_trace,
        schema_conformance=best_conformance,
        failure_buckets=final_buckets,
    )


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve the schema-first prompt on Azure GPT-4.1",
    )
    parser.add_argument(
        "--max-samples", type=int, default=15,
        help="Max cases per complexity tier (default: 15)",
    )
    parser.add_argument(
        "--max-download", type=int, default=500,
        help="Max rows to download from HuggingFace (default: 500)",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Override iteration count (default: use 'deep' config = 5)",
    )
    args = parser.parse_args()

    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  MutaGenAI — Evolving the Schema-First Prompt                ║
    ║  Single seed · Azure GPT-4.1 · "deep" algorithm              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # ── Azure GPT-4.1 backend ───────────────────────────────────────
    cfg_base = {
        "backend": LLMBackend.AZURE_OPENAI,
        "azure_use_rbac": True,
        "timeout": 60.0,
    }

    client_cfg = PromptEvolverConfig(**cfg_base)
    client = LLMClient(client_cfg)

    if not client.is_available():
        print("  ✗ Azure OpenAI not available. Check AZURE_OPENAI_ENDPOINT.")
        return

    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "N/A")
    print(f"  Backend: Azure OpenAI ({deployment})")

    # ── Load data (same params as original experiments) ──────────────
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

    # ── Show the seed prompt ─────────────────────────────────────────
    print("\n  Schema-first seed prompt:")
    print("  " + "─" * 50)
    for line in SCHEMA_FIRST_SEED.strip().split("\n"):
        print(f"  │ {line}")
    print("  " + "─" * 50)

    # ── Evolve per tier ──────────────────────────────────────────────
    all_experiments: list[ScrapeGraphExperiment] = []

    for tier_name in COMPLEXITY_TIERS:
        cases = by_tier.get(tier_name, [])
        if not cases:
            continue

        algo = "deep"
        algo_params = dict(ALGORITHM_CONFIGS[algo])
        if args.iterations:
            algo_params["iterations"] = args.iterations

        evo_cfg = PromptEvolverConfig(
            **cfg_base,
            **algo_params,
        )

        exp = run_schemafirst_evolution(
            cases=cases,
            client=client,
            config=evo_cfg,
            algorithm_name=algo,
            seed=123,
            verbose=True,
        )
        all_experiments.append(exp)
        show_prompt_evolution(exp)

    # ── Comparison table ─────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  Comparison: Schema-First Evolved vs Generic-Seed Evolved (Azure GPT-4.1)")
    print(f"{'=' * 90}")

    # Load generic-seed evolutionary results
    evo_path = Path("scrapegraph_experiment_log_azure_gpt41.json")
    generic_evo: dict[str, dict] = {}
    if evo_path.exists():
        with open(evo_path) as f:
            evo_data = json.load(f)
        for exp in evo_data.get("experiments", []):
            if exp.get("algorithm") == "deep":
                generic_evo[exp["tier"]] = {
                    "baseline": exp["baseline_score"],
                    "evolved": exp["evolved_score"],
                }

    # Load schema-first static baseline
    sf_static_path = Path("scrapegraph_schemafirst_baseline.json")
    sf_static: dict[str, float] = {}
    if sf_static_path.exists():
        with open(sf_static_path) as f:
            sf_data = json.load(f)
        for tier, r in sf_data.get("results", {}).items():
            sf_static[tier] = r["score"]

    print(
        f"  {'Tier':<8} {'Generic base':>13} {'Generic evo':>12} "
        f"{'SF static':>10} {'SF evo':>8} {'SF Δ':>8}"
    )
    print(f"  {'─' * 60}")

    for exp in all_experiments:
        tier = exp.tier
        ge = generic_evo.get(tier, {})
        sf_s = sf_static.get(tier, 0.0)
        sf_delta = exp.evolved_score - exp.baseline_score

        print(
            f"  {tier:<8} "
            f"{ge.get('baseline', 0.0):>12.1f}% "
            f"{ge.get('evolved', 0.0):>11.1f}% "
            f"{sf_s:>9.1f}% "
            f"{exp.evolved_score:>7.1f}% "
            f"{sf_delta:>+7.1f}%"
        )

    # ── Show winning prompts ─────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print("  Evolved Schema-First Prompts")
    print(f"{'=' * 90}")
    for exp in all_experiments:
        print(f"\n  --- {exp.tier} ({exp.baseline_score:.1f}% → {exp.evolved_score:.1f}%) ---")
        print(exp.best_prompt_template)

    # ── Save results ─────────────────────────────────────────────────
    log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": "scrapegraph_100k",
        "approach": "schema_first_evolved",
        "seed_prompt": "schema_first",
        "backend": "azure_openai",
        "model": deployment,
        "experiments": [
            {
                "tier": exp.tier,
                "algorithm": exp.algorithm,
                "backend": exp.backend,
                "n_cases": exp.n_cases,
                "baseline_score": exp.baseline_score,
                "evolved_score": exp.evolved_score,
                "delta": round(exp.evolved_score - exp.baseline_score, 2),
                "best_temperature": exp.best_temperature,
                "best_top_p": exp.best_top_p,
                "iterations": exp.iterations,
                "wall_time": round(exp.wall_time, 1),
                "history": exp.history,
                "prompt_evolution": exp.prompt_evolution,
                "best_prompt_template": exp.best_prompt_template,
                "schema_conformance": exp.schema_conformance,
                "failure_buckets": exp.failure_buckets,
            }
            for exp in all_experiments
        ],
    }

    out_path = Path("scrapegraph_schemafirst_evolved_gpt41.json")
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
