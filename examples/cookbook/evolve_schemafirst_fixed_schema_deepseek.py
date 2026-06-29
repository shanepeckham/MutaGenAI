#!/usr/bin/env python3
"""Evolve the schema-first prompt on a fixed-schema subset with gpt-oss.

Loads 70 HIGH-tier cases that share the same schema structure (NGOJobListing):
  - 20 cases used as the training set for evolutionary search
  - 50 cases held out as the evaluation set

The evolution trains on the 20-case set, then the winning prompt is
evaluated on the 50-case holdout to measure generalisation.

Compare with eval_schemafirst_fixed_schema_deepseek.py (same 50 eval cases,
no evolution) to isolate the effect of evolutionary prompt optimisation
on gpt-oss.

Usage:
    uv run python examples/cookbook/evolve_schemafirst_fixed_schema_deepseek.py
    uv run python examples/cookbook/evolve_schemafirst_fixed_schema_deepseek.py --iterations 8
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_evolution_scrapegraph import (
    ALGORITHM_CONFIGS,
    ScrapeGraphCase,
    ScrapeGraphExperiment,
    score_schema_conformance,
    show_prompt_evolution,
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
    get_failure_bucket_mutations,
)


# ─────────────────────────────────────────────────────────────────────────
# Schema-first seed — the ONLY seed for evolution
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
# Evolution engine — trains on 20 cases, evaluates winner on 50
# ─────────────────────────────────────────────────────────────────────────

def run_fixed_schema_evolution(
    train_cases: list[ScrapeGraphCase],
    eval_cases: list[ScrapeGraphCase],
    client: LLMClient,
    config: PromptEvolverConfig,
    algorithm_name: str = "deep",
    seed: int = 123,
    verbose: bool = True,
) -> dict:
    """Evolve on train_cases, then evaluate winner on eval_cases."""
    rng = np.random.default_rng(seed)
    problem_type = config.problem_type or ProblemType.GENERATION
    error_profile = ErrorProfile()

    # ── Evaluate a candidate ─────────────────────────────────────────
    def evaluate(
        candidate: PromptCandidate,
        cases: list[ScrapeGraphCase],
        track_buckets: bool = True,
    ) -> tuple[float, float]:
        total = 0.0
        conform_total = 0.0
        for case in cases:
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

        n = len(cases) if cases else 1
        return (total / n * 100.0, conform_total / n * 100.0)

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Fixed-Schema Evolution (NGOJobListing)")
        print(f"  Train: {len(train_cases)} cases  |  Eval: {len(eval_cases)} cases")
        print(f"  Algorithm: {algorithm_name}  |  Backend: {config.backend.value}")
        print(f"  Seed: schema-first prompt (single seed)")
        print(f"{'=' * 60}")

    t0 = time.perf_counter()
    prompt_trace: list[dict[str, Any]] = []

    # ── Init islands with schema-first seed ──────────────────────────
    islands: list[list[PromptCandidate]] = [
        [] for _ in range(config.num_islands)
    ]

    for isl_id in range(config.num_islands):
        cand = PromptCandidate(
            template=SCHEMA_FIRST_SEED,
            temperature=float(rng.uniform(*config.temperature_range)),
            top_p=float(rng.uniform(*config.top_p_range)),
            generation=0,
        )
        score, _ = evaluate(cand, train_cases)
        cand.score = score
        islands[isl_id].append(cand)

        prompt_trace.append({
            "generation": 0,
            "island": isl_id,
            "score": round(score, 1),
            "template_hash": cand.hash,
            "template_preview": cand.template[:120].replace("\n", " "),
        })

    baseline_best = max(
        (c for isl in islands for c in isl), key=lambda c: c.score,
    )
    baseline_score = baseline_best.score

    if verbose:
        print(f"  Schema-first baseline (train): {baseline_score:.1f}%")

    # ── Evolution loop (trains on train_cases only) ──────────────────
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

                score, _ = evaluate(child, train_cases)
                child.score = score
                new_cands.append(child)

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
                f"best(train)={best_overall.score:5.1f}%  "
                f"temp={best_overall.temperature:.3f}  "
                f"top_p={best_overall.top_p:.3f}"
            )
            if bucket_mutations:
                top_buckets = error_profile.worst_buckets(3)
                bucket_str = ", ".join(f"{b}({c})" for b, c in top_buckets)
                print(f"         buckets: {bucket_str}")

    train_time = time.perf_counter() - t0

    # ── Evaluate winner on the held-out 50 eval cases ────────────────
    if verbose:
        print(f"\n  Evolution complete ({train_time:.1f}s)")
        print(f"  Evaluating winner on {len(eval_cases)} held-out eval cases...")

    winner = best_overall
    eval_score, eval_conformance = evaluate(winner, eval_cases, track_buckets=False)

    # Also evaluate the original seed on eval set for comparison
    seed_cand = PromptCandidate(
        template=SCHEMA_FIRST_SEED,
        temperature=0.1,
        top_p=0.95,
        generation=0,
    )
    seed_eval_score, seed_eval_conformance = evaluate(
        seed_cand, eval_cases, track_buckets=False,
    )

    if verbose:
        print(f"\n  {'─' * 50}")
        print(f"  Holdout Evaluation (50 cases):")
        print(f"    Seed prompt:    {seed_eval_score:.1f}% (conformance: {seed_eval_conformance:.1f}%)")
        print(f"    Evolved prompt: {eval_score:.1f}% (conformance: {eval_conformance:.1f}%)")
        delta = eval_score - seed_eval_score
        print(f"    Delta:          {delta:+.1f}%")
        print(f"  {'─' * 50}")

    total_time = time.perf_counter() - t0

    return {
        "train_baseline_score": baseline_score,
        "train_evolved_score": best_overall.score,
        "eval_seed_score": seed_eval_score,
        "eval_seed_conformance": seed_eval_conformance,
        "eval_evolved_score": eval_score,
        "eval_evolved_conformance": eval_conformance,
        "eval_delta": round(eval_score - seed_eval_score, 2),
        "best_temperature": best_overall.temperature,
        "best_top_p": best_overall.top_p,
        "best_prompt_template": best_overall.template,
        "iterations": config.iterations,
        "train_time": round(train_time, 1),
        "total_time": round(total_time, 1),
        "history": history,
        "prompt_evolution": prompt_trace,
        "failure_buckets": dict(error_profile.failure_buckets),
    }


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve schema-first prompt on fixed-schema dataset (gpt-oss)",
    )
    parser.add_argument(
        "--iterations", type=int, default=None,
        help="Override iteration count (default: use 'deep' config = 5)",
    )
    args = parser.parse_args()

    print(r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  MutaGenAI — Fixed-Schema Evolution (NGOJobListing)          ║
    ║  Train: 20 cases · Eval: 50 cases · gpt-oss                 ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # ── deepseek-v3.2:cloud backend ─────────────────────────────────
    cfg_base = {
        "backend": LLMBackend.OLLAMA,
        "ollama_model": "gpt-oss",
        "timeout": 120.0,
        "max_tokens": 4096,
    }

    client_cfg = PromptEvolverConfig(**cfg_base)
    client = LLMClient(client_cfg)

    if not client.is_available():
        print("  ✗ Ollama not available. Check that 'ollama serve' is running.")
        return

    print(f"  Backend: Ollama (gpt-oss)")

    # ── Load data ───────────────────────────────────────────────────
    print("\n  Loading fixed-schema dataset...")
    train_cases = load_fixed_schema_cases("train")
    eval_cases = load_fixed_schema_cases("eval")
    print(f"  Train: {len(train_cases)} cases  |  Eval: {len(eval_cases)} cases")

    # ── Show the seed prompt ─────────────────────────────────────────
    print("\n  Schema-first seed prompt:")
    print("  " + "─" * 50)
    for line in SCHEMA_FIRST_SEED.strip().split("\n"):
        print(f"  │ {line}")
    print("  " + "─" * 50)

    # ── Evolution ───────────────────────────────────────────────────
    algo = "deep"
    algo_params = dict(ALGORITHM_CONFIGS[algo])
    if args.iterations:
        algo_params["iterations"] = args.iterations

    evo_cfg = PromptEvolverConfig(**cfg_base, **algo_params)

    results = run_fixed_schema_evolution(
        train_cases=train_cases,
        eval_cases=eval_cases,
        client=client,
        config=evo_cfg,
        algorithm_name=algo,
        seed=123,
        verbose=True,
    )

    # ── Comparison with static baseline ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Comparison Summary")
    print(f"{'=' * 60}")

    static_path = Path("scrapegraph_fixed_schema_baseline_deepseek.json")
    static_score = None
    if static_path.exists():
        with open(static_path) as f:
            static_data = json.load(f)
        static_score = static_data.get("score")

    print(f"  {'Metric':<40} {'Score':>10}")
    print(f"  {'─' * 42}")
    if static_score is not None:
        print(f"  {'Static baseline (50 eval):':<40} {static_score:>9.1f}%")
    print(f"  {'Seed prompt (50 eval):':<40} {results['eval_seed_score']:>9.1f}%")
    print(f"  {'Evolved prompt (50 eval):':<40} {results['eval_evolved_score']:>9.1f}%")
    print(f"  {'Evolution delta:':<40} {results['eval_delta']:>+9.1f}%")
    print(f"  {'Train score (20 cases):':<40} {results['train_evolved_score']:>9.1f}%")

    # ── Show winning prompt ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("  Evolved Prompt")
    print(f"{'=' * 60}")
    print(results["best_prompt_template"])

    # ── Save results ─────────────────────────────────────────────────
    log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": "scrapegraph_100k",
        "approach": "schema_first_evolved_fixed_schema",
        "dataset": "fixed_schema_70 (20 train + 50 eval)",
        "schema_family": "NGOJobListing (jobs structural family)",
        "backend": "ollama",
        "model": "gpt-oss",
        "seed_prompt": SCHEMA_FIRST_SEED,
        "n_train_cases": len(train_cases),
        "n_eval_cases": len(eval_cases),
        "train_baseline_score": round(results["train_baseline_score"], 2),
        "train_evolved_score": round(results["train_evolved_score"], 2),
        "eval_seed_score": round(results["eval_seed_score"], 2),
        "eval_seed_conformance": round(results["eval_seed_conformance"], 2),
        "eval_evolved_score": round(results["eval_evolved_score"], 2),
        "eval_evolved_conformance": round(results["eval_evolved_conformance"], 2),
        "eval_delta": results["eval_delta"],
        "best_temperature": results["best_temperature"],
        "best_top_p": results["best_top_p"],
        "best_prompt_template": results["best_prompt_template"],
        "iterations": results["iterations"],
        "train_time": results["train_time"],
        "total_time": results["total_time"],
        "history": results["history"],
        "prompt_evolution": results["prompt_evolution"],
        "failure_buckets": results["failure_buckets"],
    }

    out_path = Path("scrapegraph_fixed_schema_evolved_deepseek.json")
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
