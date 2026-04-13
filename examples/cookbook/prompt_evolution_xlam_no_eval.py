#!/usr/bin/env python3
"""
Cookbook Recipe 50 — xLAM No-Eval Prompt Evolution (Wizard-style)
=================================================================

Demonstrates the ``evosim init`` wizard approach applied to the
**Salesforce xLAM function-calling 60k** benchmark.  We pretend we
have **no ground truth**, **no domain-specific mutations**, and let
EvoSim handle everything automatically — exactly as a first-time user
would experience via ``evosim init``.

After the no-eval evolution finishes, we reveal the ground-truth labels
and evaluate the best prompts to measure how close no-eval gets.

Wizard answers simulated
------------------------
1.  Task: function-calling assistant
2.  Ground truth: **no**
3.  Test inputs: 20 xLAM queries (labels stripped)
4.  Scoring: **Composite (recommended)** — auto-configured
5.  Domain mutations: **none** (auto-generated)
6.  Human eval: **no** (fully automated)
7.  Seeds: **auto-generated** from task description
8.  Backend: Ollama (local)
9.  Config: standard (3 gen, 4 pop, 2 islands)

Results are compared against ground-truth evolution (Recipe 44) and
the default un-evolved prompt.

Usage::

    export OLLAMA_MODEL=llama3.2       # or any model you have
    uv run python examples/cookbook/prompt_evolution_xlam_no_eval.py

References
----------
* Wizard: ``evosim init``
* Dataset: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
* Recipe 44: ``prompt_evolution_xlam.py`` (ground-truth version)
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", ".."))
sys.path.insert(0, _here)

from dotenv import load_dotenv

load_dotenv(os.path.join(_here, "..", "..", ".env"))

from prompture.prompt_evolver import (
    LLMBackend,
    LLMClient,
    PromptCandidate,
    PromptEvolverConfig,
)
from prompture.strategies import (
    CompositeScorer,
    LLMJudge,
    NoEvalConfig,
    NoEvalPromptEvolver,
    ProxyCheck,
    ProxyMetricsScorer,
    Scorer,
    SelfConsistencyScorer,
)

# Reuse data-loader and scorer from Recipe 44
from prompt_evolution_xlam import (
    XLAMCase,
    _format_xlam_tools,
    _parse_function_calls,
    _score_params,
    load_xlam_dataset,
    score_xlam_case,
)


# ── Task description (what the wizard user would type) ────────────────

TASK_DESCRIPTION = (
    "You are a function-calling assistant. Given a user query and a list of "
    "available functions, produce one or more function calls to fulfil the "
    "request.  Output ONLY function calls in the format: "
    "[func_name(param=value, ...)]"
)

# ── Auto-generated seed templates (wizard default) ────────────────────

SEED_TEMPLATES = [
    TASK_DESCRIPTION,
    TASK_DESCRIPTION + "\n\nThink step-by-step before answering.",
    TASK_DESCRIPTION + "\n\nBe concise. Output only the function call.",
    TASK_DESCRIPTION + "\n\nFollow the format exactly. No extra text.",
]

# ── Auto-generated mutations (wizard default — generic) ──────────────

AUTO_MUTATIONS = [
    "Add chain-of-thought reasoning",
    "Enforce strict output format",
    "Add error recovery instructions",
    "Inject few-shot examples",
    "Emphasise parameter extraction",
    "Add role-play framing",
    "Shorten to single paragraph",
    "Add 'think before answering' preamble",
    "Specify forbidden output patterns",
    "Add edge-case handling rules",
    "Rewrite in imperative voice",
    "Add output validation step",
]


# ── Proxy checks (auto-generated — format-level) ─────────────────────

def _is_valid_function_call(text: str) -> bool:
    """Check if output looks like a function call."""
    return "[" in text and "(" in text and ")" in text


def _has_params(text: str) -> bool:
    """Check if the call includes parameters."""
    return "=" in text


def _is_concise(text: str) -> bool:
    """Check if output is concise (no prose explanation)."""
    return len(text.splitlines()) <= 3


def _no_refusal(text: str) -> bool:
    """Check that the model doesn't refuse the task."""
    lower = text.lower()
    return not any(
        phrase in lower
        for phrase in ["i cannot", "i can't", "sorry", "i'm unable", "not possible"]
    )


PROXY_CHECKS = [
    ProxyCheck(name="bracket_format", check_fn=_is_valid_function_call, weight=2.0),
    ProxyCheck(name="has_params", check_fn=_has_params, weight=1.5),
    ProxyCheck(name="concise", check_fn=_is_concise, weight=1.0),
    ProxyCheck(name="no_refusal", check_fn=_no_refusal, weight=1.0),
]


# ── Scorer construction (what the wizard generates) ──────────────────

def build_scorer(client: LLMClient) -> Scorer:
    """Build the Composite scorer — wizard 'recommended' default."""
    judge = LLMJudge(
        rubric=(
            "Rate this output for a function-calling task. Score 0-10 on: "
            "1) Correct function name chosen, 2) Parameters extracted "
            "accurately, 3) Output format [func(param=val)], 4) Conciseness."
        ),
    )
    consistency = SelfConsistencyScorer(num_samples=3)
    proxy = ProxyMetricsScorer(checks=PROXY_CHECKS)

    return CompositeScorer([
        (judge, 0.35),
        (consistency, 0.35),
        (proxy, 0.30),
    ])


# ── Prepare test inputs from xLAM (strip labels) ────────────────────

def prepare_test_inputs(
    cases: list[XLAMCase],
    n: int = 20,
) -> tuple[list[str], list[XLAMCase]]:
    """Select *n* cases and build unlabelled test input strings.

    Returns the formatted test inputs AND the original cases (for later
    ground-truth evaluation).
    """
    rng = np.random.default_rng(42)
    indices = rng.choice(len(cases), size=min(n, len(cases)), replace=False)
    selected = [cases[int(i)] for i in sorted(indices)]

    test_inputs: list[str] = []
    for case in selected:
        tool_text = _format_xlam_tools(case.tools)
        test_inputs.append(
            f"Available functions:\n{tool_text}\n\nUser query: {case.query}"
        )

    return test_inputs, selected


# ── Ground-truth evaluation ──────────────────────────────────────────

def evaluate_on_gt(
    prompt: str,
    cases: list[XLAMCase],
    client: LLMClient,
) -> dict[str, float]:
    """Evaluate a prompt against ground-truth xLAM cases."""
    scores: list[float] = []
    name_hits = 0
    total_expected = 0
    param_scores: list[float] = []

    for case in cases:
        tool_text = _format_xlam_tools(case.tools)
        system = prompt
        user_msg = (
            f"Available functions:\n{tool_text}\n\nUser query: {case.query}"
        )

        try:
            response = client.complete(
                system_prompt=system,
                user_message=user_msg,
                temperature=0.7,
            )
        except Exception:
            response = ""

        if not response:
            response = ""

        score = score_xlam_case(response, case)
        scores.append(score)

        # Name accuracy and param accuracy
        parsed = _parse_function_calls(response)
        for exp in case.answers:
            total_expected += 1
            exp_name = exp["name"]
            exp_args = exp.get("arguments", {})
            matched_name = False
            for p in parsed:
                if (
                    p[0] == exp_name
                    or p[0].replace("_", ".") == exp_name
                    or p[0].replace(".", "_") == exp_name.replace(".", "_")
                ):
                    matched_name = True
                    param_scores.append(_score_params(p[1], exp_args))
                    break
            if matched_name:
                name_hits += 1
            else:
                param_scores.append(0.0)

    overall = float(np.mean(scores)) if scores else 0.0
    name_acc = name_hits / max(total_expected, 1)
    param_acc = float(np.mean(param_scores)) if param_scores else 0.0

    return {
        "overall": overall,
        "name_accuracy": name_acc,
        "param_accuracy": param_acc,
        "n_cases": len(cases),
    }


# ── Main ─────────────────────────────────────────────────────────────

def main() -> None:
    print()
    print("    ╔═══════════════════════════════════════════════════════════════╗")
    print("    ║  EvoSim × xLAM — Wizard-Style No-Eval Evolution             ║")
    print("    ║  No ground truth • No domain mutations • Fully automatic     ║")
    print("    ╚═══════════════════════════════════════════════════════════════╝")
    print()

    # ── Load data ────────────────────────────────────────
    print("  Loading xLAM dataset ...")
    try:
        by_cat = load_xlam_dataset(max_per_category=30)
    except Exception as exc:
        print(f"  ERROR: {exc}")
        sys.exit(1)

    all_cases: list[XLAMCase] = []
    for cat, cases in sorted(by_cat.items()):
        all_cases.extend(cases)
        print(f"    {cat}: {len(cases)} cases")
    print(f"  Total: {len(all_cases)} cases across {len(by_cat)} categories")

    # ── Prepare test inputs (strip labels) ───────────────
    test_inputs, gt_cases = prepare_test_inputs(all_cases, n=20)
    print(f"  Using {len(test_inputs)} unlabelled test inputs for no-eval\n")

    # ── Backend config ───────────────────────────────────
    backend = LLMBackend.OLLAMA
    model = os.getenv("OLLAMA_MODEL", "llama3.2")

    llm_cfg = PromptEvolverConfig(
        backend=backend,
        ollama_model=model,
    )
    client = LLMClient(llm_cfg)

    if not client.is_available():
        print(f"  ERROR: Cannot connect to Ollama ({model}).")
        print("  Start Ollama with: ollama serve")
        sys.exit(1)

    print(f"  Backend: {backend.value} ({model})")

    # ── Phase 0: Default prompt baseline ─────────────────
    print("\n  Phase 0: Default Prompt Baseline")
    print("  " + "─" * 50)

    default_prompt = TASK_DESCRIPTION
    print("  Evaluating default prompt on ground-truth ...")
    gt_default = evaluate_on_gt(default_prompt, gt_cases, client)
    print(f"  Default baseline: {gt_default['overall']:.1%} overall, "
          f"{gt_default['name_accuracy']:.1%} name accuracy, "
          f"{gt_default['param_accuracy']:.1%} param accuracy\n")

    # ── Phase 1: No-eval composite evolution ─────────────
    print("  Phase 1: No-Eval Composite Evolution (wizard default)")
    print("  " + "═" * 50)
    print()

    config = NoEvalConfig(
        iterations=3,
        population_size=4,
        num_islands=2,
        elite_size=3,
        backend=backend,
    )

    scorer = build_scorer(client)

    evolver = NoEvalPromptEvolver(
        task_description=TASK_DESCRIPTION,
        test_inputs=test_inputs,
        scorer=scorer,
        config=config,
        seed_templates=SEED_TEMPLATES,
    )

    t0 = time.time()
    result = evolver.run()
    elapsed = time.time() - t0

    print(f"\n  No-eval fitness: {result.best_score / 100:.1%}")
    print(f"  Temperature: {result.best_temperature:.3f}")
    print(f"  Wall time: {elapsed:.1f}s")

    # ── Evaluate best prompt on ground-truth ─────────────
    print("\n  Evaluating best no-eval prompt on ground-truth ...")
    gt_noeval = evaluate_on_gt(result.best_prompt, gt_cases, client)
    print(f"  GT overall: {gt_noeval['overall']:.1%}")
    print(f"  GT name accuracy: {gt_noeval['name_accuracy']:.1%}")
    print(f"  GT param accuracy: {gt_noeval['param_accuracy']:.1%}")

    # ── Phase 2: Run each individual strategy for comparison ─────
    print("\n  Phase 2: Individual Strategy Comparison")
    print("  " + "═" * 50)

    strategies: dict[str, tuple[Scorer, str]] = {
        "LLM-as-Judge": (
            LLMJudge(rubric=(
                "Rate this function-calling output 0-10. Criteria: correct "
                "function name, correct parameters, [func(p=v)] format, concise."
            )),
            "LLMJudge",
        ),
        "Self-Consistency": (
            SelfConsistencyScorer(num_samples=3),
            "SelfConsistencyScorer",
        ),
        "Proxy Metrics": (
            ProxyMetricsScorer(checks=PROXY_CHECKS),
            "ProxyMetricsScorer",
        ),
    }

    strategy_results: list[dict[str, Any]] = []

    for name, (scorer_obj, scorer_cls) in strategies.items():
        print(f"\n  ───────────────────────────────────────────────────────")
        print(f"  Strategy: {name}")
        print(f"  ───────────────────────────────────────────────────────")

        evo = NoEvalPromptEvolver(
            task_description=TASK_DESCRIPTION,
            test_inputs=test_inputs,
            scorer=scorer_obj,
            config=config,
            seed_templates=SEED_TEMPLATES,
        )
        t0 = time.time()
        res = evo.run()
        t_elapsed = time.time() - t0

        print(f"  Evaluating on ground-truth ...")
        gt_result = evaluate_on_gt(res.best_prompt, gt_cases, client)

        print(f"  No-eval fitness: {res.best_score / 100:.1%}  |  "
              f"GT score: {gt_result['overall']:.1%}  |  "
              f"Name acc: {gt_result['name_accuracy']:.1%}  |  "
              f"Param acc: {gt_result['param_accuracy']:.1%}")

        strategy_results.append({
            "strategy": name,
            "noeval_fitness": res.best_score,
            "gt_overall": gt_result["overall"],
            "gt_name_acc": gt_result["name_accuracy"],
            "gt_param_acc": gt_result["param_accuracy"],
            "wall_time": t_elapsed,
            "best_prompt": res.best_prompt[:200],
        })

    # Add composite result
    strategy_results.insert(0, {
        "strategy": "Composite (wizard default)",
        "noeval_fitness": result.best_score,
        "gt_overall": gt_noeval["overall"],
        "gt_name_acc": gt_noeval["name_accuracy"],
        "gt_param_acc": gt_noeval["param_accuracy"],
        "wall_time": elapsed,
        "best_prompt": result.best_prompt[:200],
    })

    # ── Comparison table ─────────────────────────────────
    print("\n")
    print("=" * 98)
    print("  xLAM: Wizard-Style No-Eval Results")
    print("=" * 98)
    print(f"  {'Approach':<30} {'Fitness':>8} {'GT Score':>10} "
          f"{'Func Name':>10} {'Param Acc':>10} {'Time':>8}")
    print("  " + "─" * 92)
    print(f"  {'Default prompt (baseline)':<30} {'—':>8} "
          f"{gt_default['overall']:>9.1%} "
          f"{gt_default['name_accuracy']:>9.1%} "
          f"{gt_default['param_accuracy']:>9.1%} {'—':>8}")

    for sr in strategy_results:
        print(f"  {sr['strategy']:<30} {sr['noeval_fitness'] / 100:>7.1%} "
              f"{sr['gt_overall']:>9.1%} "
              f"{sr['gt_name_acc']:>9.1%} "
              f"{sr['gt_param_acc']:>9.1%} {sr['wall_time']:>7.1f}s")

    print("  " + "─" * 80)

    # Best no-eval
    best = max(strategy_results, key=lambda x: x["gt_overall"])
    lift = best["gt_overall"] - gt_default["overall"]
    print(f"\n  Best no-eval strategy: {best['strategy']} "
          f"(GT: {best['gt_overall']:.1%})")
    print(f"  Lift over default: {'+' if lift >= 0 else ''}{lift:.1%}")

    # ── Save log ─────────────────────────────────────────
    log = {
        "benchmark": "xlam",
        "approach": "wizard_no_eval",
        "model": model,
        "config": {
            "iterations": config.iterations,
            "population_size": config.population_size,
            "num_islands": config.num_islands,
        },
        "default_baseline": gt_default,
        "strategies": strategy_results,
        "best_strategy": best["strategy"],
        "best_gt_score": best["gt_overall"],
        "lift_over_default": lift,
    }

    log_path = Path(__file__).parent.parent.parent / "xlam_noeval_comparison_log.json"
    log_path.write_text(json.dumps(log, indent=2, default=str))
    print(f"\n  Log saved to {log_path}")

    # ── Best prompt ──────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Best No-Eval Prompt ({best['strategy']})")
    print(f"  GT Score: {best['gt_overall']:.1%}")
    print(f"{'=' * 60}")
    print(f"    {best['best_prompt']}")
    print("\n  Done.")


if __name__ == "__main__":
    main()
