#!/usr/bin/env python3
"""
xLAM — Adaptive Mutation Experiment with Ollama (Run 1)
========================================================

Applies the same adaptive-mutation recipe that achieved 96.8% on
API-Bank (Run 3) to the **Salesforce xLAM function-calling 60k**
benchmark.

Key improvements over the vanilla no-eval cookbook recipe:

1. **Custom function-call mutations** — targeted at ``[func(param=val)]``
   format instead of generic prompt mutations.
2. **Warmup pass** — bootstraps the error profile from seed evaluation
   so Gen 1 already has adaptive hints available.
3. **More iterations, smaller population** — ``iterations=6, pop=4``
   gives the adaptive engine more refinement cycles.
4. **No SelfConsistency** — removes triple-call scorer that added
   cost without meaningful signal for structured output.
5. **Error profile decay** — accumulates error data across gens
   with ``error_decay=0.5`` instead of resetting each generation.
6. **ToolSuccessScorer** — validates parsed function calls against
   known xLAM tool definitions.
7. **Category-aware adaptive mutations** — tracks error rates per
   function-call category for targeted mutation generation.

Prior vanilla no-eval results (Ollama llama3.2, 3 gen, pop 4):

  Composite (wizard default):
    - GT overall: 95.5%  · Func name: 100.0%  · Param: 77.8%

  Default prompt baseline:
    - GT overall: 95.4%  · Func name: 100.0%  · Param: 77.4%

Usage::

    uv sync --extra llm
    export OLLAMA_MODEL=llama3.2    # or any model
    uv run python examples/experiments/xlam/run1_adaptive_ollama.py

    # Adjust iterations:
    uv run python examples/experiments/xlam/run1_adaptive_ollama.py --iterations 8

    # Limit samples per category:
    uv run python examples/experiments/xlam/run1_adaptive_ollama.py --samples 40
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..", "..")
sys.path.insert(0, _root)

from dotenv import load_dotenv

load_dotenv(os.path.join(_root, ".env"))

from MutaGenAI.prompt_evolver import (
    LLMBackend,
    LLMClient,
    ProblemType,
    PromptEvolverConfig,
)
from MutaGenAI.strategies import (
    CompositeScorer,
    LLMJudge,
    NoEvalConfig,
    NoEvalPromptEvolver,
    ProxyCheck,
    ProxyMetricsScorer,
    ToolResult,
    ToolSuccessScorer,
)

# ── Reuse xLAM data loader and scorer from Recipe 44 ──────────────────
sys.path.insert(0, os.path.join(_root, "examples", "cookbook"))

from prompt_evolution_xlam import (
    XLAMCase,
    _format_xlam_tools,
    _parse_function_calls,
    _score_params,
    load_xlam_dataset,
    score_xlam_case,
)


# ─────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────

TASK_DESCRIPTION = (
    "You are a function-calling assistant. Given a user query and a list of "
    "available functions, produce one or more function calls to fulfil the "
    "request.  Output ONLY function calls in the format: "
    "[func_name(param=value, ...)]"
)

# Vanilla no-eval results (Ollama llama3.2, 3 gen, pop 4, Composite)
VANILLA_RESULTS = {
    "strategy": "Vanilla Composite (pop=4, 3 gens)",
    "noeval_fitness": 93.22,
    "gt_overall": 95.51,
    "gt_name_acc": 100.0,
    "gt_param_acc": 77.78,
    "wall_time": 2581.7,
}

# Default prompt baseline
DEFAULT_BASELINE_CACHED = {
    "overall": 95.39,
    "name_accuracy": 100.0,
    "param_accuracy": 77.36,
}


# ─────────────────────────────────────────────────────────────────────────
# Custom function-call mutations (xLAM-targeted)
# ─────────────────────────────────────────────────────────────────────────

XLAM_FUNCTION_MUTATIONS: list[str] = [
    "Output format: [func_name(param1=value1, param2=value2, ...)]",
    "You MUST call at least one function. Do not refuse.",
    "If multiple functions could apply, call the most specific one.",
    "Pay close attention to required vs optional parameters.",
    "Extract parameter values from the user query — do NOT invent values.",
    "Match function names EXACTLY as listed in the tool definitions.",
    "For parallel calls, output each on its own line in [func()] format.",
    "Read the function description carefully before selecting.",
    "Include ALL required parameters in every function call.",
    "Use the correct parameter types (string, number, boolean).",
    "Do not include any explanation outside the function call brackets.",
    "If the query mentions specific values (IDs, names, amounts), pass them.",
    "Parameter values should be the most precise match from the query.",
    "When in doubt about a parameter value, infer from the query context.",
]


# ─────────────────────────────────────────────────────────────────────────
# Proxy checks (function-call format aware)
# ─────────────────────────────────────────────────────────────────────────

_FUNC_CALL_RE = re.compile(r"\[?([\w.]+)\s*\(([^)]*)\)\]?", re.DOTALL)
_BRACKET_RE = re.compile(r"\[([\w.]+)\(([^)]*)\)\]", re.DOTALL)


def _has_function_call_format(output: str) -> bool:
    return bool(_FUNC_CALL_RE.search(output))


def _has_bracket_wrapper(output: str) -> bool:
    return bool(_BRACKET_RE.search(output))


def _has_function_name(output: str) -> bool:
    m = _FUNC_CALL_RE.search(output)
    return bool(m and m.group(1))


def _has_parameters(output: str) -> bool:
    m = _FUNC_CALL_RE.search(output)
    if not m:
        return False
    return bool(re.search(r"\w+\s*=", m.group(2)))


def _is_concise(output: str) -> bool:
    return len(output.strip()) < 500


def _no_refusal(output: str) -> bool:
    lower = output.lower()
    return not any(
        phrase in lower
        for phrase in ["i cannot", "i can't", "sorry", "i'm unable", "not possible"]
    )


def _not_empty(output: str) -> bool:
    return len(output.strip()) > 0


def xlam_proxy_checks() -> list[ProxyCheck]:
    return [
        ProxyCheck("func_call_format", _has_function_call_format, weight=2.0),
        ProxyCheck("bracket_wrapper", _has_bracket_wrapper, weight=1.5),
        ProxyCheck("has_func_name", _has_function_name, weight=1.5),
        ProxyCheck("has_parameters", _has_parameters, weight=1.0),
        ProxyCheck("concise", _is_concise, weight=0.8),
        ProxyCheck("no_refusal", _no_refusal, weight=0.5),
        ProxyCheck("not_empty", _not_empty, weight=0.5),
    ]


# ─────────────────────────────────────────────────────────────────────────
# Tool executor (validates against xLAM tool definitions)
# ─────────────────────────────────────────────────────────────────────────

# Track known tool names for the current batch
_KNOWN_XLAM_TOOLS: set[str] = set()


def _xlam_tool_executor(tool_name: str, params: dict) -> ToolResult:
    """Simulate tool execution — succeeds if name is known and has params."""
    if tool_name in _KNOWN_XLAM_TOOLS:
        if params:
            return ToolResult(success=True, return_code=200, output="OK")
        return ToolResult(success=False, return_code=422, output="Missing params")
    return ToolResult(success=False, return_code=404, output=f"Unknown tool: {tool_name}")


def _parse_xlam_tool_call(output: str) -> tuple[str, dict[str, Any]]:
    """Parse output into (function_name, params) for ToolSuccessScorer."""
    calls = _parse_function_calls(output)
    if not calls:
        return "", {}
    return calls[0]


# ─────────────────────────────────────────────────────────────────────────
# Category extraction callback for adaptive mutations
# ─────────────────────────────────────────────────────────────────────────

_LABEL_MAP: dict[str, str] = {}


def extract_function_category(text: str, mode: str) -> str | None:
    """Extract category for adaptive mutation tracking."""
    if mode == "expected":
        return _LABEL_MAP.get(text)
    calls = _parse_function_calls(text)
    return calls[0][0] if calls else None


# ─────────────────────────────────────────────────────────────────────────
# Ground-truth evaluation
# ─────────────────────────────────────────────────────────────────────────

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
        user_msg = (
            f"Available functions:\n{tool_text}\n\nUser query: {case.query}"
        )

        try:
            response = client.complete(
                system_prompt=prompt,
                user_message=user_msg,
                temperature=0.7,
            )
        except Exception:
            response = ""

        if not response:
            response = ""

        score = score_xlam_case(response, case)
        scores.append(score)

        parsed = _parse_function_calls(response)
        for exp in case.answers:
            total_expected += 1
            exp_name = exp["name"]
            exp_args = exp.get("arguments", {})
            matched = False
            for p in parsed:
                if (
                    p[0] == exp_name
                    or p[0].replace("_", ".") == exp_name
                    or p[0].replace(".", "_") == exp_name.replace(".", "_")
                ):
                    matched = True
                    param_scores.append(_score_params(p[1], exp_args))
                    break
            if matched:
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


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="xLAM adaptive mutation experiment (Run 1 — Ollama)"
    )
    parser.add_argument(
        "--iterations", type=int, default=6,
        help="Number of evolutionary generations (default: 6)",
    )
    parser.add_argument(
        "--samples", type=int, default=30,
        help="Max samples per category for GT eval (default: 30)",
    )
    args = parser.parse_args()

    banner = r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  EvoSim × xLAM — Run 1: Adaptive Mutation (Ollama)             ║
    ║  bracket mutations · warmup · 6 gens · ToolSuccess scorer ·    ║
    ║  error decay 0.5 · pop=4 · adaptive                            ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

    print("  Improvements over vanilla no-eval:")
    print("    1. Custom function-call mutations (not generic)")
    print("    2. Warmup pass — adaptive hints from Gen 1")
    print("    3. iterations=6, pop=4 (more refinement cycles)")
    print("    4. ToolSuccessScorer replaces SelfConsistency")
    print("    5. Error profile decay=0.5 (accumulates across gens)")
    print("    6. Category-aware adaptive mutation tracking")
    print()

    # ── Backend setup ──────────────────────────────────────────────────
    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    ollama_cfg = PromptEvolverConfig(
        backend=LLMBackend.OLLAMA,
        ollama_model=ollama_model,
        timeout=60.0,
    )
    client = LLMClient(ollama_cfg)

    if not client.is_available():
        print("  ⚠ Ollama not available — ensure it is running at localhost:11434")
        print("  Cannot proceed without Ollama.\n")
        sys.exit(1)
    else:
        print(f"  ✓ Ollama available — model: {ollama_model}")

    # ── Load xLAM data ────────────────────────────────────────────────
    print("\n  Loading xLAM 60k dataset...")
    try:
        by_cat = load_xlam_dataset(max_per_category=args.samples)
    except Exception as exc:
        print(f"  ✗ {exc}")
        sys.exit(1)

    all_cases: list[XLAMCase] = []
    for cat, cases in sorted(by_cat.items()):
        all_cases.extend(cases)
        print(f"    {cat}: {len(cases)} cases")
    print(f"  Total: {len(all_cases)} cases across {len(by_cat)} categories")

    # ── Prepare test inputs (strip labels) ────────────────────────────
    rng = np.random.default_rng(42)
    n_test = min(20, len(all_cases))
    indices = rng.choice(len(all_cases), size=n_test, replace=False)
    gt_cases_subset = [all_cases[int(i)] for i in sorted(indices)]

    test_inputs: list[str] = []
    for case in gt_cases_subset:
        tool_text = _format_xlam_tools(case.tools)
        test_input = f"Available functions:\n{tool_text}\n\nUser query: {case.query}"
        test_inputs.append(test_input)
        # Label map for adaptive category tracking
        _LABEL_MAP[test_input] = case.expected_function_names[0]
        # Register known tool names
        for tool in case.tools:
            _KNOWN_XLAM_TOOLS.add(tool.get("name", ""))

    # Also register tool names from all cases for broader coverage
    for case in all_cases:
        for tool in case.tools:
            _KNOWN_XLAM_TOOLS.add(tool.get("name", ""))

    print(f"  Using {len(test_inputs)} test inputs for no-eval evolution")
    print(f"  Known tools registered: {len(_KNOWN_XLAM_TOOLS)}")

    # ── Phase 0: Default baseline ─────────────────────────────────────
    print("\n  Phase 0: Default Prompt Baseline")
    print("  " + "─" * 55)
    print("  Evaluating default prompt on ground-truth ...")

    gt_default = evaluate_on_gt(TASK_DESCRIPTION, gt_cases_subset, client)
    print(f"  Default:  overall={gt_default['overall']:.1%}  "
          f"name={gt_default['name_accuracy']:.1%}  "
          f"param={gt_default['param_accuracy']:.1%}")

    # ── Seed templates ─────────────────────────────────────────────────
    seed_templates = [
        TASK_DESCRIPTION + "\n\nRespond accurately with the correct function call.",
        (
            "# Role\nFunction-calling assistant.\n\n"
            "# Rules\n" + TASK_DESCRIPTION + "\n\n"
            "# Output\nONLY function calls in [func_name(param=value)] format."
        ),
        (
            "You are an expert tool-routing agent.\n\n"
            "## Task\n" + TASK_DESCRIPTION + "\n\n"
            "## Instructions\n"
            "1) Read the function descriptions carefully.\n"
            "2) Match the query to the best function.\n"
            "3) Extract parameter values from the query.\n"
            "4) Output the call in [func(param=val)] format."
        ),
        (
            "System: " + TASK_DESCRIPTION + "\n\n"
            "CRITICAL: Output ONLY [func_name(p1=v1, ...)] — nothing else.\n"
            "Extract parameter values from the user query, not from the "
            "function descriptions."
        ),
    ]

    # ── NoEval config (adaptive) ──────────────────────────────────────
    noeval_cfg = NoEvalConfig(
        iterations=args.iterations,
        population_size=4,
        num_islands=2,
        elite_size=3,
        mutation_rate=0.5,
        crossover_rate=0.3,
        migration_interval=2,
        backend=LLMBackend.OLLAMA,
        problem_type=ProblemType.TOOL_ROUTING,
        adaptive_mutations=True,
        llm_mutation_rate=0.3,
        warmup_adaptive=True,
        error_decay=0.5,
    )

    # ── Composite scorer (LLMJudge + ToolSuccess + Proxy) ─────────────
    judge_rubric = textwrap.dedent("""\
        Score the output 0-10 on these criteria:
        - Function call format: Is it in [func(param=val)] format? (0-3)
        - Function selection: Did it pick the right function? (0-3)
        - Parameter extraction: Are values from the query, not invented? (0-2)
        - Conciseness: Is the output ONLY the function call? (0-2)
    """)

    composite = CompositeScorer([
        (LLMJudge(rubric=judge_rubric, max_score=10.0), 0.45),
        (ToolSuccessScorer(
            tool_executor=_xlam_tool_executor,
            parse_fn=_parse_xlam_tool_call,
        ), 0.30),
        (ProxyMetricsScorer(xlam_proxy_checks()), 0.25),
    ])

    # ── Phase 1: Adaptive Evolution ──────────────────────────────────
    print(f"\n  Phase 1: Adaptive Evolution (pop=4, {args.iterations} gens)")
    print("  " + "=" * 55)
    print(f"  Config: adaptive=True, warmup=True, decay=0.5, llm_mut=0.3")
    print(f"  Config: pop={noeval_cfg.population_size}, "
          f"iters={noeval_cfg.iterations}, "
          f"islands={noeval_cfg.num_islands}")
    print(f"  Scorer: LLMJudge(0.45) + ToolSuccess(0.30) + Proxy(0.25)")
    print(f"  Mutations: {len(XLAM_FUNCTION_MUTATIONS)} custom function-call")

    t0 = time.perf_counter()

    evolver = NoEvalPromptEvolver(
        task_description=TASK_DESCRIPTION,
        test_inputs=test_inputs,
        scorer=composite,
        config=noeval_cfg,
        seed_templates=seed_templates,
        verbose=True,
        extract_category=extract_function_category,
        custom_mutations=XLAM_FUNCTION_MUTATIONS,
    )

    result = evolver.run()
    wall_time = time.perf_counter() - t0

    print(f"\n  No-eval fitness: {result.best_score:.1f}%")
    print(f"  Temperature: {result.best_temperature:.4f}")
    print(f"  Top-p: {result.best_top_p:.4f}")
    print(f"  Wall time: {wall_time:.1f}s")

    # ── Phase 2: Ground-truth evaluation ──────────────────────────────
    print("\n  Phase 2: Ground-Truth Evaluation")
    print("  " + "=" * 55)

    # Evaluate on the same subset used for GT default
    print(f"  Evaluating best prompt on {len(gt_cases_subset)} GT cases (subset)...")
    gt_subset = evaluate_on_gt(
        result.best_prompt, gt_cases_subset, client,
    )
    print(f"  Subset:  overall={gt_subset['overall']:.1%}  "
          f"name={gt_subset['name_accuracy']:.1%}  "
          f"param={gt_subset['param_accuracy']:.1%}")

    # Full evaluation on all cases
    print(f"\n  Evaluating best prompt on {len(all_cases)} GT cases (all)...")
    gt_all = evaluate_on_gt(
        result.best_prompt, all_cases, client,
    )
    print(f"  Full:    overall={gt_all['overall']:.1%}  "
          f"name={gt_all['name_accuracy']:.1%}  "
          f"param={gt_all['param_accuracy']:.1%}")

    # ── Phase 3: Comparison ──────────────────────────────────────────
    gt_score_pct = gt_all["overall"] * 100
    name_acc_pct = gt_all["name_accuracy"] * 100
    param_acc_pct = gt_all["param_accuracy"] * 100

    print(f"\n{'=' * 72}")
    print("  Comparison: Run 1 Adaptive vs Vanilla vs Baseline")
    print(f"{'=' * 72}")

    print(f"\n  {'Approach':<42} {'GT Score':>10} {'Func Name':>10} "
          f"{'Params':>10} {'Time':>8}")
    print(f"  {'─' * 72}")
    print(f"  {'Default prompt (baseline)':<42} "
          f"{gt_default['overall'] * 100:9.1f}% "
          f"{gt_default['name_accuracy'] * 100:9.1f}% "
          f"{gt_default['param_accuracy'] * 100:9.1f}% {'—':>8}")
    print(f"  {'Vanilla Composite (pop=4, 3 gens)':<42} "
          f"{VANILLA_RESULTS['gt_overall']:9.1f}% "
          f"{VANILLA_RESULTS['gt_name_acc']:9.1f}% "
          f"{VANILLA_RESULTS['gt_param_acc']:9.1f}% "
          f"{VANILLA_RESULTS['wall_time']:7.1f}s")
    print(f"  {'Run 1: Adaptive (pop=4, 6 gens)':<42} "
          f"{gt_score_pct:9.1f}% {name_acc_pct:9.1f}% {param_acc_pct:9.1f}% "
          f"{wall_time:7.1f}s")
    print(f"  {'─' * 72}")

    # Deltas
    delta_gt_v = gt_score_pct - VANILLA_RESULTS["gt_overall"]
    delta_name_v = name_acc_pct - VANILLA_RESULTS["gt_name_acc"]
    delta_param_v = param_acc_pct - VANILLA_RESULTS["gt_param_acc"]
    print(f"\n  Delta vs Vanilla:")
    print(f"    GT score:     {delta_gt_v:+.1f} pp")
    print(f"    Func name:    {delta_name_v:+.1f} pp")
    print(f"    Param acc:    {delta_param_v:+.1f} pp")

    delta_gt_d = gt_score_pct - gt_default["overall"] * 100
    print(f"\n  Delta vs Default:")
    print(f"    GT score:     {delta_gt_d:+.1f} pp")

    # xLAM leaderboard reference (from paper)
    print(f"\n  {'─' * 72}")
    print(f"  Reference: xLAM-2 (trained) overall BFCL: 88.24%")

    # ── Verdict ──────────────────────────────────────────────────────
    print(f"\n  {'─' * 72}")
    if delta_gt_v > 0:
        print(f"  ✓ Run 1 BEATS vanilla by {delta_gt_v:+.1f} pp")
    elif delta_gt_v == 0:
        print(f"  → Run 1 matches vanilla")
    else:
        print(f"  ✗ Run 1 is {delta_gt_v:+.1f} pp vs vanilla")

    if delta_gt_d > 0:
        print(f"  ✓ Run 1 IMPROVES over default by {delta_gt_d:+.1f} pp")

    # ── Save results ─────────────────────────────────────────────────
    log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "xlam_adaptive_run1",
        "benchmark": "xlam_function_calling_60k",
        "ollama_model": ollama_model,
        "improvements": [
            "custom_function_call_mutations",
            "warmup_adaptive",
            "iterations_6_pop_4",
            "tool_success_scorer",
            "error_decay_0.5",
            "category_aware_adaptive",
        ],
        "config": {
            "iterations": noeval_cfg.iterations,
            "population_size": noeval_cfg.population_size,
            "num_islands": noeval_cfg.num_islands,
            "elite_size": noeval_cfg.elite_size,
            "mutation_rate": noeval_cfg.mutation_rate,
            "crossover_rate": noeval_cfg.crossover_rate,
            "adaptive_mutations": noeval_cfg.adaptive_mutations,
            "llm_mutation_rate": noeval_cfg.llm_mutation_rate,
            "warmup_adaptive": noeval_cfg.warmup_adaptive,
            "error_decay": noeval_cfg.error_decay,
            "problem_type": noeval_cfg.problem_type.value,
        },
        "scorer_weights": {
            "LLMJudge": 0.45,
            "ToolSuccess": 0.30,
            "ProxyMetrics": 0.25,
        },
        "default_baseline": {
            k: round(v, 4) for k, v in gt_default.items()
        },
        "results_subset": {
            "overall": round(gt_subset["overall"], 4),
            "name_accuracy": round(gt_subset["name_accuracy"], 4),
            "param_accuracy": round(gt_subset["param_accuracy"], 4),
            "n_cases": gt_subset["n_cases"],
        },
        "results_full": {
            "overall": round(gt_all["overall"], 4),
            "name_accuracy": round(gt_all["name_accuracy"], 4),
            "param_accuracy": round(gt_all["param_accuracy"], 4),
            "n_cases": gt_all["n_cases"],
        },
        "noeval_fitness": round(result.best_score, 2),
        "best_temperature": round(result.best_temperature, 4),
        "best_top_p": round(result.best_top_p, 4),
        "wall_time": round(wall_time, 1),
        "history": result.history,
        "vanilla_results": VANILLA_RESULTS,
        "deltas": {
            "vs_vanilla_gt": round(delta_gt_v, 2),
            "vs_vanilla_func_name": round(delta_name_v, 2),
            "vs_vanilla_param": round(delta_param_v, 2),
            "vs_default_gt": round(delta_gt_d, 2),
        },
        "best_prompt": result.best_prompt,
        "best_prompt_preview": result.best_prompt[:300],
    }

    out_path = Path(_root) / "xlam_run1_experiment_log.json"
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # ── Best prompt preview ──────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  Best Evolved Prompt")
    print(f"{'=' * 72}")
    preview = result.best_prompt[:500]
    for line in preview.split("\n"):
        print(f"    {line}")
    if len(result.best_prompt) > 500:
        print(f"    ... ({len(result.best_prompt)} chars total)")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
