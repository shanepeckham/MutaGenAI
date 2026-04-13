#!/usr/bin/env python3
"""
Cookbook Recipe 49 — API-Bank No-Eval vs Ground-Truth Comparison
================================================================

Runs the **same** API-Bank prompt evolution task using both approaches:

1. **No-eval strategies** — pretend we have no ground-truth labels and
   evolve prompts using LLM-as-Judge, Proxy Metrics, Self-Consistency,
   and a Composite scorer.
2. **Ground-truth eval** — the standard Recipe 46 approach that scores
   against known-correct API calls.

The head-to-head comparison shows how close each no-eval strategy gets
to ground-truth–guided evolution, proving the no-eval pipeline can
find competitive prompts without any labelled data.

Usage::

    uv sync --extra llm
    uv run python examples/cookbook/prompt_evolution_apibank_no_eval.py

    # Run only no-eval strategies:
    uv run python examples/cookbook/prompt_evolution_apibank_no_eval.py --no-eval-only

    # Run only ground-truth baseline:
    uv run python examples/cookbook/prompt_evolution_apibank_no_eval.py --gt-only

    # More generations for better convergence:
    uv run python examples/cookbook/prompt_evolution_apibank_no_eval.py --iterations 5

References
----------
* Recipe 46 — API-Bank ground-truth evolution (prompt_evolution_apibank.py)
* Recipe 48 — No-eval strategies overview (prompt_evolution_no_eval.py)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import statistics
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

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
    PreferencePair,
    PreferenceScorer,
    ProxyCheck,
    ProxyMetricsScorer,
    Scorer,
    SelfConsistencyScorer,
    ToolResult,
    ToolSuccessScorer,
)


# ─────────────────────────────────────────────────────────────────────────
# Import API-Bank helpers from Recipe 46
# ─────────────────────────────────────────────────────────────────────────

from prompt_evolution_apibank import (
    ALGORITHM_CONFIGS,
    APIBankCase,
    APIBankExperiment,
    _APIBANK_MUTATIONS,
    _APIBANK_SEED_TEMPLATES,
    _crossover_apibank_templates,
    _mutate_apibank_template,
    evaluate_baseline,
    load_apibank_dataset,
    run_apibank_evolution,
    score_apibank_case,
    show_prompt_evolution,
)


# ─────────────────────────────────────────────────────────────────────────
# API-Bank–specific no-eval seed templates
# ─────────────────────────────────────────────────────────────────────────

TASK_DESCRIPTION = textwrap.dedent("""\
    You are an API-calling assistant. Given a conversation and API
    descriptions, generate the correct API request in the format
    [ApiName(key1='value1', key2='value2', ...)].

    Rules:
    - Output EXACTLY ONE API call in the bracket format shown above.
    - Match API names exactly as described.
    - Extract parameter values from the conversation — do NOT invent values.
    - Include ALL required parameters.
    - Use single quotes around parameter values.
    - Output ONLY the API call, nothing else.
""").strip()


# ─────────────────────────────────────────────────────────────────────────
# API-Bank–specific proxy checks
# ─────────────────────────────────────────────────────────────────────────

_API_CALL_RE = re.compile(r"\[(\w+)\((.*?)\)\]", re.DOTALL)


def _has_api_bracket_format(output: str) -> bool:
    """Check output uses [ApiName(...)] format."""
    return bool(_API_CALL_RE.search(output))


def _has_single_quotes(output: str) -> bool:
    """Check parameter values use single quotes."""
    m = _API_CALL_RE.search(output)
    if not m:
        return False
    params_str = m.group(2)
    # At least one param with single quotes
    return bool(re.search(r"\w+\s*=\s*'[^']*'", params_str))


def _has_api_name(output: str) -> bool:
    """Check output contains a parseable API name."""
    m = _API_CALL_RE.search(output)
    return bool(m and m.group(1))


def _has_parameters(output: str) -> bool:
    """Check output contains at least one key='value' parameter."""
    m = _API_CALL_RE.search(output)
    if not m:
        return False
    return bool(re.search(r"\w+\s*=\s*'", m.group(2)))


def _no_extra_text(output: str) -> bool:
    """Check output is mostly the API call (< 20 chars of extra text)."""
    m = _API_CALL_RE.search(output)
    if not m:
        return False
    api_call_len = len(m.group(0))
    return len(output.strip()) - api_call_len < 20


def _under_300_chars(output: str) -> bool:
    """API calls should be concise."""
    return len(output.strip()) < 300


def apibank_proxy_checks() -> list[ProxyCheck]:
    """API-Bank–specific structural checks."""
    return [
        ProxyCheck("bracket_format", _has_api_bracket_format, weight=2.0),
        ProxyCheck("has_api_name", _has_api_name, weight=1.5),
        ProxyCheck("has_parameters", _has_parameters, weight=1.0),
        ProxyCheck("single_quotes", _has_single_quotes, weight=1.0),
        ProxyCheck("no_extra_text", _no_extra_text, weight=0.8),
        ProxyCheck("under_300_chars", _under_300_chars, weight=0.5),
        ProxyCheck("not_empty", lambda o: len(o.strip()) > 0, weight=0.5),
    ]


# ─────────────────────────────────────────────────────────────────────────
# API-Bank–specific tool executor (simulated)
# ─────────────────────────────────────────────────────────────────────────

# Known API-Bank tool names for validation
_KNOWN_APIBANK_TOOLS = {
    "ToolSearcher", "GetUserToken", "AddAgenda", "DeleteAgenda",
    "ModifyAgenda", "QueryAgenda", "SendEmail", "DeleteEmail",
    "QueryBalance", "TransferMoney", "QueryHealthData", "BookHotel",
    "CancelHotel", "BookFlight", "CancelFlight", "GetTrainTicket",
    "CancelTrainTicket", "QueryStock", "BuyStock", "SellStock",
    "OpenBankAccount", "QueryMedicalRecord", "ModifyRegistration",
    "QueryRegistration", "PlayMusic", "PauseMusic", "AddAlarm",
    "DeleteAlarm", "QueryAlarm", "ModifyAlarm", "GetWeather",
    "QueryNews", "TranslateText", "RecordHealthData",
}


def _apibank_tool_executor(tool_name: str, params: dict) -> ToolResult:
    """Simulate API-Bank tool execution."""
    if tool_name in _KNOWN_APIBANK_TOOLS:
        if params:
            return ToolResult(success=True, return_code=200, output="OK")
        return ToolResult(success=False, return_code=422, output="Missing params")
    return ToolResult(success=False, return_code=404, output="Unknown API")


def _parse_apibank_tool_call(output: str) -> tuple[str, dict[str, Any]]:
    """Parse [ApiName(key='val',...)] format for ToolSuccessScorer."""
    m = _API_CALL_RE.search(output)
    if not m:
        return "", {}
    api_name = m.group(1)
    params_str = m.group(2)
    params = dict(re.findall(r"(\w+)\s*=\s*'([^']*)'", params_str))
    return api_name, params


# ─────────────────────────────────────────────────────────────────────────
# Build preference pairs from task knowledge (no labels needed)
# ─────────────────────────────────────────────────────────────────────────

def _build_preference_pairs(test_inputs: list[str]) -> list[PreferencePair]:
    """Build preference pairs from general knowledge of API-Bank format.

    These are NOT ground-truth labels — they just show the format
    preference (bracket-style API call vs. prose/JSON).
    """
    pairs = []
    for inp in test_inputs[:5]:  # Use first 5 inputs
        pairs.append(PreferencePair(
            input_text=inp,
            good_output="[SomeApi(param1='value1', param2='value2')]",
            bad_output='I think you should call the SomeApi function with param1 set to value1.',
        ))
    return pairs


# ─────────────────────────────────────────────────────────────────────────
# No-eval evolution with API-Bank–tuned seed templates
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class NoEvalAPIBankResult:
    """Result from a no-eval API-Bank evolution run."""

    strategy_name: str
    evolved_score_noeval: float  # The no-eval fitness score (0-100)
    gt_score: float              # Ground-truth score when tested against labels
    best_prompt: str
    best_temperature: float
    best_top_p: float
    wall_time: float
    history: list[tuple[int, float]]
    api_name_accuracy: float = 0.0
    param_accuracy: float = 0.0


def evaluate_prompt_on_gt(
    prompt_template: str,
    temperature: float,
    top_p: float,
    cases: list[APIBankCase],
    client: LLMClient,
) -> tuple[float, float, float]:
    """Evaluate a prompt against ground-truth API-Bank cases.

    Returns (overall_score, api_name_acc, param_acc).
    """
    total = 0.0
    name_hits = 0
    param_total = 0.0

    for case in cases:
        # The no-eval prompts use {instruction} and {input} placeholders
        # but the GT cases have case.instruction and case.input_text
        sys_prompt = prompt_template

        # If the prompt has placeholders, fill them
        if "{apibank_instruction}" in sys_prompt:
            sys_prompt = sys_prompt.replace(
                "{apibank_instruction}", case.instruction
            )
        if "{apibank_input}" in sys_prompt:
            sys_prompt = sys_prompt.replace(
                "{apibank_input}", case.input_text
            )

        # Build user message from instruction + input if no placeholders
        if "{apibank_instruction}" not in prompt_template and "{apibank_input}" not in prompt_template:
            user_message = f"{case.instruction}\n\n{case.input_text}\n\nGenerate API Request:"
        else:
            user_message = "Generate API Request:"

        response = client.complete(
            system_prompt=sys_prompt,
            user_message=user_message,
            temperature=temperature,
            top_p=top_p,
        )
        if response is None:
            continue

        score, detail = score_apibank_case(response, case)
        total += score
        if detail["api_name_match"]:
            name_hits += 1
        param_total += detail["param_score"]

    n = len(cases) if cases else 1
    return (
        total / n * 100.0,
        name_hits / n * 100.0,
        param_total / n * 100.0,
    )


def run_no_eval_strategy(
    strategy_name: str,
    scorer: Scorer,
    test_inputs: list[str],
    gt_cases: list[APIBankCase],
    client: LLMClient,
    config: NoEvalConfig,
    verbose: bool = True,
) -> NoEvalAPIBankResult:
    """Run a no-eval strategy and then evaluate the winner on ground truth."""

    if verbose:
        print(f"\n  {'─' * 55}")
        print(f"  Strategy: {strategy_name}")
        print(f"  {'─' * 55}")

    # Build API-Bank–specific seed templates (with instruction/input
    # placeholders stripped since no-eval uses raw test_inputs)
    seed_templates = [
        TASK_DESCRIPTION + "\n\nRespond accurately with the correct API call.",
        (
            "# Role\nAPI-calling assistant.\n\n"
            "# Rules\n" + TASK_DESCRIPTION + "\n\n"
            "# Output\nONLY the API call in [ApiName(key='val')] format."
        ),
        (
            "You are an expert API orchestration agent.\n\n"
            "## Task\n" + TASK_DESCRIPTION + "\n\n"
            "## Instructions\n"
            "Think step by step: 1) identify API, 2) extract params, 3) output call."
        ),
        (
            "System: " + TASK_DESCRIPTION + "\n\n"
            "CRITICAL: Output ONLY [ApiName(key1='value1', ...)] — nothing else.\n"
            "Extract parameter values verbatim from the conversation."
        ),
    ]

    evolver = NoEvalPromptEvolver(
        task_description=TASK_DESCRIPTION,
        test_inputs=test_inputs,
        scorer=scorer,
        config=config,
        seed_templates=seed_templates,
        verbose=verbose,
    )

    result = evolver.run()

    # Now evaluate the evolved prompt on ground-truth
    if verbose:
        print(f"  Evaluating best prompt on ground-truth ({len(gt_cases)} cases)...")

    gt_score, name_acc, param_acc = evaluate_prompt_on_gt(
        result.best_prompt,
        result.best_temperature,
        result.best_top_p,
        gt_cases,
        client,
    )

    if verbose:
        print(
            f"  No-eval fitness: {result.best_score:.1f}%  |  "
            f"GT score: {gt_score:.1f}%  |  "
            f"API name: {name_acc:.1f}%  |  "
            f"Params: {param_acc:.1f}%"
        )

    return NoEvalAPIBankResult(
        strategy_name=strategy_name,
        evolved_score_noeval=result.best_score,
        gt_score=gt_score,
        best_prompt=result.best_prompt,
        best_temperature=result.best_temperature,
        best_top_p=result.best_top_p,
        wall_time=result.wall_time,
        history=result.history,
        api_name_accuracy=name_acc,
        param_accuracy=param_acc,
    )


# ─────────────────────────────────────────────────────────────────────────
# Comparison table
# ─────────────────────────────────────────────────────────────────────────


def show_comparison_table(
    no_eval_results: list[NoEvalAPIBankResult],
    gt_experiment: Optional[APIBankExperiment],
    default_baseline: float,
) -> None:
    """Print head-to-head comparison of no-eval vs ground-truth evolution."""

    print(f"\n{'=' * 90}")
    print("  API-Bank: No-Eval vs Ground-Truth Comparison")
    print(f"{'=' * 90}")
    print(
        f"  {'Approach':<28} {'Fitness':>9} {'GT Score':>10} "
        f"{'API Name':>10} {'Params':>10} {'Time':>8}"
    )
    print(f"  {'─' * 85}")

    # Default baseline
    print(
        f"  {'Default prompt (baseline)':<28} {'—':>9} "
        f"{default_baseline:9.1f}% {'—':>10} {'—':>10} {'—':>8}"
    )

    # No-eval results
    for r in no_eval_results:
        print(
            f"  {r.strategy_name:<28} {r.evolved_score_noeval:8.1f}% "
            f"{r.gt_score:9.1f}% {r.api_name_accuracy:9.1f}% "
            f"{r.param_accuracy:9.1f}% {r.wall_time:7.1f}s"
        )

    # Ground-truth result
    if gt_experiment:
        print(f"  {'─' * 85}")
        print(
            f"  {'Ground-truth evolution':<28} "
            f"{gt_experiment.evolved_score:8.1f}% "
            f"{gt_experiment.evolved_score:9.1f}% "
            f"{gt_experiment.api_name_accuracy:9.1f}% "
            f"{gt_experiment.param_accuracy:9.1f}% "
            f"{gt_experiment.wall_time:7.1f}s"
        )

    print(f"  {'─' * 85}")

    # Analysis
    if gt_experiment and no_eval_results:
        best_noeval = max(no_eval_results, key=lambda r: r.gt_score)
        gap = gt_experiment.evolved_score - best_noeval.gt_score
        print(
            f"\n  Best no-eval strategy: {best_noeval.strategy_name} "
            f"(GT score: {best_noeval.gt_score:.1f}%)"
        )
        print(
            f"  Ground-truth evolution: {gt_experiment.evolved_score:.1f}%"
        )
        print(
            f"  Gap: {gap:.1f} pp  "
            f"({'no-eval wins!' if gap <= 0 else f'ground-truth ahead by {gap:.1f} pp'})"
        )
        if best_noeval.gt_score > default_baseline:
            lift = best_noeval.gt_score - default_baseline
            print(
                f"  No-eval lift over default baseline: +{lift:.1f} pp"
            )


# ─────────────────────────────────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────────────────────────────────


def save_comparison_log(
    no_eval_results: list[NoEvalAPIBankResult],
    gt_experiment: Optional[APIBankExperiment],
    default_baseline: float,
    path: str = "apibank_noeval_comparison_log.json",
) -> None:
    """Save comparison results to JSON."""
    log: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "default_baseline": round(default_baseline, 2),
        "no_eval_results": [],
        "ground_truth_result": None,
    }

    for r in no_eval_results:
        log["no_eval_results"].append({
            "strategy": r.strategy_name,
            "noeval_fitness": round(r.evolved_score_noeval, 2),
            "gt_score": round(r.gt_score, 2),
            "api_name_accuracy": round(r.api_name_accuracy, 2),
            "param_accuracy": round(r.param_accuracy, 2),
            "wall_time": round(r.wall_time, 1),
            "best_temperature": round(r.best_temperature, 4),
            "best_top_p": round(r.best_top_p, 4),
            "history": r.history,
            "best_prompt_preview": r.best_prompt[:200],
        })

    if gt_experiment:
        log["ground_truth_result"] = {
            "evolved_score": round(gt_experiment.evolved_score, 2),
            "api_name_accuracy": round(gt_experiment.api_name_accuracy, 2),
            "param_accuracy": round(gt_experiment.param_accuracy, 2),
            "wall_time": round(gt_experiment.wall_time, 1),
            "baseline_score": round(gt_experiment.baseline_score, 2),
            "history": gt_experiment.history,
        }

    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Comparison log saved to {path}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="API-Bank: No-Eval vs Ground-Truth comparison"
    )
    parser.add_argument(
        "--no-eval-only", action="store_true",
        help="Run only the no-eval strategies (skip ground-truth evolution)",
    )
    parser.add_argument(
        "--gt-only", action="store_true",
        help="Run only the ground-truth evolution (skip no-eval strategies)",
    )
    parser.add_argument(
        "--iterations", type=int, default=3,
        help="Number of evolutionary generations (default: 3)",
    )
    parser.add_argument(
        "--level", type=str, default="level_1",
        choices=["level_1", "level_2"],
        help="API-Bank level to test (default: level_1)",
    )
    args = parser.parse_args()

    banner = r"""
    ╔═══════════════════════════════════════════════════════════════╗
    ║  EvoSim × API-Bank — No-Eval vs Ground-Truth Comparison      ║
    ║  Same benchmark, same model, same evolution — different       ║
    ║  fitness signals.  How close can no-eval get?                 ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)

    # ── Backend setup ──────────────────────────────────────────────────
    ollama_cfg = PromptEvolverConfig(
        backend=LLMBackend.OLLAMA,
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        timeout=60.0,
    )
    client = LLMClient(ollama_cfg)

    if not client.is_available():
        print("  ⚠ Ollama not available — ensure it is running.")
        print("  Running in mock mode for demonstration.\n")

    # ── Load dataset ───────────────────────────────────────────────────
    print("  Loading API-Bank benchmark data...")
    try:
        by_level = load_apibank_dataset(
            max_per_level=30, levels=[args.level]
        )
    except Exception as exc:
        print(f"  ✗ {exc}")
        return

    gt_cases = by_level.get(args.level, [])
    if not gt_cases:
        print(f"  No cases loaded for {args.level}")
        return

    print(f"  {args.level}: {len(gt_cases)} cases loaded")

    # ── Extract test inputs (strip labels — pretend we don't have them) ─
    # Take a representative subset as unlabelled test inputs
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(gt_cases), size=min(10, len(gt_cases)), replace=False
    )
    test_inputs = []
    for idx in sample_indices:
        case = gt_cases[int(idx)]
        # Combine instruction + input as the "unlabelled" test input
        test_input = f"{case.instruction}\n\n{case.input_text}"
        test_inputs.append(test_input)

    print(f"  Using {len(test_inputs)} unlabelled test inputs for no-eval strategies")

    # ── Default baseline ───────────────────────────────────────────────
    print("\n  Phase 0: Default Prompt Baseline")
    print("  " + "─" * 50)
    default_baseline = evaluate_baseline(gt_cases, args.level, client)

    # ── No-eval config ─────────────────────────────────────────────────
    noeval_cfg = NoEvalConfig(
        iterations=args.iterations,
        population_size=4,
        num_islands=2,
        elite_size=3,
        mutation_rate=0.5,
        crossover_rate=0.3,
        migration_interval=3,
        backend=LLMBackend.OLLAMA,
    )

    no_eval_results: list[NoEvalAPIBankResult] = []
    gt_experiment: Optional[APIBankExperiment] = None

    # ── Phase 1: No-eval strategies ────────────────────────────────────
    if not args.gt_only:
        print("\n  Phase 1: No-Eval Strategies")
        print("  " + "=" * 50)

        # Strategy 1: LLM-as-Judge
        judge_rubric = textwrap.dedent("""\
            Score the output 0-10 on these criteria:
            - API call format: Is it in [ApiName(key='value', ...)] format? (0-4)
            - Parameter extraction: Are values from the conversation, not invented? (0-3)
            - API selection: Does the chosen API match the user's intent? (0-2)
            - Conciseness: Is it ONLY the API call with no extra text? (0-1)
        """)
        result = run_no_eval_strategy(
            "LLM-as-Judge",
            LLMJudge(rubric=judge_rubric, max_score=10.0),
            test_inputs, gt_cases, client, noeval_cfg,
        )
        no_eval_results.append(result)

        # Strategy 3: Tool-Use Success
        result = run_no_eval_strategy(
            "Tool-Use Success",
            ToolSuccessScorer(
                tool_executor=_apibank_tool_executor,
                parse_fn=_parse_apibank_tool_call,
            ),
            test_inputs, gt_cases, client, noeval_cfg,
        )
        no_eval_results.append(result)

        # Strategy 4: Self-Consistency
        result = run_no_eval_strategy(
            "Self-Consistency",
            SelfConsistencyScorer(num_samples=3),
            test_inputs, gt_cases, client, noeval_cfg,
        )
        no_eval_results.append(result)

        # Strategy 5: Proxy Metrics (API-Bank specific)
        result = run_no_eval_strategy(
            "Proxy Metrics (API-Bank)",
            ProxyMetricsScorer(apibank_proxy_checks()),
            test_inputs, gt_cases, client, noeval_cfg,
        )
        no_eval_results.append(result)

        # Strategy 6: Preference Scoring
        pref_pairs = _build_preference_pairs(test_inputs)
        result = run_no_eval_strategy(
            "Preference Scoring",
            PreferenceScorer(pref_pairs),
            test_inputs, gt_cases, client, noeval_cfg,
        )
        no_eval_results.append(result)

        # Composite: Judge 40% + Tool Success 25% + Proxy 25% + Consistency 10%
        composite = CompositeScorer([
            (LLMJudge(rubric=judge_rubric, max_score=10.0), 0.4),
            (ToolSuccessScorer(
                tool_executor=_apibank_tool_executor,
                parse_fn=_parse_apibank_tool_call,
            ), 0.25),
            (ProxyMetricsScorer(apibank_proxy_checks()), 0.25),
            (SelfConsistencyScorer(num_samples=3), 0.1),
        ])
        result = run_no_eval_strategy(
            "Composite (recommended)",
            composite,
            test_inputs, gt_cases, client, noeval_cfg,
        )
        no_eval_results.append(result)

    # ── Phase 2: Ground-truth evolution ────────────────────────────────
    if not args.no_eval_only:
        print("\n  Phase 2: Ground-Truth Evolution")
        print("  " + "=" * 50)

        algo_params = ALGORITHM_CONFIGS["standard"]
        gt_cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            timeout=60.0,
            iterations=args.iterations,
            population_size=algo_params["population_size"],
            num_islands=algo_params["num_islands"],
            elite_size=algo_params["elite_size"],
            mutation_rate=algo_params["mutation_rate"],
            crossover_rate=algo_params["crossover_rate"],
            eval_sample_size=algo_params["eval_sample_size"],
        )

        gt_experiment = run_apibank_evolution(
            cases=gt_cases,
            client=client,
            config=gt_cfg,
            algorithm_name="ground-truth",
            verbose=True,
        )
        show_prompt_evolution(gt_experiment)

    # ── Comparison ─────────────────────────────────────────────────────
    show_comparison_table(no_eval_results, gt_experiment, default_baseline)

    # ── Save log ───────────────────────────────────────────────────────
    log_path = os.path.join(
        os.path.dirname(__file__), "..", "..",
        "apibank_noeval_comparison_log.json",
    )
    save_comparison_log(no_eval_results, gt_experiment, default_baseline, log_path)

    # ── Best no-eval prompt ────────────────────────────────────────────
    if no_eval_results:
        best = max(no_eval_results, key=lambda r: r.gt_score)
        print(f"\n{'=' * 60}")
        print(f"  Best No-Eval Prompt ({best.strategy_name})")
        print(f"  GT Score: {best.gt_score:.1f}%  |  No-eval fitness: {best.evolved_score_noeval:.1f}%")
        print(f"{'=' * 60}")
        preview = best.best_prompt[:500]
        for line in preview.split("\n"):
            print(f"    {line}")
        if len(best.best_prompt) > 500:
            print(f"    ... ({len(best.best_prompt)} chars total)")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
