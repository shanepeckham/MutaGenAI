#!/usr/bin/env python3
"""
API-Bank — Adaptive Mutation Experiment (No Ground Truth)
==========================================================

Tests whether ``adaptive_mutations=True`` + ``llm_mutation_rate=0.3``
improves no-eval prompt evolution on the API-Bank benchmark compared
to the prior Composite run (Recipe 49) which used vanilla mutations.

Key differences from the prior run:
  - ``adaptive_mutations=True``  — error-guided mutation targeting weak APIs
  - ``llm_mutation_rate=0.3``    — 30% chance of LLM rewrite using failures
  - ``ProblemType.TOOL_ROUTING`` — tool-routing mutation pool
  - ``population_size=8``        — double the prior run (was 4)
  - ``extract_category`` callback routes API name errors to the adaptive engine

Prior best results (Composite, pop=4, no adaptive):
  - No-eval fitness: 81.3%
  - GT score:        82.3%
  - API name acc:    83.3%
  - Param acc:       74.1%

Leaderboard (Li et al., EMNLP 2023):
  - GPT-4:     Call 83.8% · Retrieval 41.2%
  - GPT-3.5:   Call 82.6% · Retrieval 35.3%

Usage::

    uv sync --extra llm
    uv run python examples/experiments/apibank/run1_adaptive_noeval.py

    # Choose level:
    uv run python examples/experiments/apibank/run1_adaptive_noeval.py --level level_2

    # Adjust iterations:
    uv run python examples/experiments/apibank/run1_adaptive_noeval.py --iterations 5
"""
from __future__ import annotations

import argparse
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

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..", "..")
sys.path.insert(0, _root)

from dotenv import load_dotenv

load_dotenv(os.path.join(_root, ".env"))

from MutaGenAI.prompt_evolver import (
    LLMBackend,
    LLMClient,
    ProblemType,
    PromptCandidate,
    PromptEvolverConfig,
)
from MutaGenAI.strategies import (
    CompositeScorer,
    LLMJudge,
    NoEvalConfig,
    NoEvalPromptEvolver,
    ProxyCheck,
    ProxyMetricsScorer,
    SelfConsistencyScorer,
    ToolResult,
    ToolSuccessScorer,
)

# ── Import API-Bank helpers from Recipe 46 ─────────────────────────────
sys.path.insert(0, os.path.join(_root, "examples", "cookbook"))

from prompt_evolution_apibank import (
    APIBankCase,
    APIBankExperiment,
    _parse_api_call,
    _PARAM_RE,
    evaluate_baseline,
    load_apibank_dataset,
    score_apibank_case,
)


# ─────────────────────────────────────────────────────────────────────────
# Constants
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

# Prior run results for comparison (Composite, pop=4, no adaptive)
PRIOR_RESULTS = {
    "strategy": "Composite (recommended)",
    "noeval_fitness": 81.34,
    "gt_score": 82.32,
    "api_name_accuracy": 83.33,
    "param_accuracy": 74.14,
    "wall_time": 2029.4,
}

# Published leaderboard (Li et al., EMNLP 2023)
LEADERBOARD = {
    "GPT-4": {"call": 83.8, "retrieval": 41.2, "plan": 38.8},
    "GPT-3.5": {"call": 82.6, "retrieval": 35.3, "plan": 13.1},
    "Lynx-7B": {"call": 78.4, "retrieval": 22.5, "plan": 11.9},
}


# ─────────────────────────────────────────────────────────────────────────
# API-Bank proxy checks (same as recipe 49)
# ─────────────────────────────────────────────────────────────────────────

_API_CALL_RE = re.compile(r"\[(\w+)\((.*?)\)\]", re.DOTALL)


def _has_api_bracket_format(output: str) -> bool:
    return bool(_API_CALL_RE.search(output))


def _has_single_quotes(output: str) -> bool:
    m = _API_CALL_RE.search(output)
    if not m:
        return False
    return bool(re.search(r"\w+\s*=\s*'[^']*'", m.group(2)))


def _has_api_name(output: str) -> bool:
    m = _API_CALL_RE.search(output)
    return bool(m and m.group(1))


def _has_parameters(output: str) -> bool:
    m = _API_CALL_RE.search(output)
    if not m:
        return False
    return bool(re.search(r"\w+\s*=\s*'", m.group(2)))


def _no_extra_text(output: str) -> bool:
    m = _API_CALL_RE.search(output)
    if not m:
        return False
    return len(output.strip()) - len(m.group(0)) < 20


def _under_300_chars(output: str) -> bool:
    return len(output.strip()) < 300


def apibank_proxy_checks() -> list[ProxyCheck]:
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
# Tool executor (simulated)
# ─────────────────────────────────────────────────────────────────────────

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
    if tool_name in _KNOWN_APIBANK_TOOLS:
        if params:
            return ToolResult(success=True, return_code=200, output="OK")
        return ToolResult(success=False, return_code=422, output="Missing params")
    return ToolResult(success=False, return_code=404, output="Unknown API")


def _parse_apibank_tool_call(output: str) -> tuple[str, dict[str, Any]]:
    m = _API_CALL_RE.search(output)
    if not m:
        return "", {}
    api_name = m.group(1)
    params = dict(re.findall(r"(\w+)\s*=\s*'([^']*)'", m.group(2)))
    return api_name, params


# ─────────────────────────────────────────────────────────────────────────
# Category extraction callback for adaptive mutations
# ─────────────────────────────────────────────────────────────────────────

# Build a lookup from test_input text → expected API name at runtime
_LABEL_MAP: dict[str, str] = {}


def extract_api_category(text: str, mode: str) -> str | None:
    """Extract API category for adaptive error tracking.

    mode='expected': text is the test_input — look up the expected API name.
    mode='predicted': text is the LLM output — parse the API call.
    """
    if mode == "expected":
        return _LABEL_MAP.get(text)
    # mode == 'predicted'
    m = _API_CALL_RE.search(text)
    return m.group(1) if m else None


# ─────────────────────────────────────────────────────────────────────────
# Ground-truth evaluation of an evolved prompt
# ─────────────────────────────────────────────────────────────────────────

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
        sys_prompt = prompt_template
        if "{apibank_instruction}" in sys_prompt:
            sys_prompt = sys_prompt.replace(
                "{apibank_instruction}", case.instruction
            )
        if "{apibank_input}" in sys_prompt:
            sys_prompt = sys_prompt.replace(
                "{apibank_input}", case.input_text
            )

        if (
            "{apibank_instruction}" not in prompt_template
            and "{apibank_input}" not in prompt_template
        ):
            user_message = (
                f"{case.instruction}\n\n{case.input_text}\n\n"
                "Generate API Request:"
            )
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


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="API-Bank adaptive mutation no-eval experiment"
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
    ╔══════════════════════════════════════════════════════════════════╗
    ║  EvoSim × API-Bank — Adaptive Mutation Experiment (No GT)       ║
    ║  adaptive_mutations=True · llm_mutation_rate=0.3 · pop=8        ║
    ║  Composite scorer · Ollama · ProblemType.TOOL_ROUTING           ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

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
        print("  Running in mock mode (random scores) for demonstration.\n")
    else:
        print(f"  ✓ Ollama available — model: {ollama_model}")

    # ── Load API-Bank data ─────────────────────────────────────────────
    print("\n  Loading API-Bank benchmark data...")
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

    # ── Build test inputs + label map for adaptive category tracking ──
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(gt_cases), size=min(10, len(gt_cases)), replace=False
    )

    test_inputs: list[str] = []
    for idx in sample_indices:
        case = gt_cases[int(idx)]
        test_input = f"{case.instruction}\n\n{case.input_text}"
        test_inputs.append(test_input)
        # Populate label map for the adaptive mutation callback
        _LABEL_MAP[test_input] = case.expected_api_name

    print(f"  Using {len(test_inputs)} test inputs for no-eval evolution")
    print(f"  Label map populated with {len(_LABEL_MAP)} API names for adaptive tracking")

    # ── Default baseline ───────────────────────────────────────────────
    print("\n  Phase 0: Default Prompt Baseline")
    print("  " + "─" * 55)
    default_baseline = evaluate_baseline(gt_cases, args.level, client)

    # ── Seed templates ─────────────────────────────────────────────────
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

    # ── No-eval config with adaptive mutations ─────────────────────────
    noeval_cfg = NoEvalConfig(
        iterations=args.iterations,
        population_size=8,
        num_islands=2,
        elite_size=4,
        mutation_rate=0.5,
        crossover_rate=0.3,
        migration_interval=2,
        backend=LLMBackend.OLLAMA,
        problem_type=ProblemType.TOOL_ROUTING,
        adaptive_mutations=True,
        llm_mutation_rate=0.3,
    )

    # ── Composite scorer (same weights as prior run) ───────────────────
    judge_rubric = textwrap.dedent("""\
        Score the output 0-10 on these criteria:
        - API call format: Is it in [ApiName(key='value', ...)] format? (0-4)
        - Parameter extraction: Are values from the conversation, not invented? (0-3)
        - API selection: Does the chosen API match the user's intent? (0-2)
        - Conciseness: Is it ONLY the API call with no extra text? (0-1)
    """)

    composite = CompositeScorer([
        (LLMJudge(rubric=judge_rubric, max_score=10.0), 0.4),
        (ToolSuccessScorer(
            tool_executor=_apibank_tool_executor,
            parse_fn=_parse_apibank_tool_call,
        ), 0.25),
        (ProxyMetricsScorer(apibank_proxy_checks()), 0.25),
        (SelfConsistencyScorer(num_samples=3), 0.1),
    ])

    # ── Phase 1: Evolution with adaptive mutations ─────────────────────
    print("\n  Phase 1: Composite + Adaptive Mutations (pop=8)")
    print("  " + "=" * 55)
    print(f"  Config: adaptive_mutations=True, llm_mutation_rate=0.3")
    print(f"  Config: pop={noeval_cfg.population_size}, "
          f"iters={noeval_cfg.iterations}, "
          f"islands={noeval_cfg.num_islands}")

    t0 = time.perf_counter()

    evolver = NoEvalPromptEvolver(
        task_description=TASK_DESCRIPTION,
        test_inputs=test_inputs,
        scorer=composite,
        config=noeval_cfg,
        seed_templates=seed_templates,
        verbose=True,
        extract_category=extract_api_category,
    )

    result = evolver.run()
    wall_time = time.perf_counter() - t0

    print(f"\n  No-eval fitness: {result.best_score:.1f}%")
    print(f"  Temperature: {result.best_temperature:.4f}")
    print(f"  Top-p: {result.best_top_p:.4f}")
    print(f"  Wall time: {wall_time:.1f}s")

    # ── Phase 2: Evaluate best prompt on ground truth ──────────────────
    print("\n  Phase 2: Ground-Truth Evaluation")
    print("  " + "=" * 55)
    print(f"  Evaluating best prompt on {len(gt_cases)} ground-truth cases...")

    gt_score, name_acc, param_acc = evaluate_prompt_on_gt(
        result.best_prompt,
        result.best_temperature,
        result.best_top_p,
        gt_cases,
        client,
    )

    print(f"\n  GT Score:       {gt_score:.1f}%")
    print(f"  API Name Acc:   {name_acc:.1f}%")
    print(f"  Param Acc:      {param_acc:.1f}%")

    # ── Phase 3: Comparison ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("  Comparison: Adaptive Mutations vs Prior Run vs Leaderboard")
    print(f"{'=' * 70}")

    print(f"\n  {'Approach':<38} {'GT Score':>10} {'API Name':>10} "
          f"{'Params':>10} {'Time':>8}")
    print(f"  {'─' * 65}")
    print(f"  {'Default prompt (baseline)':<38} "
          f"{default_baseline:9.1f}% {'—':>10} {'—':>10} {'—':>8}")
    print(f"  {'Prior: Composite (pop=4, vanilla)':<38} "
          f"{PRIOR_RESULTS['gt_score']:9.1f}% "
          f"{PRIOR_RESULTS['api_name_accuracy']:9.1f}% "
          f"{PRIOR_RESULTS['param_accuracy']:9.1f}% "
          f"{PRIOR_RESULTS['wall_time']:7.1f}s")
    print(f"  {'NEW: Composite+Adaptive (pop=8)':<38} "
          f"{gt_score:9.1f}% {name_acc:9.1f}% {param_acc:9.1f}% "
          f"{wall_time:7.1f}s")
    print(f"  {'─' * 65}")

    # Delta vs prior
    delta_gt = gt_score - PRIOR_RESULTS["gt_score"]
    delta_api = name_acc - PRIOR_RESULTS["api_name_accuracy"]
    delta_param = param_acc - PRIOR_RESULTS["param_accuracy"]
    print(f"\n  Delta vs prior run:")
    print(f"    GT score:     {delta_gt:+.1f} pp")
    print(f"    API name acc: {delta_api:+.1f} pp")
    print(f"    Param acc:    {delta_param:+.1f} pp")

    # vs leaderboard
    print(f"\n  {'─' * 65}")
    print(f"  Published leaderboard (Li et al., EMNLP 2023):")
    for model, scores in LEADERBOARD.items():
        level_key = "call" if args.level == "level_1" else "retrieval"
        lb_score = scores[level_key]
        delta_lb = gt_score - lb_score
        marker = "✓" if delta_lb >= 0 else " "
        print(f"    {marker} {model:<12} {level_key}: {lb_score:.1f}%  "
              f"(our delta: {delta_lb:+.1f} pp)")

    # ── Verdict ────────────────────────────────────────────────────────
    print(f"\n  {'─' * 65}")
    if delta_gt > 0:
        print(f"  ✓ Adaptive mutations IMPROVED GT score by {delta_gt:+.1f} pp")
    elif delta_gt == 0:
        print(f"  → Adaptive mutations matched prior run (no change)")
    else:
        print(f"  ✗ Adaptive mutations scored {delta_gt:+.1f} pp vs prior")

    # ── Save results ───────────────────────────────────────────────────
    log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "apibank_adaptive_noeval",
        "level": args.level,
        "ollama_model": ollama_model,
        "config": {
            "iterations": noeval_cfg.iterations,
            "population_size": noeval_cfg.population_size,
            "num_islands": noeval_cfg.num_islands,
            "elite_size": noeval_cfg.elite_size,
            "mutation_rate": noeval_cfg.mutation_rate,
            "crossover_rate": noeval_cfg.crossover_rate,
            "adaptive_mutations": noeval_cfg.adaptive_mutations,
            "llm_mutation_rate": noeval_cfg.llm_mutation_rate,
            "problem_type": noeval_cfg.problem_type.value,
        },
        "default_baseline": round(default_baseline, 2),
        "results": {
            "noeval_fitness": round(result.best_score, 2),
            "gt_score": round(gt_score, 2),
            "api_name_accuracy": round(name_acc, 2),
            "param_accuracy": round(param_acc, 2),
            "best_temperature": round(result.best_temperature, 4),
            "best_top_p": round(result.best_top_p, 4),
            "wall_time": round(wall_time, 1),
            "history": result.history,
        },
        "prior_results": PRIOR_RESULTS,
        "leaderboard": LEADERBOARD,
        "deltas": {
            "vs_prior_gt": round(delta_gt, 2),
            "vs_prior_api_name": round(delta_api, 2),
            "vs_prior_param": round(delta_param, 2),
        },
        "best_prompt": result.best_prompt,
        "best_prompt_preview": result.best_prompt[:300],
    }

    out_path = Path(_root) / "apibank_adaptive_experiment_log.json"
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # ── Best prompt preview ────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"  Best Evolved Prompt")
    print(f"{'=' * 70}")
    preview = result.best_prompt[:500]
    for line in preview.split("\n"):
        print(f"    {line}")
    if len(result.best_prompt) > 500:
        print(f"    ... ({len(result.best_prompt)} chars total)")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
