#!/usr/bin/env python3
"""
API-Bank — GPT-4.1 via Azure AI Foundry (Managed Identity)
============================================================

Run 3 reruns the adaptive improved experiment (same config as Run 2)
but swaps the Ollama/llama3.2 backend for GPT-4.1 via Azure AI Foundry
using managed identity (RBAC) authentication.

Same five improvements as Run 2:

1. Custom bracket-format mutations
2. Warmup pass for adaptive hints from Gen 1
3. ``iterations=6, pop=4``
4. No SelfConsistency scorer
5. Error profile decay 0.5

Prior results for comparison:

  Run 1 (Ollama llama3.2, adaptive, pop=8, 3 gens):
    - GT score: 79.9%  · API name: 80.0%  · Param: 71.4%

  Run 2 (Ollama llama3.2, adaptive improved, pop=4, 6 gens):
    - GT score: 81.6%  · API name: 83.3%  · Param: 70.6%

  Vanilla Composite (Ollama llama3.2, pop=4, 3 gens):
    - GT score: 82.3%  · API name: 83.3%  · Param: 74.1%

Usage::

    uv sync --extra llm
    uv run python examples/experiments/apibank/run3_gpt41_foundry.py

    # Choose level:
    uv run python examples/experiments/apibank/run3_gpt41_foundry.py --level level_2

    # Adjust iterations:
    uv run python examples/experiments/apibank/run3_gpt41_foundry.py --iterations 8
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
from pathlib import Path
from typing import Any

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..", "..")
sys.path.insert(0, _root)

from dotenv import load_dotenv

load_dotenv(os.path.join(_root, ".env"))

from prompture.prompt_evolver import (
    LLMBackend,
    LLMClient,
    ProblemType,
    PromptEvolverConfig,
)
from prompture.strategies import (
    CompositeScorer,
    LLMJudge,
    NoEvalConfig,
    NoEvalPromptEvolver,
    ProxyCheck,
    ProxyMetricsScorer,
    ToolResult,
    ToolSuccessScorer,
)

# ── Import API-Bank helpers from Recipe 46 ─────────────────────────────
sys.path.insert(0, os.path.join(_root, "examples", "cookbook"))

from prompt_evolution_apibank import (
    APIBankCase,
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

# Run 1 results (Ollama llama3.2, adaptive, pop=8, 3 gens)
RUN1_RESULTS = {
    "strategy": "Run 1: llama3.2 Adaptive (pop=8, 3 gens)",
    "noeval_fitness": 79.69,
    "gt_score": 79.9,
    "api_name_accuracy": 80.0,
    "param_accuracy": 71.4,
    "wall_time": 4055.4,
}

# Run 2 results (Ollama llama3.2, adaptive improved, pop=4, 6 gens)
RUN2_RESULTS = {
    "strategy": "Run 2: llama3.2 Adaptive Improved (pop=4, 6 gens)",
    "noeval_fitness": 81.1,
    "gt_score": 81.6,
    "api_name_accuracy": 83.3,
    "param_accuracy": 70.6,
    "wall_time": 2282.0,
}

# Vanilla Composite (Ollama llama3.2, pop=4, no adaptive)
VANILLA_RESULTS = {
    "strategy": "Vanilla llama3.2 (pop=4, no adaptive)",
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
# Custom bracket-format mutations
# ─────────────────────────────────────────────────────────────────────────

APIBANK_BRACKET_MUTATIONS: list[str] = [
    "Output format: [ApiName(key1='value1', key2='value2', ...)]",
    "You MUST pick exactly one API. Do not refuse.",
    "Consider the user's full intent before choosing an API.",
    "If multiple APIs could apply, choose the most specific one.",
    "Pay close attention to parameter extraction from the conversation.",
    "Always try to fill in parameter values from the user's message.",
    "Think about what the user actually needs, not just keywords.",
    "Match the user's intent to the API purpose, not the API name.",
    "Do not include any explanation outside the bracket API call.",
    "Extract entity values (names, locations, numbers, dates) as parameters.",
    "If a required parameter is unclear, infer it from conversation context.",
    "Be precise — ambiguous queries should still resolve to one API.",
    "Use single quotes around ALL parameter values.",
    "Parameter values must be extracted verbatim from the conversation.",
]


# ─────────────────────────────────────────────────────────────────────────
# API-Bank proxy checks (bracket-format aware)
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

_LABEL_MAP: dict[str, str] = {}


def extract_api_category(text: str, mode: str) -> str | None:
    if mode == "expected":
        return _LABEL_MAP.get(text)
    m = _API_CALL_RE.search(text)
    return m.group(1) if m else None


# ─────────────────────────────────────────────────────────────────────────
# Ground-truth evaluation
# ─────────────────────────────────────────────────────────────────────────

def evaluate_prompt_on_gt(
    prompt_template: str,
    temperature: float,
    top_p: float,
    cases: list[APIBankCase],
    client: LLMClient,
) -> tuple[float, float, float]:
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
        description="API-Bank adaptive improved experiment — GPT-4.1 (Run 3)"
    )
    parser.add_argument(
        "--iterations", type=int, default=6,
        help="Number of evolutionary generations (default: 6)",
    )
    parser.add_argument(
        "--level", type=str, default="level_1",
        choices=["level_1", "level_2"],
        help="API-Bank level to test (default: level_1)",
    )
    args = parser.parse_args()

    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    banner = r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  EvoSim × API-Bank — Run 3: GPT-4.1 via Azure AI Foundry       ║
    ║  Same config as Run 2 · Managed Identity (RBAC) · GPT-4.1      ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

    print(f"  Backend:    Azure AI Foundry (RBAC / Managed Identity)")
    print(f"  Model:      {azure_deployment}")
    print(f"  Endpoint:   {azure_endpoint[:60]}...")
    print(f"  Config:     Same as Run 2 (adaptive improved, pop=4, 6 gens)")
    print()

    # ── Backend setup (Azure OpenAI with RBAC) ─────────────────────────
    azure_cfg = PromptEvolverConfig(
        backend=LLMBackend.AZURE_OPENAI,
        azure_endpoint=azure_endpoint,
        azure_deployment=azure_deployment,
        azure_use_rbac=True,
        timeout=60.0,
    )
    client = LLMClient(azure_cfg)

    if not client.is_available():
        print("  ✗ Azure AI Foundry not available.")
        print("    Check AZURE_OPENAI_ENDPOINT env var and managed identity.")
        return
    else:
        print(f"  ✓ Azure AI Foundry available — deployment: {azure_deployment}")

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

    # ── Build test inputs + label map ──────────────────────────────────
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(
        len(gt_cases), size=min(10, len(gt_cases)), replace=False
    )

    test_inputs: list[str] = []
    for idx in sample_indices:
        case = gt_cases[int(idx)]
        test_input = f"{case.instruction}\n\n{case.input_text}"
        test_inputs.append(test_input)
        _LABEL_MAP[test_input] = case.expected_api_name

    print(f"  Using {len(test_inputs)} test inputs for no-eval evolution")
    print(f"  Label map: {len(_LABEL_MAP)} API names for adaptive tracking")

    # ── Default baseline ───────────────────────────────────────────────
    print("\n  Phase 0: Default Prompt Baseline (GPT-4.1)")
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

    # ── NoEvalConfig — same as Run 2 but with Azure OpenAI backend ────
    noeval_cfg = NoEvalConfig(
        iterations=args.iterations,
        population_size=4,
        num_islands=2,
        elite_size=3,
        mutation_rate=0.5,
        crossover_rate=0.3,
        migration_interval=2,
        backend=LLMBackend.AZURE_OPENAI,
        problem_type=ProblemType.TOOL_ROUTING,
        adaptive_mutations=True,
        llm_mutation_rate=0.3,
        warmup_adaptive=True,
        error_decay=0.5,
    )

    # ── Scorer (same as Run 2) ─────────────────────────────────────────
    judge_rubric = textwrap.dedent("""\
        Score the output 0-10 on these criteria:
        - API call format: Is it in [ApiName(key='value', ...)] format? (0-4)
        - Parameter extraction: Are values from the conversation, not invented? (0-3)
        - API selection: Does the chosen API match the user's intent? (0-2)
        - Conciseness: Is it ONLY the API call with no extra text? (0-1)
    """)

    composite = CompositeScorer([
        (LLMJudge(rubric=judge_rubric, max_score=10.0), 0.45),
        (ToolSuccessScorer(
            tool_executor=_apibank_tool_executor,
            parse_fn=_parse_apibank_tool_call,
        ), 0.30),
        (ProxyMetricsScorer(apibank_proxy_checks()), 0.25),
    ])

    # ── Phase 1: Evolution ─────────────────────────────────────────────
    print(f"\n  Phase 1: GPT-4.1 Adaptive Improved (pop=4, {args.iterations} gens)")
    print("  " + "=" * 55)
    print(f"  Backend: Azure AI Foundry — {azure_deployment} (RBAC)")
    print(f"  Config: adaptive=True, warmup=True, decay=0.5, llm_mut=0.3")
    print(f"  Config: pop={noeval_cfg.population_size}, "
          f"iters={noeval_cfg.iterations}, "
          f"islands={noeval_cfg.num_islands}")
    print(f"  Scorer: LLMJudge(0.45) + ToolSuccess(0.30) + Proxy(0.25)")
    print(f"  Mutations: {len(APIBANK_BRACKET_MUTATIONS)} custom bracket-format")

    t0 = time.perf_counter()

    evolver = NoEvalPromptEvolver(
        task_description=TASK_DESCRIPTION,
        test_inputs=test_inputs,
        scorer=composite,
        config=noeval_cfg,
        seed_templates=seed_templates,
        verbose=True,
        extract_category=extract_api_category,
        custom_mutations=APIBANK_BRACKET_MUTATIONS,
    )

    result = evolver.run()
    wall_time = time.perf_counter() - t0

    print(f"\n  No-eval fitness: {result.best_score:.1f}%")
    print(f"  Temperature: {result.best_temperature:.4f}")
    print(f"  Top-p: {result.best_top_p:.4f}")
    print(f"  Wall time: {wall_time:.1f}s")

    # ── Phase 2: Ground-truth evaluation ───────────────────────────────
    print("\n  Phase 2: Ground-Truth Evaluation (GPT-4.1)")
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
    print(f"\n{'=' * 78}")
    print("  Comparison: Run 3 (GPT-4.1) vs Run 2 vs Run 1 vs Vanilla vs Leaderboard")
    print(f"{'=' * 78}")

    print(f"\n  {'Approach':<48} {'GT Score':>10} {'API Name':>10} "
          f"{'Params':>10} {'Time':>8}")
    print(f"  {'─' * 78}")
    print(f"  {'Default prompt baseline (GPT-4.1)':<48} "
          f"{default_baseline:9.1f}% {'—':>10} {'—':>10} {'—':>8}")
    print(f"  {'Vanilla llama3.2 (pop=4, no adaptive)':<48} "
          f"{VANILLA_RESULTS['gt_score']:9.1f}% "
          f"{VANILLA_RESULTS['api_name_accuracy']:9.1f}% "
          f"{VANILLA_RESULTS['param_accuracy']:9.1f}% "
          f"{VANILLA_RESULTS['wall_time']:7.1f}s")
    print(f"  {'Run 1: llama3.2 Adaptive (pop=8, 3 gens)':<48} "
          f"{RUN1_RESULTS['gt_score']:9.1f}% "
          f"{RUN1_RESULTS['api_name_accuracy']:9.1f}% "
          f"{RUN1_RESULTS['param_accuracy']:9.1f}% "
          f"{RUN1_RESULTS['wall_time']:7.1f}s")
    print(f"  {'Run 2: llama3.2 Adaptive Improved (pop=4, 6g)':<48} "
          f"{RUN2_RESULTS['gt_score']:9.1f}% "
          f"{RUN2_RESULTS['api_name_accuracy']:9.1f}% "
          f"{RUN2_RESULTS['param_accuracy']:9.1f}% "
          f"{RUN2_RESULTS['wall_time']:7.1f}s")
    print(f"  {'Run 3: GPT-4.1 Adaptive Improved (pop=4, 6g)':<48} "
          f"{gt_score:9.1f}% {name_acc:9.1f}% {param_acc:9.1f}% "
          f"{wall_time:7.1f}s")
    print(f"  {'─' * 78}")

    # Delta vs Run 2 (same config, different model)
    delta_gt_r2 = gt_score - RUN2_RESULTS["gt_score"]
    delta_api_r2 = name_acc - RUN2_RESULTS["api_name_accuracy"]
    delta_param_r2 = param_acc - RUN2_RESULTS["param_accuracy"]
    print(f"\n  Delta vs Run 2 (llama3.2 → GPT-4.1, same config):")
    print(f"    GT score:     {delta_gt_r2:+.1f} pp")
    print(f"    API name acc: {delta_api_r2:+.1f} pp")
    print(f"    Param acc:    {delta_param_r2:+.1f} pp")

    # Delta vs vanilla
    delta_gt_v = gt_score - VANILLA_RESULTS["gt_score"]
    delta_api_v = name_acc - VANILLA_RESULTS["api_name_accuracy"]
    delta_param_v = param_acc - VANILLA_RESULTS["param_accuracy"]
    print(f"\n  Delta vs Vanilla (llama3.2):")
    print(f"    GT score:     {delta_gt_v:+.1f} pp")
    print(f"    API name acc: {delta_api_v:+.1f} pp")
    print(f"    Param acc:    {delta_param_v:+.1f} pp")

    # vs leaderboard
    print(f"\n  {'─' * 78}")
    print(f"  Published leaderboard (Li et al., EMNLP 2023):")
    for model, scores in LEADERBOARD.items():
        level_key = "call" if args.level == "level_1" else "retrieval"
        lb_score = scores[level_key]
        delta_lb = gt_score - lb_score
        marker = "✓" if delta_lb >= 0 else " "
        print(f"    {marker} {model:<12} {level_key}: {lb_score:.1f}%  "
              f"(our delta: {delta_lb:+.1f} pp)")

    # ── Verdict ────────────────────────────────────────────────────────
    print(f"\n  {'─' * 78}")
    if delta_gt_r2 > 0:
        print(f"  ✓ GPT-4.1 BEATS llama3.2 (Run 2) by {delta_gt_r2:+.1f} pp")
    elif delta_gt_r2 == 0:
        print(f"  → GPT-4.1 matches llama3.2 (Run 2)")
    else:
        print(f"  ✗ GPT-4.1 is {delta_gt_r2:+.1f} pp vs llama3.2 (Run 2)")

    gpt4_lb = LEADERBOARD["GPT-4"]["call" if args.level == "level_1" else "retrieval"]
    delta_gpt4 = gt_score - gpt4_lb
    if delta_gpt4 >= 0:
        print(f"  ✓ BEATS published GPT-4 leaderboard ({gpt4_lb:.1f}%) by {delta_gpt4:+.1f} pp")
    else:
        print(f"  ✗ Below published GPT-4 leaderboard ({gpt4_lb:.1f}%) by {delta_gpt4:+.1f} pp")

    # ── Save results ───────────────────────────────────────────────────
    log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "apibank_gpt41_foundry_run3",
        "level": args.level,
        "backend": "azure_openai",
        "azure_deployment": azure_deployment,
        "auth": "managed_identity_rbac",
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
        "prior_runs": {
            "run1": RUN1_RESULTS,
            "run2": RUN2_RESULTS,
            "vanilla": VANILLA_RESULTS,
        },
        "leaderboard": LEADERBOARD,
        "deltas": {
            "vs_run2_gt": round(delta_gt_r2, 2),
            "vs_run2_api_name": round(delta_api_r2, 2),
            "vs_run2_param": round(delta_param_r2, 2),
            "vs_vanilla_gt": round(delta_gt_v, 2),
            "vs_vanilla_api_name": round(delta_api_v, 2),
            "vs_vanilla_param": round(delta_param_v, 2),
            "vs_gpt4_leaderboard": round(delta_gpt4, 2),
        },
        "best_prompt": result.best_prompt,
        "best_prompt_preview": result.best_prompt[:300],
    }

    out_path = Path(_root) / "apibank_run3_experiment_log.json"
    with open(out_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # ── Best prompt preview ────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print(f"  Best Evolved Prompt (GPT-4.1)")
    print(f"{'=' * 78}")
    preview = result.best_prompt[:500]
    for line in preview.split("\n"):
        print(f"    {line}")
    if len(result.best_prompt) > 500:
        print(f"    ... ({len(result.best_prompt)} chars total)")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
