#!/usr/bin/env python3
"""
API-Bank — GPT-4.1 Holdout Evaluation (Run 4)
===============================================

Evaluates the **winning evolved prompt from Run 3** (GPT-4.1 via Azure
AI Foundry) on a *holdout subset* of the API-Bank dataset that was NOT
exposed during the evolution process.

Run 3 evolved on 30 cases sampled with ``rng(42)`` from the full
level-1 dataset.  This script:

1. Loads the full API-Bank level-1 dataset.
2. Reproduces the exact 30-case evolution sample (same seed).
3. Builds the holdout set: every case NOT in that sample.
4. Evaluates the **evolved prompt** (with its tuned temperature/top_p)
   and the **default prompt** (baseline) on the holdout set.
5. Saves a detailed log to ``logs/apibank_run4_holdout_log.json``.

This determines whether the evolved prompt *generalises* beyond the
cases the EA saw.

Prior Run 3 results (on evolution set):
  GT score: 96.79 %  ·  API name: 96.67 %  ·  Param: 95.31 %

Usage::

    uv sync --extra llm
    uv run python examples/experiments/apibank/run4_gpt41_holdout.py
"""
from __future__ import annotations

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

from prompture.prompt_evolver import (
    LLMBackend,
    LLMClient,
    PromptEvolverConfig,
)

# ── Import API-Bank helpers from cookbook ────────────────────────────────
sys.path.insert(0, os.path.join(_root, "examples", "cookbook"))

from prompt_evolution_apibank import (
    APIBankCase,
    load_apibank_dataset,
    score_apibank_case,
)


# ─────────────────────────────────────────────────────────────────────────
# The winning evolved prompt from Run 3
# ─────────────────────────────────────────────────────────────────────────

EVOLVED_PROMPT = textwrap.dedent("""\
    You are an expert API orchestration agent.

    ## Task
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

    ## Instructions
    Think step by step: 1) identify API, 2) extract params, 3) output call.
""").strip()

EVOLVED_TEMPERATURE = 0.9697
EVOLVED_TOP_P = 0.9243

# ─────────────────────────────────────────────────────────────────────────
# Default (unevolved) baseline prompt
# ─────────────────────────────────────────────────────────────────────────

DEFAULT_PROMPT = textwrap.dedent("""\
    You are a helpful assistant. Given the conversation history and API \
    descriptions, generate the correct API request in the format \
    [ApiName(key1='value1', ...)].

    {apibank_instruction}

    {apibank_input}
""").strip()

DEFAULT_TEMPERATURE = 0.1
DEFAULT_TOP_P = 0.95


# ─────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────

def evaluate_prompt(
    prompt_template: str,
    temperature: float,
    top_p: float,
    cases: list[APIBankCase],
    client: LLMClient,
    label: str = "",
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate a prompt template on a list of API-Bank cases.

    Returns a dict with gt_score, api_name_accuracy, param_accuracy,
    per-case details, and error breakdown.
    """
    total = 0.0
    name_hits = 0
    param_total = 0.0
    per_case: list[dict[str, Any]] = []
    errors: dict[str, int] = {}

    for i, case in enumerate(cases):
        # Build the system prompt — handle both templated and plain prompts
        if "{apibank_instruction}" in prompt_template:
            sys_prompt = prompt_template.replace(
                "{apibank_instruction}", case.instruction
            ).replace(
                "{apibank_input}", case.input_text
            )
            user_message = "Generate API Request:"
        else:
            sys_prompt = prompt_template
            user_message = (
                f"{case.instruction}\n\n{case.input_text}\n\n"
                "Generate API Request:"
            )

        response = client.complete(
            system_prompt=sys_prompt,
            user_message=user_message,
            temperature=temperature,
            top_p=top_p,
        )

        if response is None:
            per_case.append({
                "case_id": case.case_id,
                "score": 0.0,
                "error": "NO_RESPONSE",
            })
            errors["NO_RESPONSE"] = errors.get("NO_RESPONSE", 0) + 1
            continue

        score, detail = score_apibank_case(response, case)
        total += score
        if detail["api_name_match"]:
            name_hits += 1
        param_total += detail["param_score"]

        err = detail.get("error")
        if err:
            errors[err] = errors.get(err, 0) + 1

        per_case.append({
            "case_id": case.case_id,
            "score": round(score, 4),
            "api_name_match": detail["api_name_match"],
            "param_score": round(detail["param_score"], 4),
            "format_ok": detail["format_ok"],
            "error": err,
            "response_preview": response[:200],
        })

        if verbose and (i + 1) % 10 == 0:
            running_pct = total / (i + 1) * 100.0
            print(f"    [{label}] {i + 1}/{len(cases)}  running score: {running_pct:.1f}%")

    n = len(cases) if cases else 1
    gt_score = total / n * 100.0
    api_name_accuracy = name_hits / n * 100.0
    param_accuracy = param_total / n * 100.0

    return {
        "gt_score": round(gt_score, 2),
        "api_name_accuracy": round(api_name_accuracy, 2),
        "param_accuracy": round(param_accuracy, 2),
        "n_cases": len(cases),
        "errors": errors,
        "per_case": per_case,
    }


# ─────────────────────────────────────────────────────────────────────────
# Holdout split
# ─────────────────────────────────────────────────────────────────────────

def build_holdout_split(
    all_cases: list[APIBankCase],
    evolution_sample_size: int = 30,
    seed: int = 42,
) -> tuple[list[APIBankCase], list[APIBankCase]]:
    """Reproduce the evolution sample and return (evolution_set, holdout_set).

    Uses the same RNG seed and sampling logic as load_apibank_dataset to
    identify which cases the EA saw, then returns all remaining cases as
    the holdout.
    """
    # load_apibank_dataset samples with rng(42) when len > max_per_level.
    # We replicate that logic here to get the exact same indices.
    rng = np.random.default_rng(seed)

    if len(all_cases) > evolution_sample_size:
        evo_indices = set(
            int(i) for i in rng.choice(
                len(all_cases), size=evolution_sample_size, replace=False
            )
        )
    else:
        evo_indices = set(range(len(all_cases)))

    evo_cases = [all_cases[i] for i in sorted(evo_indices)]
    holdout_cases = [
        all_cases[i] for i in range(len(all_cases))
        if i not in evo_indices
    ]

    return evo_cases, holdout_cases


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    banner = r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  Prompture × API-Bank — Run 4: GPT-4.1 Holdout Evaluation      ║
    ║  Evolved prompt vs Default prompt on unseen holdout cases       ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    print(f"  Backend:    Azure AI Foundry (RBAC / Managed Identity)")
    print(f"  Model:      {azure_deployment}")
    print(f"  Endpoint:   {azure_endpoint[:60]}...")
    print()

    # ── Backend setup ──────────────────────────────────────────────────
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
    print(f"  ✓ Azure AI Foundry available — deployment: {azure_deployment}")

    # ── Load full API-Bank dataset (no subsampling) ────────────────────
    print("\n  Loading full API-Bank level_1 dataset (no subsampling)...")
    try:
        # Load ALL cases — set max_per_level very high
        by_level = load_apibank_dataset(
            max_per_level=10_000, levels=["level_1"]
        )
    except Exception as exc:
        print(f"  ✗ {exc}")
        return

    all_cases = by_level.get("level_1", [])
    if not all_cases:
        print("  No cases loaded for level_1")
        return

    print(f"  Total level_1 cases: {len(all_cases)}")

    # ── Split into evolution set and holdout ────────────────────────────
    evo_cases, holdout_cases = build_holdout_split(
        all_cases, evolution_sample_size=30, seed=42
    )
    print(f"  Evolution set: {len(evo_cases)} cases (seen during Run 3)")
    print(f"  Holdout set:   {len(holdout_cases)} cases (never seen)")

    if not holdout_cases:
        print("  ✗ No holdout cases — dataset too small for a split.")
        return

    # ── Phase 1: Evaluate evolved prompt on holdout ────────────────────
    print(f"\n  Phase 1: Evolved Prompt on Holdout ({len(holdout_cases)} cases)")
    print("  " + "─" * 55)
    print(f"  Temperature: {EVOLVED_TEMPERATURE:.4f}  |  Top-p: {EVOLVED_TOP_P:.4f}")

    t0 = time.perf_counter()
    evolved_results = evaluate_prompt(
        EVOLVED_PROMPT, EVOLVED_TEMPERATURE, EVOLVED_TOP_P,
        holdout_cases, client, label="Evolved",
    )
    evolved_time = time.perf_counter() - t0

    print(f"\n  Evolved prompt (holdout):")
    print(f"    GT Score:          {evolved_results['gt_score']:.2f}%")
    print(f"    API Name Accuracy: {evolved_results['api_name_accuracy']:.2f}%")
    print(f"    Param Accuracy:    {evolved_results['param_accuracy']:.2f}%")
    print(f"    Wall time:         {evolved_time:.1f}s")
    if evolved_results["errors"]:
        print(f"    Error breakdown:   {evolved_results['errors']}")

    # ── Phase 2: Evaluate default prompt on holdout ────────────────────
    print(f"\n  Phase 2: Default Prompt on Holdout ({len(holdout_cases)} cases)")
    print("  " + "─" * 55)

    t1 = time.perf_counter()
    default_results = evaluate_prompt(
        DEFAULT_PROMPT, DEFAULT_TEMPERATURE, DEFAULT_TOP_P,
        holdout_cases, client, label="Default",
    )
    default_time = time.perf_counter() - t1

    print(f"\n  Default prompt (holdout):")
    print(f"    GT Score:          {default_results['gt_score']:.2f}%")
    print(f"    API Name Accuracy: {default_results['api_name_accuracy']:.2f}%")
    print(f"    Param Accuracy:    {default_results['param_accuracy']:.2f}%")
    print(f"    Wall time:         {default_time:.1f}s")
    if default_results["errors"]:
        print(f"    Error breakdown:   {default_results['errors']}")

    # ── Phase 3: Summary comparison ────────────────────────────────────
    gt_delta = evolved_results["gt_score"] - default_results["gt_score"]
    name_delta = evolved_results["api_name_accuracy"] - default_results["api_name_accuracy"]
    param_delta = evolved_results["param_accuracy"] - default_results["param_accuracy"]

    print(f"\n{'=' * 65}")
    print("  HOLDOUT RESULTS SUMMARY")
    print(f"{'=' * 65}")
    print(f"  {'Metric':<22} {'Evolved':>10} {'Default':>10} {'Delta':>10}")
    print(f"  {'─' * 60}")
    print(f"  {'GT Score':<22} {evolved_results['gt_score']:>9.2f}% {default_results['gt_score']:>9.2f}% {gt_delta:>+9.2f}%")
    print(f"  {'API Name Accuracy':<22} {evolved_results['api_name_accuracy']:>9.2f}% {default_results['api_name_accuracy']:>9.2f}% {name_delta:>+9.2f}%")
    print(f"  {'Param Accuracy':<22} {evolved_results['param_accuracy']:>9.2f}% {default_results['param_accuracy']:>9.2f}% {param_delta:>+9.2f}%")
    print(f"  {'─' * 60}")
    print(f"  Holdout cases: {len(holdout_cases)}  |  Evolution cases: {len(evo_cases)}")
    print()

    if gt_delta > 0:
        print("  ✓ Evolved prompt GENERALISES — outperforms default on unseen data.")
    elif gt_delta == 0:
        print("  ≈ Evolved prompt ties with default on holdout.")
    else:
        print("  ✗ Evolved prompt underperforms default on holdout (possible overfit).")

    # ── Run 3 comparison (evolution set results) ───────────────────────
    run3_gt = 96.79
    print(f"\n  Run 3 (evolution set): GT {run3_gt:.2f}%")
    print(f"  Run 4 (holdout set):   GT {evolved_results['gt_score']:.2f}%")
    gen_gap = run3_gt - evolved_results["gt_score"]
    print(f"  Generalisation gap:    {gen_gap:+.2f}%")

    # ── Save log ───────────────────────────────────────────────────────
    log = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "experiment": "API-Bank Run 4: GPT-4.1 Holdout Evaluation",
        "level": "level_1",
        "backend": "azure_openai",
        "azure_deployment": azure_deployment,
        "auth": "managed_identity_rbac",
        "dataset": {
            "total_cases": len(all_cases),
            "evolution_cases": len(evo_cases),
            "holdout_cases": len(holdout_cases),
            "evolution_seed": 42,
            "evolution_sample_size": 30,
        },
        "evolved_prompt": {
            "text": EVOLVED_PROMPT,
            "temperature": EVOLVED_TEMPERATURE,
            "top_p": EVOLVED_TOP_P,
            "source": "Run 3 best prompt",
        },
        "holdout_results": {
            "evolved": {
                "gt_score": evolved_results["gt_score"],
                "api_name_accuracy": evolved_results["api_name_accuracy"],
                "param_accuracy": evolved_results["param_accuracy"],
                "wall_time": round(evolved_time, 1),
                "errors": evolved_results["errors"],
            },
            "default": {
                "gt_score": default_results["gt_score"],
                "api_name_accuracy": default_results["api_name_accuracy"],
                "param_accuracy": default_results["param_accuracy"],
                "wall_time": round(default_time, 1),
                "errors": default_results["errors"],
            },
            "deltas": {
                "gt_score": round(gt_delta, 2),
                "api_name_accuracy": round(name_delta, 2),
                "param_accuracy": round(param_delta, 2),
            },
        },
        "generalisation": {
            "run3_evolution_gt": run3_gt,
            "run4_holdout_gt": evolved_results["gt_score"],
            "gap": round(gen_gap, 2),
        },
        "per_case_evolved": evolved_results["per_case"],
        "per_case_default": default_results["per_case"],
    }

    log_path = Path(_root) / "logs" / "apibank_run4_holdout_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Log saved → {log_path}")


if __name__ == "__main__":
    main()
