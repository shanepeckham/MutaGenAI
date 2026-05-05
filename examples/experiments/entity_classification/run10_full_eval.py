#!/usr/bin/env python3
"""
Run 10 — Full-scale evaluation of the winning evolved prompt (679 val + 1000 test).

Compares against the STATIC baseline results already saved in
full_eval_1000_results.json (not re-run).
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..", "..")
sys.path.insert(0, _root)

from dotenv import load_dotenv
load_dotenv(os.path.join(_root, ".env"))

from MutaGenAI.prompt_evolver import LLMBackend, LLMClient, PromptEvolverConfig

from examples.experiments.entity_classification import (
    ENTITY_TYPES,
    evaluate_prompt,
    extract_entity,
)

# ── Entity types ─────────────────────────────────────────────────
EVOLVED_PROMPT = """\
Classify each input using exactly one word from: Agent, Task, Tool, Input, Output, Human.

Agent: An autonomous entity or software that initiates actions or makes decisions independently.
Tool: A named system, program, service, or function that performs actions but does not act autonomously; often described by its capabilities.
Task: An explicit request, command, instruction, or activity to be performed; often starts with action verbs or describes a need.
Input: Data, parameters, or categories provided for processing; raw information or content submitted to a system, agent, or tool.
Output: The response, result, or information produced after executing a task or processing input.
Human: Any reference to a person, their comments, requests, or opinions; content clearly authored or attributed to a human.

Classify based on the primary role or purpose in context. If the input describes a functionality, request, or action needed, use Task. If it refers to a named system or function, use Tool. If it is a person's comment or request, use Human."""

EVOLVED_TEMPERATURE = 0.5491
EVOLVED_TOP_P = 0.7

VAL_SIZE = 679   # full validation set
TEST_SIZE = 1000  # same as prior full eval

# ── Prior static results (from full_eval_1000_results.json) ───────────────
PRIOR_RESULTS_PATH = Path(_root) / "full_eval_1000_results.json"

# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("Run 10 Full Eval — Evolved Prompt (679 val + 1000 test)")
    print("  (Static baseline loaded from prior run — not re-evaluated)")
    print("=" * 70)

    # ── Load prior static results ─────────────────────────────────────
    if PRIOR_RESULTS_PATH.exists():
        prior = json.loads(PRIOR_RESULTS_PATH.read_text())
        static_val_acc = prior["static"]["validation_accuracy"]
        static_test_acc = prior["static"]["test_accuracy"]
        static_per_class_val = prior["static"]["per_class_validation"]
        static_per_class_test = prior["static"]["per_class_test"]
        static_counts_val = prior["static"].get("per_class_counts_val", {})
        static_counts_test = prior["static"].get("per_class_counts_test", {})
        print(f"\n  Loaded prior static results from {PRIOR_RESULTS_PATH.name}")
        print(f"    Static val:  {static_val_acc:.1f}%")
        print(f"    Static test: {static_test_acc:.1f}%")
    else:
        print(f"\n  ⚠  No prior results found at {PRIOR_RESULTS_PATH}")
        print("     Run full_eval_1000.py first for static baseline.")
        static_val_acc = None
        static_test_acc = None
        static_per_class_val = {}
        static_per_class_test = {}
        static_counts_val = {}
        static_counts_test = {}

    # ── Load dataset ──────────────────────────────────────────────────
    try:
        from datasets import load_dataset as _hf_load
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print("\nLoading dataset …")
    ds = _hf_load("holistic-ai/entity-classification-agentic-ai")
    val_data = [{"content": r["content"], "expected_entity": r["expected_entity"]} for r in ds["validation"]]
    test_data = [{"content": r["content"], "expected_entity": r["expected_entity"]} for r in ds["test"]]
    print(f"  validation: {len(val_data)} total samples")
    print(f"  test:       {len(test_data)} total samples")

    rng = np.random.default_rng(42)

    val_size = min(VAL_SIZE, len(val_data))
    test_size = min(TEST_SIZE, len(test_data))

    # Use same seed=42 as prior run to get identical subsets
    val_indices = rng.choice(len(val_data), size=val_size, replace=False)
    val_subset = [val_data[int(i)] for i in val_indices]

    test_indices = rng.choice(len(test_data), size=test_size, replace=False)
    test_subset = [test_data[int(i)] for i in test_indices]

    print(f"\n  Eval validation subset: {len(val_subset)} samples")
    print(f"  Eval test subset:       {len(test_subset)} samples")

    client = LLMClient(PromptEvolverConfig(backend=LLMBackend.AZURE_OPENAI, max_tokens=10))

    # ── Evolved prompt (Run 10 winner) ────────────────────────────────
    print("\n" + "─" * 70)
    print("Evaluating Run 10 EVOLVED prompt (adaptive + LLM mutations)")
    print("─" * 70)
    print(f"  Temperature: {EVOLVED_TEMPERATURE}")
    print(f"  Top-p:       {EVOLVED_TOP_P}")

    t0 = time.time()
    evolved_val = evaluate_prompt(client, EVOLVED_PROMPT, val_subset, EVOLVED_TEMPERATURE, EVOLVED_TOP_P, "evolved-val")
    evolved_test = evaluate_prompt(client, EVOLVED_PROMPT, test_subset, EVOLVED_TEMPERATURE, EVOLVED_TOP_P, "evolved-test")
    evolved_time = time.time() - t0
    print(f"\n  Evolved wall time: {evolved_time:.0f}s")

    # ── Results ───────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("RESULTS — Run 10 Evolved vs Static Baseline")
    print(f"  ({val_size} val + {test_size} test samples, seed=42)")
    print("═" * 70)

    print(f"\n  Run 10 Evolved:")
    print(f"    Validation: {evolved_val['accuracy']:.1f}%  ({evolved_val['correct']}/{evolved_val['total']})")
    print(f"    Test:       {evolved_test['accuracy']:.1f}%  ({evolved_test['correct']}/{evolved_test['total']})")
    print(f"    Per-class (validation):")
    for e in ENTITY_TYPES:
        c = evolved_val["per_class_counts"][e]
        print(f"      {e:8s}: {evolved_val['per_class_acc'][e]:5.1f}%  ({c['correct']}/{c['total']})")
    print(f"    Per-class (test):")
    for e in ENTITY_TYPES:
        c = evolved_test["per_class_counts"][e]
        print(f"      {e:8s}: {evolved_test['per_class_acc'][e]:5.1f}%  ({c['correct']}/{c['total']})")

    # ── Comparison table ──────────────────────────────────────────────
    if static_val_acc is not None:
        val_delta = evolved_val["accuracy"] - static_val_acc
        test_delta = evolved_test["accuracy"] - static_test_acc

        print(f"\n  {'':30s} {'Validation':>12s} {'Test':>12s}")
        print(f"  {'─' * 55}")
        print(f"  {'Static (prior run)':30s} {static_val_acc:11.1f}% {static_test_acc:11.1f}%")
        print(f"  {'Run 10 Evolved (adaptive)':30s} {evolved_val['accuracy']:11.1f}% {evolved_test['accuracy']:11.1f}%")
        print(f"  {'─' * 55}")
        print(f"  {'Δ (evolved − static)':30s} {val_delta:+10.1f}% {test_delta:+10.1f}%")

        winner = "EVOLVED" if test_delta > 0 else ("STATIC" if test_delta < 0 else "TIE")
        print(f"\n  Winner (by test): {winner}")

        # Per-class comparison
        print(f"\n  Per-class comparison (validation):")
        print(f"  {'Class':10s} {'Static':>8s} {'Evolved':>8s} {'Delta':>8s}")
        print(f"  {'─' * 36}")
        for e in ENTITY_TYPES:
            s = static_per_class_val.get(e, 0.0)
            ev = evolved_val["per_class_acc"][e]
            print(f"  {e:10s} {s:7.1f}% {ev:7.1f}% {ev - s:+7.1f}%")

        print(f"\n  Per-class comparison (test):")
        print(f"  {'Class':10s} {'Static':>8s} {'Evolved':>8s} {'Delta':>8s}")
        print(f"  {'─' * 36}")
        for e in ENTITY_TYPES:
            s = static_per_class_test.get(e, 0.0)
            ev = evolved_test["per_class_acc"][e]
            print(f"  {e:10s} {s:7.1f}% {ev:7.1f}% {ev - s:+7.1f}%")

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "experiment": "Run 10 full-scale evaluation (evolved only)",
        "model": "gpt-4.1 (Azure OpenAI)",
        "val_size": val_size,
        "test_size": test_size,
        "evolved_prompt": EVOLVED_PROMPT,
        "evolved_temperature": EVOLVED_TEMPERATURE,
        "evolved_top_p": EVOLVED_TOP_P,
        "evolved": {
            "validation_accuracy": evolved_val["accuracy"],
            "test_accuracy": evolved_test["accuracy"],
            "per_class_validation": evolved_val["per_class_acc"],
            "per_class_test": evolved_test["per_class_acc"],
            "per_class_counts_val": evolved_val["per_class_counts"],
            "per_class_counts_test": evolved_test["per_class_counts"],
            "wall_time_s": evolved_time,
        },
        "static_baseline": {
            "validation_accuracy": static_val_acc,
            "test_accuracy": static_test_acc,
            "per_class_validation": static_per_class_val,
            "per_class_test": static_per_class_test,
            "source": str(PRIOR_RESULTS_PATH.name),
        },
        "comparison": {
            "val_delta": evolved_val["accuracy"] - static_val_acc if static_val_acc else None,
            "test_delta": evolved_test["accuracy"] - static_test_acc if static_test_acc else None,
        },
    }

    out_path = Path(_root) / "run10_full_eval_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print("\n✓ Run 10 full evaluation complete.")


if __name__ == "__main__":
    main()
