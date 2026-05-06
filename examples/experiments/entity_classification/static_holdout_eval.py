#!/usr/bin/env python3
"""
Static prompt on held-out data — same 479 val + 500 test samples
that were excluded from the Run 10 evolution, for fair comparison.
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

# ── Static prompt (same as full_eval_1000.py) ─────────────────────────────
STATIC_PROMPT = """\
You are an expert classifier for agentic AI systems.
Your task is to classify the given text into exactly ONE of the following entity types:
- Agent: An autonomous system or actor that performs actions (e.g., AI assistant, bot, system component)
- Task: A goal, instruction, or objective to be completed
- Tool: A function, API, software, or resource used to perform a task
- Input: Data or information provided to a system
- Output: Data or information produced by a system
- Human: A person or human actor involved in the process
---
Instructions:
- Choose the SINGLE best matching entity type
- Do NOT explain your answer
- Do NOT output anything except the label
- If uncertain, choose the closest match based on intent"""

STATIC_TEMPERATURE = 0.7
STATIC_TOP_P = 0.95


def main():
    print("=" * 70)
    print("STATIC prompt on held-out data (479 val + 500 test)")
    print("  Same samples excluded from Run 10 evolution")
    print("=" * 70)

    from datasets import load_dataset as _hf_load

    print("\nLoading dataset …")
    ds = _hf_load("holistic-ai/entity-classification-agentic-ai")
    val_data = [{"content": r["content"], "expected_entity": r["expected_entity"]} for r in ds["validation"]]
    test_data = [{"content": r["content"], "expected_entity": r["expected_entity"]} for r in ds["test"]]
    print(f"  validation total: {len(val_data)}")
    print(f"  test total:       {len(test_data)}")

    # ── Reproduce evolution RNG to find excluded indices ──────────────
    evo_rng = np.random.default_rng(42)
    evo_val_indices = set(evo_rng.choice(len(val_data), size=200, replace=False).tolist())
    evo_test_indices = set(evo_rng.choice(len(test_data), size=200, replace=False).tolist())

    remaining_val = [i for i in range(len(val_data)) if i not in evo_val_indices]
    remaining_test = [i for i in range(len(test_data)) if i not in evo_test_indices]

    # Same holdout seed=99 as the evolved holdout eval
    holdout_rng = np.random.default_rng(99)
    holdout_val_idx = holdout_rng.choice(remaining_val, size=min(500, len(remaining_val)), replace=False)
    holdout_test_idx = holdout_rng.choice(remaining_test, size=min(500, len(remaining_test)), replace=False)

    val_subset = [val_data[int(i)] for i in holdout_val_idx]
    test_subset = [test_data[int(i)] for i in holdout_test_idx]
    print(f"  Holdout: {len(val_subset)} val + {len(test_subset)} test")

    overlap_val = evo_val_indices & set(holdout_val_idx.tolist())
    overlap_test = evo_test_indices & set(holdout_test_idx.tolist())
    assert len(overlap_val) == 0
    assert len(overlap_test) == 0
    print("  ✓ Zero overlap with evolution samples confirmed")

    # ── Evaluate static prompt ────────────────────────────────────────
    client = LLMClient(PromptEvolverConfig(backend=LLMBackend.AZURE_OPENAI, max_tokens=10))

    print("\n" + "─" * 70)
    print("Evaluating STATIC prompt on held-out samples")
    print("─" * 70)
    print(f"  Temperature: {STATIC_TEMPERATURE}")
    print(f"  Top-p:       {STATIC_TOP_P}")

    t0 = time.time()
    static_val = evaluate_prompt(client, STATIC_PROMPT, val_subset, STATIC_TEMPERATURE, STATIC_TOP_P, "static-val")
    static_test = evaluate_prompt(client, STATIC_PROMPT, test_subset, STATIC_TEMPERATURE, STATIC_TOP_P, "static-test")
    elapsed = time.time() - t0
    print(f"\n  Wall time: {elapsed:.0f}s")

    # ── Load evolved holdout results for comparison ───────────────────
    evolved_path = Path(_root) / "run10_holdout_eval_results.json"
    evolved = json.loads(evolved_path.read_text()) if evolved_path.exists() else None

    # ── Results ───────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("RESULTS — Static vs Evolved on HELD-OUT data")
    print(f"  ({len(val_subset)} val + {len(test_subset)} test, zero evo overlap)")
    print("═" * 70)

    print(f"\n  Static prompt:")
    print(f"    Validation: {static_val['accuracy']:.1f}%  ({static_val['correct']}/{static_val['total']})")
    print(f"    Test:       {static_test['accuracy']:.1f}%  ({static_test['correct']}/{static_test['total']})")
    print(f"    Per-class (validation):")
    for e in ENTITY_TYPES:
        c = static_val["per_class_counts"][e]
        print(f"      {e:8s}: {static_val['per_class_acc'][e]:5.1f}%  ({c['correct']}/{c['total']})")
    print(f"    Per-class (test):")
    for e in ENTITY_TYPES:
        c = static_test["per_class_counts"][e]
        print(f"      {e:8s}: {static_test['per_class_acc'][e]:5.1f}%  ({c['correct']}/{c['total']})")

    if evolved:
        ev_val = evolved["evolved"]["validation_accuracy"]
        ev_test = evolved["evolved"]["test_accuracy"]
        ev_pc_val = evolved["evolved"]["per_class_validation"]
        ev_pc_test = evolved["evolved"]["per_class_test"]
        ev_counts_val = evolved["evolved"]["per_class_counts_val"]
        ev_counts_test = evolved["evolved"]["per_class_counts_test"]

        print(f"\n  {'':35s} {'Validation':>12s} {'Test':>12s}")
        print(f"  {'─' * 60}")
        print(f"  {'Static':35s} {static_val['accuracy']:11.1f}% {static_test['accuracy']:11.1f}%")
        print(f"  {'Evolved (Run 10)':35s} {ev_val:11.1f}% {ev_test:11.1f}%")
        print(f"  {'─' * 60}")
        vd = ev_val - static_val["accuracy"]
        td = ev_test - static_test["accuracy"]
        print(f"  {'Δ (evolved − static)':35s} {vd:+10.1f}% {td:+10.1f}%")
        winner = "EVOLVED" if td > 0 else ("STATIC" if td < 0 else "TIE")
        print(f"\n  Winner (by test): {winner}")

        print(f"\n  Per-class comparison (test, held-out):")
        print(f"  {'Class':10s} {'Static':>8s} {'Evolved':>8s} {'Delta':>8s}")
        print(f"  {'─' * 36}")
        for e in ENTITY_TYPES:
            s = static_test["per_class_acc"][e]
            ev = ev_pc_test[e]
            print(f"  {e:10s} {s:7.1f}% {ev:7.1f}% {ev - s:+7.1f}%")

    # ── Save ──────────────────────────────────────────────────────────
    out = {
        "description": "Static prompt on held-out data (same samples as evolved holdout eval)",
        "holdout_seed": 99,
        "evolution_seed": 42,
        "val_size": len(val_subset),
        "test_size": len(test_subset),
        "static": {
            "validation_accuracy": static_val["accuracy"],
            "test_accuracy": static_test["accuracy"],
            "per_class_validation": static_val["per_class_acc"],
            "per_class_test": static_test["per_class_acc"],
            "per_class_counts_val": static_val["per_class_counts"],
            "per_class_counts_test": static_test["per_class_counts"],
        },
        "wall_time_s": elapsed,
    }
    out_path = Path(_root) / "static_holdout_eval_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved → {out_path}")
    print("\n✓ Static holdout evaluation complete.")


if __name__ == "__main__":
    main()
