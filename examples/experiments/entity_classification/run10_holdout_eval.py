#!/usr/bin/env python3
"""
Run 10 — Held-out evaluation.

Picks 500 val + 500 test samples that were NOT in the 200-sample subsets
used during the Run 10 evolution (seed=42).  This gives a clean, truly
unseen evaluation of the evolved prompt.
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

# ── Run 10 winning prompt ─────────────────────────────────────────────────
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

HOLDOUT_SIZE = 500  # per split


def main() -> None:
    print("=" * 70)
    print("Run 10 HELD-OUT Eval — 500 val + 500 test (unseen by evolution)")
    print("=" * 70)

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
    print(f"  validation total: {len(val_data)}")
    print(f"  test total:       {len(test_data)}")

    # ── Reproduce the evolution's RNG to find the 200 indices it used ─
    # The evolution script (run10_adaptive_mutations.py) does:
    #   rng = np.random.default_rng(42)
    #   val_indices  = rng.choice(len(val_data), size=200, replace=False)  # call 1
    #   test_indices = rng.choice(len(test_data), size=200, replace=False) # call 2
    evo_rng = np.random.default_rng(42)
    evo_val_indices = set(evo_rng.choice(len(val_data), size=200, replace=False).tolist())
    evo_test_indices = set(evo_rng.choice(len(test_data), size=200, replace=False).tolist())
    print(f"\n  Evolution used {len(evo_val_indices)} val + {len(evo_test_indices)} test indices (seed=42)")

    # ── Select held-out samples (exclude evolution indices) ───────────
    remaining_val = [i for i in range(len(val_data)) if i not in evo_val_indices]
    remaining_test = [i for i in range(len(test_data)) if i not in evo_test_indices]
    print(f"  Remaining val (unseen):  {len(remaining_val)}")
    print(f"  Remaining test (unseen): {len(remaining_test)}")

    # Use a different seed (99) so the holdout sample is deterministic
    # but independent of the evolution seed
    holdout_rng = np.random.default_rng(99)
    holdout_val_idx = holdout_rng.choice(remaining_val, size=min(HOLDOUT_SIZE, len(remaining_val)), replace=False)
    holdout_test_idx = holdout_rng.choice(remaining_test, size=min(HOLDOUT_SIZE, len(remaining_test)), replace=False)

    val_subset = [val_data[int(i)] for i in holdout_val_idx]
    test_subset = [test_data[int(i)] for i in holdout_test_idx]
    print(f"  Holdout eval: {len(val_subset)} val + {len(test_subset)} test")

    # Sanity: confirm zero overlap
    overlap_val = evo_val_indices & set(holdout_val_idx.tolist())
    overlap_test = evo_test_indices & set(holdout_test_idx.tolist())
    assert len(overlap_val) == 0, f"Val overlap: {overlap_val}"
    assert len(overlap_test) == 0, f"Test overlap: {overlap_test}"
    print("  ✓ Zero overlap with evolution samples confirmed")

    # ── Evaluate ──────────────────────────────────────────────────────
    client = LLMClient(PromptEvolverConfig(backend=LLMBackend.AZURE_OPENAI, max_tokens=10))

    print("\n" + "─" * 70)
    print("Evaluating Run 10 EVOLVED prompt on held-out samples")
    print("─" * 70)
    print(f"  Temperature: {EVOLVED_TEMPERATURE}")
    print(f"  Top-p:       {EVOLVED_TOP_P}")

    t0 = time.time()
    evolved_val = evaluate_prompt(client, EVOLVED_PROMPT, val_subset, EVOLVED_TEMPERATURE, EVOLVED_TOP_P, "holdout-val")
    evolved_test = evaluate_prompt(client, EVOLVED_PROMPT, test_subset, EVOLVED_TEMPERATURE, EVOLVED_TOP_P, "holdout-test")
    elapsed = time.time() - t0
    print(f"\n  Wall time: {elapsed:.0f}s")

    # ── Results ───────────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("RESULTS — Held-out Evaluation (samples never seen during evolution)")
    print(f"  ({len(val_subset)} val + {len(test_subset)} test, holdout seed=99)")
    print("═" * 70)

    print(f"\n  Evolved on held-out data:")
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

    # ── Comparison with full-eval numbers ─────────────────────────────
    full_eval_path = Path(_root) / "run10_full_eval_results.json"
    if full_eval_path.exists():
        fe = json.loads(full_eval_path.read_text())
        fe_val = fe["evolved"]["validation_accuracy"]
        fe_test = fe["evolved"]["test_accuracy"]
        print(f"\n  {'':35s} {'Validation':>12s} {'Test':>12s}")
        print(f"  {'─' * 60}")
        print(f"  {'Full eval (679+1000, incl. evo)':35s} {fe_val:11.1f}% {fe_test:11.1f}%")
        print(f"  {'Held-out (500+500, excl. evo)':35s} {evolved_val['accuracy']:11.1f}% {evolved_test['accuracy']:11.1f}%")
        print(f"  {'─' * 60}")
        val_d = evolved_val["accuracy"] - fe_val
        test_d = evolved_test["accuracy"] - fe_test
        print(f"  {'Δ (held-out − full)':35s} {val_d:+10.1f}% {test_d:+10.1f}%")

    # ── Save ──────────────────────────────────────────────────────────
    out = {
        "description": "Held-out evaluation — samples excluded from evolution",
        "holdout_seed": 99,
        "evolution_seed": 42,
        "evo_val_indices_excluded": len(evo_val_indices),
        "evo_test_indices_excluded": len(evo_test_indices),
        "val_size": len(val_subset),
        "test_size": len(test_subset),
        "evolved": {
            "validation_accuracy": evolved_val["accuracy"],
            "test_accuracy": evolved_test["accuracy"],
            "per_class_validation": evolved_val["per_class_acc"],
            "per_class_test": evolved_test["per_class_acc"],
            "per_class_counts_val": evolved_val["per_class_counts"],
            "per_class_counts_test": evolved_test["per_class_counts"],
        },
        "wall_time_s": elapsed,
    }
    out_path = Path(_root) / "run10_holdout_eval_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n  Results saved → {out_path}")
    print("\n✓ Held-out evaluation complete.")


if __name__ == "__main__":
    main()
