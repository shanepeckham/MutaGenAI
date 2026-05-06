#!/usr/bin/env python3
"""
Run 9 Full Eval — Evaluate the winning classification-mutations prompt
against all 679 validation + first 1000 test samples.

Compares against prior static and Run 7 evolved results.
"""
from __future__ import annotations

import json
import os
import sys
import time

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

# ── The winning Run 9 prompt ───────────────────────────────────────────────

WINNING_PROMPT = (
    "Return exactly one word from: Agent, Task, Tool, Input, Output, Human.\n"
    "Consider the semantic meaning of the input, not just keywords.\n"
    "Choose the single most specific category that fits.\n"
    "Return exactly one category label — no extra text."
)
WINNING_TEMP = 0.7037
WINNING_TOP_P = 0.927

# ── Prior results for comparison ───────────────────────────────────────────

PRIOR_STATIC = {
    "val_acc": 86.0, "val_correct": 584, "val_total": 679,
    "test_acc": 86.0, "test_correct": 860, "test_total": 1000,
    "per_class_test": {
        "Agent": 80.4, "Task": 91.2, "Tool": 81.8,
        "Input": 97.9, "Output": 95.8, "Human": 0.0,
    },
}
PRIOR_EVOLVED_R7 = {
    "val_acc": 88.1, "val_correct": 598, "val_total": 679,
    "test_acc": 88.0, "test_correct": 880, "test_total": 1000,
    "per_class_test": {
        "Agent": 78.4, "Task": 85.6, "Tool": 96.8,
        "Input": 95.3, "Output": 95.3, "Human": 2.2,
    },
}

# ── Main ───────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("Run 9 Full Eval — Classification Mutations Winning Prompt")
    print("  679 validation + 1000 test (same as prior experiments)")
    print("=" * 70)

    # Load dataset
    from datasets import load_dataset as _hf_load
    print("\nLoading dataset from HuggingFace …")
    ds = _hf_load("holistic-ai/entity-classification-agentic-ai")
    val = [{"content": r["content"], "expected_entity": r["expected_entity"]} for r in ds["validation"]]
    test_all = [{"content": r["content"], "expected_entity": r["expected_entity"]} for r in ds["test"]]
    test = test_all[:1000]  # first 1000 as in prior experiments
    print(f"  validation: {len(val)} samples")
    print(f"  test:       {len(test)} samples (first 1000 of {len(test_all)})")

    print(f"\nWinning prompt:\n{'─' * 40}")
    print(WINNING_PROMPT)
    print(f"{'─' * 40}")
    print(f"  temperature: {WINNING_TEMP}")
    print(f"  top_p:       {WINNING_TOP_P}")

    llm_config = PromptEvolverConfig(backend=LLMBackend.AZURE_OPENAI, max_tokens=10)
    client = LLMClient(llm_config)

    # ── Validation ───────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"Evaluating on validation ({len(val)} samples) …")
    print(f"{'─' * 70}")
    t0 = time.perf_counter()
    val_result = evaluate_prompt(client, WINNING_PROMPT, val, WINNING_TEMP, WINNING_TOP_P, label="val")
    val_time = time.perf_counter() - t0
    print(f"\n  Validation accuracy: {val_result['accuracy']:.1f}% "
          f"({val_result['correct']}/{val_result['total']})")
    print(f"  Time: {val_time:.0f}s")

    # ── Test ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print(f"Evaluating on test ({len(test)} samples) …")
    print(f"{'─' * 70}")
    t0 = time.perf_counter()
    test_result = evaluate_prompt(client, WINNING_PROMPT, test, WINNING_TEMP, WINNING_TOP_P, label="test")
    test_time = time.perf_counter() - t0
    print(f"\n  Test accuracy: {test_result['accuracy']:.1f}% "
          f"({test_result['correct']}/{test_result['total']})")
    print(f"  Time: {test_time:.0f}s")

    # ── Per-class ────────────────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("Per-class breakdown (test set):")
    print(f"{'─' * 70}")
    print(f"  {'Class':<10} {'Static':>10} {'R7 Evolved':>12} {'R9 ClassMut':>12} {'R9 vs Static':>13} {'R9 vs R7':>10}")
    print(f"  {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 12} {'─' * 13} {'─' * 10}")

    for e in ENTITY_TYPES:
        static_pct = PRIOR_STATIC["per_class_test"][e]
        r7_pct = PRIOR_EVOLVED_R7["per_class_test"][e]
        r9_pct = test_result["per_class_acc"][e]
        delta_static = r9_pct - static_pct
        delta_r7 = r9_pct - r7_pct
        r9_detail = f"{r9_pct:.1f}%"
        print(f"  {e:<10} {static_pct:>9.1f}% {r7_pct:>11.1f}% {r9_detail:>12} "
              f"{delta_static:>+12.1f}% {delta_r7:>+9.1f}%")

    # ── Summary comparison table ─────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("Head-to-Head: Static vs R7 Evolved vs R9 Classification Mutations")
    print(f"{'═' * 70}")
    print(f"\n  {'':30} {'Val (679)':>12} {'Test (1000)':>12}")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12}")
    print(f"  {'Static':30} {PRIOR_STATIC['val_acc']:>11.1f}% {PRIOR_STATIC['test_acc']:>11.1f}%")
    print(f"  {'R7 Evolved (tool muts)':30} {PRIOR_EVOLVED_R7['val_acc']:>11.1f}% {PRIOR_EVOLVED_R7['test_acc']:>11.1f}%")
    print(f"  {'R9 Evolved (class muts)':30} {val_result['accuracy']:>11.1f}% {test_result['accuracy']:>11.1f}%")
    print(f"  {'─' * 30} {'─' * 12} {'─' * 12}")
    r9_vs_static_val = val_result["accuracy"] - PRIOR_STATIC["val_acc"]
    r9_vs_static_test = test_result["accuracy"] - PRIOR_STATIC["test_acc"]
    r9_vs_r7_val = val_result["accuracy"] - PRIOR_EVOLVED_R7["val_acc"]
    r9_vs_r7_test = test_result["accuracy"] - PRIOR_EVOLVED_R7["test_acc"]
    print(f"  {'R9 vs Static':30} {r9_vs_static_val:>+11.1f}% {r9_vs_static_test:>+11.1f}%")
    print(f"  {'R9 vs R7 Evolved':30} {r9_vs_r7_val:>+11.1f}% {r9_vs_r7_test:>+11.1f}%")

    # ── Per-class validation breakdown ───────────────────────────────
    print(f"\n{'─' * 70}")
    print("Per-class breakdown (validation set):")
    print(f"{'─' * 70}")
    print(f"  {'Class':<10} {'R9 Accuracy':>12} {'Correct/Total':>15}")
    print(f"  {'─' * 10} {'─' * 12} {'─' * 15}")
    for e in ENTITY_TYPES:
        pc = val_result["per_class_counts"][e]
        pct = val_result["per_class_acc"][e]
        print(f"  {e:<10} {pct:>11.1f}% {pc['correct']:>7}/{pc['total']}")

    # ── Save log ─────────────────────────────────────────────────────
    log = {
        "experiment": "Run 9 Full Eval — Classification Mutations Winning Prompt",
        "winning_prompt": WINNING_PROMPT,
        "temperature": WINNING_TEMP,
        "top_p": WINNING_TOP_P,
        "validation": val_result,
        "test": test_result,
        "val_time_s": round(val_time, 1),
        "test_time_s": round(test_time, 1),
        "prior_static": PRIOR_STATIC,
        "prior_evolved_r7": PRIOR_EVOLVED_R7,
    }
    log_path = os.path.join(_root, "run9_full_eval_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Log saved → {os.path.abspath(log_path)}")
    print("\n✓ Run 9 full eval complete.")


if __name__ == "__main__":
    main()
