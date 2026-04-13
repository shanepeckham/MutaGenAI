#!/usr/bin/env python3
"""
Full 1000-sample head-to-head evaluation: Static vs Evolved prompt.
Runs both prompts on 1000 validation + 1000 test samples using gpt-4.1.
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

from prompture.prompt_evolver import LLMBackend, LLMClient, PromptEvolverConfig

from examples.experiments.entity_classification import (
    ENTITY_TYPES,
    evaluate_prompt,
    extract_entity,
)

# ── Prompts ───────────────────────────────────────────────────────────────
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

EVOLVED_PROMPT = """\
Tell me the entity type: Agent, Task, Tool, Input, Output, or Human.

Available tools:
{tool_schemas}"""

EVOLVED_TEMPERATURE = 0.7067
EVOLVED_TOP_P = 0.7588

STATIC_TEMPERATURE = 0.7
STATIC_TOP_P = 0.95

VAL_SIZE = 1000
TEST_SIZE = 1000

# ── Helpers ───────────────────────────────────────────────────────────────


def print_results(name: str, val_res: dict, test_res: dict) -> None:
    print(f"\n  {name}:")
    print(f"    Validation: {val_res['accuracy']:.1f}%  ({val_res['correct']}/{val_res['total']})")
    print(f"    Test:       {test_res['accuracy']:.1f}%  ({test_res['correct']}/{test_res['total']})")
    print(f"    Per-class (validation):")
    for e in ENTITY_TYPES:
        c = val_res["per_class_counts"][e]
        print(f"      {e:8s}: {val_res['per_class_acc'][e]:5.1f}%  ({c['correct']}/{c['total']})")
    print(f"    Per-class (test):")
    for e in ENTITY_TYPES:
        c = test_res["per_class_counts"][e]
        print(f"      {e:8s}: {test_res['per_class_acc'][e]:5.1f}%  ({c['correct']}/{c['total']})")


def main() -> None:
    print("=" * 70)
    print("Full 1000-sample Evaluation: Static vs Evolved (gpt-4.1)")
    print("=" * 70)

    # ── Load dataset ──────────────────────────────────────────────────
    try:
        from datasets import load_dataset as _hf_load
    except ImportError:
        print("ERROR: pip install datasets")
        sys.exit(1)

    print("Loading dataset …")
    ds = _hf_load("holistic-ai/entity-classification-agentic-ai")
    val_data = [{"content": r["content"], "expected_entity": r["expected_entity"]} for r in ds["validation"]]
    test_data = [{"content": r["content"], "expected_entity": r["expected_entity"]} for r in ds["test"]]
    print(f"  validation: {len(val_data)} total samples")
    print(f"  test:       {len(test_data)} total samples")

    rng = np.random.default_rng(42)

    val_size = min(VAL_SIZE, len(val_data))
    test_size = min(TEST_SIZE, len(test_data))

    val_indices = rng.choice(len(val_data), size=val_size, replace=False)
    val_subset = [val_data[int(i)] for i in val_indices]

    test_indices = rng.choice(len(test_data), size=test_size, replace=False)
    test_subset = [test_data[int(i)] for i in test_indices]

    print(f"\n  Eval validation subset: {len(val_subset)} samples")
    print(f"  Eval test subset:       {len(test_subset)} samples")

    client = LLMClient(PromptEvolverConfig(backend=LLMBackend.AZURE_OPENAI, max_tokens=10))

    # ── Static prompt ─────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Evaluating STATIC prompt")
    print("─" * 70)

    t0 = time.time()
    static_val = evaluate_prompt(client, STATIC_PROMPT, val_subset, STATIC_TEMPERATURE, STATIC_TOP_P, "static-val")
    static_test = evaluate_prompt(client, STATIC_PROMPT, test_subset, STATIC_TEMPERATURE, STATIC_TOP_P, "static-test")
    static_time = time.time() - t0
    print(f"  Static wall time: {static_time:.0f}s")

    # ── Evolved prompt ────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("Evaluating EVOLVED prompt")
    print("─" * 70)

    t0 = time.time()
    evolved_val = evaluate_prompt(client, EVOLVED_PROMPT, val_subset, EVOLVED_TEMPERATURE, EVOLVED_TOP_P, "evolved-val")
    evolved_test = evaluate_prompt(client, EVOLVED_PROMPT, test_subset, EVOLVED_TEMPERATURE, EVOLVED_TOP_P, "evolved-test")
    evolved_time = time.time() - t0
    print(f"  Evolved wall time: {evolved_time:.0f}s")

    # ── Head to head ──────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("HEAD-TO-HEAD RESULTS (1000 samples each)")
    print("═" * 70)

    print_results("Static prompt", static_val, static_test)
    print_results("Evolved prompt (EvoSim)", evolved_val, evolved_test)

    val_delta = evolved_val["accuracy"] - static_val["accuracy"]
    test_delta = evolved_test["accuracy"] - static_test["accuracy"]

    print(f"\n  {'':25s} {'Validation':>12s} {'Test':>12s}")
    print(f"  {'─' * 50}")
    print(f"  {'Static prompt':25s} {static_val['accuracy']:11.1f}% {static_test['accuracy']:11.1f}%")
    print(f"  {'Evolved prompt':25s} {evolved_val['accuracy']:11.1f}% {evolved_test['accuracy']:11.1f}%")
    print(f"  {'─' * 50}")
    print(f"  {'Δ (evolved − static)':25s} {val_delta:+10.1f}% {test_delta:+10.1f}%")

    winner = "EVOLVED" if test_delta > 0 else ("STATIC" if test_delta < 0 else "TIE")
    print(f"\n  Winner (by test): {winner}")

    # Per-class comparison
    print(f"\n  Per-class comparison (validation):")
    print(f"  {'Class':10s} {'Static':>8s} {'Evolved':>8s} {'Delta':>8s}")
    print(f"  {'─' * 36}")
    for e in ENTITY_TYPES:
        s = static_val["per_class_acc"][e]
        ev = evolved_val["per_class_acc"][e]
        print(f"  {e:10s} {s:7.1f}% {ev:7.1f}% {ev - s:+7.1f}%")

    print(f"\n  Per-class comparison (test):")
    print(f"  {'Class':10s} {'Static':>8s} {'Evolved':>8s} {'Delta':>8s}")
    print(f"  {'─' * 36}")
    for e in ENTITY_TYPES:
        s = static_test["per_class_acc"][e]
        ev = evolved_test["per_class_acc"][e]
        print(f"  {e:10s} {s:7.1f}% {ev:7.1f}% {ev - s:+7.1f}%")

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "experiment": "Full 1000-sample evaluation",
        "model": "gpt-4.1 (Azure OpenAI)",
        "val_size": val_size,
        "test_size": test_size,
        "static": {
            "prompt": STATIC_PROMPT,
            "temperature": STATIC_TEMPERATURE,
            "top_p": STATIC_TOP_P,
            "validation_accuracy": static_val["accuracy"],
            "test_accuracy": static_test["accuracy"],
            "per_class_validation": static_val["per_class_acc"],
            "per_class_test": static_test["per_class_acc"],
            "per_class_counts_val": static_val["per_class_counts"],
            "per_class_counts_test": static_test["per_class_counts"],
        },
        "evolved": {
            "prompt": EVOLVED_PROMPT,
            "temperature": EVOLVED_TEMPERATURE,
            "top_p": EVOLVED_TOP_P,
            "validation_accuracy": evolved_val["accuracy"],
            "test_accuracy": evolved_test["accuracy"],
            "per_class_validation": evolved_val["per_class_acc"],
            "per_class_test": evolved_test["per_class_acc"],
            "per_class_counts_val": evolved_val["per_class_counts"],
            "per_class_counts_test": evolved_test["per_class_counts"],
        },
        "winner": winner,
        "val_delta": val_delta,
        "test_delta": test_delta,
        "total_wall_time_s": static_time + evolved_time,
    }

    out_path = Path(_root) / "full_eval_1000_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {out_path}")
    print("\n✓ Full evaluation complete.")


if __name__ == "__main__":
    main()
