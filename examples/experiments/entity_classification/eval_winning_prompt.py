#!/usr/bin/env python3
"""Evaluate the winning evolved prompt on the entity-classification train split."""
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
)

# ── Winning prompt ────────────────────────────────────────────────────────

WINNING_PROMPT = """\
You are an AI assistant.

## Task
You are an agent that classifies inbound text into one of Agent, Task, Tool, Input, Output, Human

## Output
Provide a clear, direct response."""

TEMPERATURE = 0.7
TOP_P = 0.95

# Backend — switch between OLLAMA and AZURE_OPENAI
BACKEND = LLMBackend.OLLAMA
OLLAMA_MODEL = "llama3.2"

# How many train samples to evaluate (set to None for all)
EVAL_SIZE = 200


def load_train_split() -> list[dict]:
    """Load the train split from HuggingFace."""
    from datasets import load_dataset as _hf_load

    print("Loading train split from HuggingFace …")
    ds = _hf_load("holistic-ai/entity-classification-agentic-ai")
    train = [
        {"content": r["content"], "expected_entity": r["expected_entity"]}
        for r in ds["train"]
    ]
    print(f"  train: {len(train)} samples")
    return train


def main() -> None:
    print("=" * 70)
    print("Evaluate Winning Prompt on Train Split")
    print("=" * 70)

    train_data = load_train_split()

    rng = np.random.default_rng(42)
    if EVAL_SIZE and EVAL_SIZE < len(train_data):
        indices = rng.choice(len(train_data), size=EVAL_SIZE, replace=False)
        subset = [train_data[int(i)] for i in indices]
        print(f"\nEvaluating on {len(subset)} / {len(train_data)} train samples")
    else:
        subset = train_data
        print(f"\nEvaluating on all {len(subset)} train samples")

    client = LLMClient(PromptEvolverConfig(backend=BACKEND, ollama_model=OLLAMA_MODEL, max_tokens=10))
    if not client.is_available():
        print("\n⚠  Azure OpenAI not reachable.")
        return

    print(f"\nPrompt:\n{WINNING_PROMPT}")
    print(f"\nTemperature: {TEMPERATURE}  Top-p: {TOP_P}")
    print()

    t0 = time.perf_counter()
    result = evaluate_prompt(
        client,
        WINNING_PROMPT,
        subset,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        label="winning-train",
        log_interval=20,
    )
    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 70}")
    print(f"Results")
    print(f"{'=' * 70}")
    print(f"  Accuracy: {result['accuracy']:.1f}%  ({result['correct']}/{result['total']})")
    print(f"  Time:     {elapsed:.0f}s")
    print(f"\n  Per-class accuracy:")
    for e in ENTITY_TYPES:
        c = result["per_class_counts"][e]
        print(f"    {e:8s}: {result['per_class_acc'][e]:5.1f}%  ({c['correct']}/{c['total']})")

    log_path = Path(_root) / "logs" / "winning_prompt_train_eval.json"
    log = {
        "prompt": WINNING_PROMPT,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "split": "train",
        "eval_size": len(subset),
        "accuracy": round(result["accuracy"], 2),
        "correct": result["correct"],
        "total": result["total"],
        "per_class_acc": {k: round(v, 2) for k, v in result["per_class_acc"].items()},
        "per_class_counts": result["per_class_counts"],
        "elapsed_seconds": round(elapsed, 1),
    }
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\n  Log saved to {log_path}")


if __name__ == "__main__":
    main()
