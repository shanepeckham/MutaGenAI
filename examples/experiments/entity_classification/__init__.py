"""Shared utilities for entity classification experiments.

Consolidates the duplicated extract_entity, evaluate_prompt,
load_hf_dataset, ClassificationAccuracyScorer, and holdout-sample
selection that were previously copy-pasted across every experiment script.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np

ENTITY_TYPES = ["Agent", "Task", "Tool", "Input", "Output", "Human"]

DATASET_ID = "holistic-ai/entity-classification-agentic-ai"


def extract_entity(output: str) -> str:
    """Parse a single entity label from LLM output.

    Tries exact match, prefix match, then word-boundary match.
    """
    if not output:
        return ""
    cleaned = output.strip()
    lowered = cleaned.lower()
    for entity in ENTITY_TYPES:
        if lowered == entity.lower():
            return entity
    for entity in ENTITY_TYPES:
        if lowered.startswith(entity.lower()):
            return entity
    for entity in ENTITY_TYPES:
        if re.search(r"\b" + entity + r"\b", cleaned, re.IGNORECASE):
            return entity
    return ""


def load_hf_dataset() -> tuple[list[dict], list[dict]]:
    """Load the entity-classification dataset from HuggingFace.

    Returns (validation_samples, test_samples) as lists of dicts
    with keys ``content`` and ``expected_entity``.
    """
    from datasets import load_dataset as _hf_load

    print("Loading dataset from HuggingFace …")
    ds = _hf_load(DATASET_ID)
    val = [{"content": r["content"], "expected_entity": r["expected_entity"]} for r in ds["validation"]]
    test = [{"content": r["content"], "expected_entity": r["expected_entity"]} for r in ds["test"]]
    print(f"  validation: {len(val)} samples, test: {len(test)} samples")
    return val, test


def evaluate_prompt(
    client,
    prompt: str,
    samples: list[dict],
    temperature: float,
    top_p: float,
    label: str = "",
    log_interval: int = 50,
) -> dict:
    """Evaluate a prompt on a list of samples, returning accuracy and per-class stats.

    Returns a dict with keys: accuracy, correct, total, per_class_acc, per_class_counts.
    """
    correct = 0
    total = len(samples)
    per_class: dict[str, dict[str, int]] = {
        e: {"correct": 0, "total": 0} for e in ENTITY_TYPES
    }

    for i, sample in enumerate(samples):
        output = client.complete(
            system_prompt=prompt,
            user_message=sample["content"],
            temperature=temperature,
            top_p=top_p,
        )
        predicted = extract_entity(output or "")
        expected = sample["expected_entity"]
        hit = predicted == expected
        if hit:
            correct += 1
        per_class[expected]["total"] += 1
        if hit:
            per_class[expected]["correct"] += 1
        if (i + 1) % log_interval == 0 and label:
            print(f"    [{label}] {i + 1}/{total}  acc={correct / (i + 1) * 100:.1f}%")

    accuracy = correct / total * 100 if total else 0.0
    per_class_acc = {}
    per_class_counts = {}
    for e, v in per_class.items():
        per_class_acc[e] = v["correct"] / v["total"] * 100 if v["total"] else 0.0
        per_class_counts[e] = {"correct": v["correct"], "total": v["total"]}
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class_acc": per_class_acc,
        "per_class_counts": per_class_counts,
    }


def select_holdout_samples(
    val_data: list[dict],
    test_data: list[dict],
    evo_seed: int = 42,
    evo_val_size: int = 200,
    evo_test_size: int = 200,
    holdout_seed: int = 99,
    holdout_val_size: int = 500,
    holdout_test_size: int = 500,
) -> tuple[list[dict], list[dict]]:
    """Select held-out samples by excluding the evolution-seed indices.

    Reproduces the evolution RNG to identify which indices were used,
    then samples from the remainder with a separate seed.  Asserts
    zero overlap.
    """
    evo_rng = np.random.default_rng(evo_seed)
    evo_val_idx = set(evo_rng.choice(len(val_data), size=evo_val_size, replace=False).tolist())
    evo_test_idx = set(evo_rng.choice(len(test_data), size=evo_test_size, replace=False).tolist())

    remaining_val = [i for i in range(len(val_data)) if i not in evo_val_idx]
    remaining_test = [i for i in range(len(test_data)) if i not in evo_test_idx]

    holdout_rng = np.random.default_rng(holdout_seed)
    hval_idx = holdout_rng.choice(remaining_val, size=min(holdout_val_size, len(remaining_val)), replace=False)
    htest_idx = holdout_rng.choice(remaining_test, size=min(holdout_test_size, len(remaining_test)), replace=False)

    val_subset = [val_data[int(i)] for i in hval_idx]
    test_subset = [test_data[int(i)] for i in htest_idx]

    assert not (evo_val_idx & set(hval_idx.tolist())), "Val overlap with evo samples"
    assert not (evo_test_idx & set(htest_idx.tolist())), "Test overlap with evo samples"
    print(f"  Holdout: {len(val_subset)} val + {len(test_subset)} test")
    print("  ✓ Zero overlap with evolution samples confirmed")
    return val_subset, test_subset


try:
    from MutaGenAI.strategies import Scorer

    class ClassificationAccuracyScorer(Scorer):
        """Scorer that checks entity classification accuracy against a label map."""

        def __init__(self, label_map: dict[str, str]):
            self._label_map = label_map

        def score(self, prompt, test_input, output, client):
            expected = self._label_map.get(test_input, "")
            predicted = extract_entity(output or "")
            return 1.0 if predicted == expected else 0.0

        def name(self):
            return "classification_accuracy"

except ImportError:
    pass
