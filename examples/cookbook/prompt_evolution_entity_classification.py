#!/usr/bin/env python3
"""
Cookbook Recipe 51 — Entity Classification: Static vs Evolved Prompt
====================================================================

**Hypothesis**: Starting with a basic prompt and evolving it with EvoSim's
deep evolutionary search yields better classification accuracy than
starting with a sophisticated AI-generated prompt, because EvoSim
explores a wider prompt space.

Setup
-----
- **Dataset**: `holistic-ai/entity-classification-agentic-ai
  <https://huggingface.co/datasets/holistic-ai/entity-classification-agentic-ai>`_
- **Classes**: Agent · Task · Tool · Input · Output · Human (6 types)
- **Model**: Ollama ``llama3.2`` (local, 3 B parameters)
- **Scoring split**: ``validation`` (1 250 samples)
- **Test split**: ``test`` (2 500 samples)

Experiment
----------
1. **Static AI-generated prompt** — a sophisticated zero-shot
   classification prompt (not evolved).  Evaluated on validation,
   then on test.
2. **EvoSim evolution** — deep search (5 generations × 10 pop × 2
   islands) from eight diverse seed prompts loaded from
   ``seed_templates/entity_classification.json``.  Fitness =
   classification accuracy on a validation subsample.  Winner evaluated
   on full validation, then on test.
3. **Comparison** — accuracy delta on validation and test sets.

All prompts, mutations, and parameters are logged to
``entity_classification_evolution_log.json``.

Usage::

    pip install datasets          # one-time
    uv run python examples/cookbook/prompt_evolution_entity_classification.py

References
----------
* Wizard presets:  ``evosim init``
* Dataset:         https://huggingface.co/datasets/holistic-ai/entity-classification-agentic-ai
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from abc import ABC
from pathlib import Path
from typing import Any

import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, "..", ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(_here, "..", "..", ".env"))

from MutaGenAI.prompt_evolver import (
    LLMBackend,
    LLMClient,
    PromptCandidate,
    PromptEvolverConfig,
)
from MutaGenAI.seed_loader import load_seed_templates
from MutaGenAI.strategies import (
    NoEvalConfig,
    NoEvalPromptEvolver,
    Scorer,
)

# ── Constants ──────────────────────────────────────────────────────────────

ENTITY_TYPES = ["Agent", "Task", "Tool", "Input", "Output", "Human"]

# Sample sizes — increase for production runs (slows proportionally)
EVOLUTION_SAMPLE_SIZE = 40     # validation samples used during evolution
VALIDATION_EVAL_SIZE = 200     # validation samples for final scoring
TEST_EVAL_SIZE = 200           # test samples for final scoring

# ── Prompts ────────────────────────────────────────────────────────────────

# Sophisticated AI-generated prompt (our *static* baseline)
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

# Diverse seed prompts loaded from external config.
# See seed_templates/entity_classification.json for the full list.
SEED_TEMPLATES = load_seed_templates("entity_classification")

# Evolution config — deep search, 2 islands.
# population_size=8 with 8 diverse seed archetypes, gpt-4.1.
EVOLUTION_CONFIG = NoEvalConfig(
    iterations=5,
    population_size=8,
    num_islands=2,
    elite_size=4,
    mutation_rate=0.5,
    crossover_rate=0.3,
    migration_interval=2,
    backend=LLMBackend.AZURE_OPENAI,
    max_tokens=10,
    refine_after_splice=False,
)

# ── Dataset loading ────────────────────────────────────────────────────────


def load_hf_dataset() -> tuple[list[dict], list[dict]]:
    """Load the entity classification dataset from HuggingFace.

    Returns (validation_samples, test_samples) where each sample is
    ``{"content": str, "expected_entity": str}``.
    """
    try:
        from datasets import load_dataset as _hf_load
    except ImportError:
        print("ERROR: Install the 'datasets' library first:")
        print("  pip install datasets")
        sys.exit(1)

    print("Loading dataset from HuggingFace …")
    ds = _hf_load("holistic-ai/entity-classification-agentic-ai")

    val = [
        {"content": r["content"], "expected_entity": r["expected_entity"]}
        for r in ds["validation"]
    ]
    test = [
        {"content": r["content"], "expected_entity": r["expected_entity"]}
        for r in ds["test"]
    ]
    print(f"  validation: {len(val)} samples")
    print(f"  test:       {len(test)} samples")
    return val, test


# ── Helpers ────────────────────────────────────────────────────────────────


def extract_entity(output: str) -> str:
    """Extract predicted entity type from LLM output."""
    if not output:
        return ""
    cleaned = output.strip()
    # Exact match (case-insensitive)
    for entity in ENTITY_TYPES:
        if cleaned.lower() == entity.lower():
            return entity
    # Starts with entity name
    for entity in ENTITY_TYPES:
        if cleaned.lower().startswith(entity.lower()):
            return entity
    # Word-boundary search
    for entity in ENTITY_TYPES:
        if re.search(r"\b" + entity + r"\b", cleaned, re.IGNORECASE):
            return entity
    return ""


def evaluate_prompt(
    client: LLMClient,
    prompt: str,
    samples: list[dict],
    temperature: float = 0.7,
    top_p: float = 0.95,
    label: str = "",
) -> dict:
    """Evaluate a classification prompt on labelled samples.

    Returns dict with ``accuracy``, ``correct``, ``total``,
    ``per_class``, and ``details``.
    """
    correct = 0
    total = len(samples)
    per_class: dict[str, dict[str, int]] = {
        e: {"correct": 0, "total": 0} for e in ENTITY_TYPES
    }
    details: list[dict] = []

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
        details.append(
            {
                "content": sample["content"][:120],
                "expected": expected,
                "predicted": predicted,
                "correct": hit,
            }
        )
        if (i + 1) % 20 == 0 and label:
            acc_so_far = correct / (i + 1) * 100
            print(f"    [{label}] {i + 1}/{total}  acc={acc_so_far:.1f}%")

    accuracy = correct / total * 100 if total else 0.0
    per_class_acc = {
        e: (
            v["correct"] / v["total"] * 100 if v["total"] else 0.0
        )
        for e, v in per_class.items()
    }
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class": per_class_acc,
        "details": details,
    }


# ── Custom scorer ──────────────────────────────────────────────────────────


class ClassificationAccuracyScorer(Scorer):
    """Fitness scorer: classification accuracy against known labels.

    During evolution the evolver calls ``score(prompt, test_input, output,
    client)`` for every test input.  We look up the expected label from
    ``label_map`` (keyed by content string) and return 1.0 for a correct
    match, 0.0 otherwise.
    """

    def __init__(self, label_map: dict[str, str]) -> None:
        self._label_map = label_map

    def score(
        self,
        prompt: str,
        test_input: str,
        output: str,
        client: Any,
    ) -> float:
        expected = self._label_map.get(test_input, "")
        predicted = extract_entity(output or "")
        return 1.0 if predicted == expected else 0.0

    def name(self) -> str:
        return "classification_accuracy"


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    print("=" * 70)
    print("EvoSim Cookbook Recipe 51")
    print("Entity Classification: Static vs Evolved Prompt")
    print("=" * 70)

    # ── 0. Load dataset ──────────────────────────────────────────────
    val_data, test_data = load_hf_dataset()

    rng = np.random.default_rng(42)

    # Sub-sample for speed
    val_indices = rng.choice(len(val_data), size=min(VALIDATION_EVAL_SIZE, len(val_data)), replace=False)
    val_eval_subset = [val_data[int(i)] for i in val_indices]

    test_indices = rng.choice(len(test_data), size=min(TEST_EVAL_SIZE, len(test_data)), replace=False)
    test_eval_subset = [test_data[int(i)] for i in test_indices]

    # Separate subset for evolution scoring
    evo_indices = rng.choice(len(val_data), size=min(EVOLUTION_SAMPLE_SIZE, len(val_data)), replace=False)
    evo_subset = [val_data[int(i)] for i in evo_indices]

    print(f"\nEvolution scoring subset:  {len(evo_subset)} samples")
    print(f"Validation eval subset:   {len(val_eval_subset)} samples")
    print(f"Test eval subset:         {len(test_eval_subset)} samples")

    # Build label map for evolution scorer
    label_map = {s["content"]: s["expected_entity"] for s in evo_subset}
    test_inputs = [s["content"] for s in evo_subset]

    # LLM client — limit to 10 tokens (entity names are 1 word)
    llm_config = PromptEvolverConfig(backend=LLMBackend.AZURE_OPENAI, max_tokens=10)
    client = LLMClient(llm_config)

    if not client.is_available():
        print("\n⚠  Azure OpenAI not reachable — results will be random.")

    # ── 1. Static AI-generated prompt ────────────────────────────────
    print("\n" + "─" * 70)
    print("Phase 1: Static AI-generated prompt (baseline)")
    print("─" * 70)
    print("\nStatic prompt:")
    print(STATIC_PROMPT[:200] + " …")

    t0 = time.perf_counter()
    static_val = evaluate_prompt(
        client, STATIC_PROMPT, val_eval_subset,
        temperature=0.7, label="static-val",
    )
    static_test = evaluate_prompt(
        client, STATIC_PROMPT, test_eval_subset,
        temperature=0.7, label="static-test",
    )
    static_time = time.perf_counter() - t0

    print(f"\n  Static — validation accuracy: {static_val['accuracy']:.1f}%")
    print(f"  Static — test accuracy:       {static_test['accuracy']:.1f}%")
    print(f"  Wall time: {static_time:.0f}s")
    print("  Per-class (validation):")
    for cls, acc in static_val["per_class"].items():
        print(f"    {cls:8s}: {acc:5.1f}%")

    # ── 2. EvoSim evolution from basic seed ──────────────────────────
    print("\n" + "─" * 70)
    print("Phase 2: EvoSim deep evolution (2 islands)")
    print("─" * 70)
    print(f"\nConfig: {EVOLUTION_CONFIG.iterations} gen × "
          f"{EVOLUTION_CONFIG.population_size} pop × "
          f"{EVOLUTION_CONFIG.num_islands} islands  "
          f"(elite={EVOLUTION_CONFIG.elite_size}, "
          f"μ={EVOLUTION_CONFIG.mutation_rate}, "
          f"cx={EVOLUTION_CONFIG.crossover_rate}, "
          f"migrate every {EVOLUTION_CONFIG.migration_interval} gen)")
    print("\nSeed prompts:")
    for i, s in enumerate(SEED_TEMPLATES):
        print(f"  [{i}] {s}")

    scorer = ClassificationAccuracyScorer(label_map)

    evolver = NoEvalPromptEvolver(
        task_description=(
            "Classify text into one of six entity types for agentic AI: "
            "Agent, Task, Tool, Input, Output, Human. "
            "Return exactly one entity type name."
        ),
        test_inputs=test_inputs,
        scorer=scorer,
        config=EVOLUTION_CONFIG,
        seed_templates=SEED_TEMPLATES,
        seed=42,
        verbose=True,
    )

    t1 = time.perf_counter()
    result = evolver.run()
    evo_time_search = time.perf_counter() - t1

    print(f"\n  Best evolved score (evo subset): {result.best_score:.1f}%")
    print(f"  Best temperature:  {result.best_temperature:.3f}")
    print(f"  Best top-p:        {result.best_top_p:.3f}")
    print(f"  Candidates tried:  {len(result.all_candidates)}")
    print(f"  Search time:       {evo_time_search:.0f}s")
    print(f"\n  Best evolved prompt:\n{'─' * 40}")
    print(result.best_prompt)
    print("─" * 40)

    # Evaluate evolved winner on full validation + test
    t2 = time.perf_counter()
    evolved_val = evaluate_prompt(
        client, result.best_prompt, val_eval_subset,
        temperature=result.best_temperature,
        top_p=result.best_top_p,
        label="evolved-val",
    )
    evolved_test = evaluate_prompt(
        client, result.best_prompt, test_eval_subset,
        temperature=result.best_temperature,
        top_p=result.best_top_p,
        label="evolved-test",
    )
    evo_time_eval = time.perf_counter() - t2

    print(f"\n  Evolved — validation accuracy: {evolved_val['accuracy']:.1f}%")
    print(f"  Evolved — test accuracy:       {evolved_test['accuracy']:.1f}%")
    print(f"  Eval time: {evo_time_eval:.0f}s")
    print("  Per-class (validation):")
    for cls, acc in evolved_val["per_class"].items():
        print(f"    {cls:8s}: {acc:5.1f}%")

    # ── 3. Comparison ────────────────────────────────────────────────
    print("\n" + "═" * 70)
    print("Comparison: Static vs Evolved")
    print("═" * 70)

    val_lift = evolved_val["accuracy"] - static_val["accuracy"]
    test_lift = evolved_test["accuracy"] - static_test["accuracy"]
    winner = "evolved" if test_lift > 0 else ("static" if test_lift < 0 else "tie")

    print(f"\n  {'':20s} {'Validation':>12s}  {'Test':>12s}")
    print(f"  {'─' * 48}")
    print(f"  {'Static prompt':20s} {static_val['accuracy']:11.1f}%  {static_test['accuracy']:11.1f}%")
    print(f"  {'Evolved prompt':20s} {evolved_val['accuracy']:11.1f}%  {evolved_test['accuracy']:11.1f}%")
    print(f"  {'─' * 48}")
    print(f"  {'Δ (evolved − static)':20s} {val_lift:+11.1f}%  {test_lift:+11.1f}%")
    print(f"\n  Winner: {winner.upper()}")

    # ── 4. Build and save log ────────────────────────────────────────
    candidate_log = []
    for c in result.all_candidates:
        candidate_log.append(
            {
                "hash": c.hash,
                "generation": c.generation,
                "island_id": c.island_id,
                "operation": c.operation,
                "parent_hashes": c.parent_hashes,
                "score": round(c.score, 2),
                "temperature": round(c.temperature, 4),
                "top_p": round(c.top_p, 4),
                "template": c.template,
            }
        )

    log = {
        "experiment": "Entity Classification: Static vs Evolved Prompt",
        "recipe": 51,
        "dataset": "holistic-ai/entity-classification-agentic-ai",
        "model": "llama3.2 (Ollama)",
        "entity_types": ENTITY_TYPES,
        "sample_sizes": {
            "evolution": len(evo_subset),
            "validation_eval": len(val_eval_subset),
            "test_eval": len(test_eval_subset),
        },
        "config": {
            "iterations": EVOLUTION_CONFIG.iterations,
            "population_size": EVOLUTION_CONFIG.population_size,
            "num_islands": EVOLUTION_CONFIG.num_islands,
            "elite_size": EVOLUTION_CONFIG.elite_size,
            "mutation_rate": EVOLUTION_CONFIG.mutation_rate,
            "crossover_rate": EVOLUTION_CONFIG.crossover_rate,
            "migration_interval": EVOLUTION_CONFIG.migration_interval,
        },
        "static_prompt": {
            "prompt": STATIC_PROMPT,
            "validation_accuracy": round(static_val["accuracy"], 2),
            "test_accuracy": round(static_test["accuracy"], 2),
            "per_class_validation": static_val["per_class"],
            "per_class_test": static_test["per_class"],
            "wall_time_s": round(static_time, 1),
        },
        "evolution": {
            "seed_prompts": SEED_TEMPLATES,
            "best_prompt": result.best_prompt,
            "best_score_evo_subset": round(result.best_score, 2),
            "best_temperature": round(result.best_temperature, 4),
            "best_top_p": round(result.best_top_p, 4),
            "validation_accuracy": round(evolved_val["accuracy"], 2),
            "test_accuracy": round(evolved_test["accuracy"], 2),
            "per_class_validation": evolved_val["per_class"],
            "per_class_test": evolved_test["per_class"],
            "search_time_s": round(evo_time_search, 1),
            "eval_time_s": round(evo_time_eval, 1),
            "total_candidates": len(result.all_candidates),
            "history": result.history,
            "all_candidates": candidate_log,
        },
        "comparison": {
            "validation_static": round(static_val["accuracy"], 2),
            "validation_evolved": round(evolved_val["accuracy"], 2),
            "validation_lift": round(val_lift, 2),
            "test_static": round(static_test["accuracy"], 2),
            "test_evolved": round(evolved_test["accuracy"], 2),
            "test_lift": round(test_lift, 2),
            "winner": winner,
        },
    }

    log_path = Path(__file__).resolve().parent.parent.parent / "entity_classification_evolution_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\n  Log saved → {log_path}")

    print(f"\n✓ Cookbook Recipe 51 complete.")


if __name__ == "__main__":
    main()
