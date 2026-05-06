#!/usr/bin/env python3
"""
Run 8 — Evolution-only (no static prompt), GPT-4.1, pop=8.
Uses the updated code with conditional {tool_schemas} injection.
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
)

from examples.experiments.entity_classification import (
    ENTITY_TYPES,
    ClassificationAccuracyScorer,
    evaluate_prompt,
    extract_entity,
    load_hf_dataset,
)

EVOLUTION_SAMPLE_SIZE = 40
VALIDATION_EVAL_SIZE = 200
TEST_EVAL_SIZE = 200

SEED_TEMPLATES = load_seed_templates("entity_classification")

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

# ── Constants ──────────────────────────────────────────────────────────────


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("Run 8 — Evolution Only (no static), GPT-4.1, pop=8")
    print("  conditional {tool_schemas} fix applied")
    print("=" * 70)

    val_data, test_data = load_hf_dataset()
    rng = np.random.default_rng(42)

    val_indices = rng.choice(len(val_data), size=min(VALIDATION_EVAL_SIZE, len(val_data)), replace=False)
    val_eval_subset = [val_data[int(i)] for i in val_indices]

    test_indices = rng.choice(len(test_data), size=min(TEST_EVAL_SIZE, len(test_data)), replace=False)
    test_eval_subset = [test_data[int(i)] for i in test_indices]

    evo_indices = rng.choice(len(val_data), size=min(EVOLUTION_SAMPLE_SIZE, len(val_data)), replace=False)
    evo_subset = [val_data[int(i)] for i in evo_indices]

    print(f"\nEvolution scoring subset:  {len(evo_subset)} samples")
    print(f"Validation eval subset:   {len(val_eval_subset)} samples")
    print(f"Test eval subset:         {len(test_eval_subset)} samples")

    label_map = {s["content"]: s["expected_entity"] for s in evo_subset}
    test_inputs = [s["content"] for s in evo_subset]

    llm_config = PromptEvolverConfig(backend=LLMBackend.AZURE_OPENAI, max_tokens=10)
    client = LLMClient(llm_config)

    if not client.is_available():
        print("\n⚠  Azure OpenAI not reachable — results will be random.")

    # ── Evolution ────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("EvoSim deep evolution (2 islands, NO forced {tool_schemas})")
    print("─" * 70)
    print(f"\nConfig: {EVOLUTION_CONFIG.iterations} gen × "
          f"{EVOLUTION_CONFIG.population_size} pop × "
          f"{EVOLUTION_CONFIG.num_islands} islands")
    print(f"\nSeed prompts ({len(SEED_TEMPLATES)}):")
    for i, s in enumerate(SEED_TEMPLATES):
        print(f"  [{i}] {s[:100]}...")

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

    # Check if {tool_schemas} leaked in
    if "{tool_schemas}" in result.best_prompt:
        print("\n  ⚠  WARNING: {tool_schemas} is present in evolved prompt!")
    else:
        print("\n  ✓  {tool_schemas} NOT present — fix is working")

    # ── Evaluate evolved winner ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("Evaluating evolved prompt on val/test subsets")
    print("─" * 70)

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
    print("\n  Per-class (validation):")
    for cls, acc in evolved_val["per_class"].items():
        print(f"    {cls:8s}: {acc:5.1f}%")
    print("\n  Per-class (test):")
    for cls, acc in evolved_test["per_class"].items():
        print(f"    {cls:8s}: {acc:5.1f}%")

    # ── Compare with Run 7 reference ─────────────────────────────────
    # Run 7 (200-sample): Static 83.0/87.0, Evolved 87.5/85.5
    # Full 1000-sample:   Static 86.0/86.0, Evolved 88.1/88.0
    print("\n" + "═" * 70)
    print("Run 8 vs Run 7 Reference (200-sample)")
    print("═" * 70)
    print(f"\n  {'':22s} {'Validation':>12s}  {'Test':>12s}")
    print(f"  {'─' * 50}")
    print(f"  {'Run 7 Static':22s} {'83.0':>11s}%  {'87.0':>11s}%")
    print(f"  {'Run 7 Evolved':22s} {'87.5':>11s}%  {'85.5':>11s}%")
    print(f"  {'Run 8 Evolved (no TS)':22s} {evolved_val['accuracy']:11.1f}%  {evolved_test['accuracy']:11.1f}%")

    # ── Save log ─────────────────────────────────────────────────────
    candidate_log = []
    for c in result.all_candidates:
        candidate_log.append({
            "hash": c.hash,
            "generation": c.generation,
            "island_id": c.island_id,
            "operation": c.operation,
            "parent_hashes": c.parent_hashes,
            "score": round(c.score, 2),
            "temperature": round(c.temperature, 4),
            "top_p": round(c.top_p, 4),
            "template": c.template,
        })

    log = {
        "experiment": "Run 8 — Evolution Only, no forced {tool_schemas}",
        "model": "gpt-4.1 (Azure OpenAI)",
        "dataset": "holistic-ai/entity-classification-agentic-ai",
        "entity_types": ENTITY_TYPES,
        "tool_schemas_forced": False,
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
            "tool_schemas_in_best": "{tool_schemas}" in result.best_prompt,
        },
    }

    log_path = Path(_root) / "run8_evolution_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\n  Log saved → {log_path}")
    print(f"\n✓ Run 8 complete.")


if __name__ == "__main__":
    main()
