#!/usr/bin/env python3
"""
Run 10 — Evolution with adaptive mutations + LLM-assisted mutation, GPT-4.1, pop=8.

New features enabled:
  - adaptive_mutations=True  → error-guided mutation snippets targeting weak classes
  - llm_mutation_rate=0.3    → 30% chance of LLM-based prompt rewrite using failures
  - ProblemType.CLASSIFICATION
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
    ProblemType,
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
    problem_type=ProblemType.CLASSIFICATION,
    adaptive_mutations=True,
    llm_mutation_rate=0.3,
)

# ── Constants ──────────────────────────────────────────────────────────────


def extract_category(text: str, mode: str) -> str | None:
    """Extract category callback for adaptive mutations.

    When mode='expected', the text is the original test_input —
    we look it up in the global label_map.
    When mode='predicted', the text is the LLM output —
    we parse the entity from it.
    """
    if mode == "expected":
        return _LABEL_MAP.get(text)
    return extract_entity(text) or None


# Module-level reference set by main() before evolution starts
_LABEL_MAP: dict[str, str] = {}


# ── Main ───────────────────────────────────────────────────────────────────


def main():
    global _LABEL_MAP

    print("=" * 70)
    print("Run 10 — Adaptive Mutations + LLM Mutation, GPT-4.1, pop=8")
    print(f"  problem_type       = {EVOLUTION_CONFIG.problem_type.value}")
    print(f"  adaptive_mutations = {EVOLUTION_CONFIG.adaptive_mutations}")
    print(f"  llm_mutation_rate  = {EVOLUTION_CONFIG.llm_mutation_rate}")
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

    _LABEL_MAP = {s["content"]: s["expected_entity"] for s in evo_subset}
    test_inputs = [s["content"] for s in evo_subset]

    llm_config = PromptEvolverConfig(backend=LLMBackend.AZURE_OPENAI, max_tokens=10)
    client = LLMClient(llm_config)

    if not client.is_available():
        print("\n⚠  Azure OpenAI not reachable — results will be random.")

    # ── Evolution ────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("EvoSim evolution with ADAPTIVE + LLM mutations")
    print("─" * 70)
    print(f"\nConfig: {EVOLUTION_CONFIG.iterations} gen × "
          f"{EVOLUTION_CONFIG.population_size} pop × "
          f"{EVOLUTION_CONFIG.num_islands} islands")
    print(f"Problem type: {EVOLUTION_CONFIG.problem_type.value}")
    print(f"Adaptive mutations: {EVOLUTION_CONFIG.adaptive_mutations}")
    print(f"LLM mutation rate: {EVOLUTION_CONFIG.llm_mutation_rate}")
    print(f"\nSeed prompts ({len(SEED_TEMPLATES)}):")
    for i, s in enumerate(SEED_TEMPLATES):
        print(f"  [{i}] {s[:100]}...")

    scorer = ClassificationAccuracyScorer(_LABEL_MAP)

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
        extract_category=extract_category,
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

    # Check for tool-routing pollution
    tool_routing_markers = ["{tool_schemas}", "tool_name", "parameters", "JSON object"]
    leaks = [m for m in tool_routing_markers if m.lower() in result.best_prompt.lower()]
    if leaks:
        print(f"\n  ⚠  WARNING: Tool-routing language detected: {leaks}")
    else:
        print("\n  ✓  No tool-routing language in evolved prompt")

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

    # ── Compare with prior runs ──────────────────────────────────────
    print("\n" + "═" * 70)
    print("Run 10 vs Prior Runs (200-sample)")
    print("═" * 70)
    print(f"\n  {'':35s} {'Validation':>12s}  {'Test':>12s}")
    print(f"  {'─' * 63}")
    print(f"  {'Run 7 Evolved (tool muts)':35s} {'87.5':>11s}%  {'85.5':>11s}%")
    print(f"  {'Run 9 Evolved (class muts)':35s} {'89.0':>11s}%  {'86.5':>11s}%")
    print(f"  {'Run 10 Evolved (adaptive+LLM)':35s} {evolved_val['accuracy']:11.1f}%  {evolved_test['accuracy']:11.1f}%")

    # ── Save log ─────────────────────────────────────────────────────
    # Inspect error profile from the evolver
    error_profile_data = {}
    if hasattr(evolver, "_error_profile"):
        ep = evolver._error_profile
        error_profile_data = {
            "total": dict(ep.total),
            "errors": dict(ep.errors),
            "worst_categories": [
                {"category": c, "error_rate": round(r, 4)}
                for c, r in ep.worst_categories(top_k=6)
            ],
        }

    # Count operations in candidates
    operation_counts: dict[str, int] = {}
    for c in result.all_candidates:
        op = c.operation or "unknown"
        operation_counts[op] = operation_counts.get(op, 0) + 1

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
        "experiment": "Run 10 — Adaptive Mutations + LLM Mutation, GPT-4.1, pop=8",
        "model": "gpt-4.1 (Azure OpenAI)",
        "dataset": "holistic-ai/entity-classification-agentic-ai",
        "entity_types": ENTITY_TYPES,
        "problem_type": EVOLUTION_CONFIG.problem_type.value,
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
            "problem_type": EVOLUTION_CONFIG.problem_type.value,
            "adaptive_mutations": EVOLUTION_CONFIG.adaptive_mutations,
            "llm_mutation_rate": EVOLUTION_CONFIG.llm_mutation_rate,
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
            "operation_counts": operation_counts,
            "error_profile_final_gen": error_profile_data,
            "history": result.history,
            "all_candidates": candidate_log,
            "tool_schemas_in_best": "{tool_schemas}" in result.best_prompt,
        },
    }

    log_path = Path(_root) / "run10_adaptive_log.json"
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\n  Log saved → {log_path}")
    print(f"\n✓ Run 10 complete.")


if __name__ == "__main__":
    main()
