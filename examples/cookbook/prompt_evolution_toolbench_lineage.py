#!/usr/bin/env python3
"""
ToolBench G1 — Adaptive Mutation with Lineage Tracking
=======================================================

Reruns ToolBench G1 prompt evolution with:
- Adaptive mutations enabled
- Population size 8
- Full lineage tracking via ``result.lineage_json()``

Outputs:
- ``logs/toolbench_g1_lineage.json``      — lineage tree for the visualiser
- ``logs/toolbench_g1_lineage_log.json``   — experiment results log

Usage::

    uv run python examples/cookbook/prompt_evolution_toolbench_lineage.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from MutaGenAI import (
    LLMBackend,
    LLMClient,
    PromptCandidate,
    PromptEvolver,
    PromptEvolverConfig,
)
from MutaGenAI.prompt_evolver import EvalSample, Tool
import MutaGenAI.prompt_evolver as _pe

# Re-use helpers from the existing ToolBench recipe
from prompt_evolution_toolbench import (
    _format_tool_list,
    _TOOLBENCH_DEFAULT_PROMPT,
    _TOOLBENCH_SEED_TEMPLATES,
    evaluate_baseline,
    load_toolbench_dataset,
    score_toolbench_case,
    ToolBenchCase,
)

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")


class ToolBenchEvolver(PromptEvolver):
    """PromptEvolver subclass that scores via ToolBench API matching."""

    def __init__(
        self,
        config: PromptEvolverConfig,
        eval_cases: list[ToolBenchCase],
        client: LLMClient,
    ) -> None:
        # Pass empty tools/eval_dataset — we override _evaluate_candidate
        super().__init__(tools=[], eval_dataset=[], config=config)
        self._tb_cases = eval_cases
        self._tb_client = client
        self._tb_rng = np.random.default_rng(42)
        # Use the caller's pre-existing client instead of creating a new one
        self._client = client

    def _evaluate_candidate(self, candidate: PromptCandidate) -> float:
        total = 0.0
        for case in self._tb_cases:
            fn_text = _format_tool_list(case.api_list)
            # ToolBench templates use {toolbench_apis}; core seeds use
            # {tool_schemas}. Support both.
            sys_prompt = candidate.template.replace(
                "{tool_schemas}", fn_text
            ).replace("{toolbench_apis}", fn_text)
            resp = self._tb_client.complete(
                system_prompt=sys_prompt,
                user_message=case.query,
                temperature=candidate.temperature,
                top_p=candidate.top_p,
            )
            if resp is None:
                total += float(self._tb_rng.uniform(0, 0.3))
                continue
            total += score_toolbench_case(resp, case)
        n = len(self._tb_cases)
        return (total / n * 100.0) if n else 0.0


def main() -> None:
    banner = r"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║  MutagenAI × ToolBench G1 — Adaptive Mutation + Lineage        ║
    ║  Population 8 · Adaptive mutations · Full lineage tracking      ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

    # ── Backend ────────────────────────────────────────────────────────
    base_cfg = PromptEvolverConfig(
        backend=LLMBackend.OLLAMA,
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        timeout=60.0,
    )
    client = LLMClient(base_cfg)

    if not client.is_available():
        print("  ⚠ Ollama not reachable at localhost:11434")
        print("    Start it with: ollama serve")
        return

    # ── Load G1 data ──────────────────────────────────────────────────
    print("  Loading ToolBench G1 instruction data...")
    by_split = load_toolbench_dataset(max_per_split=20, splits=["g1_instruction"])
    cases = by_split.get("g1_instruction", [])
    if not cases:
        print("  ✗ No G1 cases loaded.")
        return
    multi = sum(1 for c in cases if c.is_multi_tool)
    print(f"    g1_instruction: {len(cases)} cases "
          f"({multi} multi-tool, {len(cases) - multi} single-tool)")

    # ── Baseline ──────────────────────────────────────────────────────
    print("\n  Phase 1: Default prompt baseline")
    print("  " + "─" * 50)
    default_score = evaluate_baseline(cases, "g1_instruction", client)

    # ── Subsample eval cases ──────────────────────────────────────────
    rng = np.random.default_rng(42)
    eval_size = min(12, len(cases))
    idx = rng.choice(len(cases), size=eval_size, replace=False)
    eval_cases = [cases[int(i)] for i in idx]

    # ── Config ────────────────────────────────────────────────────────
    config = PromptEvolverConfig(
        iterations=4,
        population_size=8,
        num_islands=2,
        elite_size=5,
        mutation_rate=0.6,
        crossover_rate=0.3,
        eval_sample_size=12,
        adaptive_mutations=True,
        llm_mutation_rate=0.3,
        backend=LLMBackend.OLLAMA,
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        timeout=60.0,
    )

    # ── Inject ToolBench seed templates into the module ────────────────
    original_seeds = list(_pe._SEED_TEMPLATES)
    _pe._SEED_TEMPLATES = list(_TOOLBENCH_SEED_TEMPLATES)

    # ── Evolution ─────────────────────────────────────────────────────
    print("\n  Phase 2: Adaptive Evolution (pop=8, adaptive_mutations=True)")
    print("  " + "─" * 50)

    t0 = time.perf_counter()
    evolver = ToolBenchEvolver(config, eval_cases, client)
    result = evolver.run()
    wall_time = time.perf_counter() - t0

    # Restore original seeds
    _pe._SEED_TEMPLATES = original_seeds

    best_prompt = result.best_prompt

    print(f"\n  {'═' * 60}")
    print(f"  ⭐ RESULTS")
    print(f"  {'═' * 60}")
    print(f"  Default baseline:  {default_score:.1f}%")
    print(f"  Evolved best:      {result.best_score:.1f}%")
    print(f"  Delta:            {result.best_score - default_score:+.1f} pp")
    print(f"  Wall time:         {wall_time:.1f}s")
    print(f"  Total candidates:  {len(result.all_candidates)}")
    print(f"\n  Best prompt (temperature={result.best_temperature:.4f}, "
          f"top_p={result.best_top_p:.4f}):")
    print(f"  {'─' * 60}")
    for line in best_prompt.split("\n"):
        print(f"    {line}")

    # ── Export lineage ────────────────────────────────────────────────
    lineage = result.lineage_json()
    os.makedirs(LOG_DIR, exist_ok=True)
    lineage_path = os.path.join(LOG_DIR, "toolbench_g1_lineage.json")
    with open(lineage_path, "w") as f:
        json.dump(lineage, f, indent=2)
    print(f"\n  ✓ Lineage data saved to {lineage_path}")
    print(f"    ({len(lineage)} candidates — load in docs/lineage_tree.html)")

    # ── Experiment log ────────────────────────────────────────────────
    log_path = os.path.join(LOG_DIR, "toolbench_g1_lineage_log.json")
    log: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "benchmark": "toolbench_g1_instruction",
        "backend": "ollama",
        "model": os.environ.get("OLLAMA_MODEL", "llama3.2"),
        "config": {
            "iterations": config.iterations,
            "population_size": config.population_size,
            "num_islands": config.num_islands,
            "elite_size": config.elite_size,
            "mutation_rate": config.mutation_rate,
            "crossover_rate": config.crossover_rate,
            "adaptive_mutations": config.adaptive_mutations,
            "llm_mutation_rate": config.llm_mutation_rate,
        },
        "default_baseline": round(default_score, 2),
        "evolved_score": round(result.best_score, 2),
        "delta": round(result.best_score - default_score, 2),
        "best_temperature": round(result.best_temperature, 4),
        "best_top_p": round(result.best_top_p, 4),
        "wall_time": round(wall_time, 1),
        "total_candidates": len(lineage),
        "best_prompt": best_prompt,
        "history": [
            {"generation": g, "best_score": s}
            for g, s in result.history
        ],
    }
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  ✓ Experiment log saved to {log_path}")


if __name__ == "__main__":
    main()
