#!/usr/bin/env python3
"""Quick validation of token optimization with Ollama.

Runs a tiny evolution (2 generations, pop 4, 1 island, 5 eval turns)
to confirm the token minimization logic works end-to-end.

Usage:
    cd /path/to/Prompture
    MUTAGENAI_BACKEND=ollama MUTAGENAI_MINIMIZE_TOKENS=1 \
        uv run python examples/cookbook/browsergym/validate_token_opt.py
"""
from __future__ import annotations

import os
import sys
import time

# Ensure repo root on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ["PYTHONUNBUFFERED"] = "1"
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# Force Ollama backend and token optimization ON
os.environ.setdefault("MUTAGENAI_BACKEND", "ollama")
os.environ.setdefault("MUTAGENAI_MINIMIZE_TOKENS", "1")
os.environ.setdefault("MUTAGENAI_TOKEN_WEIGHT", "0.15")
os.environ.setdefault("MUTAGENAI_MAX_PROMPT_TOKENS", "500")
os.environ.setdefault("OLLAMA_MODEL", "qwen3:8b")

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env"))

# Now import the cookbook module components
from prompt_evolution_browsergym import (
    BACKEND,
    MODEL,
    MINIMIZE_TOKENS,
    TOKEN_WEIGHT,
    MAX_PROMPT_TOKENS,
    _count_prompt_tokens,
    _DEFAULT_PROMPT,
    BrowserGymEvolver,
    BrowsingTurn,
    SEED_TEMPLATES,
    DOMAIN_MUTATIONS,
    evaluate_baseline,
    load_browsing_turns,
    format_browsing_context,
)
from MutaGenAI.prompt_evolver import (
    LLMBackend,
    LLMClient,
    PromptEvolverConfig,
)
import MutaGenAI.prompt_evolver as _pe


def main() -> None:
    print("=" * 60)
    print("  Token Optimization Validation (Ollama)")
    print("=" * 60)
    print(f"  Backend:          {BACKEND.value}")
    print(f"  Model:            {MODEL}")
    print(f"  MINIMIZE_TOKENS:  {MINIMIZE_TOKENS}")
    print(f"  TOKEN_WEIGHT:     {TOKEN_WEIGHT}")
    print(f"  MAX_PROMPT_TOKENS:{MAX_PROMPT_TOKENS}")
    print()

    # Token counting sanity check
    baseline_tokens = _count_prompt_tokens(_DEFAULT_PROMPT)
    print(f"  Baseline prompt tokens: {baseline_tokens}")
    print(f"  Baseline prompt length: {len(_DEFAULT_PROMPT)} chars")

    # A deliberately verbose prompt (should be penalized)
    verbose_prompt = (
        "You are a highly capable web browsing agent with extensive expertise "
        "in navigating and interacting with web pages. Your primary mission is "
        "to assist the user by performing the exact browser action that matches "
        "their stated goal. You should carefully analyze the accessibility tree, "
        "identify the correct target element using its unique identifier, and "
        "produce the correct action. Always reason step by step before acting.\n\n"
        "Output your action as: ACTION: action_type(key=\"value\")\n\n"
        "Valid actions: click(uid=\"...\"), text_input(text=\"...\", uid=\"...\"), "
        "say(speaker=\"navigator\", utterance=\"...\"), scroll(x=0, y=N), "
        "submit(uid=\"...\"), load(url=\"...\")\n\n"
        "Important guidelines:\n"
        "1. Always prefer clicking over scrolling\n"
        "2. Only use text_input when the user explicitly requests typing\n"
        "3. Use say() for communicating with the user\n"
        "4. Consider the full context before deciding\n"
        "5. Match elements by their uid attribute carefully\n"
    )
    verbose_tokens = _count_prompt_tokens(verbose_prompt)
    print(f"  Verbose prompt tokens:  {verbose_tokens}")
    print()

    # Demonstrate the scoring difference
    print("  Scoring impact (assuming same accuracy of 60%):")
    raw_accuracy = 60.0
    for label, tokens in [("baseline", baseline_tokens), ("verbose", verbose_tokens)]:
        length_ratio = min(tokens / MAX_PROMPT_TOKENS, 1.0)
        brevity_bonus = (1.0 - length_ratio) * 100.0
        blended = raw_accuracy * (1 - TOKEN_WEIGHT) + brevity_bonus * TOKEN_WEIGHT
        print(f"    {label:10s}: tokens={tokens:3d}  brevity_bonus={brevity_bonus:5.1f}  "
              f"blended_score={blended:.1f}  (vs raw {raw_accuracy:.1f})")
    print()

    # Load a small dataset
    print("  Loading dataset (5 demos, 3 turns/demo)...")
    by_category = load_browsing_turns(max_demos=5, max_turns_per_demo=3)
    all_turns: list[BrowsingTurn] = []
    for turns in by_category.values():
        all_turns.extend(turns)
    print(f"    Loaded {len(all_turns)} turns")
    print()

    # Check Ollama connectivity
    if BACKEND == LLMBackend.OLLAMA:
        cfg = PromptEvolverConfig(
            backend=BACKEND,
            ollama_model=MODEL,
            timeout=120.0,
        )
    else:
        cfg = PromptEvolverConfig(
            backend=BACKEND,
            azure_deployment=MODEL,
            azure_use_rbac=True,
            timeout=60.0,
        )
    client = LLMClient(cfg)

    print("  Checking LLM backend...", end=" ", flush=True)
    if client.is_available():
        print("OK")
    else:
        print("UNAVAILABLE — cannot validate, exiting.")
        return

    # Quick baseline
    print("\n  Phase 1: Baseline (5 turns only)")
    print("  " + "-" * 50)
    baseline_score, _ = evaluate_baseline(all_turns[:5], _DEFAULT_PROMPT, client)
    print(f"    Baseline score: {baseline_score:.1f}%")
    print(f"    Baseline tokens: {baseline_tokens}")
    print()

    # Run tiny evolution (2 generations, pop 4, 1 island)
    print("  Phase 2: Mini evolution (2 gen, pop 4, 1 island)")
    print("  " + "-" * 50)

    evo_cfg = PromptEvolverConfig(
        iterations=2,
        population_size=4,
        num_islands=1,
        elite_size=2,
        mutation_rate=0.7,
        crossover_rate=0.3,
        eval_sample_size=5,
        adaptive_mutations=False,
        llm_mutation_rate=0.0,
        refine_after_splice=False,
        describe_entities=False,
        backend=BACKEND,
        **({"ollama_model": MODEL} if BACKEND == LLMBackend.OLLAMA else {
            "azure_deployment": MODEL,
            "azure_use_rbac": True,
        }),
        timeout=120.0,
    )

    # Inject seeds
    original_seeds = list(_pe._SEED_TEMPLATES)
    _pe._SEED_TEMPLATES = list(SEED_TEMPLATES)

    t0 = time.perf_counter()
    evolver = BrowserGymEvolver(evo_cfg, all_turns[:10], client)
    result = evolver.run()
    wall_time = time.perf_counter() - t0

    _pe._SEED_TEMPLATES = original_seeds

    best_prompt = result.best_prompt
    evolved_tokens = _count_prompt_tokens(best_prompt)

    print(f"\n    Best score:     {result.best_score:.1f}%")
    print(f"    Evolved tokens: {evolved_tokens}")
    print(f"    Token delta:    {evolved_tokens - baseline_tokens:+d}")
    print(f"    Wall time:      {wall_time:.1f}s")
    print(f"    Candidates:     {len(result.all_candidates)}")
    print()

    # Show all candidate scores and token counts
    print("  All candidates (score, tokens):")
    for c in sorted(result.all_candidates, key=lambda x: x.score, reverse=True)[:8]:
        c_tokens = _count_prompt_tokens(c.template)
        print(f"    score={c.score:5.1f}%  tokens={c_tokens:3d}  "
              f"op={c.operation:12s}  gen={c.generation}")

    print()
    print("  " + "=" * 60)
    print("  VALIDATION COMPLETE")
    print("  " + "=" * 60)
    print(f"    Token optimization: {'ENABLED' if MINIMIZE_TOKENS else 'DISABLED'}")
    print(f"    Baseline tokens:    {baseline_tokens}")
    print(f"    Evolved tokens:     {evolved_tokens} ({evolved_tokens - baseline_tokens:+d})")
    token_saving = (1 - evolved_tokens / max(1, baseline_tokens)) * 100
    print(f"    Token saving:       {token_saving:+.1f}%")
    print()
    print("  Best evolved prompt:")
    print("  " + "-" * 60)
    for line in best_prompt.split("\n"):
        print(f"    {line}")
    print("  " + "-" * 60)


if __name__ == "__main__":
    main()
