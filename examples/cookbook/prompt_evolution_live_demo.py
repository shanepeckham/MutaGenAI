"""Live demo — evolve a smart-home routing prompt with the full toolbox.

This single recipe exercises every recent MutaGenAI feature at once:

  • Leaderboard seeds      — warm-start from a best-known agent-routing prompt
  • Bandit operators       — UCB picks mutation / crossover / LLM ops adaptively
  • Semantic LLM crossover  — merges the strengths of two strong parents
  • Response caching        — identical re-evaluations cost no tokens
  • Budget guardrails       — stop cleanly at a spend / call ceiling
  • Live dashboard (SSE)    — watch convergence + lineage update in real time
  • Quality diversity       — Pareto front + MAP-Elites printed at the end

Run it::

    uv run python examples/cookbook/prompt_evolution_live_demo.py

A browser tab opens with the live dashboard. The demo works even without an
LLM backend (it falls back to mock scoring) so the dashboard always streams;
start Ollama (``ollama serve`` + ``ollama pull llama3.2``) for real evolution.
"""

from __future__ import annotations

import os

from MutaGenAI import (
    EvalSample,
    LLMBackend,
    OperatorSelection,
    ProblemType,
    PromptEvolver,
    PromptEvolverConfig,
    Tool,
    build_map_elites,
    count_prompt_tokens,
    leaderboard_seeds,
    run_with_live_dashboard,
)

# ── The task: route a smart-home request to the right device tools ─────────

TOOLS = [
    Tool("set_thermostat", "Set the target temperature for a room",
         {"room": "string", "temperature": "int"}),
    Tool("set_lights", "Turn lights on/off or dim them in a room",
         {"room": "string", "state": "string", "brightness": "int"}),
    Tool("play_music", "Play music on a speaker",
         {"room": "string", "genre": "string"}),
    Tool("lock_door", "Lock or unlock a door",
         {"door": "string", "state": "string"}),
    Tool("arm_security", "Arm or disarm the security system",
         {"mode": "string"}),
    Tool("get_status", "Report the current status of a device",
         {"device": "string"}),
]

DATASET = [
    EvalSample("Make the living room 22 degrees", "set_thermostat",
               {"room": "living room", "temperature": "22"}),
    EvalSample("It's too cold in the bedroom, set it to 24", "set_thermostat",
               {"room": "bedroom", "temperature": "24"}),
    EvalSample("Turn off the kitchen lights", "set_lights",
               {"room": "kitchen", "state": "off"}),
    EvalSample("Dim the living room lights to 30%", "set_lights",
               {"room": "living room", "state": "on", "brightness": "30"}),
    EvalSample("Play some jazz in the office", "play_music",
               {"room": "office", "genre": "jazz"}),
    EvalSample("Start classical music in the bedroom", "play_music",
               {"room": "bedroom", "genre": "classical"}),
    EvalSample("Lock the front door", "lock_door",
               {"door": "front", "state": "lock"}),
    EvalSample("Unlock the back door please", "lock_door",
               {"door": "back", "state": "unlock"}),
    EvalSample("Arm the alarm, we're leaving", "arm_security",
               {"mode": "away"}),
    EvalSample("Disarm the security system", "arm_security",
               {"mode": "off"}),
    EvalSample("Is the front door locked?", "get_status",
               {"device": "front door"}),
    EvalSample("What's the thermostat set to?", "get_status",
               {"device": "thermostat"}),
    EvalSample("Warm up the kitchen to 21", "set_thermostat",
               {"room": "kitchen", "temperature": "21"}),
    EvalSample("Lights on in the garage", "set_lights",
               {"room": "garage", "state": "on"}),
    EvalSample("Put on rock music in the living room", "play_music",
               {"room": "living room", "genre": "rock"}),
    EvalSample("Lock the garage door", "lock_door",
               {"door": "garage", "state": "lock"}),
]


def main() -> None:
    # Warm-start from the leaderboard's agent-routing prompt, then add a
    # couple of contrasting seeds so the islands have crossover material.
    seeds = leaderboard_seeds("agent_routing") + [
        "Route the smart-home request to exactly one device tool. "
        "Return JSON: {\"tool\": \"...\", \"parameters\": {...}}.",
        "Pick the single tool that fulfils the request and extract its "
        "parameters from the user's words. Output JSON only.",
    ]

    config = PromptEvolverConfig(
        iterations=8,
        population_size=4,
        num_islands=3,
        elite_size=3,
        backend=LLMBackend.OLLAMA,           # falls back to mock if offline
        problem_type=ProblemType.TOOL_ROUTING,
        # Adaptive operator selection + semantic crossover
        operator_selection=OperatorSelection.UCB,
        llm_mutation_rate=0.3,
        llm_crossover_rate=0.4,
        crossover_rate=0.4,
        # Spend controls
        use_cache=True,
        max_calls=4000,
        budget_usd=2.0,
        cost_per_1k_input_tokens=0.0005,
        cost_per_1k_output_tokens=0.0015,
        max_concurrency=8,
    )

    evolver = PromptEvolver(
        tools=TOOLS,
        eval_dataset=DATASET,
        config=config,
        seed=42,
        verbose=True,
        seed_templates=seeds,
    )

    print("Opening the live dashboard… watch convergence + lineage stream in.")
    result, server = run_with_live_dashboard(evolver, open_browser=True)

    try:
        # ── Post-run quality-diversity + spend report ──────────────────────
        print("\n" + "=" * 64)
        print(result.summary().split("Best prompt")[0].rstrip())
        print("=" * 64)

        print("\nOperator bandit — what actually helped:")
        for arm, stats in (result.operator_stats or {}).items():
            print(f"  {arm:14} pulls={stats['count']:3d}  "
                  f"mean_reward={stats['mean_reward']:.3f}  "
                  f"share={stats['share']:.0%}")

        print("\nPareto front (accuracy vs. prompt length):")
        for cand in result.pareto_front()[:6]:
            print(f"  score={cand.score:5.1f}  "
                  f"tokens={count_prompt_tokens(cand.template):4d}  "
                  f"{cand.template[:60].replace(chr(10), ' ')}…")

        archive = build_map_elites(result.all_candidates)
        print(f"\nMAP-Elites — {archive.coverage} behaviour cells covered:")
        for rec in archive.to_json()[:6]:
            print(f"  [{rec['style']:13} · {rec['tokens']:4d} tok]  "
                  f"score={rec['score']:.1f}")

        print(f"\nSpend: {result.llm_calls} calls · "
              f"cache hits {result.cache_hits} · "
              f"est. ${result.estimated_cost_usd or 0:.4f} · "
              f"stop_reason={result.stop_reason}")

        print(f"\nDashboard still live at {server.url}")
        if os.environ.get("DEMO_AUTO_EXIT") != "1":
            input("Press Enter to stop the dashboard server and exit… ")
    finally:
        server.stop()


if __name__ == "__main__":
    main()
