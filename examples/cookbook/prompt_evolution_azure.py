#!/usr/bin/env python3
"""
Cookbook Recipe 40 — Azure OpenAI vs Ollama Prompt Evolution Comparison
=======================================================================

Runs the same baseline + PromptEvolver experiment from Recipe 39 against
both **Ollama** (local) and **Azure OpenAI** (cloud), then prints a
side-by-side comparison table.

Authentication
--------------
RBAC (recommended, no API key needed)::

    # 1. Log in to Azure
    az login

    # 2. Assign yourself the OpenAI User role (one-time)
    az role assignment create \\
      --assignee <your-object-id> \\
      --role "Cognitive Services OpenAI User" \\
      --scope /subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<resource>

    # 3. Set these in .env (see .env.example)
    AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
    AZURE_OPENAI_DEPLOYMENT=<deployment>

API key auth (alternative)::

    AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com
    AZURE_OPENAI_DEPLOYMENT=<deployment>
    AZURE_OPENAI_API_KEY=<key>
    AZURE_OPENAI_USE_RBAC=false

Usage::

    uv sync --extra llm   # installs httpx + azure-identity
    uv run python examples/cookbook/prompt_evolution_azure.py
"""
from __future__ import annotations

import os
import sys
import time

# Ensure repo root is on path for dev installs
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

# Load .env before anything reads os.environ
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from MutaGenAI.prompt_evolver import (
    EvalSample,
    LLMBackend,
    LLMClient,
    PromptEvolver,
    PromptEvolverConfig,
    Tool,
    parse_tool_response,
    score_response,
)


# ── Shared fixtures ──────────────────────────────────────────────────────

TOOLS = [
    Tool("get_weather", "Get current weather for a location",
         {"location": "string"}),
    Tool("send_email", "Compose and send an email",
         {"to": "string", "subject": "string"}),
    Tool("schedule_meeting", "Schedule a calendar meeting",
         {"title": "string", "date": "string"}),
    Tool("translate_text", "Translate text to another language",
         {"text": "string", "target_language": "string"}),
    Tool("calculate", "Evaluate a mathematical expression",
         {"expression": "string"}),
    Tool("search_web", "Search the internet for information",
         {"query": "string"}),
]

DATASET = [
    # get_weather
    EvalSample("What's the weather like in London?", "get_weather",
               {"location": "London"}),
    EvalSample("Is it going to rain in Tokyo?", "get_weather",
               {"location": "Tokyo"}),
    EvalSample("Temperature in New York right now?", "get_weather",
               {"location": "New York"}),
    EvalSample("What's the forecast for Berlin this weekend?", "get_weather",
               {"location": "Berlin"}),
    # send_email
    EvalSample("Send an email to bob@acme.com about the project",
               "send_email", {"to": "bob@acme.com", "subject": "project"}),
    EvalSample("Email Alice the meeting notes", "send_email",
               {"to": "Alice"}),
    EvalSample("Write a message to the team about deadlines", "send_email",
               {"subject": "deadlines"}),
    EvalSample("Drop a note to carol@company.io confirming the call",
               "send_email", {"to": "carol@company.io"}),
    # schedule_meeting
    EvalSample("Set up a meeting with design on Tuesday at 2pm",
               "schedule_meeting", {"date": "Tuesday at 2pm"}),
    EvalSample("Book a standup every Monday morning",
               "schedule_meeting", {"title": "standup"}),
    EvalSample("Arrange a sync with engineering for tomorrow",
               "schedule_meeting", {"date": "tomorrow"}),
    EvalSample("Schedule a brainstorm for next Friday",
               "schedule_meeting", {"date": "next Friday"}),
    # translate_text
    EvalSample("Translate 'hello world' to Spanish", "translate_text",
               {"text": "hello world", "target_language": "Spanish"}),
    EvalSample("How do you say 'thank you' in Japanese?", "translate_text",
               {"text": "thank you", "target_language": "Japanese"}),
    EvalSample("Convert this to French: 'Where is the station?'",
               "translate_text",
               {"text": "Where is the station?", "target_language": "French"}),
    EvalSample("Say 'goodbye' in German", "translate_text",
               {"text": "goodbye", "target_language": "German"}),
    # calculate
    EvalSample("What is 15% of 230?", "calculate",
               {"expression": "0.15 * 230"}),
    EvalSample("Calculate 42 times 18", "calculate",
               {"expression": "42 * 18"}),
    EvalSample("What's the square root of 144?", "calculate",
               {"expression": "sqrt(144)"}),
    EvalSample("Add 567 and 433", "calculate",
               {"expression": "567 + 433"}),
    # search_web
    EvalSample("Search for the latest Python release notes", "search_web",
               {"query": "latest Python release notes"}),
    EvalSample("Look up information about quantum computing", "search_web",
               {"query": "quantum computing"}),
    EvalSample("Find articles about climate change", "search_web",
               {"query": "climate change"}),
    EvalSample("What is the tallest building in the world?", "search_web",
               {"query": "tallest building in the world"}),
]


def build_baselines(tool_schemas: str) -> dict[str, tuple[str, float, float]]:
    """Return the five baseline prompts (name → prompt, temp, top_p)."""
    return {
        "Naive (tool list only)": (
            f"You are a helpful assistant.\n\n"
            f"Available tools:\n{tool_schemas}",
            0.7, 0.95,
        ),
        "Minimal JSON instruction": (
            f"Pick a tool for the user's request. "
            f"Respond with JSON: {{\"tool\": \"<name>\", \"parameters\": {{...}}}}\n\n"
            f"Tools:\n{tool_schemas}",
            0.3, 0.9,
        ),
        "Verbose (kitchen sink)": (
            f"You are an AI assistant with access to the following tools. "
            f"The user will give you a natural language query. You should "
            f"carefully read the query, think about which tool is most "
            f"appropriate, consider all possible options, weigh the pros "
            f"and cons, and then select the single best tool. Extract any "
            f"relevant parameter values from the query. If you cannot "
            f"determine a parameter value, leave it empty. Return your "
            f"answer as a JSON object with 'tool' and 'parameters' keys. "
            f"Do not return anything else.\n\n"
            f"Tools:\n{tool_schemas}",
            0.5, 0.95,
        ),
        "High temperature (creative)": (
            f"Route the user's request to the correct tool.\n\n"
            f"Tools:\n{tool_schemas}\n\n"
            f"Respond with JSON: {{\"tool\": \"<name>\", \"parameters\": {{...}}}}",
            0.9, 0.99,
        ),
        "Zero temperature (greedy)": (
            f"Route the user's request to the correct tool.\n\n"
            f"Tools:\n{tool_schemas}\n\n"
            f"Respond with JSON: {{\"tool\": \"<name>\", \"parameters\": {{...}}}}",
            0.0, 1.0,
        ),
    }


def evaluate_baselines(
    baselines: dict[str, tuple[str, float, float]],
    client: LLMClient,
    tool_names: list[str],
) -> dict[str, float]:
    """Evaluate each baseline and return {name: accuracy%}."""
    scores: dict[str, float] = {}
    for name, (prompt, temp, top_p) in baselines.items():
        total = 0.0
        for sample in DATASET:
            response = client.complete(
                system_prompt=prompt,
                user_message=sample.query,
                temperature=temp,
                top_p=top_p,
            )
            if response is None:
                continue
            pred_tool, pred_params = parse_tool_response(response, tool_names)
            total += score_response(
                pred_tool, pred_params,
                sample.expected_tool, sample.expected_params,
            )
        accuracy = (total / len(DATASET)) * 100 if DATASET else 0.0
        scores[name] = accuracy
        print(f"    {name:<30s}  {accuracy:5.1f}%  "
              f"temp={temp:.1f}  top_p={top_p:.2f}")
    return scores


def run_evolver(backend: LLMBackend) -> tuple[dict[str, float], float, float]:
    """Run baselines + evolution for one backend.

    Returns (baseline_scores, evolved_score, wall_time).
    """
    config = PromptEvolverConfig(backend=backend)
    client = LLMClient(config)

    if not client.is_available():
        print(f"  ⚠  {backend.value} is not available — skipping\n")
        return {}, 0.0, 0.0

    tool_schemas = "\n".join(f"  - {t.schema_str()}" for t in TOOLS)
    tool_names = [t.name for t in TOOLS]
    baselines = build_baselines(tool_schemas)

    print(f"  Baselines ({backend.value}):")
    baseline_scores = evaluate_baselines(baselines, client, tool_names)

    best_bl_name = max(baseline_scores, key=baseline_scores.get) if baseline_scores else "N/A"
    best_bl_score = max(baseline_scores.values()) if baseline_scores else 0.0
    print(f"    Best baseline: {best_bl_name} ({best_bl_score:.1f}%)\n")

    evo_config = PromptEvolverConfig(
        iterations=5,
        population_size=4,
        num_islands=2,
        elite_size=3,
        mutation_rate=0.6,
        crossover_rate=0.3,
        eval_sample_size=6,
        backend=backend,
    )

    print(f"  PromptEvolver ({backend.value}, {evo_config.iterations} gens):")
    t0 = time.perf_counter()
    evolver = PromptEvolver(
        tools=TOOLS,
        eval_dataset=DATASET,
        config=evo_config,
        seed=42,
        verbose=True,
    )
    result = evolver.run()
    wall = time.perf_counter() - t0
    print(f"    Evolved best: {result.best_score:.1f}%  "
          f"(temp={result.best_temperature:.3f}, top_p={result.best_top_p:.3f})")
    print(f"    Wall time: {wall:.1f}s\n")

    return baseline_scores, result.best_score, wall


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    print("EvoSim Cookbook Recipe 40 — Azure OpenAI vs Ollama Comparison")
    print("=" * 62)
    print(f"  Tools:   {len(TOOLS)}")
    print(f"  Dataset: {len(DATASET)} samples")

    # Show Azure config (redacted)
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "")
    has_key = bool(os.environ.get("AZURE_OPENAI_API_KEY", ""))
    use_rbac = os.environ.get("AZURE_OPENAI_USE_RBAC", "true").lower() not in ("0", "false", "no")
    auth_mode = "RBAC (DefaultAzureCredential)" if use_rbac else "API key"
    print(f"  Azure endpoint:   {endpoint or '(not set)'}")
    print(f"  Azure deployment: {deployment or '(not set)'}")
    print(f"  Azure auth:       {auth_mode}")
    if not use_rbac:
        print(f"  Azure key set:    {has_key}")
    print()

    results: dict[str, tuple[dict[str, float], float, float]] = {}

    # ── Run Ollama ───────────────────────────────────────────────────────
    print("─" * 62)
    print("  Ollama (local)")
    print("─" * 62)
    results["Ollama"] = run_evolver(LLMBackend.OLLAMA)

    # ── Run Azure OpenAI ─────────────────────────────────────────────────
    print("─" * 62)
    print("  Azure OpenAI (cloud)")
    print("─" * 62)
    results["Azure OpenAI"] = run_evolver(LLMBackend.AZURE_OPENAI)

    # ── Side-by-side comparison table ────────────────────────────────────
    print("=" * 62)
    print("  Side-by-Side Comparison")
    print("=" * 62)

    ollama_bl, ollama_evo, ollama_time = results.get("Ollama", ({}, 0.0, 0.0))
    azure_bl, azure_evo, azure_time = results.get("Azure OpenAI", ({}, 0.0, 0.0))

    all_bl_names = list(dict.fromkeys(list(ollama_bl.keys()) + list(azure_bl.keys())))

    print(f"  {'Prompt':<32s} {'Ollama':>8s} {'Azure':>8s} {'Delta':>8s}")
    print(f"  {'-' * 32} {'-' * 8} {'-' * 8} {'-' * 8}")

    for name in all_bl_names:
        o = ollama_bl.get(name, 0.0)
        a = azure_bl.get(name, 0.0)
        d = a - o
        print(f"  {name:<32s} {o:>7.1f}% {a:>7.1f}% {d:>+7.1f}%")

    print(f"  {'-' * 32} {'-' * 8} {'-' * 8} {'-' * 8}")
    d_evo = azure_evo - ollama_evo
    print(f"  {'EVOLVED (EvoSim)':<32s} {ollama_evo:>7.1f}% "
          f"{azure_evo:>7.1f}% {d_evo:>+7.1f}%")

    print()
    print(f"  Wall time:  Ollama={ollama_time:.0f}s  "
          f"Azure={azure_time:.0f}s")

    # Winner
    if azure_evo > ollama_evo:
        print(f"\n  Azure OpenAI evolved prompt wins by "
              f"+{azure_evo - ollama_evo:.1f}%")
    elif ollama_evo > azure_evo:
        print(f"\n  Ollama evolved prompt wins by "
              f"+{ollama_evo - azure_evo:.1f}%")
    else:
        print("\n  Both backends tied on evolved prompt accuracy")

    print("\n✓ Cookbook Recipe 40 complete.")


if __name__ == "__main__":
    main()
