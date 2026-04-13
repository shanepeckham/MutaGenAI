#!/usr/bin/env python3
"""
Cookbook Recipe 41 — Agentic Prompt Evolution Across Workloads
=============================================================

Evolves system prompts for **three diverse agentic workloads** using both
Ollama (local) and Azure OpenAI (cloud), compares against hand-crafted
baselines, and **prints the full prompt evolution trace** so you can see
how the prompt text mutates across generations.

Scenarios
---------
1. **Customer-Support Triage** — 9 overlapping tools, ambiguous tickets
2. **Code-Assistant Agent**    — 9 dev-tools, nuanced code queries
3. **Data-Pipeline Orchestrator** — 8 data-ops tools, analytical queries

Usage::

    uv sync --extra llm
    uv run python examples/cookbook/prompt_evolution_agentic.py
"""
from __future__ import annotations

import os
import sys
import textwrap
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from prompture.prompt_evolver import (
    EvalSample,
    LLMBackend,
    LLMClient,
    PromptCandidate,
    PromptEvolver,
    PromptEvolverConfig,
    PromptEvolverResult,
    Tool,
    parse_tool_response,
    score_response,
)

# ─────────────────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class Scenario:
    name: str
    tools: list[Tool]
    dataset: list[EvalSample]


def _support_scenario() -> Scenario:
    """Customer-support triage with 9 overlapping tools."""
    tools = [
        Tool("lookup_order", "Look up order status by order ID or email",
             {"order_id": "string"}),
        Tool("refund_order", "Process a refund for a specific order",
             {"order_id": "string", "reason": "string"}),
        Tool("create_ticket", "Create a new support ticket",
             {"subject": "string", "priority": "string"}),
        Tool("update_ticket", "Update an existing support ticket",
             {"ticket_id": "string", "status": "string"}),
        Tool("escalate_to_human", "Transfer conversation to a human agent",
             {"reason": "string"}),
        Tool("search_knowledge_base", "Search help articles and FAQs",
             {"query": "string"}),
        Tool("send_notification", "Send an email or SMS notification",
             {"recipient": "string", "message": "string"}),
        Tool("apply_discount", "Apply a discount code to an order",
             {"order_id": "string", "code": "string"}),
        Tool("cancel_order", "Cancel a pending order",
             {"order_id": "string"}),
    ]
    dataset = [
        # refund_order — easy to confuse with cancel_order
        EvalSample("I want my money back for order #9912",
                   "refund_order", {"order_id": "#9912"}),
        EvalSample("Please refund order 5543, the item was damaged",
                   "refund_order", {"order_id": "5543", "reason": "damaged"}),
        # cancel_order — easy to confuse with refund_order
        EvalSample("Cancel my order 7721, I changed my mind",
                   "cancel_order", {"order_id": "7721"}),
        EvalSample("I placed order 3300 by mistake, please cancel it",
                   "cancel_order", {"order_id": "3300"}),
        # lookup_order
        EvalSample("Where is my package? Order #4455",
                   "lookup_order", {"order_id": "#4455"}),
        EvalSample("Can you check the status of order 1122?",
                   "lookup_order", {"order_id": "1122"}),
        # escalate_to_human — tricky phrasing
        EvalSample("I need to speak with a real person right now",
                   "escalate_to_human", {"reason": "customer request"}),
        EvalSample("This bot isn't helping, connect me to a manager",
                   "escalate_to_human", {"reason": "customer request"}),
        # search_knowledge_base
        EvalSample("How do I change my shipping address?",
                   "search_knowledge_base", {"query": "change shipping address"}),
        EvalSample("What is your return policy?",
                   "search_knowledge_base", {"query": "return policy"}),
        # create_ticket
        EvalSample("I have an issue that nobody has been able to fix",
                   "create_ticket", {"subject": "unresolved issue"}),
        EvalSample("Open a high-priority ticket about my billing error",
                   "create_ticket", {"subject": "billing error", "priority": "high"}),
        # apply_discount
        EvalSample("I have a promo code SAVE20 for order 6789",
                   "apply_discount", {"order_id": "6789", "code": "SAVE20"}),
        EvalSample("Apply my coupon to order #8800",
                   "apply_discount", {"order_id": "#8800"}),
        # send_notification
        EvalSample("Text me at 555-1234 when the order ships",
                   "send_notification", {"recipient": "555-1234"}),
        EvalSample("Email the receipt to alice@example.com",
                   "send_notification", {"recipient": "alice@example.com"}),
        # update_ticket
        EvalSample("Mark ticket TK-100 as resolved",
                   "update_ticket", {"ticket_id": "TK-100", "status": "resolved"}),
        EvalSample("Change the status of TK-205 to in-progress",
                   "update_ticket", {"ticket_id": "TK-205", "status": "in-progress"}),
        # Ambiguous — refund or cancel?
        EvalSample("I don't want order 2211 anymore, give me my money back",
                   "refund_order", {"order_id": "2211"}),
        # Ambiguous — knowledge base or create ticket?
        EvalSample("My account keeps getting locked out, what do I do?",
                   "search_knowledge_base", {"query": "account locked out"}),
    ]
    return Scenario("Customer-Support Triage", tools, dataset)


def _code_assistant_scenario() -> Scenario:
    """Code-assistant agent with 9 dev-tools."""
    tools = [
        Tool("run_tests", "Run test suite for a module or file",
             {"target": "string"}),
        Tool("search_codebase", "Search source code by text or regex",
             {"query": "string", "scope": "string"}),
        Tool("edit_file", "Edit lines in an existing file",
             {"file": "string", "changes": "string"}),
        Tool("create_file", "Create a new source file",
             {"path": "string", "content": "string"}),
        Tool("run_linter", "Run linter/type-checker on a file or project",
             {"target": "string"}),
        Tool("git_commit", "Stage and commit changes",
             {"message": "string"}),
        Tool("open_pull_request", "Open a pull request on GitHub",
             {"title": "string", "branch": "string"}),
        Tool("explain_code", "Explain what a code snippet does",
             {"code": "string"}),
        Tool("refactor_function", "Refactor a function for clarity or perf",
             {"function_name": "string", "file": "string"}),
    ]
    dataset = [
        # run_tests
        EvalSample("Run the tests for the auth module",
                   "run_tests", {"target": "auth"}),
        EvalSample("Check if the payment tests pass",
                   "run_tests", {"target": "payment"}),
        # search_codebase
        EvalSample("Find all usages of deprecated_api() in the repo",
                   "search_codebase", {"query": "deprecated_api"}),
        EvalSample("Where is the database connection string configured?",
                   "search_codebase", {"query": "database connection string"}),
        # edit_file
        EvalSample("Add type hints to utils.py",
                   "edit_file", {"file": "utils.py"}),
        EvalSample("Fix the typo on line 42 of main.py",
                   "edit_file", {"file": "main.py"}),
        # create_file
        EvalSample("Create a new test file for the billing service",
                   "create_file", {"path": "test_billing.py"}),
        EvalSample("Set up an empty config.yaml in the project root",
                   "create_file", {"path": "config.yaml"}),
        # run_linter
        EvalSample("Run mypy on the whole project",
                   "run_linter", {"target": "project"}),
        EvalSample("Check for lint errors in server.py",
                   "run_linter", {"target": "server.py"}),
        # git_commit
        EvalSample("Commit these changes with message 'fix auth bug'",
                   "git_commit", {"message": "fix auth bug"}),
        EvalSample("Stage and commit the refactored module",
                   "git_commit", {"message": "refactored module"}),
        # open_pull_request
        EvalSample("Open a PR for the feature/login branch",
                   "open_pull_request", {"branch": "feature/login"}),
        EvalSample("Create a pull request titled 'Add caching layer'",
                   "open_pull_request", {"title": "Add caching layer"}),
        # explain_code
        EvalSample("What does this regex do: r'^[a-z]+\\d{3}$'?",
                   "explain_code", {"code": "r'^[a-z]+\\d{3}$'"}),
        EvalSample("Explain this decorator pattern",
                   "explain_code", {}),
        # refactor_function
        EvalSample("Refactor process_data() in etl.py for readability",
                   "refactor_function",
                   {"function_name": "process_data", "file": "etl.py"}),
        EvalSample("Clean up the parse_config function in loader.py",
                   "refactor_function",
                   {"function_name": "parse_config", "file": "loader.py"}),
        # Ambiguous — edit or refactor?
        EvalSample("Simplify the calculate_tax function in billing.py",
                   "refactor_function",
                   {"function_name": "calculate_tax", "file": "billing.py"}),
        # Ambiguous — search or explain?
        EvalSample("Show me how the caching middleware works",
                   "explain_code", {}),
    ]
    return Scenario("Code-Assistant Agent", tools, dataset)


def _data_pipeline_scenario() -> Scenario:
    """Data-pipeline orchestrator with 8 data-ops tools."""
    tools = [
        Tool("query_database", "Run a SQL query against the data warehouse",
             {"sql": "string", "database": "string"}),
        Tool("transform_data", "Apply a transformation to a dataset",
             {"dataset": "string", "operation": "string"}),
        Tool("export_csv", "Export query results to CSV",
             {"source": "string", "destination": "string"}),
        Tool("send_alert", "Send an alert via Slack or email",
             {"channel": "string", "message": "string"}),
        Tool("schedule_job", "Schedule a recurring ETL job",
             {"job_name": "string", "cron": "string"}),
        Tool("validate_schema", "Check a dataset against its schema",
             {"dataset": "string"}),
        Tool("merge_datasets", "Join or union two datasets",
             {"left": "string", "right": "string", "join_key": "string"}),
        Tool("compute_metrics", "Calculate aggregate metrics on a dataset",
             {"dataset": "string", "metrics": "string"}),
    ]
    dataset = [
        # query_database
        EvalSample("Show me all orders from last month",
                   "query_database", {"database": "warehouse"}),
        EvalSample("Run a query to count active users in the analytics db",
                   "query_database", {"database": "analytics"}),
        # transform_data
        EvalSample("Normalize the revenue column in the sales dataset",
                   "transform_data",
                   {"dataset": "sales", "operation": "normalize"}),
        EvalSample("Remove duplicate rows from the events table",
                   "transform_data",
                   {"dataset": "events", "operation": "deduplicate"}),
        # export_csv
        EvalSample("Export the quarterly report to a CSV file",
                   "export_csv", {"source": "quarterly_report"}),
        EvalSample("Download the user_metrics data as CSV",
                   "export_csv", {"source": "user_metrics"}),
        # send_alert
        EvalSample("Alert the data team on Slack that the pipeline failed",
                   "send_alert", {"channel": "Slack"}),
        EvalSample("Send an email alert if row count drops below threshold",
                   "send_alert", {"channel": "email"}),
        # schedule_job
        EvalSample("Set up a daily ETL run at midnight",
                   "schedule_job", {"cron": "0 0 * * *"}),
        EvalSample("Schedule the aggregation job to run every hour",
                   "schedule_job", {"cron": "0 * * * *"}),
        # validate_schema
        EvalSample("Check if the new data matches our expected schema",
                   "validate_schema", {}),
        EvalSample("Validate the ingested customer_data format",
                   "validate_schema", {"dataset": "customer_data"}),
        # merge_datasets
        EvalSample("Combine the sales and inventory tables on product_id",
                   "merge_datasets",
                   {"left": "sales", "right": "inventory",
                    "join_key": "product_id"}),
        EvalSample("Join users with orders by user_id",
                   "merge_datasets",
                   {"left": "users", "right": "orders",
                    "join_key": "user_id"}),
        # compute_metrics
        EvalSample("Calculate conversion rates for last quarter",
                   "compute_metrics",
                   {"dataset": "conversions", "metrics": "conversion_rate"}),
        EvalSample("What's the average order value in the sales data?",
                   "compute_metrics",
                   {"dataset": "sales", "metrics": "average_order_value"}),
        # Ambiguous — query or compute_metrics?
        EvalSample("How many signups did we get last week?",
                   "compute_metrics",
                   {"dataset": "signups", "metrics": "count"}),
        # Ambiguous — transform or merge?
        EvalSample("Take the raw logs and enrich them with user profiles",
                   "merge_datasets",
                   {"left": "raw_logs", "right": "user_profiles"}),
        # Ambiguous — export or query?
        EvalSample("Pull the retention numbers and save as a spreadsheet",
                   "export_csv", {"source": "retention"}),
        # Ambiguous — alert or schedule?
        EvalSample("Set it up so I'm notified whenever latency spikes",
                   "send_alert", {"channel": "Slack"}),
    ]
    return Scenario("Data-Pipeline Orchestrator", tools, dataset)


ALL_SCENARIOS = [_support_scenario, _code_assistant_scenario, _data_pipeline_scenario]


# ─────────────────────────────────────────────────────────────────────────
# Prompt evolution display
# ─────────────────────────────────────────────────────────────────────────


def show_prompt_evolution(result: PromptEvolverResult, top_n: int = 3) -> None:
    """Print how the best prompt evolved across generations.

    Groups all_candidates by generation, picks the best per gen, and
    shows the progression of the prompt template text.
    """
    candidates = result.all_candidates
    if not candidates:
        print("    (no candidate data available)")
        return

    # Group by generation
    by_gen: dict[int, list[PromptCandidate]] = {}
    for c in candidates:
        by_gen.setdefault(c.generation, []).append(c)

    gens_sorted = sorted(by_gen.keys())
    if not gens_sorted:
        return

    best_per_gen: list[tuple[int, PromptCandidate]] = []
    for g in gens_sorted:
        best = max(by_gen[g], key=lambda c: c.score)
        best_per_gen.append((g, best))

    # Select milestones: first, middle, last (+1 extra if enough gens)
    milestones: list[int] = [0]
    if len(best_per_gen) > 2:
        milestones.append(len(best_per_gen) // 2)
    if len(best_per_gen) > 1:
        milestones.append(len(best_per_gen) - 1)
    milestones = sorted(set(milestones))[:top_n]

    print()
    print("    ┌─ Prompt Evolution Trace ─────────────────────────────────")
    prev_lines: list[str] | None = None
    for idx in milestones:
        gen, cand = best_per_gen[idx]
        print(f"    │")
        print(f"    │  Generation {gen}   score={cand.score:.1f}%  "
              f"temp={cand.temperature:.3f}  top_p={cand.top_p:.3f}")
        print(f"    │  ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌")

        # Show the template (truncate long lines)
        template = cand.template
        # Remove the tool schemas block to keep output readable
        cut = template.find("{tool_schemas}")
        if cut != -1:
            # Show only the instruction part before {tool_schemas}
            pre = template[:cut].rstrip()
            post = template[cut + len("{tool_schemas}"):].strip()
            display = pre + "\n  [... tool schemas ...]" + (
                f"\n{post}" if post else "")
        else:
            display = template

        lines = display.splitlines()
        cur_lines = lines

        if prev_lines is not None:
            # Mark new/changed lines with +
            for line in cur_lines:
                stripped = line.strip()
                marker = "    │    "
                if stripped and stripped not in [pl.strip() for pl in prev_lines]:
                    marker = "    │  + "
                wrapped = textwrap.shorten(line, width=64, placeholder="…")
                print(f"{marker}{wrapped}")
        else:
            for line in cur_lines:
                wrapped = textwrap.shorten(line, width=64, placeholder="…")
                print(f"    │    {wrapped}")

        prev_lines = cur_lines

    print(f"    │")
    print(f"    └─────────────────────────────────────────────────────────")

    # Convergence history
    if result.history:
        print()
        print("    Convergence:  ", end="")
        for gen, score in result.history:
            print(f"G{gen}={score:.0f}%  ", end="")
        print()


# ─────────────────────────────────────────────────────────────────────────
# Baseline prompts
# ─────────────────────────────────────────────────────────────────────────


def build_baselines(tool_schemas: str) -> dict[str, tuple[str, float, float]]:
    """Five baseline prompts varying in specificity and temperature."""
    return {
        "Naive (tool list only)": (
            f"You are a helpful assistant.\n\nAvailable tools:\n{tool_schemas}",
            0.7, 0.95,
        ),
        "Minimal JSON instruction": (
            f"Pick a tool for the user's request. "
            f'Respond with JSON: {{"tool": "<name>", "parameters": {{...}}}}\n\n'
            f"Tools:\n{tool_schemas}",
            0.3, 0.9,
        ),
        "Verbose (kitchen sink)": (
            f"You are an AI assistant with access to the following tools. "
            f"Carefully read the user's query, consider all tools, and select "
            f"the single best tool. Extract parameter values from the query. "
            f"Return JSON with 'tool' and 'parameters' keys. "
            f"Do not return anything else.\n\nTools:\n{tool_schemas}",
            0.5, 0.95,
        ),
        "High temperature (creative)": (
            f"Route the user's request to the correct tool.\n\n"
            f"Tools:\n{tool_schemas}\n\n"
            f'Respond with JSON: {{"tool": "<name>", "parameters": {{...}}}}',
            0.9, 0.99,
        ),
        "Zero temperature (greedy)": (
            f"Route the user's request to the correct tool.\n\n"
            f"Tools:\n{tool_schemas}\n\n"
            f'Respond with JSON: {{"tool": "<name>", "parameters": {{...}}}}',
            0.0, 1.0,
        ),
    }


# ─────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────


def evaluate_baselines(
    baselines: dict[str, tuple[str, float, float]],
    dataset: list[EvalSample],
    client: LLMClient,
    tool_names: list[str],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for name, (prompt, temp, top_p) in baselines.items():
        total = 0.0
        for sample in dataset:
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
        accuracy = (total / len(dataset)) * 100 if dataset else 0.0
        scores[name] = accuracy
        print(f"      {name:<30s}  {accuracy:5.1f}%  "
              f"temp={temp:.1f}  top_p={top_p:.2f}")
    return scores


def run_scenario_on_backend(
    scenario: Scenario, backend: LLMBackend
) -> tuple[dict[str, float], PromptEvolverResult | None, float]:
    """Run baselines + evolution for one scenario on one backend.

    Returns (baseline_scores, evolver_result, wall_time).
    """
    config = PromptEvolverConfig(backend=backend)
    client = LLMClient(config)

    if not client.is_available():
        print(f"      ⚠  {backend.value} is not available — skipping")
        return {}, None, 0.0

    tool_schemas = "\n".join(f"  - {t.schema_str()}" for t in scenario.tools)
    tool_names = [t.name for t in scenario.tools]
    baselines = build_baselines(tool_schemas)

    print(f"    Baselines:")
    baseline_scores = evaluate_baselines(
        baselines, scenario.dataset, client, tool_names
    )
    best_bl_name = max(baseline_scores, key=baseline_scores.get) if baseline_scores else "N/A"
    best_bl_score = max(baseline_scores.values()) if baseline_scores else 0.0
    print(f"      Best baseline: {best_bl_name} ({best_bl_score:.1f}%)\n")

    evo_config = PromptEvolverConfig(
        iterations=5,
        population_size=4,
        num_islands=2,
        elite_size=3,
        mutation_rate=0.6,
        crossover_rate=0.3,
        eval_sample_size=8,
        backend=backend,
    )

    print(f"    PromptEvolver ({evo_config.iterations} gens):")
    t0 = time.perf_counter()
    evolver = PromptEvolver(
        tools=scenario.tools,
        eval_dataset=scenario.dataset,
        config=evo_config,
        seed=42,
        verbose=True,
    )
    result = evolver.run()
    wall = time.perf_counter() - t0
    print(f"      Evolved best: {result.best_score:.1f}%  "
          f"(temp={result.best_temperature:.3f}, top_p={result.best_top_p:.3f})")
    print(f"      Wall time: {wall:.1f}s")

    # Show prompt evolution
    show_prompt_evolution(result)
    print()

    return baseline_scores, result, wall


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────


def main() -> None:
    print("EvoSim Cookbook Recipe 41 — Agentic Prompt Evolution")
    print("=" * 62)
    print()

    # All scenario × backend results
    all_results: dict[
        str,
        dict[str, tuple[dict[str, float], PromptEvolverResult | None, float]],
    ] = {}

    for scenario_fn in ALL_SCENARIOS:
        scenario = scenario_fn()
        print("━" * 62)
        print(f"  Scenario: {scenario.name}")
        print(f"  Tools: {len(scenario.tools)}   Samples: {len(scenario.dataset)}")
        print("━" * 62)
        print()

        scenario_results: dict[
            str, tuple[dict[str, float], PromptEvolverResult | None, float]
        ] = {}

        for backend in [LLMBackend.OLLAMA, LLMBackend.AZURE_OPENAI]:
            label = "Ollama" if backend == LLMBackend.OLLAMA else "Azure OpenAI"
            print(f"  ── {label} {'─' * (56 - len(label))}")
            scenario_results[label] = run_scenario_on_backend(scenario, backend)

        all_results[scenario.name] = scenario_results

    # ── Summary table ────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  Summary — Evolved Accuracy Across Scenarios & Backends")
    print("=" * 62)
    print()
    print(f"  {'Scenario':<30s} {'Ollama':>10s} {'Azure':>10s} {'Best BL*':>10s}")
    print(f"  {'─' * 30} {'─' * 10} {'─' * 10} {'─' * 10}")

    for scenario_name, backends in all_results.items():
        ollama_data = backends.get("Ollama", ({}, None, 0.0))
        azure_data = backends.get("Azure OpenAI", ({}, None, 0.0))

        o_evo = ollama_data[1].best_score if ollama_data[1] else 0.0
        a_evo = azure_data[1].best_score if azure_data[1] else 0.0

        # Best baseline across both backends
        all_bl = list(ollama_data[0].values()) + list(azure_data[0].values())
        best_bl = max(all_bl) if all_bl else 0.0

        short_name = scenario_name[:30]
        print(f"  {short_name:<30s} {o_evo:>9.1f}% {a_evo:>9.1f}% {best_bl:>9.1f}%")

    print()
    print("  * Best BL = highest-scoring hand-crafted baseline across both backends")
    print()
    print("✓ Cookbook Recipe 41 complete.")


if __name__ == "__main__":
    main()
