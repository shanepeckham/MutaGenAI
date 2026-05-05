#!/usr/bin/env python3
"""
Cookbook Recipe 43 — τ-Bench Conversational Agent Prompt Evolution
=================================================================

Evolves system prompts on the **τ-bench** (Sierra Research) benchmark
for conversational customer-service agents.  τ-bench evaluates agents
across airline, retail, and telecom domains, scoring on three axes:

* **Tool-call correctness** — did the agent invoke the right API calls?
* **Information communication** — did the agent relay required data?
* **Policy compliance** — did the agent follow domain rules?

Top models score only ~46 % pass@1 on airline and ~69 % on retail,
making this one of the hardest agentic benchmarks available.

Categories benchmarked
----------------------
1. **Airline**   (50 tasks) — booking, modification, cancellation, refunds
2. **Retail**    (50 tasks) — orders, returns, product inquiries

Algorithm experiments
---------------------
* **Standard** — 3 iterations, pop 4, 2 islands  (balanced)
* **Deep**     — 5 iterations, pop 5, 2 islands  (thorough)

Usage::

    uv sync --extra llm
    uv run python examples/cookbook/prompt_evolution_tau_bench.py

The script saves an experiment log to ``tau_bench_experiment_log.json``
for dashboard consumption.

References
----------
* τ-bench repo: https://github.com/sierra-research/tau2-bench
* Paper: https://arxiv.org/abs/2506.07982
* Leaderboard: https://taubench.com
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import textwrap
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from MutaGenAI.prompt_evolver import (
    LLMBackend,
    LLMClient,
    PromptCandidate,
    PromptEvolverConfig,
)

# ─────────────────────────────────────────────────────────────────────────
# τ-bench data loading
# ─────────────────────────────────────────────────────────────────────────

_TAU_DATA_DIR: Optional[Path] = None

_RAW_BASE = (
    "https://raw.githubusercontent.com/sierra-research/tau2-bench/main/data/tau2/domains"
)


def _tau_data_dir() -> Path:
    """Return the local cache directory for τ-bench data."""
    global _TAU_DATA_DIR
    if _TAU_DATA_DIR is not None:
        return _TAU_DATA_DIR

    cache = Path(__file__).resolve().parent.parent.parent / ".tau_bench_cache"
    cache.mkdir(exist_ok=True)
    _TAU_DATA_DIR = cache
    return cache


def _download_if_missing(url: str, dest: Path) -> Path:
    """Download a file from *url* to *dest* if it does not exist."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"    Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)  # noqa: S310
    return dest


@dataclass
class TauBenchTask:
    """A single τ-bench task."""

    id: str
    domain: str
    purpose: str
    reason_for_call: str
    known_info: str
    task_instructions: str
    expected_actions: list[dict[str, Any]]
    communicate_info: list[str]
    nl_assertions: list[str]
    reward_basis: list[str]

    @property
    def user_message(self) -> str:
        """Build a realistic first-turn user message from the scenario."""
        parts: list[str] = []
        if self.reason_for_call:
            parts.append(self.reason_for_call.strip())
        if self.known_info:
            # Expose *some* known info naturally
            for line in self.known_info.strip().split("\n"):
                line = line.strip()
                if line and not line.startswith("You do not know"):
                    parts.append(line)
        return "\n".join(parts)

    @property
    def expected_tool_names(self) -> list[str]:
        return [a["name"] for a in self.expected_actions if "name" in a]


def load_tau_bench_domain(
    domain: str, max_tasks: int | None = None
) -> tuple[list[TauBenchTask], str]:
    """Load τ-bench tasks and the domain policy.

    Downloads from GitHub to a local cache on first use.

    Returns
    -------
    tasks : list[TauBenchTask]
    policy : str
    """
    cache = _tau_data_dir()
    domain_dir = cache / domain

    # Download tasks.json and policy.md
    tasks_url = f"{_RAW_BASE}/{domain}/tasks.json"
    policy_url = f"{_RAW_BASE}/{domain}/policy.md"

    tasks_path = _download_if_missing(tasks_url, domain_dir / "tasks.json")
    policy_path = _download_if_missing(policy_url, domain_dir / "policy.md")

    # Load policy
    policy = policy_path.read_text(encoding="utf-8")

    # Load tasks
    with open(tasks_path, encoding="utf-8") as f:
        raw_tasks = json.load(f)

    tasks: list[TauBenchTask] = []
    for raw in raw_tasks:
        scenario = raw.get("user_scenario", {})
        instructions = scenario.get("instructions", {})
        eval_criteria = raw.get("evaluation_criteria", {})

        tasks.append(
            TauBenchTask(
                id=str(raw["id"]),
                domain=domain,
                purpose=raw.get("description", {}).get("purpose", ""),
                reason_for_call=instructions.get("reason_for_call", ""),
                known_info=instructions.get("known_info", ""),
                task_instructions=instructions.get("task_instructions", ""),
                expected_actions=eval_criteria.get("actions", []),
                communicate_info=eval_criteria.get("communicate_info", []),
                nl_assertions=eval_criteria.get("nl_assertions", []),
                reward_basis=eval_criteria.get("reward_basis", []),
            )
        )

    if max_tasks and max_tasks < len(tasks):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(tasks), size=max_tasks, replace=False)
        tasks = [tasks[int(i)] for i in sorted(indices)]

    return tasks, policy


# ─────────────────────────────────────────────────────────────────────────
# τ-bench tool descriptions (simplified for prompt-mode evaluation)
# ─────────────────────────────────────────────────────────────────────────

_AIRLINE_TOOLS = """\
Available tools:
- get_user_details(user_id) — Retrieve user profile, membership level, payment methods
- get_reservation_details(reservation_id) — Get full reservation details
- search_direct_flight(origin, destination, date) — Search for direct flights
- book_reservation(user_id, origin, destination, flight_type, cabin, flights, \
passengers, payment_methods, total_baggages, nonfree_baggages, insurance) — Book a new flight
- cancel_reservation(reservation_id) — Cancel an existing reservation
- update_reservation_flights(reservation_id, cabin, flights, payment_id) — Modify flights/cabin
- update_reservation_passengers(reservation_id, passengers) — Update passenger info
- update_reservation_baggages(reservation_id, total_baggages, nonfree_baggages, \
payment_id) — Update baggage
- calculate(expression) — Calculate a math expression
- transfer_to_human_agents(summary) — Transfer to human agent with summary
- send_certificate(user_id, amount) — Issue a compensation certificate
"""

_RETAIL_TOOLS = """\
Available tools:
- get_user_details(user_id) — Retrieve user profile and order history
- get_order_details(order_id) — Get full order details including items and status
- get_product_details(product_id) — Get product details including price and stock
- list_all_product_types() — List all available product categories
- modify_pending_order(order_id, item_ids, new_item_ids, payment_method_id) \
— Modify a pending order
- cancel_pending_order(order_id, reason) — Cancel a pending order
- return_delivered_order(order_id, item_ids, payment_method_id) — Return items
- exchange_delivered_order(order_id, item_ids, new_item_ids, payment_method_id) \
— Exchange items
- transfer_to_human_agents(summary) — Transfer to human agent with summary
"""

_DOMAIN_TOOLS: dict[str, str] = {
    "airline": _AIRLINE_TOOLS,
    "retail": _RETAIL_TOOLS,
}


# ─────────────────────────────────────────────────────────────────────────
# τ-bench scoring
# ─────────────────────────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    """Lower-case + strip whitespace for fuzzy matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _extract_tool_mentions(response: str) -> list[str]:
    """Extract tool/function names mentioned in the response."""
    tools: list[str] = []

    # Match patterns: tool_name(...), tool_name:, **tool_name**, `tool_name`
    patterns = [
        r"\b(get_user_details|get_reservation_details|search_direct_flight|"
        r"book_reservation|cancel_reservation|update_reservation_flights|"
        r"update_reservation_passengers|update_reservation_baggages|"
        r"calculate|transfer_to_human_agents|send_certificate|"
        r"get_order_details|get_product_details|list_all_product_types|"
        r"modify_pending_order|cancel_pending_order|return_delivered_order|"
        r"exchange_delivered_order)\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, response, re.IGNORECASE):
            tools.append(m.group(1).lower())

    return list(dict.fromkeys(tools))  # deduplicate, preserve order


def score_tau_bench_task(
    response: str,
    task: TauBenchTask,
) -> dict[str, float]:
    """Score a model response against a τ-bench task.

    Returns a dict of sub-scores and a composite score, all in [0, 1].
    """
    if not response:
        return {"tool_score": 0.0, "info_score": 0.0, "assertion_score": 0.0, "composite": 0.0}

    resp_norm = _normalise(response)

    # ── 1. Tool-call accuracy ──────────────────────────────────────────
    tool_score = 0.0
    if task.expected_actions:
        expected_tools = [a["name"] for a in task.expected_actions if "name" in a]
        unique_expected = list(dict.fromkeys(expected_tools))
        mentioned = _extract_tool_mentions(response)

        if unique_expected:
            # Credit for each expected tool mentioned in the response
            matches = sum(1 for t in unique_expected if t.lower() in mentioned)
            tool_score = matches / len(unique_expected)

            # Also check for argument mentions (partial credit)
            for action in task.expected_actions:
                args = action.get("arguments", {})
                for val in args.values():
                    if isinstance(val, str) and val.lower() in resp_norm:
                        tool_score = min(1.0, tool_score + 0.05)
    else:
        # No expected actions — task tests policy compliance (refusing actions)
        # Agent should NOT suggest mutating tools
        mutating = ["cancel_reservation", "book_reservation",
                     "update_reservation_flights", "update_reservation_passengers",
                     "update_reservation_baggages", "modify_pending_order",
                     "cancel_pending_order", "return_delivered_order",
                     "exchange_delivered_order", "send_certificate"]
        mentioned = _extract_tool_mentions(response)
        bad_mentions = sum(1 for t in mentioned if t in mutating)
        tool_score = 1.0 if bad_mentions == 0 else max(0.0, 1.0 - bad_mentions * 0.3)

    # ── 2. Information communication ───────────────────────────────────
    info_score = 0.0
    if task.communicate_info:
        matches = 0
        for info_val in task.communicate_info:
            # Check for numeric values in various formats
            info_str = str(info_val).strip()
            if info_str in response:
                matches += 1
            elif info_str.replace(",", "") in response.replace(",", ""):
                matches += 1
            elif info_str.startswith("$"):
                if info_str[1:] in response:
                    matches += 0.8
        info_score = matches / len(task.communicate_info)
    else:
        info_score = 1.0  # No info required — full credit

    # ── 3. NL assertion matching ───────────────────────────────────────
    assertion_score = 0.0
    if task.nl_assertions:
        matches = 0.0
        for assertion in task.nl_assertions:
            a_norm = _normalise(assertion)

            # Check for positive assertions (agent should do X)
            is_negative = any(
                neg in a_norm
                for neg in ["should not", "should refuse", "does not",
                            "should deny", "does not offer", "does not cancel",
                            "does not allow", "should not cancel",
                            "should not offer", "should not approve",
                            "should not change", "should not book"]
            )

            if is_negative:
                # For negative assertions, check the agent's response avoids
                # suggesting the forbidden action
                action_words = re.findall(
                    r"cancel|refund|compensation|approve|change|offer|book|modify|"
                    r"certificate|waiv",
                    a_norm,
                )
                violated = False
                for word in action_words:
                    if word in resp_norm:
                        # Check context — did agent promise/confirm the action?
                        affirm = re.search(
                            rf"(i.{{0,20}}{word}|will\s+{word}|let me\s+{word}|"
                            rf"proceed.{{0,15}}{word}|confirm.{{0,15}}{word}|"
                            rf"happy to\s+{word}|going to\s+{word})",
                            resp_norm,
                        )
                        if affirm:
                            violated = True
                            break
                matches += 0.0 if violated else 1.0
            else:
                # Positive assertion — extract key entities and check
                key_terms = re.findall(r"\b[A-Z0-9]{4,}\b", assertion)  # IDs
                key_terms += re.findall(r"\b\d{2,}\b", assertion)  # numbers
                key_actions = re.findall(
                    r"cancel|book|upgrade|downgrade|update|search|refund|"
                    r"communicate|check|detect|identify|confirm|verify",
                    a_norm,
                )

                if key_terms:
                    term_matches = sum(
                        1 for t in key_terms if t.lower() in resp_norm
                    )
                    matches += min(1.0, term_matches / len(key_terms))
                elif key_actions:
                    action_matches = sum(
                        1 for a in key_actions if a in resp_norm
                    )
                    matches += min(1.0, action_matches / len(key_actions))
                else:
                    # Fallback: check for any word overlap
                    words = set(a_norm.split()) - {"the", "a", "an", "is", "that",
                                                    "should", "agent", "user", "to",
                                                    "and", "or", "not", "be", "for",
                                                    "of", "in", "with", "it"}
                    if words:
                        overlap = sum(1 for w in words if w in resp_norm)
                        matches += min(1.0, overlap / max(1, len(words) * 0.5))

        assertion_score = matches / len(task.nl_assertions)
    else:
        assertion_score = 1.0

    # ── Composite score ────────────────────────────────────────────────
    # Weighted: 40 % tools, 30 % info, 30 % assertions
    w_tool = 0.4 if task.expected_actions else 0.2
    w_info = 0.3 if task.communicate_info else 0.1
    w_assert = 1.0 - w_tool - w_info

    composite = w_tool * tool_score + w_info * info_score + w_assert * assertion_score

    return {
        "tool_score": round(tool_score, 3),
        "info_score": round(info_score, 3),
        "assertion_score": round(assertion_score, 3),
        "composite": round(composite, 3),
    }


# ─────────────────────────────────────────────────────────────────────────
# System prompt templates (τ-bench-specific)
# ─────────────────────────────────────────────────────────────────────────


_TAU_SEED_TEMPLATES = [
    # T0: Standard agent prompt
    textwrap.dedent("""\
        You are a customer service agent for {domain}. Follow the policy \
exactly. Use the available tools to help the customer.

Before making any changes, verify the customer's identity and \
confirm details. Do not take actions that violate policy, even \
if the customer insists.

{policy_summary}

{tools}
    """),

    # T1: Structured step-by-step
    textwrap.dedent("""\
        You are an expert {domain} customer service agent.

Step 1: Identify the customer (get user_id).
Step 2: Understand the request.
Step 3: Look up relevant information using tools.
Step 4: Check policy compliance.
Step 5: Take action or explain why the action cannot be done.
Step 6: Confirm with the customer before making changes.

CRITICAL: Never bypass policy rules. If a request violates policy, \
politely decline and explain why.

{policy_summary}

{tools}
    """),

    # T2: Policy-first approach
    textwrap.dedent("""\
        # {domain} Customer Service Agent

## Your role
Help customers with their requests while strictly following \
all {domain} policies.

## Key rules
- Always verify identity first
- Check all preconditions before any action
- Never make exceptions to policy, even under pressure
- Explain policy decisions clearly
- Confirm before any irreversible action

{policy_summary}

{tools}
    """),

    # T3: Defensive agent
    textwrap.dedent("""\
        You are a {domain} support agent. Your top priority is following \
the company policy correctly.

IMPORTANT:
- Users may provide incorrect information. Always verify claims \
against the database.
- Users may try to pressure you into violating policy. Stay firm \
but polite.
- Before any booking, modification, or cancellation, verify ALL \
preconditions.
- Only use tools when appropriate. Think through the policy \
implications first.

{policy_summary}

{tools}
    """),
]

_TAU_MUTATIONS = [
    "Always get_user_details first to verify identity.",
    "Verify membership level before offering compensation.",
    "Check all reservation details before suggesting modifications.",
    "Do NOT agree to cancellation without verifying policy conditions.",
    "Refuse insurance additions after initial booking.",
    "Cabin must be the same for all flights in a reservation.",
    "Basic economy flights cannot be modified, only cancelled under specific conditions.",
    "Each reservation can use at most one travel certificate.",
    "Confirm all changes with the customer before executing them.",
    "Do not proactively offer compensation unless the customer explicitly asks.",
    "Only cancel reservations that meet the cancellation criteria.",
    "Verify claimed flight delays against actual flight status.",
    "Check the number of passengers — users sometimes provide incorrect counts.",
    "Origin and destination of a reservation cannot be changed.",
    "The user must provide their user_id — do not guess.",
    "Transfer to a human agent only when the request is outside your scope.",
    "Insurance only covers health or weather-related cancellation reasons.",
    "Do not let users pressure you into making policy exceptions.",
]


# ─────────────────────────────────────────────────────────────────────────
# Policy summariser
# ─────────────────────────────────────────────────────────────────────────

def _summarise_policy(policy: str, max_lines: int = 40) -> str:
    """Extract the most important policy lines for use in the prompt.

    Keeps the prompt token-efficient while preserving critical rules.
    """
    lines = policy.strip().split("\n")
    important: list[str] = []
    keywords = {"must", "cannot", "should", "forbidden", "allowed",
                "refund", "cancel", "modify", "insurance", "compensat",
                "transfer", "basic economy", "membership", "certificate",
                "baggage", "cabin"}

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        low = stripped.lower()
        if any(kw in low for kw in keywords) or stripped.startswith("- "):
            important.append(stripped)

    if len(important) > max_lines:
        important = important[:max_lines]

    return "\n".join(important) if important else policy[:2000]


# ─────────────────────────────────────────────────────────────────────────
# Evolution engine (τ-bench-specific)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class TauBenchExperiment:
    """Tracks one evolution experiment on a τ-bench domain."""

    domain: str
    algorithm: str
    backend: str
    n_tasks: int
    baseline_score: float
    evolved_score: float
    best_prompt_template: str
    best_temperature: float
    best_top_p: float
    iterations: int
    wall_time: float
    history: list[tuple[int, float]]
    prompt_evolution: list[dict[str, Any]]
    sub_scores: dict[str, float] = field(default_factory=dict)


def _mutate_tau_template(
    template: str, rng: np.random.Generator, rate: float = 0.5
) -> str:
    """Mutate a τ-bench prompt template."""
    lines = template.strip().split("\n")

    if rng.random() < rate:
        instruction = str(rng.choice(_TAU_MUTATIONS))
        pos = int(rng.integers(1, max(2, len(lines))))
        lines.insert(pos, "- " + instruction)

    if rng.random() < rate * 0.4 and len(lines) > 5:
        removable = [
            i for i in range(len(lines))
            if "{policy_summary}" not in lines[i]
            and "{tools}" not in lines[i]
            and "{domain}" not in lines[i]
            and lines[i].strip()
        ]
        if removable:
            idx = int(rng.choice(removable))
            lines.pop(idx)

    if rng.random() < rate * 0.3 and len(lines) > 3:
        idx = int(rng.integers(0, len(lines) - 1))
        lines[idx], lines[idx + 1] = lines[idx + 1], lines[idx]

    result = "\n".join(lines)
    # Ensure required placeholders
    if "{policy_summary}" not in result:
        result += "\n\n{policy_summary}"
    if "{tools}" not in result:
        result += "\n\n{tools}"
    return result


def _crossover_tau_templates(
    a: str, b: str, rng: np.random.Generator
) -> str:
    """Crossover two τ-bench prompt templates."""
    la = a.strip().split("\n")
    lb = b.strip().split("\n")
    ca = int(rng.integers(1, max(2, len(la))))
    cb = int(rng.integers(1, max(2, len(lb))))
    child = "\n".join(la[:ca] + lb[cb:])
    if "{policy_summary}" not in child:
        child += "\n\n{policy_summary}"
    if "{tools}" not in child:
        child += "\n\n{tools}"
    return child


def run_tau_bench_evolution(
    tasks: list[TauBenchTask],
    policy: str,
    domain: str,
    client: LLMClient,
    config: PromptEvolverConfig,
    algorithm_name: str = "standard",
    seed: int = 42,
    verbose: bool = True,
) -> TauBenchExperiment:
    """Run prompt evolution on τ-bench tasks.

    Returns a TauBenchExperiment with full tracking.
    """
    rng = np.random.default_rng(seed)
    policy_summary = _summarise_policy(policy)
    tools_text = _DOMAIN_TOOLS.get(domain, "")

    # ── Evaluate a candidate ───────────────────────────────────────────
    def evaluate(candidate: PromptCandidate, eval_tasks: list[TauBenchTask]) -> float:
        total = 0.0
        for task in eval_tasks:
            sys_prompt = (
                candidate.template
                .replace("{domain}", domain)
                .replace("{policy_summary}", policy_summary)
                .replace("{tools}", tools_text)
            )

            response = client.complete(
                system_prompt=sys_prompt,
                user_message=task.user_message,
                temperature=candidate.temperature,
                top_p=candidate.top_p,
            )
            if response is None:
                total += float(rng.uniform(0, 0.1))
                continue

            scores = score_tau_bench_task(response, task)
            total += scores["composite"]

        return (total / len(eval_tasks) * 100.0) if eval_tasks else 0.0

    # ── Subsample for evaluation ───────────────────────────────────────
    if config.eval_sample_size and config.eval_sample_size < len(tasks):
        eval_indices = rng.choice(
            len(tasks), size=config.eval_sample_size, replace=False
        )
        eval_tasks = [tasks[int(i)] for i in eval_indices]
    else:
        eval_tasks = tasks

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  τ-bench Domain: {domain}  |  Algorithm: {algorithm_name}")
        print(f"  Tasks: {len(eval_tasks)}  |  Backend: {config.backend.value}")
        print(f"{'=' * 60}")

    t0 = time.perf_counter()
    prompt_trace: list[dict[str, Any]] = []

    # Init islands
    islands: list[list[PromptCandidate]] = [
        [] for _ in range(config.num_islands)
    ]

    for i, tmpl in enumerate(_TAU_SEED_TEMPLATES):
        cand = PromptCandidate(
            template=tmpl,
            temperature=float(rng.uniform(*config.temperature_range)),
            top_p=float(rng.uniform(*config.top_p_range)),
            generation=0,
        )
        cand.score = evaluate(cand, eval_tasks)
        islands[i % config.num_islands].append(cand)

        prompt_trace.append({
            "generation": 0,
            "score": round(cand.score, 1),
            "template_hash": cand.hash,
            "template_preview": cand.template[:120].replace("\n", " "),
        })

    baseline_best = max(
        (c for isl in islands for c in isl), key=lambda c: c.score
    )
    baseline_score = baseline_best.score

    if verbose:
        print(f"  Baseline best: {baseline_score:.1f}%")

    # ── Evolution loop ─────────────────────────────────────────────────
    best_overall = copy.deepcopy(baseline_best)
    history: list[tuple[int, float]] = [(0, baseline_score)]

    for gen in range(1, config.iterations + 1):
        for isl_id in range(config.num_islands):
            island = islands[isl_id]
            if not island:
                continue

            new_cands: list[PromptCandidate] = []
            for _ in range(config.population_size):
                k = min(3, len(island))
                idxs = rng.choice(len(island), size=k, replace=False)
                parent_a = max(
                    (island[int(i)] for i in idxs), key=lambda c: c.score
                )

                if rng.random() < config.crossover_rate and len(island) > 1:
                    idxs2 = rng.choice(len(island), size=k, replace=False)
                    parent_b = max(
                        (island[int(i)] for i in idxs2), key=lambda c: c.score
                    )
                    child_tmpl = _crossover_tau_templates(
                        parent_a.template, parent_b.template, rng
                    )
                else:
                    child_tmpl = parent_a.template

                if rng.random() < config.mutation_rate:
                    child_tmpl = _mutate_tau_template(
                        child_tmpl, rng, config.mutation_rate
                    )

                temp = parent_a.temperature + float(rng.normal(0, 0.1))
                temp = float(np.clip(temp, *config.temperature_range))
                top_p = parent_a.top_p + float(rng.normal(0, 0.05))
                top_p = float(np.clip(top_p, *config.top_p_range))

                child = PromptCandidate(
                    template=child_tmpl,
                    temperature=temp,
                    top_p=top_p,
                    generation=gen,
                )
                child.score = evaluate(child, eval_tasks)
                new_cands.append(child)

            combined = island + new_cands
            combined.sort(key=lambda c: c.score, reverse=True)
            islands[isl_id] = combined[: config.elite_size]

        # Migration every 3 gens
        if gen % 3 == 0 and config.num_islands > 1:
            for src in range(config.num_islands):
                if not islands[src]:
                    continue
                best_src = max(islands[src], key=lambda c: c.score)
                dest = (src + 1) % config.num_islands
                migrant = PromptCandidate(
                    template=best_src.template,
                    temperature=best_src.temperature,
                    top_p=best_src.top_p,
                    generation=best_src.generation,
                    score=best_src.score,
                )
                islands[dest].append(migrant)

        gen_best = max(
            (c for isl in islands for c in isl), key=lambda c: c.score
        )
        if gen_best.score > best_overall.score:
            best_overall = copy.deepcopy(gen_best)

        history.append((gen, best_overall.score))

        prompt_trace.append({
            "generation": gen,
            "score": round(best_overall.score, 1),
            "template_hash": best_overall.hash,
            "template_preview": best_overall.template[:120].replace("\n", " "),
        })

        if verbose:
            print(
                f"  Gen {gen:2d}/{config.iterations}  "
                f"best={best_overall.score:5.1f}%  "
                f"temp={best_overall.temperature:.3f}  "
                f"top_p={best_overall.top_p:.3f}"
            )

    wall_time = time.perf_counter() - t0

    # Final detailed scoring on all tasks with the best prompt
    final_scores: dict[str, list[float]] = {
        "tool_score": [], "info_score": [], "assertion_score": []
    }
    for task in eval_tasks:
        sys_prompt = (
            best_overall.template
            .replace("{domain}", domain)
            .replace("{policy_summary}", policy_summary)
            .replace("{tools}", tools_text)
        )
        response = client.complete(
            system_prompt=sys_prompt,
            user_message=task.user_message,
            temperature=best_overall.temperature,
            top_p=best_overall.top_p,
        )
        if response:
            sc = score_tau_bench_task(response, task)
            for k in final_scores:
                final_scores[k].append(sc[k])

    sub_scores = {
        k: round(float(np.mean(v)) * 100, 1) if v else 0.0
        for k, v in final_scores.items()
    }

    return TauBenchExperiment(
        domain=domain,
        algorithm=algorithm_name,
        backend=config.backend.value,
        n_tasks=len(eval_tasks),
        baseline_score=baseline_score,
        evolved_score=best_overall.score,
        best_prompt_template=best_overall.template,
        best_temperature=best_overall.temperature,
        best_top_p=best_overall.top_p,
        iterations=config.iterations,
        wall_time=wall_time,
        history=history,
        prompt_evolution=prompt_trace,
        sub_scores=sub_scores,
    )


# ─────────────────────────────────────────────────────────────────────────
# Baseline evaluation
# ─────────────────────────────────────────────────────────────────────────

_TAU_DEFAULT_PROMPT = textwrap.dedent("""\
    You are a customer service agent. Help the customer with their request.

{policy_summary}

{tools}
""")


def evaluate_baseline(
    tasks: list[TauBenchTask],
    policy: str,
    domain: str,
    client: LLMClient,
    verbose: bool = True,
) -> float:
    """Evaluate the default τ-bench system prompt on tasks."""
    if verbose:
        print(f"  Evaluating default τ-bench prompt baseline ({domain})...")

    policy_summary = _summarise_policy(policy)
    tools_text = _DOMAIN_TOOLS.get(domain, "")
    sys_prompt = (
        _TAU_DEFAULT_PROMPT
        .replace("{domain}", domain)
        .replace("{policy_summary}", policy_summary)
        .replace("{tools}", tools_text)
    )

    total = 0.0
    rng = np.random.default_rng(42)
    for task in tasks:
        response = client.complete(
            system_prompt=sys_prompt,
            user_message=task.user_message,
            temperature=0.1,
            top_p=0.95,
        )
        if response is None:
            total += float(rng.uniform(0, 0.1))
            continue
        scores = score_tau_bench_task(response, task)
        total += scores["composite"]

    score = (total / len(tasks) * 100.0) if tasks else 0.0
    if verbose:
        print(f"  Default prompt baseline ({domain}): {score:.1f}%")
    return score


# ─────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────


def show_prompt_evolution(experiment: TauBenchExperiment) -> None:
    """Print the prompt evolution trace for an experiment."""
    print(f"\n{'─' * 60}")
    print(f"  Prompt Evolution — {experiment.domain} ({experiment.algorithm})")
    print(f"{'─' * 60}")

    seen_hashes: set[str] = set()
    for entry in experiment.prompt_evolution:
        h = entry["template_hash"]
        marker = " *NEW*" if h not in seen_hashes else ""
        seen_hashes.add(h)
        print(
            f"  Gen {entry['generation']:2d}  "
            f"Score {entry['score']:5.1f}%  "
            f"[{h[:8]}]{marker}"
        )
        if marker:
            print(f"    → {entry['template_preview']}")

    print(
        f"\n  Baseline: {experiment.baseline_score:.1f}%  →  "
        f"Evolved: {experiment.evolved_score:.1f}%  "
        f"(Δ{experiment.evolved_score - experiment.baseline_score:+.1f}%)"
    )
    if experiment.sub_scores:
        print(
            f"  Sub-scores → Tools: {experiment.sub_scores.get('tool_score', 0):.1f}%  "
            f"Info: {experiment.sub_scores.get('info_score', 0):.1f}%  "
            f"Assertions: {experiment.sub_scores.get('assertion_score', 0):.1f}%"
        )


def show_results_table(
    experiments: list[TauBenchExperiment],
    default_baselines: dict[str, float],
) -> None:
    """Print a summary table of all experiments."""
    print(f"\n{'=' * 85}")
    print("  τ-Bench Prompt Evolution Results")
    print(f"{'=' * 85}")
    print(
        f"  {'Domain':<12} {'Algorithm':<12} {'Backend':<14} "
        f"{'Default':>8} {'Base':>8} {'Evolved':>8} {'Δ':>7} {'Time':>7}"
    )
    print(f"  {'─' * 80}")

    for exp in experiments:
        dflt = default_baselines.get(exp.domain, 0.0)
        delta = exp.evolved_score - exp.baseline_score
        print(
            f"  {exp.domain:<12} {exp.algorithm:<12} {exp.backend:<14} "
            f"{dflt:7.1f}% {exp.baseline_score:7.1f}% "
            f"{exp.evolved_score:7.1f}% {delta:+6.1f}% "
            f"{exp.wall_time:6.1f}s"
        )

    print(f"  {'─' * 80}")


# ─────────────────────────────────────────────────────────────────────────
# Experiment log persistence
# ─────────────────────────────────────────────────────────────────────────


def save_experiment_log(
    experiments: list[TauBenchExperiment],
    default_baselines: dict[str, float],
    path: str = "tau_bench_experiment_log.json",
) -> None:
    """Save experiment results to JSON for dashboard consumption."""
    log: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "default_baselines": default_baselines,
        "experiments": [],
    }
    for exp in experiments:
        entry = {
            "domain": exp.domain,
            "algorithm": exp.algorithm,
            "backend": exp.backend,
            "n_tasks": exp.n_tasks,
            "baseline_score": round(exp.baseline_score, 2),
            "evolved_score": round(exp.evolved_score, 2),
            "delta": round(exp.evolved_score - exp.baseline_score, 2),
            "best_temperature": round(exp.best_temperature, 4),
            "best_top_p": round(exp.best_top_p, 4),
            "iterations": exp.iterations,
            "wall_time": round(exp.wall_time, 1),
            "history": exp.history,
            "prompt_evolution": exp.prompt_evolution,
            "best_prompt_template": exp.best_prompt_template,
            "sub_scores": exp.sub_scores,
        }
        log["experiments"].append(entry)

    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Experiment log saved to {path}")


# ─────────────────────────────────────────────────────────────────────────
# Main — run all experiments
# ─────────────────────────────────────────────────────────────────────────

ALGORITHM_CONFIGS: dict[str, dict[str, Any]] = {
    "standard": {
        "iterations": 3,
        "population_size": 4,
        "num_islands": 2,
        "elite_size": 3,
        "mutation_rate": 0.6,
        "crossover_rate": 0.3,
        "eval_sample_size": 10,
    },
    "deep": {
        "iterations": 5,
        "population_size": 5,
        "num_islands": 2,
        "elite_size": 3,
        "mutation_rate": 0.5,
        "crossover_rate": 0.4,
        "eval_sample_size": 12,
    },
}

BENCHMARK_DOMAINS: list[tuple[str, int]] = [
    ("airline", 20),
    ("retail", 20),
]


def main() -> None:
    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  EvoSim × τ-Bench — Conversational Agent Prompt Evolution   ║
    ║  Evolving system prompts for tool-agent-user interaction     ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

    # ── Backend setup ──────────────────────────────────────────────────
    ollama_cfg = PromptEvolverConfig(
        backend=LLMBackend.OLLAMA,
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        timeout=60.0,
    )
    ollama_client = LLMClient(ollama_cfg)

    if not ollama_client.is_available():
        print("  ⚠ Ollama not available — ensure it is running at localhost:11434")
        print("  Running in mock mode (random scores) for demonstration.\n")

    # ── Load τ-bench data ──────────────────────────────────────────────
    print("  Loading τ-bench data (downloading from GitHub if needed)...")
    domain_data: dict[str, tuple[list[TauBenchTask], str]] = {}
    for domain, n_tasks in BENCHMARK_DOMAINS:
        try:
            tasks, policy = load_tau_bench_domain(domain, max_tasks=n_tasks)
            domain_data[domain] = (tasks, policy)
            print(f"    {domain}: {len(tasks)} tasks loaded")
        except Exception as exc:
            print(f"    {domain}: SKIPPED ({exc})")

    if not domain_data:
        print("  No τ-bench data loaded — exiting.")
        return

    # ── Phase 1: Default prompt baselines ──────────────────────────────
    print("\n  Phase 1: Default Prompt Baselines")
    print("  " + "─" * 50)
    default_baselines: dict[str, float] = {}
    for domain, (tasks, policy) in domain_data.items():
        score = evaluate_baseline(tasks, policy, domain, ollama_client)
        default_baselines[domain] = score

    # ── Phase 2: Ollama evolution ──────────────────────────────────────
    print("\n  Phase 2: Prompt Evolution (Ollama)")
    print("  " + "─" * 50)

    all_experiments: list[TauBenchExperiment] = []

    # Standard on all domains
    for domain, (tasks, policy) in domain_data.items():
        algo = "standard"
        algo_params = ALGORITHM_CONFIGS[algo]
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            timeout=60.0,
            **algo_params,
        )

        exp = run_tau_bench_evolution(
            tasks=tasks,
            policy=policy,
            domain=domain,
            client=ollama_client,
            config=cfg,
            algorithm_name=algo,
            verbose=True,
        )
        all_experiments.append(exp)
        show_prompt_evolution(exp)

    # Deep on airline (hardest domain)
    if "airline" in domain_data:
        algo = "deep"
        algo_params = ALGORITHM_CONFIGS[algo]
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            timeout=60.0,
            **algo_params,
        )

        tasks, policy = domain_data["airline"]
        exp = run_tau_bench_evolution(
            tasks=tasks,
            policy=policy,
            domain="airline",
            client=ollama_client,
            config=cfg,
            algorithm_name=algo,
            seed=123,
            verbose=True,
        )
        all_experiments.append(exp)
        show_prompt_evolution(exp)

    # ── Phase 3: Azure OpenAI (GPT-4.1) ───────────────────────────────
    print("\n  Phase 3: Azure OpenAI (GPT-4.1) Prompt Evolution")
    print("  " + "─" * 50)

    azure_cfg = PromptEvolverConfig(
        backend=LLMBackend.AZURE_OPENAI,
        timeout=30.0,
    )
    azure_client = LLMClient(azure_cfg)
    azure_available = azure_client.is_available()

    if not azure_available:
        print(
            "  ⚠ Azure OpenAI not available — set AZURE_OPENAI_ENDPOINT "
            "and AZURE_OPENAI_DEPLOYMENT env vars."
        )
        print("  Skipping Azure OpenAI experiments.\n")
    else:
        print(
            f"  ✓ Azure OpenAI connected: {azure_cfg.azure_deployment} "
            f"(RBAC={azure_cfg.azure_use_rbac})"
        )

        # Azure baselines
        azure_baselines: dict[str, float] = {}
        for domain, (tasks, policy) in domain_data.items():
            score = evaluate_baseline(tasks, policy, domain, azure_client)
            azure_baselines[domain] = score
            default_baselines[f"{domain}_azure"] = score

        # Standard evolution on all domains
        for domain, (tasks, policy) in domain_data.items():
            algo = "standard"
            algo_params = ALGORITHM_CONFIGS[algo]
            az_cfg = PromptEvolverConfig(
                backend=LLMBackend.AZURE_OPENAI,
                timeout=30.0,
                **algo_params,
            )

            exp = run_tau_bench_evolution(
                tasks=tasks,
                policy=policy,
                domain=domain,
                client=azure_client,
                config=az_cfg,
                algorithm_name=algo,
                verbose=True,
            )
            all_experiments.append(exp)
            show_prompt_evolution(exp)

        # Deep on airline
        if "airline" in domain_data:
            algo = "deep"
            algo_params = ALGORITHM_CONFIGS[algo]
            az_cfg = PromptEvolverConfig(
                backend=LLMBackend.AZURE_OPENAI,
                timeout=30.0,
                **algo_params,
            )

            tasks, policy = domain_data["airline"]
            exp = run_tau_bench_evolution(
                tasks=tasks,
                policy=policy,
                domain="airline",
                client=azure_client,
                config=az_cfg,
                algorithm_name=algo,
                seed=123,
                verbose=True,
            )
            all_experiments.append(exp)
            show_prompt_evolution(exp)

    # ── Results ────────────────────────────────────────────────────────
    show_results_table(all_experiments, default_baselines)

    # ── Save experiment log ────────────────────────────────────────────
    log_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "tau_bench_experiment_log.json"
    )
    save_experiment_log(all_experiments, default_baselines, log_path)

    # ── Print best evolved prompt ──────────────────────────────────────
    if all_experiments:
        best_exp = max(all_experiments, key=lambda e: e.evolved_score)
        print(f"\n{'=' * 60}")
        print(f"  Best Evolved Prompt ({best_exp.domain} / {best_exp.algorithm})")
        print(f"  Score: {best_exp.evolved_score:.1f}%")
        print(f"{'=' * 60}")
        preview = best_exp.best_prompt_template[:500]
        for line in preview.split("\n"):
            print(f"    {line}")
        if len(best_exp.best_prompt_template) > 500:
            print(f"    ... ({len(best_exp.best_prompt_template)} chars total)")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
