#!/usr/bin/env python3
"""
Cookbook Recipe 47 — Browser Agent Tool-Calling Prompt Evolution
================================================================

Evolves system prompts on the **Tool-Calling Browser Agent Tasks**
benchmark (DataCreatorAI) — 1,062 multi-turn conversations covering
practical agentic workflows:

* **Ticket booking** — train reservations with parameter gathering
* **Form filling** — dynamic multi-field forms, document uploads
* **Payment processing** — UPI/financial transactions
* **App integration** — GitHub, Figma, Gmail, Trello workflows
* **Failure recovery** — payment failures, retries, invalid params,
  waitlists, graceful degradation

Published by DataCreatorAI — synthetic but high-fidelity multi-turn
dialogues with structured ``tool_calls`` and simulated tool responses
including error states.

Algorithm experiments
---------------------
* **Standard** — 3 iterations, pop 4, 2 islands  (balanced)
* **Deep**     — 5 iterations, pop 5, 2 islands  (thorough)

Categories
----------
* **normal**           — successful tool-call conversations
* **failure_recovery** — conversations with error/retry scenarios
* **multi_tool**       — conversations with 3+ tool call turns

Usage::

    uv sync --extra llm
    export OLLAMA_MODEL=llama3.2
    uv run python examples/cookbook/prompt_evolution_browser_agent.py

The script saves an experiment log to
``browser_agent_experiment_log.json`` for dashboard consumption,
including per-generation prompt snapshots.

References
----------
* Dataset : https://huggingface.co/datasets/DataCreatorAI/tool-calling-browser-agent-tasks
* License : CC-BY-4.0
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

from prompture.prompt_evolver import (
    LLMBackend,
    LLMClient,
    PromptCandidate,
    PromptEvolverConfig,
)


# ─────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class BrowserAgentCase:
    """A single browser-agent benchmark test case."""

    case_id: str
    conversation_context: list[dict[str, Any]]  # messages up to the eval point
    expected_tool_call: dict[str, Any]           # expected tool_calls[0]
    expected_reply: str                          # expected assistant content
    category: str                                # normal / failure_recovery / multi_tool
    num_tool_turns: int                          # total tool-call turns in conversation

    @property
    def expected_function_name(self) -> str:
        """Extract the expected function name."""
        try:
            return self.expected_tool_call["function"]["name"]
        except (KeyError, TypeError):
            return ""

    @property
    def expected_arguments(self) -> dict[str, Any]:
        """Extract the expected function arguments."""
        try:
            args = self.expected_tool_call["function"]["arguments"]
            if isinstance(args, str):
                return json.loads(args)
            return args if isinstance(args, dict) else {}
        except (KeyError, TypeError, json.JSONDecodeError):
            return {}


@dataclass
class BrowserAgentExperiment:
    """Result container for one evolution run."""

    category: str
    algorithm: str
    backend: str
    n_cases: int
    baseline_score: float
    evolved_score: float
    best_prompt_template: str
    best_temperature: float
    best_top_p: float
    iterations: int
    wall_time: float
    history: list[tuple[int, float]] = field(default_factory=list)
    prompt_evolution: list[dict[str, Any]] = field(default_factory=list)
    prompt_mutations: list[dict[str, Any]] = field(default_factory=list)
    func_name_accuracy: float = 0.0
    arg_accuracy: float = 0.0


# ─────────────────────────────────────────────────────────────────────────
# Tool-call parsing
# ─────────────────────────────────────────────────────────────────────────

# Match JSON tool_calls in various formats the model may produce
_FUNC_CALL_RE = re.compile(
    r'"name"\s*:\s*"(\w+)".*?"arguments"\s*:\s*(\{[^}]*\})',
    re.DOTALL,
)

# Simpler pattern: function_name(args) or {"name": "x", "arguments": {...}}
_SIMPLE_FUNC_RE = re.compile(r'(\w+)\s*\(([^)]*)\)')


def _parse_tool_call(text: str) -> tuple[str, dict[str, Any]]:
    """Parse a model response to extract function name and arguments.

    Tries multiple formats:
    1. JSON tool_calls format
    2. function_name(arg=val, ...) format
    3. Plain function name mention
    """
    if not text:
        return "", {}

    # Try JSON format first
    m = _FUNC_CALL_RE.search(text)
    if m:
        name = m.group(1)
        try:
            args = json.loads(m.group(2))
            return name, args
        except json.JSONDecodeError:
            return name, {}

    # Try to find JSON in the response
    try:
        # Look for tool_calls array
        tc_match = re.search(r'"tool_calls"\s*:\s*\[(.+?)\]', text, re.DOTALL)
        if tc_match:
            inner = tc_match.group(1)
            m2 = _FUNC_CALL_RE.search(inner)
            if m2:
                name = m2.group(1)
                try:
                    args = json.loads(m2.group(2))
                    return name, args
                except json.JSONDecodeError:
                    return name, {}
    except Exception:
        pass

    # Try JSON object with name and arguments
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            if "function" in data and isinstance(data["function"], dict):
                name = data["function"].get("name", "")
                args = data["function"].get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                return name, args if isinstance(args, dict) else {}
            if "name" in data:
                name = data["name"]
                args = data.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                return name, args if isinstance(args, dict) else {}
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to find just function name mentions from known function list
    known_funcs = [
        "manage_booking", "financial_services", "content_processing",
        "web_search", "summarize_text", "create_document", "download_file",
        "fill_passenger_form", "upload_file", "app_open", "resource_find",
        "action_execute", "file_download", "resource_edit", "file_upload",
        "content_update", "share_send", "shopping_ecommerce",
    ]
    text_lower = text.lower()
    for func in known_funcs:
        if func in text_lower:
            return func, {}

    return "", {}


def _normalise(s: str) -> str:
    """Normalise for comparison."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


# ─────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────


def score_browser_agent_case(
    response: str, case: BrowserAgentCase,
) -> tuple[float, dict[str, Any]]:
    """Score a response against a browser-agent test case.

    Returns (score 0.0–1.0, detail dict).

    Scoring components:
    - Function name match  (0.4 weight): did the model pick the right function?
    - Argument match       (0.4 weight): correct arg keys & values?
    - Format compliance    (0.2 weight): is the output in tool_calls format?
    """
    detail: dict[str, Any] = {
        "func_name_match": False,
        "arg_score": 0.0,
        "format_ok": False,
        "error": None,
    }

    pred_name, pred_args = _parse_tool_call(response)

    exp_name = case.expected_function_name
    exp_args = case.expected_arguments

    # Format compliance
    format_ok = bool(pred_name)
    detail["format_ok"] = format_ok
    format_score = 1.0 if format_ok else 0.0

    if not format_ok:
        detail["error"] = "NO_TOOL_CALL"
        return 0.0, detail

    # Function name match
    name_match = _normalise(pred_name) == _normalise(exp_name)
    detail["func_name_match"] = name_match
    name_score = 1.0 if name_match else 0.0

    if not name_match:
        detail["error"] = "FUNC_NAME_MISMATCH"

    # Argument match
    if exp_args:
        total_args = len(exp_args)
        matched = 0.0
        for key, exp_val in exp_args.items():
            pred_val = pred_args.get(key)
            if pred_val is None:
                continue
            if isinstance(exp_val, (list, dict)):
                # Complex args — check type match only
                if isinstance(pred_val, type(exp_val)):
                    matched += 0.7
                else:
                    matched += 0.2
            elif _normalise(str(pred_val)) == _normalise(str(exp_val)):
                matched += 1.0
            elif key in pred_args:
                matched += 0.3  # partial credit for right key
        arg_score = matched / total_args if total_args > 0 else 1.0
    else:
        arg_score = 1.0 if not pred_args else 0.5

    detail["arg_score"] = arg_score

    return 0.4 * name_score + 0.4 * arg_score + 0.2 * format_score, detail


# ─────────────────────────────────────────────────────────────────────────
# Dataset loading from HuggingFace
# ─────────────────────────────────────────────────────────────────────────

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".browser_agent_cache"

_HF_REPO = "DataCreatorAI/tool-calling-browser-agent-tasks"


def _download_browser_agent_data() -> list[dict[str, Any]]:
    """Download dataset from HuggingFace and cache locally."""
    cache_file = _CACHE_DIR / "benchmark_data.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    print("  Downloading browser-agent benchmark from HuggingFace...")

    conversations: list[dict[str, Any]] = []

    # Primary: datasets library
    try:
        from datasets import load_dataset

        ds = load_dataset(_HF_REPO, split="train")
        for row in ds:
            msgs = row["messages"]
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            conversations.append({"messages": msgs})
        print(f"    Loaded {len(conversations)} via datasets library")
    except Exception as e1:
        print(f"    datasets library failed: {e1}")
        # Fallback: parquet via huggingface_hub
        try:
            from huggingface_hub import hf_hub_download

            local_path = hf_hub_download(
                repo_id=_HF_REPO,
                filename="data/train-00000-of-00001.parquet",
                repo_type="dataset",
            )

            import pyarrow.parquet as pq

            table = pq.read_table(local_path)
            rows = table.to_pydict()
            for msgs in rows["messages"]:
                if isinstance(msgs, list):
                    conversations.append({"messages": msgs})
                else:
                    conversations.append({"messages": json.loads(msgs)})
            print(f"    Loaded {len(conversations)} via parquet")
        except Exception as e2:
            print(f"    parquet fallback failed: {e2}")
            # Final fallback: HF API (paginated)
            import urllib.request

            offset = 0
            batch_size = 100
            while True:
                url = (
                    "https://datasets-server.huggingface.co/rows?"
                    f"dataset={_HF_REPO.replace('/', '%2F')}"
                    f"&config=default&split=train&offset={offset}"
                    f"&length={batch_size}"
                )
                with urllib.request.urlopen(url, timeout=120) as resp:
                    data = json.loads(resp.read())
                batch = data.get("rows", [])
                if not batch:
                    break
                for r in batch:
                    conversations.append(r["row"])
                offset += len(batch)
                if len(batch) < batch_size:
                    break
            print(f"    Loaded {len(conversations)} via HF API")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w") as f:
        json.dump(conversations, f)
    print(f"    Cached {len(conversations)} conversations")

    return conversations


def _classify_conversation(messages: list[dict[str, Any]]) -> str:
    """Classify a conversation as normal, failure_recovery, or multi_tool."""
    tool_call_count = 0
    has_error = False

    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            tool_call_count += 1
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if isinstance(content, str) and (
                '"error"' in content.lower()
                or '"status":"error"' in content.replace(" ", "")
                or '"status": "error"' in content
                or "PAYMENT_FAILED" in content
                or "FAILED" in content
                or '"code":' in content.replace(" ", "")
            ):
                has_error = True

    if has_error:
        return "failure_recovery"
    if tool_call_count >= 3:
        return "multi_tool"
    return "normal"


def _extract_first_tool_call_case(
    messages: list[dict[str, Any]],
    case_id: str,
) -> BrowserAgentCase | None:
    """Extract the first tool-call evaluation point from a conversation.

    Returns the conversation context up to the first assistant tool_calls
    message, and the expected tool_call as ground truth.
    """
    tool_call_count = 0
    context: list[dict[str, Any]] = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        tool_calls = msg.get("tool_calls")

        if role == "assistant" and tool_calls:
            tool_call_count += 1

            # Use the first tool_calls instance as eval point
            if tool_call_count == 1:
                category = _classify_conversation(messages)
                return BrowserAgentCase(
                    case_id=case_id,
                    conversation_context=context.copy(),
                    expected_tool_call=tool_calls[0],
                    expected_reply=msg.get("content", ""),
                    category=category,
                    num_tool_turns=sum(
                        1
                        for m in messages
                        if m.get("role") == "assistant" and m.get("tool_calls")
                    ),
                )

        # Build context up to the eval point
        simplified = {"role": role, "content": msg.get("content", "")}
        context.append(simplified)

    return None


def load_browser_agent_dataset(
    max_per_category: int = 30,
    categories: list[str] | None = None,
) -> dict[str, list[BrowserAgentCase]]:
    """Load and group browser-agent benchmark data by category.

    Parameters
    ----------
    max_per_category : int
        Maximum cases per category.
    categories : list[str] | None
        Which categories to include. Defaults to all.

    Returns
    -------
    dict mapping category name to list of BrowserAgentCase
    """
    if categories is None:
        categories = ["normal", "failure_recovery", "multi_tool"]

    raw = _download_browser_agent_data()
    rng = np.random.default_rng(42)

    # Extract all cases
    all_cases: dict[str, list[BrowserAgentCase]] = {c: [] for c in categories}

    for idx, conv in enumerate(raw):
        msgs = conv.get("messages", [])
        if not msgs:
            continue

        case = _extract_first_tool_call_case(msgs, f"conv_{idx}")
        if case is None:
            continue

        if case.category in all_cases:
            all_cases[case.category].append(case)

    # Subsample
    by_category: dict[str, list[BrowserAgentCase]] = {}
    for cat_name, cases in all_cases.items():
        if not cases:
            continue
        if len(cases) > max_per_category:
            indices = rng.choice(len(cases), size=max_per_category, replace=False)
            cases = [cases[int(i)] for i in indices]
        by_category[cat_name] = cases

    return by_category


# ─────────────────────────────────────────────────────────────────────────
# Seed prompt templates
# ─────────────────────────────────────────────────────────────────────────

_SEED_TEMPLATES = [
    # T1 — Minimal
    textwrap.dedent("""\
You are a browser-based AI assistant that helps users complete tasks by \
calling the right tool functions. Given the conversation, generate the \
correct tool call as a JSON object with "name" and "arguments" fields.

Conversation:
{browser_conversation}
"""),

    # T2 — Step-by-step
    textwrap.dedent("""\
You are an expert tool-calling browser agent. Follow these steps:
1. Read the full conversation carefully.
2. Identify what the user needs done next.
3. Select the correct function to call.
4. Extract all required parameter values from the conversation.
5. Output a JSON tool call: {{"name": "function_name", "arguments": {{...}}}}

Conversation:
{browser_conversation}
"""),

    # T3 — Constraint-focused
    textwrap.dedent("""\
You are a precise tool-calling agent. Rules:
- Output EXACTLY ONE function call as JSON: {{"name": "func", "arguments": {{...}}}}
- Match function names EXACTLY from the available tools.
- Extract parameter values from the conversation — do NOT invent values.
- Include ALL required parameters with correct types.
- Output ONLY the JSON tool call, nothing else.

Conversation:
{browser_conversation}
"""),

    # T4 — Failure-recovery aware
    textwrap.dedent("""\
You are an agentic browser assistant specialised in multi-step workflows \
including ticket booking, form filling, payment processing, and app \
integration. You handle failures gracefully — retrying, adjusting \
parameters, or pivoting to alternatives when tools return errors.

Given the conversation so far, generate the NEXT tool call as JSON:
{{"name": "function_name", "arguments": {{...}}}}

Extract all values from the conversation. If a previous call failed, \
adjust your approach based on the error message.

Conversation:
{browser_conversation}
"""),
]


# ─────────────────────────────────────────────────────────────────────────
# Mutation operators
# ─────────────────────────────────────────────────────────────────────────

_MUTATIONS: list[str] = [
    # Parameter extraction
    "Add: 'Extract ALL parameter values verbatim from the user messages.'",
    "Append: 'Pay attention to names, dates, IDs, amounts, and email addresses.'",
    "Insert: 'Use exact values the user provided — do not rephrase or abbreviate.'",
    # Function selection
    "Add: 'Choose the function that best matches the user intent in this turn.'",
    "Inject: 'For booking tasks use manage_booking; for payments use financial_services.'",
    "Insert: 'Match the function name exactly — check spelling carefully.'",
    # Format compliance
    'Append: \'Output ONLY valid JSON: {"name": "func", "arguments": {...}}\'',
    "Add: 'Do not wrap the JSON in markdown code blocks or add explanatory text.'",
    "Insert: 'The arguments field must be a JSON object, not a string.'",
    # Failure recovery
    "Add: 'If a previous tool result shows an error, adjust your next call accordingly.'",
    "Inject: 'For payment failures, retry with the same or alternative payment method.'",
    "Append: 'Check previous tool responses for context before making the next call.'",
    # Multi-step reasoning
    "Add: 'Track the state across turns — booking first, then payment, then confirmation.'",
    "Prepend: 'Read the ENTIRE conversation including tool responses before acting.'",
    "Insert: 'Use information from earlier tool responses in subsequent calls.'",
    # Chain-of-thought
    "Prepend: 'Think step by step about which function to call and with what parameters.'",
    "Insert: 'First identify the task, then the function, then gather parameters.'",
    "Add: 'Reason about the user intent before outputting the tool call.'",
]


def _mutate_template(
    template: str, rng: np.random.Generator, rate: float = 0.5,
) -> str:
    """Apply a random mutation to a prompt template."""
    mutation = rng.choice(_MUTATIONS)
    lines = template.strip().split("\n")

    action = mutation.split(":")[0].strip().lower()
    content = mutation.split(":", 1)[1].strip().strip("'\"")

    if "prepend" in action:
        lines.insert(0, content)
    elif "append" in action:
        lines.append(content)
    elif "inject" in action or "insert" in action:
        pos = int(rng.integers(1, max(2, len(lines))))
        lines.insert(pos, content)
    elif "add" in action:
        pos = int(rng.integers(len(lines) // 2, max(len(lines) // 2 + 1, len(lines))))
        lines.insert(pos, content)
    else:
        lines.append(content)

    result = "\n".join(lines)

    # Ensure placeholder survives
    if "{browser_conversation}" not in result:
        result += "\n\n{browser_conversation}"

    return result


def _crossover_templates(a: str, b: str, rng: np.random.Generator) -> str:
    """Crossover two prompt templates."""
    la = a.strip().split("\n")
    lb = b.strip().split("\n")
    ca = int(rng.integers(1, max(2, len(la))))
    cb = int(rng.integers(1, max(2, len(lb))))
    child = "\n".join(la[:ca] + lb[cb:])
    if "{browser_conversation}" not in child:
        child += "\n\n{browser_conversation}"
    return child


# ─────────────────────────────────────────────────────────────────────────
# Conversation formatting
# ─────────────────────────────────────────────────────────────────────────


def _format_conversation(context: list[dict[str, Any]]) -> str:
    """Format conversation context for the prompt template."""
    lines: list[str] = []
    for msg in context:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content:
            lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────
# Evolution engine
# ─────────────────────────────────────────────────────────────────────────


def run_browser_agent_evolution(
    cases: list[BrowserAgentCase],
    client: LLMClient,
    config: PromptEvolverConfig,
    algorithm_name: str = "standard",
    seed: int = 42,
    verbose: bool = True,
    category_override: str | None = None,
) -> BrowserAgentExperiment:
    """Run prompt evolution on browser-agent test cases.

    Returns a BrowserAgentExperiment with full tracking including
    per-generation prompt mutations.
    """
    rng = np.random.default_rng(seed)
    category = category_override or (cases[0].category if cases else "unknown")

    # ── Evaluate a candidate ───────────────────────────────────────────
    def evaluate(
        candidate: PromptCandidate, eval_cases: list[BrowserAgentCase],
    ) -> tuple[float, float, float]:
        """Returns (overall_score, func_name_acc, arg_acc)."""
        total = 0.0
        name_hits = 0
        arg_total = 0.0
        for case in eval_cases:
            conversation_text = _format_conversation(case.conversation_context)
            sys_prompt = candidate.template.replace(
                "{browser_conversation}", conversation_text,
            )

            response = client.complete(
                system_prompt=sys_prompt,
                user_message="Generate the next tool call as JSON:",
                temperature=candidate.temperature,
                top_p=candidate.top_p,
            )
            if response is None:
                total += float(rng.uniform(0, 0.1))
                continue

            score, detail = score_browser_agent_case(response, case)
            total += score
            if detail["func_name_match"]:
                name_hits += 1
            arg_total += detail["arg_score"]

        n = len(eval_cases) if eval_cases else 1
        return (
            (total / n * 100.0),
            (name_hits / n * 100.0),
            (arg_total / n * 100.0),
        )

    # ── Subsample for evaluation ───────────────────────────────────────
    if config.eval_sample_size and config.eval_sample_size < len(cases):
        eval_indices = rng.choice(
            len(cases), size=config.eval_sample_size, replace=False,
        )
        eval_cases = [cases[int(i)] for i in eval_indices]
    else:
        eval_cases = cases

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  Category: {category}  |  Algorithm: {algorithm_name}")
        print(
            f"  Cases: {len(eval_cases)}  |  Backend: {config.backend.value}",
        )
        print(f"{'=' * 60}")

    t0 = time.perf_counter()
    prompt_trace: list[dict[str, Any]] = []
    mutation_trace: list[dict[str, Any]] = []

    # Init islands
    islands: list[list[PromptCandidate]] = [
        [] for _ in range(config.num_islands)
    ]

    best_name_acc = 0.0
    best_arg_acc = 0.0

    for i, tmpl in enumerate(_SEED_TEMPLATES):
        cand = PromptCandidate(
            template=tmpl,
            temperature=float(rng.uniform(*config.temperature_range)),
            top_p=float(rng.uniform(*config.top_p_range)),
            generation=0,
        )
        score, name_acc, arg_acc = evaluate(cand, eval_cases)
        cand.score = score
        islands[i % config.num_islands].append(cand)

        if score > best_name_acc:
            best_name_acc = name_acc
            best_arg_acc = arg_acc

        prompt_trace.append({
            "generation": 0,
            "score": round(score, 1),
            "template_hash": cand.hash,
            "template_preview": cand.template[:120].replace("\n", " "),
            "full_template": cand.template,
        })

    baseline_best = max(
        (c for isl in islands for c in isl), key=lambda c: c.score,
    )
    baseline_score = baseline_best.score

    if verbose:
        print(f"  Baseline best: {baseline_score:.1f}%")

    # ── Evolution loop ─────────────────────────────────────────────────
    best_overall = copy.deepcopy(baseline_best)
    history: list[tuple[int, float]] = [(0, baseline_score)]

    for gen in range(1, config.iterations + 1):
        gen_mutations: list[dict[str, str]] = []

        for isl_id in range(config.num_islands):
            island = islands[isl_id]
            if not island:
                continue

            new_cands: list[PromptCandidate] = []
            for _ in range(config.population_size):
                k = min(3, len(island))
                idxs = rng.choice(len(island), size=k, replace=False)
                parent_a = max(
                    (island[int(i)] for i in idxs), key=lambda c: c.score,
                )

                if rng.random() < config.crossover_rate and len(island) > 1:
                    idxs2 = rng.choice(len(island), size=k, replace=False)
                    parent_b = max(
                        (island[int(i)] for i in idxs2), key=lambda c: c.score,
                    )
                    child_tmpl = _crossover_templates(
                        parent_a.template, parent_b.template, rng,
                    )
                    gen_mutations.append({
                        "type": "crossover",
                        "parent_a": parent_a.hash[:8],
                        "parent_b": parent_b.hash[:8],
                    })
                else:
                    child_tmpl = parent_a.template

                if rng.random() < config.mutation_rate:
                    old_tmpl = child_tmpl
                    child_tmpl = _mutate_template(
                        child_tmpl, rng, config.mutation_rate,
                    )
                    gen_mutations.append({
                        "type": "mutation",
                        "parent": parent_a.hash[:8],
                        "diff_preview": child_tmpl[:200].replace("\n", " "),
                    })

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
                score, name_acc, arg_acc = evaluate(child, eval_cases)
                child.score = score
                new_cands.append(child)

                if score > best_overall.score:
                    best_name_acc = name_acc
                    best_arg_acc = arg_acc

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
            (c for isl in islands for c in isl), key=lambda c: c.score,
        )
        if gen_best.score > best_overall.score:
            best_overall = copy.deepcopy(gen_best)

        history.append((gen, best_overall.score))

        prompt_trace.append({
            "generation": gen,
            "score": round(best_overall.score, 1),
            "template_hash": best_overall.hash,
            "template_preview": best_overall.template[:120].replace("\n", " "),
            "full_template": best_overall.template,
        })

        mutation_trace.append({
            "generation": gen,
            "mutations": gen_mutations,
            "best_score": round(best_overall.score, 1),
            "best_hash": best_overall.hash[:8],
        })

        if verbose:
            print(
                f"  Gen {gen:2d}/{config.iterations}  "
                f"best={best_overall.score:5.1f}%  "
                f"temp={best_overall.temperature:.3f}  "
                f"top_p={best_overall.top_p:.3f}",
            )

    wall_time = time.perf_counter() - t0

    return BrowserAgentExperiment(
        category=category,
        algorithm=algorithm_name,
        backend=config.backend.value,
        n_cases=len(eval_cases),
        baseline_score=baseline_score,
        evolved_score=best_overall.score,
        best_prompt_template=best_overall.template,
        best_temperature=best_overall.temperature,
        best_top_p=best_overall.top_p,
        iterations=config.iterations,
        wall_time=wall_time,
        history=history,
        prompt_evolution=prompt_trace,
        prompt_mutations=mutation_trace,
        func_name_accuracy=best_name_acc,
        arg_accuracy=best_arg_acc,
    )


# ─────────────────────────────────────────────────────────────────────────
# Baseline evaluation
# ─────────────────────────────────────────────────────────────────────────

_DEFAULT_PROMPT = textwrap.dedent("""\
You are a helpful assistant. Given the conversation, generate the \
appropriate tool call as a JSON object with "name" and "arguments" fields.

Conversation:
{browser_conversation}
""")


def evaluate_baseline(
    cases: list[BrowserAgentCase],
    category_name: str,
    client: LLMClient,
    verbose: bool = True,
) -> float:
    """Evaluate the default prompt on cases."""
    if verbose:
        print(f"  Evaluating default prompt baseline ({category_name})...")

    rng = np.random.default_rng(42)
    total = 0.0
    for case in cases:
        conversation_text = _format_conversation(case.conversation_context)
        sys_prompt = _DEFAULT_PROMPT.replace(
            "{browser_conversation}", conversation_text,
        )

        response = client.complete(
            system_prompt=sys_prompt,
            user_message="Generate the next tool call as JSON:",
            temperature=0.1,
            top_p=0.95,
        )
        if response is None:
            total += float(rng.uniform(0, 0.1))
            continue

        score, _ = score_browser_agent_case(response, case)
        total += score

    result = (total / len(cases) * 100.0) if cases else 0.0
    if verbose:
        print(f"  Default prompt baseline ({category_name}): {result:.1f}%")
    return result


# ─────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────


def show_prompt_evolution(experiment: BrowserAgentExperiment) -> None:
    """Print the prompt evolution trace including mutations."""
    print(f"\n{'─' * 60}")
    print(f"  Prompt Evolution — {experiment.category} ({experiment.algorithm})")
    print(f"{'─' * 60}")

    seen_hashes: set[str] = set()
    for entry in experiment.prompt_evolution:
        h = entry["template_hash"]
        marker = " *NEW*" if h not in seen_hashes else ""
        seen_hashes.add(h)
        print(
            f"  Gen {entry['generation']:2d}  "
            f"Score {entry['score']:5.1f}%  "
            f"[{h[:8]}]{marker}",
        )
        if marker:
            print(f"    → {entry['template_preview']}")

    # Show mutation trace
    if experiment.prompt_mutations:
        print(f"\n  Mutation trace:")
        for mt in experiment.prompt_mutations:
            n_muts = len(mt["mutations"])
            print(
                f"    Gen {mt['generation']}: {n_muts} ops → "
                f"best={mt['best_score']}% [{mt['best_hash']}]",
            )

    print(
        f"\n  Baseline: {experiment.baseline_score:.1f}%  →  "
        f"Evolved: {experiment.evolved_score:.1f}%  "
        f"(Δ{experiment.evolved_score - experiment.baseline_score:+.1f}%)",
    )
    print(
        f"  Func Name Accuracy: {experiment.func_name_accuracy:.1f}%  |  "
        f"Arg Accuracy: {experiment.arg_accuracy:.1f}%",
    )


def show_results_table(
    experiments: list[BrowserAgentExperiment],
    default_baselines: dict[str, float],
) -> None:
    """Print a summary table of all experiments."""
    print(f"\n{'=' * 90}")
    print("  Browser Agent Prompt Evolution Results")
    print(f"{'=' * 90}")
    print(
        f"  {'Category':<20} {'Algorithm':<12} {'Backend':<14} "
        f"{'Default':>8} {'Base':>8} {'Evolved':>8} {'Δ':>7} {'Time':>7}",
    )
    print(f"  {'─' * 85}")

    for exp in experiments:
        dflt = default_baselines.get(exp.category, 0.0)
        delta = exp.evolved_score - exp.baseline_score
        print(
            f"  {exp.category:<20} {exp.algorithm:<12} {exp.backend:<14} "
            f"{dflt:7.1f}% {exp.baseline_score:7.1f}% "
            f"{exp.evolved_score:7.1f}% {delta:+6.1f}% "
            f"{exp.wall_time:6.1f}s",
        )

    print(f"  {'─' * 85}")


# ─────────────────────────────────────────────────────────────────────────
# Experiment log persistence
# ─────────────────────────────────────────────────────────────────────────


def save_experiment_log(
    experiments: list[BrowserAgentExperiment],
    default_baselines: dict[str, float],
    path: str = "browser_agent_experiment_log.json",
) -> None:
    """Save experiment results to JSON for dashboard consumption."""
    log: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "default_baselines": default_baselines,
        "experiments": [],
    }
    for exp in experiments:
        entry = {
            "category": exp.category,
            "algorithm": exp.algorithm,
            "backend": exp.backend,
            "n_cases": exp.n_cases,
            "baseline_score": round(exp.baseline_score, 2),
            "evolved_score": round(exp.evolved_score, 2),
            "delta": round(exp.evolved_score - exp.baseline_score, 2),
            "best_temperature": round(exp.best_temperature, 4),
            "best_top_p": round(exp.best_top_p, 4),
            "iterations": exp.iterations,
            "wall_time": round(exp.wall_time, 1),
            "history": exp.history,
            "prompt_evolution": exp.prompt_evolution,
            "prompt_mutations": exp.prompt_mutations,
            "best_prompt_template": exp.best_prompt_template,
            "func_name_accuracy": round(exp.func_name_accuracy, 2),
            "arg_accuracy": round(exp.arg_accuracy, 2),
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

CATEGORIES = ["normal", "failure_recovery", "multi_tool"]
MAX_PER_CATEGORY = 30


def main() -> None:
    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  EvoSim × Browser Agent — Tool-Calling Prompt Evolution      ║
    ║  1,062 conversations · booking/forms/payments/recovery        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

    # Load dataset
    print("  Loading browser-agent benchmark data...")
    by_category = load_browser_agent_dataset(
        max_per_category=MAX_PER_CATEGORY,
        categories=CATEGORIES,
    )

    for cat, cases in by_category.items():
        print(f"    {cat:<20}: {len(cases):3d} cases")

    all_experiments: list[BrowserAgentExperiment] = []
    default_baselines: dict[str, float] = {}

    # ── Phase 1: Default baselines ─────────────────────────────────────
    print("\n  Phase 1: Default Prompt Baselines")
    print(f"  {'─' * 50}")

    ollama_model = os.environ.get("OLLAMA_MODEL", "llama3.2")
    ollama_cfg = PromptEvolverConfig(
        backend=LLMBackend.OLLAMA,
        ollama_model=ollama_model,
        timeout=60.0,
    )
    ollama_client = LLMClient(ollama_cfg)

    if not ollama_client.is_available():
        print("  ⚠ Ollama not available — ensure it is running at localhost:11434")
        print("  Running in mock mode (random scores) for demonstration.\n")

    for cat_name, cases in by_category.items():
        score = evaluate_baseline(cases, cat_name, ollama_client)
        default_baselines[cat_name] = round(score, 1)

    # ── Phase 2: Ollama evolution ──────────────────────────────────────
    print("\n  Phase 2: Prompt Evolution (Ollama)")
    print(f"  {'─' * 50}")

    for cat_name, cases in by_category.items():
        # Standard evolution on all categories
        cfg_std = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_model=ollama_model,
            timeout=60.0,
            **ALGORITHM_CONFIGS["standard"],
        )
        exp = run_browser_agent_evolution(
            cases, ollama_client, cfg_std,
            algorithm_name="standard",
            category_override=cat_name,
        )
        show_prompt_evolution(exp)
        all_experiments.append(exp)

    # Deep evolution on failure_recovery (most interesting for deep search)
    if "failure_recovery" in by_category:
        cfg_deep = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_model=ollama_model,
            timeout=60.0,
            **ALGORITHM_CONFIGS["deep"],
        )
        exp = run_browser_agent_evolution(
            by_category["failure_recovery"], ollama_client, cfg_deep,
            algorithm_name="deep",
            category_override="failure_recovery",
        )
        show_prompt_evolution(exp)
        all_experiments.append(exp)

    # ── Phase 3: Azure OpenAI (GPT-4.1) ───────────────────────────────
    print("\n  Phase 3: Azure OpenAI (GPT-4.1) Prompt Evolution")
    print(f"  {'─' * 50}")

    azure_cfg = PromptEvolverConfig(
        backend=LLMBackend.AZURE_OPENAI,
        timeout=30.0,
    )
    azure_client = LLMClient(azure_cfg)
    azure_available = azure_client.is_available()

    if not azure_available:
        print(
            "  ⚠ Azure OpenAI not available — set AZURE_OPENAI_ENDPOINT "
            "and AZURE_OPENAI_DEPLOYMENT env vars.",
        )
        print("  Skipping Azure OpenAI experiments.\n")
    else:
        print(
            f"  ✓ Azure OpenAI connected: {azure_cfg.azure_deployment} "
            f"(RBAC={azure_cfg.azure_use_rbac})",
        )

        # Azure baselines
        for cat_name, cases in by_category.items():
            score = evaluate_baseline(cases, cat_name, azure_client)
            default_baselines[f"{cat_name}_azure"] = round(score, 1)

        # Standard on all categories
        for cat_name, cases in by_category.items():
            az_std = PromptEvolverConfig(
                backend=LLMBackend.AZURE_OPENAI,
                timeout=30.0,
                **ALGORITHM_CONFIGS["standard"],
            )
            exp = run_browser_agent_evolution(
                cases, azure_client, az_std,
                algorithm_name="standard",
                category_override=cat_name,
            )
            show_prompt_evolution(exp)
            all_experiments.append(exp)

        # Deep on failure_recovery
        if "failure_recovery" in by_category:
            az_deep = PromptEvolverConfig(
                backend=LLMBackend.AZURE_OPENAI,
                timeout=30.0,
                **ALGORITHM_CONFIGS["deep"],
            )
            exp = run_browser_agent_evolution(
                by_category["failure_recovery"], azure_client, az_deep,
                algorithm_name="deep",
                category_override="failure_recovery",
            )
            show_prompt_evolution(exp)
            all_experiments.append(exp)

    # ── Results ────────────────────────────────────────────────────────
    show_results_table(all_experiments, default_baselines)

    # Save log
    log_path = str(
        Path(__file__).resolve().parent.parent.parent
        / "browser_agent_experiment_log.json"
    )
    save_experiment_log(all_experiments, default_baselines, log_path)

    # Show best prompt
    if all_experiments:
        best_exp = max(all_experiments, key=lambda e: e.evolved_score)
        print(f"\n{'=' * 60}")
        print(f"  Best Evolved Prompt ({best_exp.category} / {best_exp.algorithm})")
        print(f"  Score: {best_exp.evolved_score:.1f}%")
        print(
            f"  Func Name Acc: {best_exp.func_name_accuracy:.1f}%  |  "
            f"Arg Acc: {best_exp.arg_accuracy:.1f}%",
        )
        print(f"{'=' * 60}")
        # Show without the conversation placeholder
        clean = best_exp.best_prompt_template.replace(
            "{browser_conversation}", "[conversation injected here]",
        )
        print(textwrap.indent(clean, "    "))

    # Show prompt mutation log
    print(f"\n{'=' * 60}")
    print("  Prompt Mutation Log (per generation)")
    print(f"{'=' * 60}")
    for exp in all_experiments:
        if not exp.prompt_mutations:
            continue
        print(f"\n  {exp.category} ({exp.algorithm}, {exp.backend}):")
        for entry in exp.prompt_evolution:
            gen = entry["generation"]
            score = entry["score"]
            h = entry["template_hash"][:8]
            tmpl = entry.get("full_template", "")
            # Show first 200 chars of winning prompt at each generation
            preview = tmpl[:200].replace("\n", " ↵ ") if tmpl else ""
            print(f"    Gen {gen}: {score:5.1f}% [{h}] {preview}")

    print("\n  Done.")


if __name__ == "__main__":
    main()
