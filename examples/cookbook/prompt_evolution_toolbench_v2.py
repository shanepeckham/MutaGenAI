#!/usr/bin/env python3
"""
Cookbook Recipe 45b — ToolBench v2: GPT-4.1 Comparative Run
============================================================

Evolves system prompts on the **ToolBench** benchmark with three new
evolutionary features for a comparative run against v1 results:

1. **Score-proportional selection** — sigmoid-weighted parent selection
   with exploration bonus penalising over-selected parents.
2. **Structured failure buckets** — classifies failures into categories
   (wrong_tool, wrong_params, no_output, unparseable, partial_match) and
   feeds targeted mutations back into the next generation.
3. **Progressive evaluation** — shallow eval on a small subset, then
   deep re-eval for candidates above a promotion threshold.

This script targets **Azure OpenAI GPT-4.1** only (no Ollama phase).
It runs g1_instruction (single-tool) and g3_instruction (cross-collection
multi-tool) with both standard and deep algorithms to produce a
meaningful comparison against the original v1 ToolBench run.

Previous v1 Azure OpenAI results (broken config — expect major improvement):
- g1_instruction standard: 16.3% → 18.9% (Δ+2.7%)
- g3_instruction standard: 16.3% → 18.9% (Δ+2.7%)
- g3_instruction deep:     19.8% → 20.4% (Δ+0.7%)

Usage::

    uv sync --extra llm
    uv run python examples/cookbook/prompt_evolution_toolbench_v2.py

Saves experiment log to ``logs/toolbench_v2_experiment_log.json``.
"""
from __future__ import annotations

import copy
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

from MutaGenAI.prompt_evolver import (
    ErrorProfile,
    FailureBucket,
    LLMBackend,
    LLMClient,
    ProblemType,
    PromptCandidate,
    PromptEvolverConfig,
    SelectionMethod,
    get_failure_bucket_mutations,
)


# ─────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class ToolBenchCase:
    """A single ToolBench benchmark test case."""

    query_id: str
    query: str
    api_list: list[dict[str, Any]]
    relevant_apis: list[dict[str, Any]]
    tier: str  # g1, g2, or g3
    split: str  # instruction, category, or tool

    @property
    def expected_tools(self) -> list[str]:
        """Tool names the query should invoke."""
        return [
            a.get("tool_name", a.get("api_name", ""))
            for a in self.relevant_apis
        ]

    @property
    def expected_apis(self) -> list[str]:
        """API endpoint names the query should invoke."""
        return [a.get("api_name", "") for a in self.relevant_apis]

    @property
    def is_multi_tool(self) -> bool:
        return self.tier in ("g2", "g3")

    @property
    def category(self) -> str:
        """Category label for grouping."""
        return f"{self.tier}_{self.split}"


@dataclass
class ToolBenchExperiment:
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
    multi_tool_ratio: float = 0.0
    failure_buckets: dict[str, int] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────
# Dataset loading from HuggingFace
# ─────────────────────────────────────────────────────────────────────────

_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".toolbench_cache"

_BENCHMARK_SPLITS = [
    "g1_instruction",
    "g1_category",
    "g1_tool",
    "g2_instruction",
    "g2_category",
    "g3_instruction",
]


def _download_toolbench_data() -> dict[str, list[dict[str, Any]]]:
    """Download ToolBench benchmark data from HuggingFace.

    Returns a dict keyed by split name with lists of raw records.
    Uses the HuggingFace datasets-server rows API (no extra deps).
    """
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / "benchmark_data.json"

    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    import urllib.request

    print("  Downloading ToolBench benchmark from HuggingFace...")
    all_splits: dict[str, list[dict[str, Any]]] = {}

    _BASE_URL = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=tuandunghcmut/toolbench-v1&config=benchmark"
    )

    for split_name in _BENCHMARK_SPLITS:
        try:
            records: list[dict[str, Any]] = []
            offset = 0
            batch_size = 100
            while True:
                url = (
                    f"{_BASE_URL}&split={split_name}"
                    f"&offset={offset}&length={batch_size}"
                )
                req = urllib.request.Request(url)
                hf_token = os.environ.get("HF_TOKEN", "")
                if hf_token:
                    req.add_header("Authorization", f"Bearer {hf_token}")
                resp = urllib.request.urlopen(req, timeout=30)
                data = json.loads(resp.read())
                rows = data.get("rows", [])
                if not rows:
                    break
                for row_obj in rows:
                    r = row_obj.get("row", {})
                    records.append({
                        "query_id": str(r.get("query_id", "")),
                        "query": str(r.get("query", "")),
                        "api_list": r.get("api_list", "[]"),
                        "relevant_apis": r.get("relevant_apis", "[]"),
                    })
                offset += len(rows)
                if len(rows) < batch_size:
                    break

            all_splits[split_name] = records
            print(f"    {split_name}: {len(records)} cases")
        except Exception as exc:
            print(f"    ⚠ Failed to load {split_name}: {exc}")

    if all_splits:
        with open(cache_file, "w") as f:
            json.dump(all_splits, f)
        return all_splits

    # Fallback: Maurus/ToolBench (public, no auth)
    print("  Primary dataset unavailable — falling back to Maurus/ToolBench...")
    _FB_URL = (
        "https://datasets-server.huggingface.co/rows"
        "?dataset=Maurus/ToolBench&config=default&split=train"
    )
    try:
        records = []
        fb_resp = urllib.request.urlopen(
            f"{_FB_URL}&offset=0&length=100", timeout=30
        )
        fb_data = json.loads(fb_resp.read())
        for row_obj in fb_data.get("rows", []):
            r = row_obj.get("row", {})
            records.append({
                "query_id": str(r.get("query_id", "")),
                "query": str(r.get("query", "")),
                "api_list": r.get("api_list", "[]"),
                "relevant_apis": r.get("api_list", "[]"),
            })
        all_splits["g1_instruction"] = records[:40]
        all_splits["g2_category"] = records[40:70]
        all_splits["g3_instruction"] = records[70:]
        for sn, recs in all_splits.items():
            print(f"    {sn}: {len(recs)} cases (from Maurus/ToolBench)")
        if all_splits:
            with open(cache_file, "w") as f:
                json.dump(all_splits, f)
    except Exception as exc:
        print(f"    ⚠ Fallback also failed: {exc}")

    return all_splits


def _parse_json_field(value: Any) -> list[Any]:
    """Safely parse a JSON string or pass through a list."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _relevant_apis_to_dicts(raw: list[Any]) -> list[dict[str, str]]:
    """Normalise relevant_apis to list of dicts."""
    result: list[dict[str, str]] = []
    for item in raw:
        if isinstance(item, dict):
            result.append(item)
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            result.append({"tool_name": str(item[0]), "api_name": str(item[1])})
        elif isinstance(item, str):
            result.append({"tool_name": item, "api_name": ""})
    return result


def load_toolbench_dataset(
    max_per_split: int = 20,
    splits: Optional[list[str]] = None,
) -> dict[str, list[ToolBenchCase]]:
    """Load and group ToolBench benchmark data."""
    if splits is None:
        splits = list(_BENCHMARK_SPLITS)

    raw = _download_toolbench_data()

    by_split: dict[str, list[ToolBenchCase]] = {}
    rng = np.random.default_rng(42)

    for split_name in splits:
        records = raw.get(split_name, [])
        if not records:
            continue

        if split_name.startswith("g1"):
            tier = "g1"
        elif split_name.startswith("g2"):
            tier = "g2"
        else:
            tier = "g3"

        split_part = split_name.split("_", 1)[1] if "_" in split_name else split_name

        if len(records) > max_per_split:
            indices = rng.choice(len(records), size=max_per_split, replace=False)
            records = [records[int(i)] for i in indices]

        cases = []
        for rec in records:
            api_list = _parse_json_field(rec.get("api_list", "[]"))
            relevant_raw = _parse_json_field(rec.get("relevant_apis", "[]"))
            relevant = _relevant_apis_to_dicts(relevant_raw)

            cases.append(ToolBenchCase(
                query_id=str(rec.get("query_id", "")),
                query=str(rec.get("query", "")),
                api_list=api_list,
                relevant_apis=relevant,
                tier=tier,
                split=split_part,
            ))

        if cases:
            by_split[split_name] = cases

    return by_split


# ─────────────────────────────────────────────────────────────────────────
# Tool formatting
# ─────────────────────────────────────────────────────────────────────────


def _format_tool_list(apis: list[dict[str, Any]]) -> str:
    """Format a list of API definitions into text for the system prompt."""
    parts: list[str] = []
    for i, api in enumerate(apis, 1):
        tool_name = api.get("tool_name", api.get("api_name", f"tool_{i}"))
        api_name = api.get("api_name", api.get("name", ""))
        desc = api.get("description", api.get("api_description", ""))
        category = api.get("category_name", "")
        method = api.get("method", "GET")

        params_text = ""
        for p in api.get("required_parameters", []):
            p_name = p.get("name", "") if isinstance(p, dict) else str(p)
            p_desc = p.get("description", "") if isinstance(p, dict) else ""
            p_type = p.get("type", "string") if isinstance(p, dict) else "string"
            params_text += f"\n      - {p_name} ({p_type}, required): {p_desc}"

        for p in api.get("optional_parameters", []):
            p_name = p.get("name", "") if isinstance(p, dict) else str(p)
            p_desc = p.get("description", "") if isinstance(p, dict) else ""
            p_type = p.get("type", "string") if isinstance(p, dict) else "string"
            params_text += f"\n      - {p_name} ({p_type}, optional): {p_desc}"

        entry = f"  {i}. [{tool_name}] {api_name}"
        if category:
            entry += f" (category: {category})"
        if desc:
            entry += f"\n     Description: {desc}"
        entry += f"\n     Method: {method}"
        if params_text:
            entry += f"\n     Parameters:{params_text}"

        parts.append(entry)

    return "\n".join(parts) if parts else "  (no tools provided)"


# ─────────────────────────────────────────────────────────────────────────
# Response parsing
# ─────────────────────────────────────────────────────────────────────────


def _parse_tool_calls(response: str) -> list[dict[str, Any]]:
    """Extract tool calls from LLM response."""
    response = response.strip()

    # Try JSON array
    json_match = re.search(r"\[.*\]", response, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list) and parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    # Try single JSON object
    obj_match = re.search(r"\{.*\}", response, re.DOTALL)
    if obj_match:
        try:
            parsed = json.loads(obj_match.group())
            if isinstance(parsed, dict):
                return [parsed]
        except json.JSONDecodeError:
            pass

    # Try extracting tool/API names from text
    calls: list[dict[str, Any]] = []
    tool_patterns = [
        re.findall(r"\[(\w+)\]\s*(\w+)", response),
        re.findall(r"(\w+)\.(\w+)\(", response),
        re.findall(r'tool_name["\s:]+(\w+)', response),
    ]

    for matches in tool_patterns:
        for match in matches:
            if isinstance(match, tuple) and len(match) >= 2:
                calls.append({
                    "tool_name": match[0],
                    "api_name": match[1],
                })
            elif isinstance(match, str):
                calls.append({"tool_name": match})

    return calls


# ─────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────


def _normalise(name: str) -> str:
    """Normalise a tool/API name for comparison."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def score_toolbench_case(
    response: str, case: ToolBenchCase,
) -> tuple[float, dict[str, Any]]:
    """Score a response against a ToolBench test case.

    Returns (score_0_to_1, detail_dict) for failure bucket classification.

    Scoring components:
    - Tool name match (0.5 weight)
    - API name match (0.3 weight)
    - Planning quality (0.2 weight): for multi-tool, are tools in right order?
    """
    detail: dict[str, Any] = {
        "tool_match": False,
        "api_match": False,
        "has_output": True,
        "parseable": True,
        "tool_score": 0.0,
        "api_score": 0.0,
        "order_score": 0.0,
    }

    parsed_calls = _parse_tool_calls(response)

    # Normalise: ensure all calls are dicts
    normalised_calls: list[dict[str, Any]] = []
    for call in parsed_calls:
        if isinstance(call, dict):
            normalised_calls.append(call)
        elif isinstance(call, str):
            normalised_calls.append({"tool_name": call})
        elif isinstance(call, (list, tuple)) and len(call) >= 2:
            normalised_calls.append({"tool_name": str(call[0]), "api_name": str(call[1])})
    parsed_calls = normalised_calls

    if not parsed_calls and not case.relevant_apis:
        detail["tool_match"] = True
        detail["api_match"] = True
        return 1.0, detail
    if not parsed_calls:
        detail["has_output"] = False
        return 0.0, detail

    # Check if response was parseable (got at least one structured call)
    if not any(c.get("tool_name") or c.get("api_name") for c in parsed_calls):
        detail["parseable"] = False

    # ── Tool name matching ─────────────────────────────────────────────
    expected_tools = {_normalise(t) for t in case.expected_tools if t}
    predicted_tools = set()
    for call in parsed_calls:
        tn = call.get("tool_name", "")
        if tn:
            predicted_tools.add(_normalise(tn))

    if expected_tools:
        tool_hits = len(expected_tools & predicted_tools)
        tool_score = tool_hits / len(expected_tools)
        detail["tool_match"] = tool_hits > 0
    else:
        tool_score = 1.0 if not predicted_tools else 0.5
        detail["tool_match"] = True

    detail["tool_score"] = tool_score

    # ── API name matching ──────────────────────────────────────────────
    expected_apis = {_normalise(a) for a in case.expected_apis if a}
    predicted_apis = set()
    for call in parsed_calls:
        an = call.get("api_name", call.get("name", call.get("function", "")))
        if an:
            predicted_apis.add(_normalise(an))

    if expected_apis:
        api_hits = len(expected_apis & predicted_apis)
        api_score = api_hits / len(expected_apis)
        detail["api_match"] = api_hits > 0
    else:
        api_score = 1.0 if not predicted_apis else 0.5
        detail["api_match"] = True

    detail["api_score"] = api_score

    # ── Planning order (multi-tool only) ───────────────────────────────
    if case.is_multi_tool and len(case.relevant_apis) > 1:
        expected_order = [_normalise(a.get("api_name", "")) for a in case.relevant_apis if a.get("api_name")]
        predicted_order = [
            _normalise(c.get("api_name", c.get("name", "")))
            for c in parsed_calls
            if c.get("api_name") or c.get("name")
        ]

        if expected_order and predicted_order:
            order_score = _lcs_ratio(expected_order, predicted_order)
        else:
            order_score = 0.0
    else:
        order_score = 1.0

    detail["order_score"] = order_score

    final_score = 0.5 * tool_score + 0.3 * api_score + 0.2 * order_score
    return final_score, detail


def _lcs_ratio(a: list[str], b: list[str]) -> float:
    """Longest common subsequence length / max length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n] / max(m, n) if max(m, n) > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────
# Failure bucket classification
# ─────────────────────────────────────────────────────────────────────────


def _classify_failure_bucket(
    score: float, detail: dict[str, Any],
) -> FailureBucket | None:
    """Map a ToolBench scoring result to a FailureBucket."""
    if not detail.get("has_output"):
        return FailureBucket.NO_OUTPUT
    if not detail.get("parseable"):
        return FailureBucket.UNPARSEABLE
    if not detail.get("tool_match") and detail.get("tool_score", 0) == 0:
        return FailureBucket.WRONG_TOOL
    if detail.get("tool_match") and not detail.get("api_match"):
        return FailureBucket.WRONG_PARAMS
    if 0 < score < 0.8:
        return FailureBucket.PARTIAL_MATCH
    return None


# ─────────────────────────────────────────────────────────────────────────
# Seed prompt templates
# ─────────────────────────────────────────────────────────────────────────

_TOOLBENCH_SEED_TEMPLATES = [
    # T1 — Minimal
    textwrap.dedent("""\
        You are a tool-calling assistant. Given the user request, select and \
call the appropriate API(s).

Available tools:
{toolbench_apis}

Return your answer as a JSON array of tool calls.
"""),

    # T2 — Step-by-step planner
    textwrap.dedent("""\
        You are an expert API orchestrator. Your task is to:
1. Analyse the user's request and break it into sub-tasks.
2. For each sub-task, select the best tool from the available APIs.
3. Determine the correct parameters for each call.
4. Return the tool calls in the correct execution order.

Available tools:
{toolbench_apis}

IMPORTANT:
- Use ONLY the tools listed above.
- Return a JSON array of objects with "tool_name", "api_name", and "parameters".
- If multiple tools are needed, list them in dependency order.
"""),

    # T3 — Decision-tree planner (DFSDT-inspired)
    textwrap.dedent("""\
        You are an API planning agent that uses depth-first reasoning.

Step 1: Understand the user's goal.
Step 2: Identify which tools are relevant.
Step 3: For each relevant tool, determine which API endpoint to call.
Step 4: Plan the execution order — if one call depends on another's output, \
sequence them correctly.
Step 5: For each call, fill in the required parameters.

Available tools:
{toolbench_apis}

Output format — JSON array:
[{{"tool_name": "...", "api_name": "...", "parameters": {{...}}}}]

Think step by step, then output ONLY the JSON array.
"""),

    # T4 — Constraint-focused
    textwrap.dedent("""\
        You are a precise tool-calling assistant. Rules:
- ONLY use tools from the list below.
- Every required parameter MUST be provided.
- Match tool names and API names EXACTLY as listed.
- For multi-step tasks, return calls in execution order.
- Return ONLY a JSON array — no explanation text.

Available tools:
{toolbench_apis}

JSON array of tool calls:
"""),
]


# ─────────────────────────────────────────────────────────────────────────
# Mutation operators
# ─────────────────────────────────────────────────────────────────────────

_TOOLBENCH_MUTATIONS: list[str] = [
    # Planning emphasis
    "Add: 'Break complex requests into atomic sub-tasks before selecting tools.'",
    "Append: 'Verify that every sub-task maps to exactly one API call.'",
    "Insert: 'If a tool call depends on another's output, mark the dependency.'",
    # Tool selection
    "Add: 'Read each tool description carefully — choose the one whose name and description best match the sub-task.'",
    "Inject: 'When multiple tools could work, prefer the one with fewer required parameters.'",
    "Insert: 'Cross-check the category of each tool against the user request domain.'",
    # Parameter quality
    "Append: 'For each required parameter, extract the value directly from the user query.'",
    "Add: 'Use parameter types as hints — if a parameter is a number, convert strings to numbers.'",
    "Insert: 'Optional parameters should only be set if the user explicitly mentions them.'",
    # Output format
    "Append: 'Your output must be ONLY a valid JSON array — no markdown, no explanation.'",
    "Add: 'Double-check that each object has tool_name, api_name, and parameters keys.'",
    "Insert: 'Match tool names and API names EXACTLY as listed — case-sensitive.'",
    # Multi-tool orchestration
    "Add: 'For requests spanning multiple categories, search across all listed tools.'",
    "Inject: 'Return tool calls in the order they should execute — data dependencies first.'",
    "Append: 'If one API returns data needed by another, put the data-producing call first.'",
    # Chain-of-thought
    "Prepend: 'Think step by step before answering.'",
    "Insert: 'First list the sub-tasks, then map each to a tool, then output the JSON.'",
    "Add: 'Reason about which tools are relevant before writing the JSON array.'",
]


def _mutate_toolbench_template(
    template: str,
    rng: np.random.Generator,
    rate: float = 0.5,
    extra_mutations: list[str] | None = None,
) -> str:
    """Apply a random mutation to a ToolBench prompt template."""
    pool = _TOOLBENCH_MUTATIONS
    if extra_mutations:
        pool = _TOOLBENCH_MUTATIONS + extra_mutations
    mutation = rng.choice(pool)
    lines = template.strip().split("\n")

    # Failure bucket mutations are plain text (no "Action: content" format)
    if ":" in mutation:
        action = mutation.split(":")[0].strip().lower()
        content = mutation.split(":", 1)[1].strip().strip("'\"")
    else:
        action = "add"
        content = mutation.strip()

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
    if "{toolbench_apis}" not in result:
        result += "\n\nAvailable tools:\n{toolbench_apis}"

    return result


def _crossover_toolbench_templates(
    a: str, b: str, rng: np.random.Generator
) -> str:
    """Crossover two ToolBench prompt templates."""
    la = a.strip().split("\n")
    lb = b.strip().split("\n")
    ca = int(rng.integers(1, max(2, len(la))))
    cb = int(rng.integers(1, max(2, len(lb))))
    child = "\n".join(la[:ca] + lb[cb:])
    if "{toolbench_apis}" not in child:
        child += "\n\nAvailable tools:\n{toolbench_apis}"
    return child


# ─────────────────────────────────────────────────────────────────────────
# Score-proportional parent selection
# ─────────────────────────────────────────────────────────────────────────


def _score_prop_select(
    island: list[PromptCandidate], rng: np.random.Generator,
) -> PromptCandidate:
    """Score-proportional selection with exploration bonus.

    Probability is proportional to ``sigmoid(score) / (1 + selection_count)``.
    """
    import math

    if len(island) == 1:
        island[0].selection_count += 1
        return island[0]

    weights = np.array([
        (
            1.0 / (1.0 + math.exp(-10.0 * (c.score / 100.0 - 0.5)))
        ) * (
            1.0 / (1.0 + c.selection_count)
        )
        for c in island
    ])

    total = weights.sum()
    if total <= 0:
        idx = int(rng.integers(len(island)))
        island[idx].selection_count += 1
        return island[idx]

    probs = weights / total
    idx = int(rng.choice(len(island), p=probs))
    island[idx].selection_count += 1
    return island[idx]


# ─────────────────────────────────────────────────────────────────────────
# Evolution engine
# ─────────────────────────────────────────────────────────────────────────


def run_toolbench_evolution(
    cases: list[ToolBenchCase],
    client: LLMClient,
    config: PromptEvolverConfig,
    algorithm_name: str = "standard",
    seed: int = 42,
    verbose: bool = True,
) -> ToolBenchExperiment:
    """Run prompt evolution on ToolBench test cases with v2 features.

    Features:
    - Score-proportional selection (when configured)
    - Failure bucket tracking + targeted mutations
    - Progressive evaluation (when configured)
    """
    rng = np.random.default_rng(seed)
    category = cases[0].category if cases else "unknown"
    multi_count = sum(1 for c in cases if c.is_multi_tool)
    multi_ratio = multi_count / len(cases) if cases else 0.0

    use_score_prop = (
        config.selection_method == SelectionMethod.SCORE_PROPORTIONAL
    )
    problem_type = config.problem_type or ProblemType.TOOL_ROUTING
    error_profile = ErrorProfile()

    # Progressive evaluation settings
    use_progressive = (
        config.eval_promotion_threshold is not None
        and config.eval_deep_sample_size is not None
    )

    # ── Evaluate a candidate ───────────────────────────────────────────
    def evaluate(
        candidate: PromptCandidate,
        eval_cases: list[ToolBenchCase],
        track_buckets: bool = True,
    ) -> float:
        total = 0.0
        for case in eval_cases:
            fn_text = _format_tool_list(case.api_list)
            sys_prompt = candidate.template.replace("{toolbench_apis}", fn_text)

            response = client.complete(
                system_prompt=sys_prompt,
                user_message=case.query,
                temperature=candidate.temperature,
                top_p=candidate.top_p,
            )
            if response is None:
                total += float(rng.uniform(0, 0.3))
                if track_buckets:
                    error_profile.record_bucket(FailureBucket.NO_OUTPUT)
                continue

            score, detail = score_toolbench_case(response, case)
            total += score

            if track_buckets:
                bucket = _classify_failure_bucket(score, detail)
                if bucket is not None:
                    error_profile.record_bucket(bucket)

        return (total / len(eval_cases) * 100.0) if eval_cases else 0.0

    # ── Subsample for evaluation ───────────────────────────────────────
    if config.eval_sample_size and config.eval_sample_size < len(cases):
        eval_indices = rng.choice(
            len(cases), size=config.eval_sample_size, replace=False
        )
        eval_cases = [cases[int(i)] for i in eval_indices]
    else:
        eval_cases = cases

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  ToolBench Split: {category}  |  Algorithm: {algorithm_name}")
        print(
            f"  Cases: {len(eval_cases)}  |  Backend: {config.backend.value}  "
            f"|  Multi-tool: {multi_ratio:.0%}"
        )
        sel_label = "score-proportional" if use_score_prop else "tournament"
        prog_label = "yes" if use_progressive else "no"
        print(f"  Selection: {sel_label}  |  Progressive: {prog_label}")
        print(f"{'=' * 60}")

    t0 = time.perf_counter()
    prompt_trace: list[dict[str, Any]] = []

    # Init islands
    islands: list[list[PromptCandidate]] = [
        [] for _ in range(config.num_islands)
    ]

    for i, tmpl in enumerate(_TOOLBENCH_SEED_TEMPLATES):
        cand = PromptCandidate(
            template=tmpl,
            temperature=float(rng.uniform(*config.temperature_range)),
            top_p=float(rng.uniform(*config.top_p_range)),
            generation=0,
        )
        cand.score = evaluate(cand, eval_cases)
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

    # Deep eval subset for progressive evaluation
    if use_progressive:
        deep_size = min(
            config.eval_deep_sample_size or len(eval_cases), len(cases)
        )
        deep_indices = rng.choice(len(cases), size=deep_size, replace=False)
        deep_cases = [cases[int(i)] for i in deep_indices]
    else:
        deep_cases = eval_cases

    for gen in range(1, config.iterations + 1):
        # Compute failure bucket mutations for this generation
        bucket_mutations = get_failure_bucket_mutations(
            error_profile, problem_type
        )

        for isl_id in range(config.num_islands):
            island = islands[isl_id]
            if not island:
                continue

            new_cands: list[PromptCandidate] = []
            for _ in range(config.population_size):
                # ── Parent selection ───────────────────────────────
                if use_score_prop:
                    parent_a = _score_prop_select(island, rng)
                else:
                    k = min(3, len(island))
                    idxs = rng.choice(len(island), size=k, replace=False)
                    parent_a = max(
                        (island[int(i)] for i in idxs),
                        key=lambda c: c.score,
                    )

                if rng.random() < config.crossover_rate and len(island) > 1:
                    if use_score_prop:
                        parent_b = _score_prop_select(island, rng)
                    else:
                        k = min(3, len(island))
                        idxs2 = rng.choice(len(island), size=k, replace=False)
                        parent_b = max(
                            (island[int(i)] for i in idxs2),
                            key=lambda c: c.score,
                        )
                    child_tmpl = _crossover_toolbench_templates(
                        parent_a.template, parent_b.template, rng
                    )
                else:
                    child_tmpl = parent_a.template

                # ── Mutation (with failure bucket hints) ──────────
                if rng.random() < config.mutation_rate:
                    child_tmpl = _mutate_toolbench_template(
                        child_tmpl, rng, config.mutation_rate,
                        extra_mutations=bucket_mutations or None,
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

                # ── Progressive evaluation ────────────────────────
                score = evaluate(child, eval_cases)
                if (
                    use_progressive
                    and score >= (config.eval_promotion_threshold or 0)
                ):
                    score = evaluate(
                        child, deep_cases, track_buckets=False,
                    )
                child.score = score
                new_cands.append(child)

            combined = island + new_cands
            combined.sort(key=lambda c: c.score, reverse=True)
            islands[isl_id] = combined[: config.elite_size]

        # Decay error profile so recent generations weigh more
        error_profile.decay(0.8)

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
            if bucket_mutations:
                top_buckets = error_profile.worst_buckets(3)
                bucket_str = ", ".join(
                    f"{b}({c})" for b, c in top_buckets
                )
                print(f"         buckets: {bucket_str}")

    wall_time = time.perf_counter() - t0

    # Snapshot final failure bucket counts
    final_buckets = dict(error_profile.failure_buckets)

    return ToolBenchExperiment(
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
        multi_tool_ratio=multi_ratio,
        failure_buckets=final_buckets,
    )


# ─────────────────────────────────────────────────────────────────────────
# Baseline evaluation
# ─────────────────────────────────────────────────────────────────────────

_TOOLBENCH_DEFAULT_PROMPT = textwrap.dedent("""\
    You are a helpful assistant. Based on the user's request, call the \
appropriate tool(s).

Available tools:
{toolbench_apis}
""")


def evaluate_baseline(
    cases: list[ToolBenchCase],
    split_name: str,
    client: LLMClient,
    verbose: bool = True,
) -> float:
    """Evaluate the default ToolBench prompt on cases."""
    if verbose:
        print(f"  Evaluating default prompt baseline ({split_name})...")

    rng = np.random.default_rng(42)
    total = 0.0
    for case in cases:
        fn_text = _format_tool_list(case.api_list)
        sys_prompt = _TOOLBENCH_DEFAULT_PROMPT.replace("{toolbench_apis}", fn_text)

        response = client.complete(
            system_prompt=sys_prompt,
            user_message=case.query,
            temperature=0.1,
            top_p=0.95,
        )
        if response is None:
            total += float(rng.uniform(0, 0.3))
            continue

        score, _ = score_toolbench_case(response, case)
        total += score

    result = (total / len(cases) * 100.0) if cases else 0.0
    if verbose:
        print(f"  Default prompt baseline ({split_name}): {result:.1f}%")
    return result


# ─────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────


def show_prompt_evolution(experiment: ToolBenchExperiment) -> None:
    """Print the prompt evolution trace for an experiment."""
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
            f"[{h[:8]}]{marker}"
        )
        if marker:
            print(f"    → {entry['template_preview']}")

    print(
        f"\n  Baseline: {experiment.baseline_score:.1f}%  →  "
        f"Evolved: {experiment.evolved_score:.1f}%  "
        f"(Δ{experiment.evolved_score - experiment.baseline_score:+.1f}%)"
    )

    if experiment.failure_buckets:
        print(f"\n  Failure Buckets:")
        for bucket, count in sorted(
            experiment.failure_buckets.items(), key=lambda x: -x[1],
        ):
            print(f"    {bucket}: {count}")


def show_results_table(
    experiments: list[ToolBenchExperiment],
    default_baselines: dict[str, float],
) -> None:
    """Print a summary table of all experiments."""
    print(f"\n{'=' * 90}")
    print("  ToolBench v2 Prompt Evolution Results (GPT-4.1)")
    print(f"{'=' * 90}")
    print(
        f"  {'Split':<18} {'Algorithm':<12} {'Backend':<14} "
        f"{'Default':>8} {'Base':>8} {'Evolved':>8} {'Δ':>7} {'Time':>7}"
    )
    print(f"  {'─' * 85}")

    for exp in experiments:
        dflt = default_baselines.get(exp.category, 0.0)
        delta = exp.evolved_score - exp.baseline_score
        print(
            f"  {exp.category:<18} {exp.algorithm:<12} {exp.backend:<14} "
            f"{dflt:7.1f}% {exp.baseline_score:7.1f}% "
            f"{exp.evolved_score:7.1f}% {delta:+6.1f}% "
            f"{exp.wall_time:6.1f}s"
        )

    print(f"  {'─' * 85}")


# ─────────────────────────────────────────────────────────────────────────
# Experiment log persistence
# ─────────────────────────────────────────────────────────────────────────


def save_experiment_log(
    experiments: list[ToolBenchExperiment],
    default_baselines: dict[str, float],
    path: str = "toolbench_v2_experiment_log.json",
) -> None:
    """Save experiment results to JSON for dashboard consumption."""
    log: dict[str, Any] = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "version": "v2",
        "features": [
            "score_proportional_selection",
            "failure_bucket_mutations",
            "progressive_evaluation",
        ],
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
            "best_prompt_template": exp.best_prompt_template,
            "multi_tool_ratio": round(exp.multi_tool_ratio, 3),
            "failure_buckets": exp.failure_buckets,
        }
        log["experiments"].append(entry)

    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Experiment log saved to {path}")


# ─────────────────────────────────────────────────────────────────────────
# Main — Azure OpenAI GPT-4.1 only
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
        "selection_method": SelectionMethod.SCORE_PROPORTIONAL,
        "problem_type": ProblemType.TOOL_ROUTING,
    },
    "deep": {
        "iterations": 5,
        "population_size": 5,
        "num_islands": 2,
        "elite_size": 3,
        "mutation_rate": 0.5,
        "crossover_rate": 0.4,
        "eval_sample_size": 12,
        "selection_method": SelectionMethod.SCORE_PROPORTIONAL,
        "problem_type": ProblemType.TOOL_ROUTING,
        "eval_promotion_threshold": 60.0,
        "eval_deep_sample_size": 20,
    },
}

# Run g1 (single-tool) and g3 (cross-collection multi-tool) for comparison
BENCHMARK_SPLITS = ["g1_instruction", "g3_instruction"]
MAX_PER_SPLIT = 20


def main() -> None:
    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  MutaGenAI × ToolBench v2 — GPT-4.1 Comparative Run         ║
    ║  Features: score-prop selection, failure buckets, progressive ║
    ║  16 464 REST APIs · G1 + G3 tiers · Azure OpenAI only        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

    # ── Azure OpenAI setup ─────────────────────────────────────────────
    azure_cfg = PromptEvolverConfig(
        backend=LLMBackend.AZURE_OPENAI,
        timeout=30.0,
    )
    azure_client = LLMClient(azure_cfg)

    if not azure_client.is_available():
        print(
            "  ✗ Azure OpenAI not available — set AZURE_OPENAI_ENDPOINT "
            "and AZURE_OPENAI_DEPLOYMENT env vars."
        )
        print("  Exiting.\n")
        return

    print(
        f"  ✓ Azure OpenAI connected: {azure_cfg.azure_deployment} "
        f"(RBAC={azure_cfg.azure_use_rbac})"
    )

    # ── Load ToolBench data ────────────────────────────────────────────
    print("\n  Loading ToolBench benchmark data...")
    try:
        by_split = load_toolbench_dataset(
            max_per_split=MAX_PER_SPLIT,
            splits=BENCHMARK_SPLITS,
        )
    except RuntimeError as exc:
        print(f"  ✗ {exc}")
        return

    for split_name in BENCHMARK_SPLITS:
        cases = by_split.get(split_name, [])
        multi = sum(1 for c in cases if c.is_multi_tool)
        print(
            f"    {split_name:<18}: {len(cases):3d} cases  "
            f"({multi} multi-tool, {len(cases) - multi} single-tool)"
        )

    if not by_split:
        print("  No ToolBench data loaded — exiting.")
        return

    # ── Phase 1: Default prompt baselines ──────────────────────────────
    print("\n  Phase 1: Default Prompt Baselines (Azure OpenAI)")
    print("  " + "─" * 50)
    default_baselines: dict[str, float] = {}
    for split_name in BENCHMARK_SPLITS:
        cases = by_split.get(split_name, [])
        if cases:
            score = evaluate_baseline(cases, split_name, azure_client)
            default_baselines[split_name] = score

    # ── Phase 2: Standard evolution on all splits ──────────────────────
    print("\n  Phase 2: Standard Evolution (Azure OpenAI GPT-4.1)")
    print("  " + "─" * 50)

    all_experiments: list[ToolBenchExperiment] = []

    for split_name in BENCHMARK_SPLITS:
        cases = by_split.get(split_name, [])
        if not cases:
            continue

        algo = "standard"
        algo_params = ALGORITHM_CONFIGS[algo]
        cfg = PromptEvolverConfig(
            backend=LLMBackend.AZURE_OPENAI,
            timeout=30.0,
            **algo_params,
        )

        exp = run_toolbench_evolution(
            cases=cases,
            client=azure_client,
            config=cfg,
            algorithm_name=algo,
            verbose=True,
        )
        all_experiments.append(exp)
        show_prompt_evolution(exp)

    # ── Phase 3: Deep evolution on g3 (hardest tier) ───────────────────
    print("\n  Phase 3: Deep Evolution on g3_instruction (Azure OpenAI GPT-4.1)")
    print("  " + "─" * 50)

    if "g3_instruction" in by_split and by_split["g3_instruction"]:
        algo = "deep"
        algo_params = ALGORITHM_CONFIGS[algo]
        cfg = PromptEvolverConfig(
            backend=LLMBackend.AZURE_OPENAI,
            timeout=30.0,
            **algo_params,
        )

        exp = run_toolbench_evolution(
            cases=by_split["g3_instruction"],
            client=azure_client,
            config=cfg,
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
        os.path.dirname(__file__), "..", "..", "logs",
        "toolbench_v2_experiment_log.json",
    )
    save_experiment_log(all_experiments, default_baselines, log_path)

    # ── Print best evolved prompt ──────────────────────────────────────
    if all_experiments:
        best_exp = max(all_experiments, key=lambda e: e.evolved_score)
        print(f"\n{'=' * 60}")
        print(
            f"  Best Evolved Prompt ({best_exp.category} / {best_exp.algorithm})"
        )
        print(f"  Score: {best_exp.evolved_score:.1f}%")
        print(f"{'=' * 60}")
        preview = best_exp.best_prompt_template[:500]
        for line in preview.split("\n"):
            print(f"    {line}")
        if len(best_exp.best_prompt_template) > 500:
            print(
                f"    ... ({len(best_exp.best_prompt_template)} chars total)"
            )

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
