#!/usr/bin/env python3
"""
Cookbook Recipe 44 — xLAM / APIGen Function-Calling Prompt Evolution
====================================================================

Evolves system prompts on the **Salesforce xLAM function-calling 60 k**
dataset — 60 000 verified function-calling examples across 3 673 APIs
in 21 categories (finance, social media, weather, sports, …).

Each example has a natural-language *query*, a list of *tool* definitions
(name, description, parameters), and *verified gold answers* (function
name + arguments).  The xLAM-2 models trained on this data already top
BFCL — evolving the prompts that drive them can push further.

Categories benchmarked
----------------------
We group the 21 API categories into 5 evaluation buckets:

1. **Finance**      — stock prices, currency conversion, financial APIs
2. **Social**       — social media, messaging, communication APIs
3. **Data**         — data analysis, search, information retrieval
4. **Science**      — maths, science, health, weather, environment APIs
5. **Entertainment** — music, movies, sports, food, travel APIs

Algorithm experiments
---------------------
* **Standard** — 3 iterations, pop 4, 2 islands  (balanced)
* **Deep**     — 5 iterations, pop 5, 2 islands  (thorough)

Usage::

    uv sync --extra llm
    uv run python examples/cookbook/prompt_evolution_xlam.py

The script saves an experiment log to ``xlam_experiment_log.json`` for
dashboard consumption.

References
----------
* Dataset: https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k
* Paper:  https://arxiv.org/abs/2406.18518 (APIGen)
* Models: https://huggingface.co/collections/Salesforce/xlam-models
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

from prompture.prompt_evolver import (
    LLMBackend,
    LLMClient,
    PromptCandidate,
    PromptEvolverConfig,
)

# ─────────────────────────────────────────────────────────────────────────
# xLAM data loading
# ─────────────────────────────────────────────────────────────────────────

_XLAM_DATA_PATH: Optional[Path] = None

_HF_URL = (
    "https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k"
    "/resolve/main/xlam_function_calling_60k.json"
)


@dataclass
class XLAMCase:
    """A single xLAM function-calling test case."""

    id: int
    query: str
    tools: list[dict[str, Any]]
    answers: list[dict[str, Any]]
    category: str  # assigned by bucket classifier

    @property
    def is_parallel(self) -> bool:
        return len(self.answers) > 1

    @property
    def expected_function_names(self) -> list[str]:
        return [a["name"] for a in self.answers]


# Category buckets — map API name prefixes / keywords to buckets
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "finance": [
        "stock", "finance", "currency", "exchange_rate", "crypto",
        "market", "invest", "bank", "loan", "tax", "mortgage",
        "economic", "forex", "dividend", "portfolio",
    ],
    "social": [
        "social", "twitter", "facebook", "instagram", "message",
        "email", "sms", "chat", "notification", "user_profile",
        "friend", "follow", "post", "comment", "share", "communication",
    ],
    "data": [
        "search", "database", "query", "data", "csv", "json",
        "parse", "extract", "transform", "sort", "filter", "merge",
        "aggregate", "analyze", "statistics", "info", "lookup",
    ],
    "science": [
        "math", "science", "physics", "chemistry", "biology",
        "weather", "temperature", "climate", "health", "medical",
        "calculate", "convert", "formula", "equation", "unit",
        "prime", "factorial", "environment",
    ],
    "entertainment": [
        "movie", "music", "sport", "game", "food", "recipe",
        "travel", "hotel", "flight", "restaurant", "book", "event",
        "ticket", "concert", "hobby", "fitness", "exercise",
    ],
}


def _classify_category(case_tools: list[dict[str, Any]], query: str) -> str:
    """Classify an xLAM case into one of our evaluation buckets."""
    text = query.lower()
    for tool in case_tools:
        text += " " + tool.get("name", "").lower()
        text += " " + tool.get("description", "").lower()

    scores: dict[str, int] = {}
    for cat, keywords in _CATEGORY_KEYWORDS.items():
        scores[cat] = sum(1 for kw in keywords if kw in text)

    best = max(scores, key=lambda k: scores[k])
    return best if scores[best] > 0 else "data"  # default bucket


def _xlam_cache_path() -> Path:
    """Return path to local xLAM data cache."""
    global _XLAM_DATA_PATH
    if _XLAM_DATA_PATH is not None:
        return _XLAM_DATA_PATH

    cache = Path(__file__).resolve().parent.parent.parent / ".xlam_cache"
    cache.mkdir(exist_ok=True)
    _XLAM_DATA_PATH = cache
    return cache


def _download_xlam_data() -> Path:
    """Download the xLAM 60k dataset if not already cached."""
    cache = _xlam_cache_path()
    dest = cache / "xlam_function_calling_60k.json"

    if dest.exists():
        return dest

    print("    Downloading xLAM 60k dataset from HuggingFace ...")
    print("    (This is a ~90 MB file — first run only)")

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Prefer huggingface_hub for gated-repo support
    try:
        from huggingface_hub import hf_hub_download

        hf_path = hf_hub_download(
            repo_id="Salesforce/xlam-function-calling-60k",
            filename="xlam_function_calling_60k.json",
            repo_type="dataset",
            token=token,
        )
        import shutil

        shutil.copy2(hf_path, dest)
        return dest
    except ImportError:
        pass  # fall through to urllib

    # Fallback: direct download
    req = urllib.request.Request(_HF_URL)  # noqa: S310
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            dest.write_bytes(resp.read())
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise RuntimeError(
                "xLAM dataset requires HuggingFace access. "
                "1) Accept terms at https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k "
                "2) Set HF_TOKEN env var with a read token."
            ) from e
        raise

    return dest


def load_xlam_dataset(
    max_per_category: int | None = None,
    categories: list[str] | None = None,
) -> dict[str, list[XLAMCase]]:
    """Load the xLAM 60k dataset, grouped by category bucket.

    Parameters
    ----------
    max_per_category :
        Cap per category (random sample). ``None`` = all.
    categories :
        Restrict to these category names. ``None`` = all 5.

    Returns
    -------
    dict mapping category name to list of XLAMCase
    """
    data_path = _download_xlam_data()

    with open(data_path, encoding="utf-8") as f:
        raw_data = json.load(f)

    # Group by category
    by_cat: dict[str, list[XLAMCase]] = {}
    for idx, entry in enumerate(raw_data):
        # Parse stringified fields
        query = entry.get("query", "")
        tools_raw = entry.get("tools", "[]")
        answers_raw = entry.get("answers", "[]")

        if isinstance(tools_raw, str):
            try:
                tools = json.loads(tools_raw)
            except json.JSONDecodeError:
                continue
        else:
            tools = tools_raw

        if isinstance(answers_raw, str):
            try:
                answers = json.loads(answers_raw)
            except json.JSONDecodeError:
                continue
        else:
            answers = answers_raw

        if not query or not tools or not answers:
            continue

        cat = _classify_category(tools, query)
        if categories and cat not in categories:
            continue

        case = XLAMCase(
            id=idx,
            query=query,
            tools=tools,
            answers=answers,
            category=cat,
        )
        by_cat.setdefault(cat, []).append(case)

    # Subsample
    if max_per_category:
        rng = np.random.default_rng(42)
        for cat in list(by_cat.keys()):
            cases = by_cat[cat]
            if len(cases) > max_per_category:
                indices = rng.choice(
                    len(cases), size=max_per_category, replace=False
                )
                by_cat[cat] = [cases[int(i)] for i in sorted(indices)]

    return by_cat


# ─────────────────────────────────────────────────────────────────────────
# xLAM scoring
# ─────────────────────────────────────────────────────────────────────────


def _format_xlam_tools(tools: list[dict[str, Any]]) -> str:
    """Format xLAM tool definitions as text for the system prompt."""
    lines: list[str] = []
    for tool in tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "")
        params = tool.get("parameters", {})

        if isinstance(params, dict):
            props = params.get("properties", params)
            required = set(params.get("required", []))
        else:
            props = {}
            required = set()

        param_parts: list[str] = []
        for pname, pinfo in props.items():
            if isinstance(pinfo, dict):
                ptype = pinfo.get("type", "string")
                pdesc = pinfo.get("description", "")
                req_marker = " [required]" if pname in required else ""
            else:
                ptype = "string"
                pdesc = str(pinfo)
                req_marker = ""
            param_parts.append(f"    - {pname} ({ptype}{req_marker}): {pdesc}")

        lines.append(f"  {name}: {desc}")
        if param_parts:
            lines.append("\n".join(param_parts))
    return "\n".join(lines)


def _parse_function_calls(response: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse LLM response into a list of (function_name, params) tuples.

    Handles formats:
    - Python-style:  func_name(param=value, ...)
    - JSON:          {"name": "...", "arguments": {...}}
    - List of calls: [func1(...), func2(...)]
    """
    if not response:
        return []

    calls: list[tuple[str, dict[str, Any]]] = []

    # Try JSON array first
    try:
        stripped = response.strip()
        json_match = re.search(r"\[.*\]", stripped, re.DOTALL)
        if json_match:
            arr = json.loads(json_match.group())
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict):
                        fname = (
                            item.get("name")
                            or item.get("function")
                            or item.get("tool", "")
                        )
                        params = (
                            item.get("arguments")
                            or item.get("parameters")
                            or item.get("params", {})
                        )
                        if fname:
                            calls.append(
                                (fname, params if isinstance(params, dict) else {})
                            )
                if calls:
                    return calls
    except (json.JSONDecodeError, ValueError):
        pass

    # Try single JSON object
    try:
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            obj = json.loads(json_match.group())
            fname = (
                obj.get("name")
                or obj.get("function")
                or obj.get("tool", "")
            )
            params = (
                obj.get("arguments")
                or obj.get("parameters")
                or obj.get("params", {})
            )
            if fname:
                return [(fname, params if isinstance(params, dict) else {})]
    except (json.JSONDecodeError, ValueError):
        pass

    # Python-style: func_name(key=value, ...)
    pattern = r"([\w.]+)\s*\(([^)]*)\)"
    for match in re.finditer(pattern, response):
        fname = match.group(1)
        param_str = match.group(2)

        if fname.lower() in (
            "if", "for", "while", "def", "class", "return", "print", "list",
        ):
            continue

        params: dict[str, Any] = {}
        for kv in re.finditer(
            r"""(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|\[([^\]]*)\]|([\w.+-]+))""",
            param_str,
        ):
            key = kv.group(1)
            val: Any = (
                kv.group(2) or kv.group(3) or kv.group(5) or ""
            )
            # Handle list values
            if kv.group(4) is not None:
                list_str = kv.group(4)
                try:
                    val = json.loads(f"[{list_str}]")
                except json.JSONDecodeError:
                    val = [s.strip().strip("'\"") for s in list_str.split(",")]
            else:
                # Try to convert to number/bool
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        if isinstance(val, str):
                            if val.lower() == "true":
                                val = True
                            elif val.lower() == "false":
                                val = False
            params[key] = val

        calls.append((fname, params))

    return calls


def _match_param_value(predicted: Any, expected: Any) -> bool:
    """Check if a predicted value matches the expected value."""
    if predicted == expected:
        return True
    # String comparison (case-insensitive)
    if str(predicted).lower().strip() == str(expected).lower().strip():
        return True
    # Numeric tolerance
    try:
        if abs(float(predicted) - float(expected)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    # List comparison
    if isinstance(predicted, list) and isinstance(expected, list):
        if len(predicted) == len(expected):
            return all(_match_param_value(p, e) for p, e in zip(predicted, expected))
    return False


def score_xlam_case(response: str, case: XLAMCase) -> float:
    """Score a model response against an xLAM test case.

    Returns a float in [0, 1]:
    - 1.0 = all expected calls with correct params
    - Partial credit for correct function names with wrong params
    - 0.0 = complete miss
    """
    parsed = _parse_function_calls(response)
    if not parsed:
        return 0.0

    expected = case.answers
    n_expected = len(expected)
    if n_expected == 0:
        return 1.0 if not parsed else 0.0

    # Single expected call
    if n_expected == 1:
        exp = expected[0]
        exp_name = exp["name"]
        exp_args = exp.get("arguments", {})

        best = 0.0
        for pred_name, pred_args in parsed:
            name_match = (
                pred_name == exp_name
                or pred_name.replace("_", ".") == exp_name
                or pred_name.replace(".", "_") == exp_name.replace(".", "_")
            )
            if name_match:
                param_score = _score_params(pred_args, exp_args)
                best = max(best, 0.5 + 0.5 * param_score)
            elif pred_name.lower() == exp_name.lower().replace(".", "_"):
                param_score = _score_params(pred_args, exp_args)
                best = max(best, 0.3 + 0.5 * param_score)
        return best

    # Parallel: multiple expected calls
    total = 0.0
    for exp in expected:
        exp_name = exp["name"]
        exp_args = exp.get("arguments", {})

        best_call = 0.0
        for pred_name, pred_args in parsed:
            name_match = (
                pred_name == exp_name
                or pred_name.replace("_", ".") == exp_name
                or pred_name.replace(".", "_") == exp_name.replace(".", "_")
            )
            if name_match:
                param_score = _score_params(pred_args, exp_args)
                best_call = max(best_call, 0.5 + 0.5 * param_score)

        total += best_call

    return total / n_expected


def _score_params(predicted: dict[str, Any], expected: dict[str, Any]) -> float:
    """Score predicted parameters against expected."""
    if not expected:
        return 1.0

    total = len(expected)
    if total == 0:
        return 1.0

    matches = 0.0
    for key, exp_val in expected.items():
        if key in predicted:
            if _match_param_value(predicted[key], exp_val):
                matches += 1.0
            else:
                matches += 0.3  # Partial: correct key, wrong value
        # Also check lower-case key variants
        elif key.lower() in {k.lower() for k in predicted}:
            for pk, pv in predicted.items():
                if pk.lower() == key.lower():
                    if _match_param_value(pv, exp_val):
                        matches += 0.8
                    else:
                        matches += 0.2
                    break

    return min(1.0, matches / total)


# ─────────────────────────────────────────────────────────────────────────
# Prompt templates (xLAM-specific)
# ─────────────────────────────────────────────────────────────────────────

_XLAM_SEED_TEMPLATES = [
    # T0: Minimal — direct Python call format
    textwrap.dedent("""\
        You are an expert in composing functions. Given a question and a set \
of possible functions, make one or more function calls to achieve the \
purpose. If none of the functions can be used, point it out.

You should only return the function calls in your response.
If you decide to invoke any of the function(s), you MUST put it in \
the format of [func_name(param_name=param_value, ...)]
You SHOULD NOT include any other text in the response.

Here is a list of functions you can invoke:
{xlam_functions}
    """),

    # T1: Structured with reasoning steps
    textwrap.dedent("""\
        You are an expert function-calling agent with access to \
domain-specific APIs.

Your task: given a user query and available functions, produce one or \
more function calls to fulfil the request.

Step 1: Identify the user's intent and required information.
Step 2: Select the correct function(s) from the available set.
Step 3: Extract parameter values from the query.
Step 4: Format the call(s).

Output ONLY function calls in this format:
[func_name(param=value, ...)]
For multiple calls: [func1(p1=v1), func2(p2=v2)]
Do NOT include any other text.

Available functions:
{xlam_functions}
    """),

    # T2: Domain-aware extraction focus
    textwrap.dedent("""\
        You are a precise tool router. Analyse the user's request and \
invoke the correct function(s).

Rules:
- Pick the most specific function that matches.
- Extract parameter values directly from the query.
- Use exact values from the query — do not invent or modify them.
- For numbers, dates, or identifiers, preserve the exact format.
- If a parameter is not mentioned, use a reasonable default.
- Output ONLY function calls: [func_name(param=value)]
- For multiple calls: [func1(p1=v1), func2(p2=v2)]

Functions:
{xlam_functions}
    """),

    # T3: Emphasis on parameter extraction and types
    textwrap.dedent("""\
        # Role
You are a function-calling agent. Your job is to map user requests \
to precise function invocations.

# Instructions
1. Read the user's query carefully.
2. Match it to one or more available functions.
3. Extract ALL required parameters from the query text.
4. For optional parameters, include them only if clearly mentioned.
5. Match parameter types: integers for counts, strings for names, \
lists for multiple values.
6. Return ONLY the function call(s) in Python format:
   [function_name(param1=value1, param2=value2)]

# Available Functions
{xlam_functions}
    """),
]

_XLAM_MUTATIONS = [
    "Be precise — extract exact values (numbers, names, strings) from the query.",
    "If the query mentions multiple actions, invoke multiple functions.",
    "Match parameter types: use integers for numeric values, strings for text.",
    "Pay attention to parameter names — map query entities to the correct params.",
    "If a function requires a parameter not in the query, use a sensible default.",
    "Do not refuse — always attempt to invoke at least one function.",
    "For parallel tasks, output all required calls in a single list.",
    "Output format: [func_name(param1=value1, param2=value2)]",
    "Each function call must include all required parameters.",
    "Use the function description to understand when to apply each function.",
    "If two functions have similar names, read their descriptions to choose.",
    "Never include explanations — only output the function call list.",
    "Return values exactly as they appear in the query (preserve casing).",
    "For list parameters, use Python list syntax: param=[val1, val2].",
    "Convert spelled-out numbers to digits: 'five' → 5.",
    "Dates should be in the format expected by the function parameter.",
    "For boolean parameters, use True/False (Python style).",
    "If the query asks for the 'first N' items, set the count parameter to N.",
]


# ─────────────────────────────────────────────────────────────────────────
# Evolution engine (xLAM-specific)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class XLAMExperiment:
    """Tracks one evolution experiment on an xLAM category."""

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
    history: list[tuple[int, float]]
    prompt_evolution: list[dict[str, Any]]
    parallel_ratio: float = 0.0


def _mutate_xlam_template(
    template: str, rng: np.random.Generator, rate: float = 0.5
) -> str:
    """Mutate an xLAM prompt template."""
    lines = template.strip().split("\n")

    if rng.random() < rate:
        instruction = str(rng.choice(_XLAM_MUTATIONS))
        pos = int(rng.integers(1, max(2, len(lines))))
        lines.insert(pos, instruction)

    if rng.random() < rate * 0.5 and len(lines) > 3:
        removable = [
            i for i in range(len(lines))
            if "{xlam_functions}" not in lines[i] and lines[i].strip()
        ]
        if removable:
            idx = int(rng.choice(removable))
            lines.pop(idx)

    if rng.random() < rate * 0.3 and len(lines) > 2:
        idx = int(rng.integers(0, len(lines) - 1))
        lines[idx], lines[idx + 1] = lines[idx + 1], lines[idx]

    result = "\n".join(lines)
    if "{xlam_functions}" not in result:
        result += "\n\nAvailable functions:\n{xlam_functions}"
    return result


def _crossover_xlam_templates(
    a: str, b: str, rng: np.random.Generator
) -> str:
    """Crossover two xLAM prompt templates."""
    la = a.strip().split("\n")
    lb = b.strip().split("\n")
    ca = int(rng.integers(1, max(2, len(la))))
    cb = int(rng.integers(1, max(2, len(lb))))
    child = "\n".join(la[:ca] + lb[cb:])
    if "{xlam_functions}" not in child:
        child += "\n\nAvailable functions:\n{xlam_functions}"
    return child


def run_xlam_evolution(
    cases: list[XLAMCase],
    client: LLMClient,
    config: PromptEvolverConfig,
    algorithm_name: str = "standard",
    seed: int = 42,
    verbose: bool = True,
) -> XLAMExperiment:
    """Run prompt evolution on xLAM test cases.

    Returns an XLAMExperiment with full tracking.
    """
    rng = np.random.default_rng(seed)
    category = cases[0].category if cases else "unknown"
    parallel_count = sum(1 for c in cases if c.is_parallel)
    parallel_ratio = parallel_count / len(cases) if cases else 0.0

    # ── Evaluate a candidate ───────────────────────────────────────────
    def evaluate(
        candidate: PromptCandidate, eval_cases: list[XLAMCase]
    ) -> float:
        total = 0.0
        for case in eval_cases:
            fn_text = _format_xlam_tools(case.tools)
            sys_prompt = candidate.template.replace("{xlam_functions}", fn_text)

            response = client.complete(
                system_prompt=sys_prompt,
                user_message=case.query,
                temperature=candidate.temperature,
                top_p=candidate.top_p,
            )
            if response is None:
                total += float(rng.uniform(0, 0.3))
                continue

            total += score_xlam_case(response, case)

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
        print(f"  xLAM Category: {category}  |  Algorithm: {algorithm_name}")
        print(
            f"  Cases: {len(eval_cases)}  |  Backend: {config.backend.value}  "
            f"|  Parallel: {parallel_ratio:.0%}"
        )
        print(f"{'=' * 60}")

    t0 = time.perf_counter()
    prompt_trace: list[dict[str, Any]] = []

    # Init islands
    islands: list[list[PromptCandidate]] = [
        [] for _ in range(config.num_islands)
    ]

    for i, tmpl in enumerate(_XLAM_SEED_TEMPLATES):
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
                    child_tmpl = _crossover_xlam_templates(
                        parent_a.template, parent_b.template, rng
                    )
                else:
                    child_tmpl = parent_a.template

                if rng.random() < config.mutation_rate:
                    child_tmpl = _mutate_xlam_template(
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
                child.score = evaluate(child, eval_cases)
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

    return XLAMExperiment(
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
        parallel_ratio=parallel_ratio,
    )


# ─────────────────────────────────────────────────────────────────────────
# Baseline evaluation
# ─────────────────────────────────────────────────────────────────────────

_XLAM_DEFAULT_PROMPT = textwrap.dedent("""\
    You are a helpful assistant. Based on the user's request, call the \
appropriate function(s).

Available functions:
{xlam_functions}
""")


def evaluate_baseline(
    cases: list[XLAMCase],
    category: str,
    client: LLMClient,
    verbose: bool = True,
) -> float:
    """Evaluate the default xLAM prompt on cases."""
    if verbose:
        print(f"  Evaluating default prompt baseline ({category})...")

    rng = np.random.default_rng(42)
    total = 0.0
    for case in cases:
        fn_text = _format_xlam_tools(case.tools)
        sys_prompt = _XLAM_DEFAULT_PROMPT.replace("{xlam_functions}", fn_text)

        response = client.complete(
            system_prompt=sys_prompt,
            user_message=case.query,
            temperature=0.1,
            top_p=0.95,
        )
        if response is None:
            total += float(rng.uniform(0, 0.3))
            continue

        total += score_xlam_case(response, case)

    score = (total / len(cases) * 100.0) if cases else 0.0
    if verbose:
        print(f"  Default prompt baseline ({category}): {score:.1f}%")
    return score


# ─────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────


def show_prompt_evolution(experiment: XLAMExperiment) -> None:
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


def show_results_table(
    experiments: list[XLAMExperiment],
    default_baselines: dict[str, float],
) -> None:
    """Print a summary table of all experiments."""
    print(f"\n{'=' * 90}")
    print("  xLAM / APIGen Prompt Evolution Results")
    print(f"{'=' * 90}")
    print(
        f"  {'Category':<16} {'Algorithm':<12} {'Backend':<14} "
        f"{'Default':>8} {'Base':>8} {'Evolved':>8} {'Δ':>7} {'Time':>7}"
    )
    print(f"  {'─' * 85}")

    for exp in experiments:
        dflt = default_baselines.get(exp.category, 0.0)
        delta = exp.evolved_score - exp.baseline_score
        print(
            f"  {exp.category:<16} {exp.algorithm:<12} {exp.backend:<14} "
            f"{dflt:7.1f}% {exp.baseline_score:7.1f}% "
            f"{exp.evolved_score:7.1f}% {delta:+6.1f}% "
            f"{exp.wall_time:6.1f}s"
        )

    print(f"  {'─' * 85}")


# ─────────────────────────────────────────────────────────────────────────
# Experiment log persistence
# ─────────────────────────────────────────────────────────────────────────


def save_experiment_log(
    experiments: list[XLAMExperiment],
    default_baselines: dict[str, float],
    path: str = "xlam_experiment_log.json",
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
            "best_prompt_template": exp.best_prompt_template,
            "parallel_ratio": round(exp.parallel_ratio, 3),
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

BENCHMARK_CATEGORIES = ["finance", "social", "data", "science", "entertainment"]
MAX_PER_CATEGORY = 30  # Load 30 cases per bucket from 60k total


def main() -> None:
    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  EvoSim × xLAM — APIGen Function-Calling Prompt Evolution   ║
    ║  60 000 verified examples · 3 673 APIs · 21 categories       ║
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

    # ── Load xLAM data ─────────────────────────────────────────────────
    print("  Loading xLAM 60k data (downloading from HuggingFace if needed)...")
    try:
        by_cat = load_xlam_dataset(
            max_per_category=MAX_PER_CATEGORY,
            categories=BENCHMARK_CATEGORIES,
        )
    except RuntimeError as exc:
        print(f"  ✗ {exc}")
        return

    for cat in BENCHMARK_CATEGORIES:
        cases = by_cat.get(cat, [])
        parallel = sum(1 for c in cases if c.is_parallel)
        print(
            f"    {cat:<16}: {len(cases):3d} cases  "
            f"({parallel} parallel, {len(cases) - parallel} single)"
        )

    if not by_cat:
        print("  No xLAM data loaded — exiting.")
        return

    # ── Phase 1: Default prompt baselines ──────────────────────────────
    print("\n  Phase 1: Default Prompt Baselines")
    print("  " + "─" * 50)
    default_baselines: dict[str, float] = {}
    for cat in BENCHMARK_CATEGORIES:
        cases = by_cat.get(cat, [])
        if cases:
            score = evaluate_baseline(cases, cat, ollama_client)
            default_baselines[cat] = score

    # ── Phase 2: Ollama evolution ──────────────────────────────────────
    print("\n  Phase 2: Prompt Evolution (Ollama)")
    print("  " + "─" * 50)

    all_experiments: list[XLAMExperiment] = []

    # Standard on all categories
    for cat in BENCHMARK_CATEGORIES:
        cases = by_cat.get(cat, [])
        if not cases:
            continue

        algo = "standard"
        algo_params = ALGORITHM_CONFIGS[algo]
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            timeout=60.0,
            **algo_params,
        )

        exp = run_xlam_evolution(
            cases=cases,
            client=ollama_client,
            config=cfg,
            algorithm_name=algo,
            verbose=True,
        )
        all_experiments.append(exp)
        show_prompt_evolution(exp)

    # Deep on finance (domain transfer test)
    if "finance" in by_cat and by_cat["finance"]:
        algo = "deep"
        algo_params = ALGORITHM_CONFIGS[algo]
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            timeout=60.0,
            **algo_params,
        )

        exp = run_xlam_evolution(
            cases=by_cat["finance"],
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
        for cat in BENCHMARK_CATEGORIES:
            cases = by_cat.get(cat, [])
            if cases:
                score = evaluate_baseline(cases, cat, azure_client)
                default_baselines[f"{cat}_azure"] = score

        # Standard evolution on all categories
        for cat in BENCHMARK_CATEGORIES:
            cases = by_cat.get(cat, [])
            if not cases:
                continue

            algo = "standard"
            algo_params = ALGORITHM_CONFIGS[algo]
            az_cfg = PromptEvolverConfig(
                backend=LLMBackend.AZURE_OPENAI,
                timeout=30.0,
                **algo_params,
            )

            exp = run_xlam_evolution(
                cases=cases,
                client=azure_client,
                config=az_cfg,
                algorithm_name=algo,
                verbose=True,
            )
            all_experiments.append(exp)
            show_prompt_evolution(exp)

        # Deep on finance
        if "finance" in by_cat and by_cat["finance"]:
            algo = "deep"
            algo_params = ALGORITHM_CONFIGS[algo]
            az_cfg = PromptEvolverConfig(
                backend=LLMBackend.AZURE_OPENAI,
                timeout=30.0,
                **algo_params,
            )

            exp = run_xlam_evolution(
                cases=by_cat["finance"],
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
        os.path.dirname(__file__), "..", "..", "xlam_experiment_log.json"
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
