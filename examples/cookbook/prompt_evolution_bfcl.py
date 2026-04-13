#!/usr/bin/env python3
"""
Cookbook Recipe 42 — BFCL V4 Benchmark Prompt Evolution
======================================================

Evolves system prompts on the **Berkeley Function Calling Leaderboard
(BFCL V4)**, closing the gap between native function-calling mode and
prompt mode.

BFCL V4 ships 4 500+ test cases across categories — the same model can
score 77 % in FC mode yet only 33 % in prompt mode.  Prompt evolution
narrows that gap by discovering better system-prompt phrasing.

Categories benchmarked
----------------------
1. **Simple Python** (400 cases) — single function call
2. **Multiple**      (200 cases) — pick the correct function from a set
3. **Parallel**      (200 cases) — invoke >1 function simultaneously
4. **Live Simple**   (258 cases) — real-world single calls

Algorithm experiments
---------------------
* **Fast**     — 3 iterations, pop 4, 2 islands  (smoke test)
* **Standard** — 5 iterations, pop 6, 2 islands  (balanced)
* **Deep**     — 8 iterations, pop 8, 3 islands  (thorough)

Usage::

    uv sync --extra llm
    uv pip install bfcl-eval
    uv run python examples/cookbook/prompt_evolution_bfcl.py

The script saves an experiment log to ``bfcl_experiment_log.json`` for
dashboard consumption.
"""
from __future__ import annotations

import copy
import json
import os
import re
import sys
import textwrap
import time
from dataclasses import dataclass
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
# BFCL data loading
# ─────────────────────────────────────────────────────────────────────────

_BFCL_DATA_DIR: Optional[Path] = None


def _bfcl_data_dir() -> Path:
    """Locate the installed bfcl_eval data directory."""
    global _BFCL_DATA_DIR
    if _BFCL_DATA_DIR is not None:
        return _BFCL_DATA_DIR

    # Walk site-packages to find the bfcl_eval data folder
    import site

    for base in site.getsitepackages() + [site.getusersitepackages()]:
        candidate = Path(base) / "bfcl_eval" / "data"
        if candidate.is_dir():
            _BFCL_DATA_DIR = candidate
            return candidate

    # Fallback: try relative to the virtualenv
    venv = Path(sys.prefix) / "lib"
    for p in venv.rglob("bfcl_eval/data"):
        if p.is_dir():
            _BFCL_DATA_DIR = p
            return p

    raise FileNotFoundError(
        "bfcl_eval data directory not found. Install with: uv pip install bfcl-eval"
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class BFCLTestCase:
    """A single BFCL test case with question, function defs, and expected output."""

    id: str
    user_query: str
    functions: list[dict[str, Any]]
    ground_truth: list[dict[str, dict[str, list[Any]]]]
    category: str

    @property
    def is_parallel(self) -> bool:
        return len(self.ground_truth) > 1

    @property
    def expected_function_names(self) -> list[str]:
        return [list(gt.keys())[0] for gt in self.ground_truth]


def load_bfcl_category(
    category: str, max_cases: int | None = None
) -> list[BFCLTestCase]:
    """Load BFCL V4 test cases for a category.

    Parameters
    ----------
    category:
        One of ``simple_python``, ``multiple``, ``parallel``,
        ``parallel_multiple``, ``live_simple``, ``live_multiple``, etc.
    max_cases:
        Maximum number of cases to load (random sample if smaller than total).
    """
    data_dir = _bfcl_data_dir()
    q_path = data_dir / f"BFCL_v4_{category}.json"
    a_path = data_dir / "possible_answer" / f"BFCL_v4_{category}.json"

    if not q_path.exists():
        raise FileNotFoundError(f"BFCL category file not found: {q_path}")
    if not a_path.exists():
        raise FileNotFoundError(f"BFCL answer file not found: {a_path}")

    questions = _load_jsonl(q_path)
    answers = _load_jsonl(a_path)

    # Build answer lookup
    answer_map = {a["id"]: a["ground_truth"] for a in answers}

    cases: list[BFCLTestCase] = []
    for q in questions:
        qid = q["id"]
        gt = answer_map.get(qid)
        if gt is None:
            continue  # Skip cases without answers
        # User query is in question[0][0]["content"]
        user_query = q["question"][0][0]["content"]
        cases.append(
            BFCLTestCase(
                id=qid,
                user_query=user_query,
                functions=q.get("function", []),
                ground_truth=gt,
                category=category,
            )
        )

    if max_cases and max_cases < len(cases):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(cases), size=max_cases, replace=False)
        cases = [cases[int(i)] for i in sorted(indices)]

    return cases


# ─────────────────────────────────────────────────────────────────────────
# BFCL-aware scoring
# ─────────────────────────────────────────────────────────────────────────


def _parse_function_calls(response: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse LLM response into a list of (function_name, params) tuples.

    Handles formats:
    - Python-style:  func_name(param=value, ...)
    - JSON:          {"function": "name", "parameters": {...}}
    - List of calls: [func1(...), func2(...)]
    """
    if not response:
        return []

    calls: list[tuple[str, dict[str, Any]]] = []

    # Try JSON array first
    try:
        stripped = response.strip()
        # Find JSON array or object
        json_match = re.search(r"\[.*\]", stripped, re.DOTALL)
        if json_match:
            arr = json.loads(json_match.group())
            if isinstance(arr, list):
                for item in arr:
                    if isinstance(item, dict):
                        fname = (
                            item.get("function")
                            or item.get("name")
                            or item.get("tool", "")
                        )
                        params = (
                            item.get("parameters")
                            or item.get("params")
                            or item.get("arguments", {})
                        )
                        if fname:
                            calls.append((fname, params if isinstance(params, dict) else {}))
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
                obj.get("function")
                or obj.get("name")
                or obj.get("tool", "")
            )
            params = (
                obj.get("parameters")
                or obj.get("params")
                or obj.get("arguments", {})
            )
            if fname:
                return [(fname, params if isinstance(params, dict) else {})]
    except (json.JSONDecodeError, ValueError):
        pass

    # Python-style: func_name(key=value, ...)
    # Match all function calls in the response
    pattern = r"([\w.]+)\s*\(([^)]*)\)"
    for match in re.finditer(pattern, response):
        fname = match.group(1)
        param_str = match.group(2)

        # Skip common false positives
        if fname.lower() in ("if", "for", "while", "def", "class", "return", "print"):
            continue

        params: dict[str, Any] = {}
        # Parse keyword arguments
        for kv in re.finditer(
            r"""(\w+)\s*=\s*(?:"([^"]*)"|'([^']*)'|([\w.+-]+))""",
            param_str,
        ):
            key = kv.group(1)
            val = kv.group(2) or kv.group(3) or kv.group(4) or ""
            # Try to convert to number
            try:
                val = int(val)  # type: ignore[assignment]
            except (ValueError, TypeError):
                try:
                    val = float(val)  # type: ignore[assignment]
                except (ValueError, TypeError):
                    if val.lower() == "true":
                        val = True  # type: ignore[assignment]
                    elif val.lower() == "false":
                        val = False  # type: ignore[assignment]
            params[key] = val

        calls.append((fname, params))

    return calls


def _match_param_value(predicted: Any, acceptable: list[Any]) -> bool:
    """Check if a predicted value matches any acceptable value."""
    for expected in acceptable:
        if expected == "":
            # Empty string means param is optional — any value is fine
            return True
        # Direct match
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
    return False


def score_bfcl_case(
    response: str, test_case: BFCLTestCase
) -> float:
    """Score a model response against a BFCL test case.

    Returns a float in [0, 1]:
    - 1.0 = perfect match (all expected calls with correct params)
    - Partial credit for correct function names with wrong params
    - 0.0 = complete miss
    """
    parsed_calls = _parse_function_calls(response)
    if not parsed_calls:
        return 0.0

    ground_truth = test_case.ground_truth
    n_expected = len(ground_truth)

    if n_expected == 0:
        return 1.0 if not parsed_calls else 0.0

    # For simple/multiple cases (single expected call)
    if n_expected == 1:
        expected_fn = list(ground_truth[0].keys())[0]
        expected_params = ground_truth[0][expected_fn]

        # Find the best matching call
        best_score = 0.0
        for pred_fn, pred_params in parsed_calls:
            # Normalise function name (some models use _ instead of .)
            if pred_fn.replace("_", ".") == expected_fn or pred_fn == expected_fn:
                # Function name match: 0.5 base
                param_score = _score_params(pred_params, expected_params)
                call_score = 0.5 + 0.5 * param_score
                best_score = max(best_score, call_score)
            elif pred_fn.lower() == expected_fn.lower().replace(".", "_"):
                # Approximate name match
                param_score = _score_params(pred_params, expected_params)
                call_score = 0.3 + 0.5 * param_score
                best_score = max(best_score, call_score)

        return best_score

    # For parallel cases (multiple expected calls)
    matched = [False] * n_expected
    total_score = 0.0

    for gt_idx, gt_entry in enumerate(ground_truth):
        expected_fn = list(gt_entry.keys())[0]
        expected_params = gt_entry[expected_fn]

        best_call_score = 0.0
        best_call_idx = -1

        for call_idx, (pred_fn, pred_params) in enumerate(parsed_calls):
            if pred_fn.replace("_", ".") == expected_fn or pred_fn == expected_fn:
                param_score = _score_params(pred_params, expected_params)
                call_score = 0.5 + 0.5 * param_score
                if call_score > best_call_score:
                    best_call_score = call_score
                    best_call_idx = call_idx

        total_score += best_call_score
        if best_call_idx >= 0:
            matched[gt_idx] = True

    return total_score / n_expected


def _score_params(
    predicted: dict[str, Any],
    expected: dict[str, list[Any]],
) -> float:
    """Score predicted parameters against BFCL expected params."""
    if not expected:
        return 1.0

    required_keys = [k for k, v in expected.items() if "" not in v]
    optional_keys = [k for k, v in expected.items() if "" in v]
    all_keys = required_keys + optional_keys

    if not all_keys:
        return 1.0

    matches = 0.0
    total = len(required_keys) if required_keys else len(all_keys)

    for key in required_keys:
        if key in predicted:
            if _match_param_value(predicted[key], expected[key]):
                matches += 1.0
            else:
                matches += 0.3  # Partial credit: correct key, wrong value

    # Bonus for correct optional params
    for key in optional_keys:
        if key in predicted and _match_param_value(predicted[key], expected[key]):
            matches += 0.2  # Small bonus

    return min(1.0, matches / total) if total > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────
# BFCL-specific prompt templates
# ─────────────────────────────────────────────────────────────────────────


def _format_bfcl_functions(functions: list[dict[str, Any]]) -> str:
    """Format BFCL function definitions as text for the system prompt."""
    lines: list[str] = []
    for fn in functions:
        name = fn["name"]
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))

        param_parts: list[str] = []
        for pname, pinfo in props.items():
            ptype = pinfo.get("type", "string")
            pdesc = pinfo.get("description", "")
            req = " [required]" if pname in required else " [optional]"
            param_parts.append(f"    - {pname} ({ptype}{req}): {pdesc}")

        lines.append(f"  {name}: {desc}")
        if param_parts:
            lines.append("\n".join(param_parts))
    return "\n".join(lines)


_BFCL_SEED_TEMPLATES = [
    # T0: Minimal — direct Python call format
    textwrap.dedent("""\
        You are an expert in composing functions. Given a question and a set of \
possible functions, make one or more function calls to achieve the purpose.
If none of the functions can be used, point it out.

You should only return the function calls in your response.
If you decide to invoke any of the function(s), you MUST put it in the format of \
[func_name(param_name=param_value, param_name2=param_value2...)]
You SHOULD NOT include any other text in the response.

Here is a list of functions you can invoke:
{bfcl_functions}
    """),

    # T1: Structured with reasoning
    textwrap.dedent("""\
        You are an expert in generating structured function calls.
Your task: given a user query and available functions, produce one or more \
function calls to fulfil the request.

Step 1: Identify the user's intent.
Step 2: Select the correct function(s).
Step 3: Extract parameters from the query.

Output ONLY function calls in this format:
[func_name(param=value, ...)]

If multiple calls are needed, include them all in the list:
[func1(p1=v1), func2(p2=v2)]

Do NOT include any other text.

Available functions:
{bfcl_functions}
    """),

    # T2: JSON-style output
    textwrap.dedent("""\
        You are a precise tool router. Analyse the user's request and invoke \
the correct function(s).

Rules:
- Pick the most specific function that matches.
- Extract parameter values directly from the query.
- If a parameter is not mentioned, use a reasonable default.
- Output ONLY function calls: [func_name(param=value)]
- For multiple calls: [func1(p1=v1), func2(p2=v2)]

Functions:
{bfcl_functions}
    """),

    # T3: Emphasis on parameter extraction
    textwrap.dedent("""\
        # Role
        You are a function-calling agent. Your job is to map user requests to \
precise function invocations.

# Instructions
1. Read the user's query carefully.
2. Match it to one or more available functions.
3. Extract ALL required parameters from the query text.
4. For optional parameters, include them only if clearly mentioned.
5. Return ONLY the function call(s) in Python format:
   [function_name(param1=value1, param2=value2)]

# Available Functions
{bfcl_functions}
    """),
]

_BFCL_MUTATIONS = [
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
]


# ─────────────────────────────────────────────────────────────────────────
# Evolution engine (BFCL-specific)
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class BFCLExperiment:
    """Tracks one evolution experiment on a BFCL category."""

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


def _mutate_bfcl_template(
    template: str, rng: np.random.Generator, rate: float = 0.5
) -> str:
    """Mutate a BFCL prompt template."""
    lines = template.strip().split("\n")

    if rng.random() < rate:
        instruction = str(rng.choice(_BFCL_MUTATIONS))
        pos = int(rng.integers(1, max(2, len(lines))))
        lines.insert(pos, instruction)

    if rng.random() < rate * 0.5 and len(lines) > 3:
        removable = [
            i for i in range(len(lines))
            if "{bfcl_functions}" not in lines[i] and lines[i].strip()
        ]
        if removable:
            idx = int(rng.choice(removable))
            lines.pop(idx)

    if rng.random() < rate * 0.3 and len(lines) > 2:
        idx = int(rng.integers(0, len(lines) - 1))
        lines[idx], lines[idx + 1] = lines[idx + 1], lines[idx]

    result = "\n".join(lines)
    if "{bfcl_functions}" not in result:
        result += "\n\nAvailable functions:\n{bfcl_functions}"
    return result


def _crossover_bfcl_templates(
    a: str, b: str, rng: np.random.Generator
) -> str:
    """Crossover two BFCL prompt templates."""
    la = a.strip().split("\n")
    lb = b.strip().split("\n")
    ca = int(rng.integers(1, max(2, len(la))))
    cb = int(rng.integers(1, max(2, len(lb))))
    child = "\n".join(la[:ca] + lb[cb:])
    if "{bfcl_functions}" not in child:
        child += "\n\nAvailable functions:\n{bfcl_functions}"
    return child


def run_bfcl_evolution(
    test_cases: list[BFCLTestCase],
    client: LLMClient,
    config: PromptEvolverConfig,
    algorithm_name: str = "standard",
    seed: int = 42,
    verbose: bool = True,
) -> BFCLExperiment:
    """Run prompt evolution on BFCL test cases.

    Returns a BFCLExperiment with full tracking.
    """
    rng = np.random.default_rng(seed)
    category = test_cases[0].category if test_cases else "unknown"

    # ── Evaluate a candidate on BFCL data ──────────────────────────────
    def evaluate(candidate: PromptCandidate, cases: list[BFCLTestCase]) -> float:
        total = 0.0
        for case in cases:
            fn_text = _format_bfcl_functions(case.functions)
            sys_prompt = candidate.template.replace("{bfcl_functions}", fn_text)

            response = client.complete(
                system_prompt=sys_prompt,
                user_message=case.user_query,
                temperature=candidate.temperature,
                top_p=candidate.top_p,
            )
            if response is None:
                total += float(rng.uniform(0, 0.3))
                continue

            total += score_bfcl_case(response, case)

        return (total / len(cases) * 100.0) if cases else 0.0

    # ── Subsample for evaluation ───────────────────────────────────────
    if config.eval_sample_size and config.eval_sample_size < len(test_cases):
        eval_indices = rng.choice(
            len(test_cases), size=config.eval_sample_size, replace=False
        )
        eval_cases = [test_cases[int(i)] for i in eval_indices]
    else:
        eval_cases = test_cases

    # ── Baseline: evaluate seed templates ──────────────────────────────
    if verbose:
        print(f"\n{'=' * 60}")
        print(f"  BFCL Category: {category}  |  Algorithm: {algorithm_name}")
        print(f"  Cases: {len(eval_cases)}  |  Backend: {config.backend.value}")
        print(f"{'=' * 60}")

    t0 = time.perf_counter()
    prompt_trace: list[dict[str, Any]] = []

    # Init islands
    islands: list[list[PromptCandidate]] = [
        [] for _ in range(config.num_islands)
    ]

    for i, tmpl in enumerate(_BFCL_SEED_TEMPLATES):
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
    all_candidates: list[PromptCandidate] = [
        c for isl in islands for c in isl
    ]

    for gen in range(1, config.iterations + 1):
        for isl_id in range(config.num_islands):
            island = islands[isl_id]
            if not island:
                continue

            new_cands: list[PromptCandidate] = []
            for _ in range(config.population_size):
                # Tournament select parent
                k = min(3, len(island))
                idxs = rng.choice(len(island), size=k, replace=False)
                parent_a = max(
                    (island[int(i)] for i in idxs), key=lambda c: c.score
                )

                # Crossover
                if rng.random() < config.crossover_rate and len(island) > 1:
                    idxs2 = rng.choice(len(island), size=k, replace=False)
                    parent_b = max(
                        (island[int(i)] for i in idxs2), key=lambda c: c.score
                    )
                    child_tmpl = _crossover_bfcl_templates(
                        parent_a.template, parent_b.template, rng
                    )
                else:
                    child_tmpl = parent_a.template

                # Mutate template
                if rng.random() < config.mutation_rate:
                    child_tmpl = _mutate_bfcl_template(
                        child_tmpl, rng, config.mutation_rate
                    )

                # Mutate continuous params
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
                all_candidates.append(child)

            # Select elite
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

    return BFCLExperiment(
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
    )


# ─────────────────────────────────────────────────────────────────────────
# Baseline evaluation
# ─────────────────────────────────────────────────────────────────────────

_BFCL_DEFAULT_PROMPT = textwrap.dedent("""\
    You are an expert in composing functions. You are given a question and a \
set of possible functions. Based on the question, you will need to make one \
or more function/tool calls to achieve the purpose.
If none of the functions can be used, point it out. If the given question \
lacks the parameters required by the function, also point it out.
You should only return the function calls in your response.

If you decide to invoke any of the function(s), you MUST put it in the \
format of [func_name1(params_name1=params_value1, params_name2=params_value2...), \
func_name2(params)]
You SHOULD NOT include any other text in the response.

Here is a list of functions you can invoke:
{bfcl_functions}
""")


def evaluate_baseline(
    test_cases: list[BFCLTestCase],
    client: LLMClient,
    verbose: bool = True,
) -> float:
    """Evaluate BFCL's default system prompt on test cases."""
    if verbose:
        print("  Evaluating BFCL default prompt baseline...")

    total = 0.0
    rng = np.random.default_rng(42)
    for case in test_cases:
        fn_text = _format_bfcl_functions(case.functions)
        sys_prompt = _BFCL_DEFAULT_PROMPT.replace("{bfcl_functions}", fn_text)
        response = client.complete(
            system_prompt=sys_prompt,
            user_message=case.user_query,
            temperature=0.1,
            top_p=0.95,
        )
        if response is None:
            total += float(rng.uniform(0, 0.3))
            continue
        total += score_bfcl_case(response, case)

    score = (total / len(test_cases) * 100.0) if test_cases else 0.0
    if verbose:
        print(f"  BFCL default prompt baseline: {score:.1f}%")
    return score


# ─────────────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────────────


def show_prompt_evolution(experiment: BFCLExperiment) -> None:
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


def show_results_table(experiments: list[BFCLExperiment], default_baseline: dict[str, float]) -> None:
    """Print a summary table of all experiments."""
    print(f"\n{'=' * 78}")
    print("  BFCL V4 Prompt Evolution Results")
    print(f"{'=' * 78}")
    print(
        f"  {'Category':<18} {'Algorithm':<12} {'Backend':<10} "
        f"{'Default':>8} {'Base':>8} {'Evolved':>8} {'Δ':>7} {'Time':>7}"
    )
    print(f"  {'─' * 74}")

    for exp in experiments:
        dflt = default_baseline.get(exp.category, 0.0)
        delta = exp.evolved_score - exp.baseline_score
        print(
            f"  {exp.category:<18} {exp.algorithm:<12} {exp.backend:<10} "
            f"{dflt:7.1f}% {exp.baseline_score:7.1f}% "
            f"{exp.evolved_score:7.1f}% {delta:+6.1f}% "
            f"{exp.wall_time:6.1f}s"
        )

    print(f"  {'─' * 74}")


# ─────────────────────────────────────────────────────────────────────────
# Experiment log persistence
# ─────────────────────────────────────────────────────────────────────────


def save_experiment_log(
    experiments: list[BFCLExperiment],
    default_baselines: dict[str, float],
    path: str = "bfcl_experiment_log.json",
) -> None:
    """Save experiment results to JSON for dashboard consumption."""
    log = {
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
        }
        log["experiments"].append(entry)

    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\n  Experiment log saved to {path}")


# ─────────────────────────────────────────────────────────────────────────
# Main — run all experiments
# ─────────────────────────────────────────────────────────────────────────


# Algorithm configurations to experiment with
ALGORITHM_CONFIGS: dict[str, dict[str, Any]] = {
    "fast": {
        "iterations": 2,
        "population_size": 3,
        "num_islands": 2,
        "elite_size": 2,
        "mutation_rate": 0.6,
        "crossover_rate": 0.3,
        "eval_sample_size": 8,
    },
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

# Categories to benchmark (with sample sizes for tractable evolution)
BENCHMARK_CATEGORIES: list[tuple[str, int]] = [
    ("simple_python", 20),
    ("multiple", 15),
    ("parallel", 12),
    ("live_simple", 15),
]


def main() -> None:
    banner = r"""
    ╔══════════════════════════════════════════════════════════════╗
    ║  EvoSim × BFCL V4 — Benchmark Prompt Evolution             ║
    ║  Closing the FC-vs-Prompt gap with evolutionary search      ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

    # ── Backend setup ──────────────────────────────────────────────────
    ollama_cfg = PromptEvolverConfig(
        backend=LLMBackend.OLLAMA,
        ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
        timeout=45.0,
    )
    ollama_client = LLMClient(ollama_cfg)

    if not ollama_client.is_available():
        print("  ⚠ Ollama not available — ensure it is running at localhost:11434")
        print("  Running in mock mode (random scores) for demonstration.\n")

    # ── Load BFCL data ─────────────────────────────────────────────────
    print("  Loading BFCL V4 data...")
    category_data: dict[str, list[BFCLTestCase]] = {}
    for cat, n_samples in BENCHMARK_CATEGORIES:
        try:
            cases = load_bfcl_category(cat, max_cases=n_samples)
            category_data[cat] = cases
            print(f"    {cat}: {len(cases)} cases loaded")
        except FileNotFoundError as exc:
            print(f"    {cat}: SKIPPED ({exc})")

    if not category_data:
        print("  No BFCL data loaded — exiting.")
        return

    # ── Evaluate BFCL default prompt baselines ─────────────────────────
    print("\n  Phase 1: Default BFCL Prompt Baselines")
    print("  " + "─" * 50)
    default_baselines: dict[str, float] = {}
    for cat, cases in category_data.items():
        score = evaluate_baseline(cases, ollama_client)
        default_baselines[cat] = score

    # ── Run evolution experiments ──────────────────────────────────────
    print("\n  Phase 2: Prompt Evolution Experiments")
    print("  " + "─" * 50)

    all_experiments: list[BFCLExperiment] = []

    # Run "standard" algorithm on all categories
    for cat, cases in category_data.items():
        algo = "standard"
        algo_params = ALGORITHM_CONFIGS[algo]
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            timeout=45.0,
            **algo_params,
        )

        exp = run_bfcl_evolution(
            test_cases=cases,
            client=ollama_client,
            config=cfg,
            algorithm_name=algo,
            verbose=True,
        )
        all_experiments.append(exp)
        show_prompt_evolution(exp)

    # Run "deep" algorithm on simple_python (best category for comparison)
    if "simple_python" in category_data:
        algo = "deep"
        algo_params = ALGORITHM_CONFIGS[algo]
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_model=os.environ.get("OLLAMA_MODEL", "llama3.2"),
            timeout=45.0,
            **algo_params,
        )

        exp = run_bfcl_evolution(
            test_cases=category_data["simple_python"],
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

        # Evaluate baselines with Azure OpenAI
        azure_baselines: dict[str, float] = {}
        for cat, cases in category_data.items():
            score = evaluate_baseline(cases, azure_client)
            azure_baselines[cat] = score
            default_baselines[f"{cat}_azure"] = score

        # Standard evolution on all categories
        for cat, cases in category_data.items():
            algo = "standard"
            algo_params = ALGORITHM_CONFIGS[algo]
            az_evo_cfg = PromptEvolverConfig(
                backend=LLMBackend.AZURE_OPENAI,
                timeout=30.0,
                **algo_params,
            )

            exp = run_bfcl_evolution(
                test_cases=cases,
                client=azure_client,
                config=az_evo_cfg,
                algorithm_name=algo,
                verbose=True,
            )
            all_experiments.append(exp)
            show_prompt_evolution(exp)

        # Deep evolution on simple_python
        if "simple_python" in category_data:
            algo = "deep"
            algo_params = ALGORITHM_CONFIGS[algo]
            az_evo_cfg = PromptEvolverConfig(
                backend=LLMBackend.AZURE_OPENAI,
                timeout=30.0,
                **algo_params,
            )

            exp = run_bfcl_evolution(
                test_cases=category_data["simple_python"],
                client=azure_client,
                config=az_evo_cfg,
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
        os.path.dirname(__file__), "..", "..", "bfcl_experiment_log.json"
    )
    save_experiment_log(all_experiments, default_baselines, log_path)

    # ── Print best evolved prompt ──────────────────────────────────────
    if all_experiments:
        best_exp = max(all_experiments, key=lambda e: e.evolved_score)
        print(f"\n{'=' * 60}")
        print(f"  Best Evolved Prompt ({best_exp.category} / {best_exp.algorithm})")
        print(f"  Score: {best_exp.evolved_score:.1f}%")
        print(f"{'=' * 60}")
        # Show first 500 chars of template
        preview = best_exp.best_prompt_template[:500]
        for line in preview.split("\n"):
            print(f"    {line}")
        if len(best_exp.best_prompt_template) > 500:
            print(f"    ... ({len(best_exp.best_prompt_template)} chars total)")

    print("\n  Done.\n")


if __name__ == "__main__":
    main()
