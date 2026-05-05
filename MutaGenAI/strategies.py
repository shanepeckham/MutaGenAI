"""
No-eval prompt evolution strategies for EvoSim.

When you don't have a labelled evaluation dataset, these seven strategies
provide alternative fitness signals for the evolutionary loop:

1. **LLM-as-Judge**         — A second LLM scores outputs against a rubric.
2. **Synthetic Eval**       — Auto-generate test cases from a task description.
3. **Tool-Use Success**     — Use tool/API return codes as fitness signals.
4. **Self-Consistency**     — Score prompts by output agreement across runs.
5. **Proxy Metrics**        — Structural checks (valid JSON, required fields, length).
6. **Preference Scoring**   — Score with good/bad output examples.
7. **Human Tournament**     — Human selects best output per generation.

Each strategy is a callable that returns a float in [0, 1].
They plug directly into :class:`NoEvalPromptEvolver` as the fitness function.

Quick start::

    from MutaGenAI.strategies import (
        NoEvalPromptEvolver,
        NoEvalConfig,
        LLMJudge,
        SyntheticEvalGenerator,
        ToolSuccessScorer,
        SelfConsistencyScorer,
        ProxyMetricsScorer,
        PreferenceScorer,
        HumanTournament,
        CompositeScorer,
    )

    # Simplest: just use LLM-as-Judge
    judge = LLMJudge(rubric="Score 0-10 on helpfulness and accuracy.")
    evolver = NoEvalPromptEvolver(
        task_description="You are a customer service agent ...",
        test_inputs=["How do I return an item?", "Where is my order?"],
        scorer=judge,
    )
    result = evolver.run()
    print(result.best_prompt)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import statistics
import textwrap
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from MutaGenAI.prompt_evolver import (
    ErrorProfile,
    LLMBackend,
    LLMClient,
    ProblemType,
    PromptCandidate,
    PromptEvolverConfig,
    PromptEvolverResult,
    _SEED_TEMPLATES,
    _crossover_templates,
    _has_entity_descriptions,
    _llm_describe_entities,
    _llm_mutate_template,
    _mutate_template,
    _refine_template,
    generate_adaptive_mutations,
    get_mutations_for_problem_type,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class NoEvalConfig:
    """Configuration for no-eval prompt evolution.

    Attributes
    ----------
    iterations : int
        Number of evolutionary generations.
    population_size : int
        New candidates per island per generation.
    num_islands : int
        Number of islands for diversity.
    elite_size : int
        Top candidates retained per island.
    mutation_rate : float
        Probability of mutating a candidate's template.
    crossover_rate : float
        Probability of crossover between two parents.
    temperature_range : tuple[float, float]
        Bounds for LLM temperature.
    top_p_range : tuple[float, float]
        Bounds for LLM top-p.
    migration_interval : int
        Generations between island migrations.
    backend : LLMBackend
        Which LLM backend to use.
    max_tokens : int or None
        Max tokens to generate per LLM call (``num_predict`` for Ollama).
        ``None`` means no limit (model default).
    refine_after_splice : bool
        If ``True``, run an LLM call after each mutation/crossover to
        remove duplicate lines, fix incoherent phrasing, and tighten
        the prompt while preserving its meaning.  Costs one extra LLM
        call per breed operation.  Default ``False``.
    problem_type : ProblemType
        The nature of the task being optimised (``tool_routing`` or
        ``classification``).  Determines which built-in mutation
        snippets are injected during evolution.
    adaptive_mutations : bool
        If ``True``, after each generation the evolver analyses which
        categories (classes or tools) have the highest error rate and
        uses an LLM to generate targeted mutation snippets.  These are
        appended to the static mutation pool for the next generation.
        Default ``False``.
    llm_mutation_rate : float
        Probability of applying an LLM-assisted rewrite (instead of
        random snippet mutation) during breeding.  The LLM receives the
        parent prompt plus a sample of recent failure cases and rewrites
        it to fix those errors.  ``0.0`` disables LLM mutation.
        Default ``0.0``.
    describe_entities : bool
        If ``True``, the describe-entities LLM rewrite is added to the
        mutation pool.  During breeding, a candidate may be randomly
        selected for expansion of bare agent/category names into
        ``name — description`` format.  This lets evolution discover
        whether descriptions help.  Default ``False``.
    """

    iterations: int = 5
    population_size: int = 4
    num_islands: int = 2
    elite_size: int = 3
    mutation_rate: float = 0.5
    crossover_rate: float = 0.3
    temperature_range: tuple[float, float] = (0.0, 1.0)
    top_p_range: tuple[float, float] = (0.7, 1.0)
    migration_interval: int = 3
    backend: LLMBackend = LLMBackend.OLLAMA
    max_tokens: Optional[int] = None
    refine_after_splice: bool = False
    problem_type: ProblemType = ProblemType.TOOL_ROUTING
    adaptive_mutations: bool = False
    llm_mutation_rate: float = 0.0
    describe_entities: bool = False
    warmup_adaptive: bool = False
    error_decay: float = 0.0


# ---------------------------------------------------------------------------
# Strategy interfaces
# ---------------------------------------------------------------------------


class Scorer(ABC):
    """Abstract base class for fitness scoring strategies."""

    @abstractmethod
    def score(
        self,
        prompt: str,
        test_input: str,
        output: str,
        client: LLMClient,
    ) -> float:
        """Score a single (input, output) pair.  Returns float in [0, 1]."""

    def name(self) -> str:
        return self.__class__.__name__


# ---------------------------------------------------------------------------
# Strategy 1: LLM-as-Judge
# ---------------------------------------------------------------------------


class LLMJudge(Scorer):
    """Use a second LLM call to score an output against a rubric.

    The judge sees the system prompt (task), the user input, and the
    model's output, then rates it on a 0–10 scale.  The rubric tells
    the judge *what* to look for.

    Parameters
    ----------
    rubric : str
        Free-text scoring instructions for the judge.  Example:
        ``"Score 0-10 on: (a) correctness, (b) helpfulness, (c) conciseness."``
    max_score : float
        Maximum score the judge can assign (default 10).

    Example
    -------
    >>> judge = LLMJudge(rubric="Rate 0-10 on accuracy and helpfulness.")
    >>> score = judge.score(prompt, user_input, output, client)
    """

    def __init__(self, rubric: str, max_score: float = 10.0) -> None:
        self.rubric = rubric
        self.max_score = max_score

    _JUDGE_TEMPLATE = textwrap.dedent("""\
        You are an impartial judge evaluating an AI assistant's response.

        ## Task Description (system prompt given to the assistant)
        {prompt}

        ## User Input
        {user_input}

        ## Assistant Output
        {output}

        ## Scoring Rubric
        {rubric}

        Rate the output on a scale of 0 to {max_score}.
        Respond with ONLY a JSON object: {{"score": <number>, "reason": "<brief explanation>"}}
    """)

    def score(
        self,
        prompt: str,
        test_input: str,
        output: str,
        client: LLMClient,
    ) -> float:
        judge_prompt = self._JUDGE_TEMPLATE.format(
            prompt=prompt,
            user_input=test_input,
            output=output,
            rubric=self.rubric,
            max_score=int(self.max_score),
        )
        response = client.complete(
            system_prompt="You are a fair evaluation judge. Always respond with valid JSON.",
            user_message=judge_prompt,
            temperature=0.1,
            top_p=0.95,
        )
        if response is None:
            return 0.0
        return self._parse_score(response)

    def _parse_score(self, response: str) -> float:
        """Extract numeric score from judge response."""
        # Try JSON first
        try:
            data = json.loads(response)
            raw = float(data.get("score", 0))
            return min(raw / self.max_score, 1.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        # Fallback: find first number in response
        match = re.search(r"(\d+(?:\.\d+)?)", response)
        if match:
            raw = float(match.group(1))
            return min(raw / self.max_score, 1.0)
        return 0.0


# ---------------------------------------------------------------------------
# Strategy 2: Synthetic Eval Generation
# ---------------------------------------------------------------------------


class SyntheticEvalGenerator:
    """Generate synthetic test cases from a task description using an LLM.

    Call :meth:`generate` once to produce input/output pairs, then use
    :class:`SyntheticEvalScorer` as the fitness function.

    Parameters
    ----------
    task_description : str
        What the agent should do (same text used as the system prompt).
    num_cases : int
        How many test cases to generate.

    Example
    -------
    >>> gen = SyntheticEvalGenerator(task_description="You are a SQL assistant.", num_cases=20)
    >>> cases = gen.generate(client)
    >>> scorer = SyntheticEvalScorer(cases)
    """

    _GEN_TEMPLATE = textwrap.dedent("""\
        Generate {num_cases} test cases for the following AI assistant task.
        Each test case has a user input and the ideal assistant output.

        ## Task Description
        {task_description}

        Return a JSON array of objects with "input" and "expected_output" keys.
        Example: [{{"input": "...", "expected_output": "..."}}]
        Return ONLY the JSON array, no markdown or explanation.
    """)

    def __init__(self, task_description: str, num_cases: int = 20) -> None:
        self.task_description = task_description
        self.num_cases = num_cases

    def generate(self, client: LLMClient) -> list[dict[str, str]]:
        """Generate synthetic eval cases via LLM.

        Returns
        -------
        list[dict[str, str]]
            Each dict has ``"input"`` and ``"expected_output"`` keys.
        """
        response = client.complete(
            system_prompt="You generate high-quality test data. Respond with valid JSON only.",
            user_message=self._GEN_TEMPLATE.format(
                num_cases=self.num_cases,
                task_description=self.task_description,
            ),
            temperature=0.7,
            top_p=0.95,
        )
        if response is None:
            return []
        return self._parse_cases(response)

    def _parse_cases(self, response: str) -> list[dict[str, str]]:
        """Parse the JSON array of test cases from the LLM response."""
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?\s*", "", response).strip().rstrip("`")
        try:
            cases = json.loads(cleaned)
            if isinstance(cases, list):
                return [
                    c
                    for c in cases
                    if isinstance(c, dict)
                    and "input" in c
                    and "expected_output" in c
                ]
        except json.JSONDecodeError:
            pass
        return []


class SyntheticEvalScorer(Scorer):
    """Score outputs by similarity to synthetic expected outputs.

    Uses an LLM judge to compare the actual output against the
    synthetic expected output for semantic similarity.

    Parameters
    ----------
    cases : list[dict[str, str]]
        Synthetic test cases from :class:`SyntheticEvalGenerator`.
    """

    def __init__(self, cases: list[dict[str, str]]) -> None:
        self.cases = cases
        self._case_map = {c["input"]: c["expected_output"] for c in cases}

    def score(
        self,
        prompt: str,
        test_input: str,
        output: str,
        client: LLMClient,
    ) -> float:
        expected = self._case_map.get(test_input, "")
        if not expected:
            return 0.5  # No expected output — neutral score

        compare_prompt = textwrap.dedent(f"""\
            Compare these two outputs for semantic similarity.

            Expected: {expected}
            Actual: {output}

            Rate similarity from 0 to 10 (10 = identical meaning).
            Respond with ONLY a JSON object: {{"score": <number>}}
        """)
        response = client.complete(
            system_prompt="You compare text outputs. Respond with valid JSON only.",
            user_message=compare_prompt,
            temperature=0.1,
            top_p=0.95,
        )
        if response is None:
            return 0.0
        try:
            data = json.loads(response)
            return min(float(data.get("score", 0)) / 10.0, 1.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            match = re.search(r"(\d+(?:\.\d+)?)", response or "")
            return min(float(match.group(1)) / 10.0, 1.0) if match else 0.0


# ---------------------------------------------------------------------------
# Strategy 3: Tool-Use Success Signals
# ---------------------------------------------------------------------------


@dataclass
class ToolResult:
    """Result of executing a tool call.

    Attributes
    ----------
    success : bool
        Whether the tool executed without error.
    return_code : int
        HTTP status code or exit code (0 = success).
    output : str
        Tool output or error message.
    """

    success: bool
    return_code: int = 0
    output: str = ""


class ToolSuccessScorer(Scorer):
    """Score based on whether tool calls succeed when executed.

    Provide a ``tool_executor`` callback that takes a tool name and
    parameters dict, then returns a :class:`ToolResult`.

    Parameters
    ----------
    tool_executor : callable
        ``(tool_name: str, params: dict) -> ToolResult``
    parse_fn : callable or None
        Custom parser to extract ``(tool_name, params)`` from the
        model output.  Defaults to JSON parsing.

    Example
    -------
    >>> def my_executor(name, params):
    ...     resp = requests.post(f"http://api/{name}", json=params)
    ...     return ToolResult(success=resp.ok, return_code=resp.status_code)
    >>> scorer = ToolSuccessScorer(tool_executor=my_executor)
    """

    def __init__(
        self,
        tool_executor: Callable[[str, dict[str, Any]], ToolResult],
        parse_fn: Optional[Callable[[str], tuple[str, dict[str, Any]]]] = None,
    ) -> None:
        self.tool_executor = tool_executor
        self.parse_fn = parse_fn or self._default_parse

    def score(
        self,
        prompt: str,
        test_input: str,
        output: str,
        client: LLMClient,
    ) -> float:
        try:
            tool_name, params = self.parse_fn(output)
        except Exception:
            return 0.0  # Couldn't parse a tool call

        if not tool_name:
            return 0.0

        try:
            result = self.tool_executor(tool_name, params)
        except Exception:
            return 0.0

        if result.success:
            return 1.0
        # Partial credit for parseable but failed calls
        if result.return_code in (400, 422):
            return 0.3  # Bad request — likely wrong params
        return 0.1  # Other failure

    @staticmethod
    def _default_parse(output: str) -> tuple[str, dict[str, Any]]:
        """Parse JSON tool call from model output."""
        # Try direct JSON
        cleaned = output.strip()
        # Strip markdown fences
        cleaned = re.sub(r"```(?:json)?\s*", "", cleaned).strip().rstrip("`")
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                name = data.get("tool", data.get("name", data.get("function", "")))
                params = data.get("parameters", data.get("params", data.get("arguments", {})))
                return str(name), dict(params) if isinstance(params, dict) else {}
        except json.JSONDecodeError:
            pass
        return "", {}


# ---------------------------------------------------------------------------
# Strategy 4: Self-Consistency
# ---------------------------------------------------------------------------


class SelfConsistencyScorer(Scorer):
    """Score by output agreement across multiple runs of the same prompt.

    Runs the prompt ``num_samples`` times on the same input and measures
    how often the outputs agree.  Higher consistency → higher fitness.

    Parameters
    ----------
    num_samples : int
        Number of times to run the prompt per test input (default 5).
    similarity_fn : callable or None
        Custom function ``(a: str, b: str) -> float`` in [0, 1].
        Defaults to normalized exact match.

    Example
    -------
    >>> scorer = SelfConsistencyScorer(num_samples=5)
    >>> score = scorer.score(prompt, "What is 2+2?", "4", client)
    """

    def __init__(
        self,
        num_samples: int = 5,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        self.num_samples = num_samples
        self.similarity_fn = similarity_fn or self._exact_match

    def score(
        self,
        prompt: str,
        test_input: str,
        output: str,
        client: LLMClient,
    ) -> float:
        # Collect additional outputs
        outputs = [output]
        for _ in range(self.num_samples - 1):
            response = client.complete(
                system_prompt=prompt,
                user_message=test_input,
                temperature=0.7,
                top_p=0.95,
            )
            if response is not None:
                outputs.append(response)

        if len(outputs) < 2:
            return 0.5  # Can't measure consistency with < 2 outputs

        # Pairwise similarity
        similarities = []
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                similarities.append(self.similarity_fn(outputs[i], outputs[j]))

        return statistics.mean(similarities) if similarities else 0.0

    @staticmethod
    def _exact_match(a: str, b: str) -> float:
        """Normalized comparison: strip whitespace, compare lowercased."""
        a_clean = a.strip().lower()
        b_clean = b.strip().lower()
        if a_clean == b_clean:
            return 1.0
        # Partial credit via token overlap (Jaccard similarity)
        tokens_a = set(a_clean.split())
        tokens_b = set(b_clean.split())
        if not tokens_a and not tokens_b:
            return 1.0
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Strategy 5: Proxy Metrics
# ---------------------------------------------------------------------------


@dataclass
class ProxyCheck:
    """A single structural check on model output.

    Parameters
    ----------
    name : str
        Human-readable name for this check.
    check_fn : callable
        ``(output: str) -> bool``
    weight : float
        Relative importance (default 1.0).

    Example
    -------
    >>> check = ProxyCheck("valid_json", lambda out: _is_valid_json(out))
    """

    name: str
    check_fn: Callable[[str], bool]
    weight: float = 1.0


class ProxyMetricsScorer(Scorer):
    """Score based on structural/format checks on the output.

    Each :class:`ProxyCheck` is a boolean test.  The final score is the
    weighted fraction of checks that pass.

    Parameters
    ----------
    checks : list[ProxyCheck]
        Ordered list of structural checks.

    Example
    -------
    >>> scorer = ProxyMetricsScorer(checks=[
    ...     ProxyCheck("valid_json", lambda o: is_json(o)),
    ...     ProxyCheck("has_tool_key", lambda o: '"tool"' in o),
    ...     ProxyCheck("under_500_chars", lambda o: len(o) < 500),
    ... ])
    """

    def __init__(self, checks: list[ProxyCheck]) -> None:
        self.checks = checks

    def score(
        self,
        prompt: str,
        test_input: str,
        output: str,
        client: LLMClient,
    ) -> float:
        s, _ = self.score_with_violations(output)
        return s

    def score_with_violations(self, output: str) -> tuple[float, int]:
        """Score *output* and return ``(score, penalty_violation_count)``.

        A penalty violation is a negative-weight check whose ``check_fn``
        returns ``True`` (i.e. the penalty condition fired).
        """
        if not self.checks:
            return 0.5, 0
        total_weight = sum(abs(c.weight) for c in self.checks)
        if total_weight == 0:
            return 0.5, 0
        earned = 0.0
        violations = 0
        for c in self.checks:
            passed = c.check_fn(output)
            if c.weight >= 0:
                # Positive check: earn weight when passing
                if passed:
                    earned += c.weight
            else:
                # Negative check: earn abs(weight) when NOT passing
                if not passed:
                    earned += abs(c.weight)
                else:
                    # check_fn returned True → penalty fired
                    violations += 1
        return max(0.0, min(1.0, earned / total_weight)), violations

    @staticmethod
    def common_checks() -> list[ProxyCheck]:
        """Handy pre-built checks for JSON tool-calling agents."""
        return [
            ProxyCheck("valid_json", lambda o: _is_valid_json(o)),
            ProxyCheck("has_tool_key", lambda o: '"tool"' in o.lower()),
            ProxyCheck("has_params_key", lambda o: '"parameters"' in o.lower() or '"params"' in o.lower()),
            ProxyCheck("no_markdown", lambda o: "```" not in o),
            ProxyCheck("under_500_chars", lambda o: len(o) < 500),
            ProxyCheck("not_empty", lambda o: len(o.strip()) > 0),
        ]


def _is_valid_json(text: str) -> bool:
    """Check if text is valid JSON (after stripping markdown fences)."""
    cleaned = re.sub(r"```(?:json)?\s*", "", text.strip()).rstrip("`").strip()
    try:
        json.loads(cleaned)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _feasibility_key(candidate: PromptCandidate) -> tuple[int, float]:
    """Sort key implementing feasibility-first ordering.

    Candidates with zero ``penalty_violations`` always rank above those
    with any violations.  Within the same feasibility tier, higher
    ``score`` wins.  Returns a tuple suitable for ``max(..., key=)``
    and ``sorted(..., reverse=True)``.
    """
    feasible = 1 if candidate.penalty_violations == 0 else 0
    return (feasible, candidate.score)


class PenaltyScaler:
    """Adaptive penalty weight scaling.

    Tracks how often each negative-weight :class:`ProxyCheck` fires
    across a generation.  After each generation call :meth:`end_generation`:
    any penalty whose trigger frequency exceeds *threshold* has its
    weight multiplied by *growth_factor*.

    Parameters
    ----------
    checks : list[ProxyCheck]
        The checks to monitor (only negative-weight checks are tracked).
    threshold : float
        Fraction of evaluations above which a penalty is considered
        too frequent (default ``0.5`` = fires on more than half of
        evaluations).
    growth_factor : float
        Multiplier applied to ``abs(weight)`` each generation the
        penalty exceeds the threshold (default ``1.5``).
    """

    def __init__(
        self,
        checks: list[ProxyCheck],
        *,
        threshold: float = 0.5,
        growth_factor: float = 1.5,
    ) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("threshold must be in (0, 1]")
        if growth_factor < 1.0:
            raise ValueError("growth_factor must be >= 1.0")
        self.checks = checks
        self.threshold = threshold
        self.growth_factor = growth_factor
        # Only track negative-weight checks
        self._penalty_names: list[str] = [
            c.name for c in checks if c.weight < 0
        ]
        self._fire_counts: dict[str, int] = {n: 0 for n in self._penalty_names}
        self._eval_count: int = 0

    def record(self, output: str) -> None:
        """Record one evaluation, tallying which penalties fire."""
        self._eval_count += 1
        for c in self.checks:
            if c.weight >= 0:
                continue
            if c.check_fn(output):
                self._fire_counts[c.name] = self._fire_counts.get(c.name, 0) + 1

    def end_generation(self) -> dict[str, float]:
        """Scale weights for high-frequency penalties and reset counters.

        Returns a dict mapping penalty name → new weight for every
        penalty that was scaled this generation.
        """
        scaled: dict[str, float] = {}
        if self._eval_count == 0:
            return scaled
        for c in self.checks:
            if c.weight >= 0:
                continue
            freq = self._fire_counts.get(c.name, 0) / self._eval_count
            if freq > self.threshold:
                c.weight = -(abs(c.weight) * self.growth_factor)
                scaled[c.name] = c.weight
                logger.info(
                    "  Penalty '%s' fired %.0f%% — weight scaled to %.2f",
                    c.name, freq * 100, c.weight,
                )
        # Reset for next generation
        self._fire_counts = {n: 0 for n in self._penalty_names}
        self._eval_count = 0
        return scaled


# ---------------------------------------------------------------------------
# Strategy 6: Preference / Contrastive Scoring
# ---------------------------------------------------------------------------


@dataclass
class PreferencePair:
    """A good/bad output pair for a given input.

    Parameters
    ----------
    input_text : str
        The user query.
    good_output : str
        A high-quality reference output.
    bad_output : str
        A low-quality output to contrast against.
    """

    input_text: str
    good_output: str
    bad_output: str


class PreferenceScorer(Scorer):
    """Score outputs by similarity to preferred (good) examples.

    Uses an LLM to compare the candidate output against both the good
    and bad reference outputs, returning a preference-weighted score.

    Parameters
    ----------
    pairs : list[PreferencePair]
        Contrastive examples for scoring.

    Example
    -------
    >>> pairs = [
    ...     PreferencePair(
    ...         input_text="What is 2+2?",
    ...         good_output='{"answer": 4}',
    ...         bad_output="Well, let me think... the answer might be 4 or 5.",
    ...     ),
    ... ]
    >>> scorer = PreferenceScorer(pairs)
    """

    _PREF_TEMPLATE = textwrap.dedent("""\
        Compare the candidate output against a good and bad reference.

        ## User Input
        {input_text}

        ## Good Reference (high quality)
        {good_output}

        ## Bad Reference (low quality)
        {bad_output}

        ## Candidate Output (to evaluate)
        {candidate_output}

        Is the candidate more similar to the Good or Bad reference?
        Rate from 0 to 10: 0 = identical to bad, 10 = identical to good.
        Respond with ONLY: {{"score": <number>}}
    """)

    def __init__(self, pairs: list[PreferencePair]) -> None:
        self.pairs = pairs
        self._pair_map = {p.input_text: p for p in pairs}

    def score(
        self,
        prompt: str,
        test_input: str,
        output: str,
        client: LLMClient,
    ) -> float:
        pair = self._pair_map.get(test_input)
        if pair is None:
            # No preference pair for this input — fall back to proxy check
            return 0.5

        response = client.complete(
            system_prompt="You compare outputs against references. Respond with valid JSON only.",
            user_message=self._PREF_TEMPLATE.format(
                input_text=pair.input_text,
                good_output=pair.good_output,
                bad_output=pair.bad_output,
                candidate_output=output,
            ),
            temperature=0.1,
            top_p=0.95,
        )
        if response is None:
            return 0.0
        try:
            data = json.loads(response)
            return min(float(data.get("score", 0)) / 10.0, 1.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            match = re.search(r"(\d+(?:\.\d+)?)", response or "")
            return min(float(match.group(1)) / 10.0, 1.0) if match else 0.0


# ---------------------------------------------------------------------------
# Strategy 7: Human-in-the-Loop Tournament
# ---------------------------------------------------------------------------


class HumanTournament(Scorer):
    """Present outputs to a human for A/B selection.

    During evolution, collects outputs and prompts a human to select
    the best one.  Works in batch mode: gathers all candidates for a
    generation, then asks for a ranking.

    Parameters
    ----------
    prompt_fn : callable or None
        Custom function to display candidates and get user choice.
        Signature: ``(test_input, outputs: list[str]) -> int``
        (returns index of the best output).
        Defaults to stdin/stdout interaction.

    Example
    -------
    >>> scorer = HumanTournament()
    >>> # During evolution, prints outputs and asks user to pick the best
    """

    def __init__(
        self,
        prompt_fn: Optional[Callable[[str, list[str]], int]] = None,
    ) -> None:
        self._prompt_fn = prompt_fn or self._default_prompt
        self._pending: dict[str, list[tuple[str, str]]] = {}  # input -> [(output, hash)]
        self._scores: dict[str, float] = {}  # hash -> score

    def score(
        self,
        prompt: str,
        test_input: str,
        output: str,
        client: LLMClient,
    ) -> float:
        output_hash = hashlib.md5(output.encode()).hexdigest()[:8]

        # Check if already scored
        if output_hash in self._scores:
            return self._scores[output_hash]

        # Buffer for batch scoring
        self._pending.setdefault(test_input, []).append((output, output_hash))

        # Score when we have enough candidates
        if len(self._pending[test_input]) >= 2:
            return self._score_batch(test_input)

        return 0.5  # Default until batch is scored

    def flush(self, test_input: str) -> None:
        """Force scoring of any pending outputs for a test input."""
        if test_input in self._pending and self._pending[test_input]:
            self._score_batch(test_input)

    def _score_batch(self, test_input: str) -> float:
        """Present pending outputs to human and score them."""
        entries = self._pending.pop(test_input, [])
        if not entries:
            return 0.5

        outputs = [e[0] for e in entries]
        hashes = [e[1] for e in entries]

        try:
            best_idx = self._prompt_fn(test_input, outputs)
            best_idx = max(0, min(best_idx, len(outputs) - 1))
        except (EOFError, KeyboardInterrupt):
            best_idx = 0

        # Assign scores: best gets 1.0, others get proportional scores
        n = len(outputs)
        for i, h in enumerate(hashes):
            if i == best_idx:
                self._scores[h] = 1.0
            else:
                self._scores[h] = max(0.0, 1.0 - (1.0 / n))

        return self._scores.get(hashes[-1], 0.5)

    @staticmethod
    def _default_prompt(test_input: str, outputs: list[str]) -> int:
        """Default stdin/stdout human interaction."""
        print("\n" + "=" * 60)
        print(f"  INPUT: {test_input}")
        print("=" * 60)
        for i, out in enumerate(outputs):
            print(f"\n  [{i + 1}] {textwrap.shorten(out, width=200)}")
        print()
        while True:
            try:
                choice = input(f"  Which is best? (1-{len(outputs)}): ").strip()
                idx = int(choice) - 1
                if 0 <= idx < len(outputs):
                    return idx
            except (ValueError, EOFError):
                pass
            print(f"  Please enter a number between 1 and {len(outputs)}")


# ---------------------------------------------------------------------------
# Composite scorer — combine multiple strategies
# ---------------------------------------------------------------------------


class CompositeScorer(Scorer):
    """Weighted combination of multiple scoring strategies.

    Parameters
    ----------
    scorers : list[tuple[Scorer, float]]
        List of ``(scorer, weight)`` pairs.

    Example
    -------
    >>> composite = CompositeScorer([
    ...     (LLMJudge(rubric="Rate helpfulness 0-10."), 0.5),
    ...     (ProxyMetricsScorer(ProxyMetricsScorer.common_checks()), 0.3),
    ...     (SelfConsistencyScorer(num_samples=3), 0.2),
    ... ])
    """

    def __init__(self, scorers: list[tuple[Scorer, float]]) -> None:
        self.scorers = scorers

    def score(
        self,
        prompt: str,
        test_input: str,
        output: str,
        client: LLMClient,
    ) -> float:
        total_weight = sum(w for _, w in self.scorers)
        if total_weight == 0:
            return 0.0
        total = sum(
            s.score(prompt, test_input, output, client) * w
            for s, w in self.scorers
        )
        return total / total_weight

    def name(self) -> str:
        names = [f"{s.name()}({w:.1f})" for s, w in self.scorers]
        return f"Composite[{', '.join(names)}]"


# ---------------------------------------------------------------------------
# No-Eval Prompt Evolver
# ---------------------------------------------------------------------------


class NoEvalPromptEvolver:
    """Evolutionary prompt optimiser that works without labelled data.

    Uses a pluggable :class:`Scorer` strategy as the fitness function
    instead of ground-truth labels.  Supports all seven no-eval strategies
    and combinations via :class:`CompositeScorer`.

    Parameters
    ----------
    task_description : str
        Describes what the agent should do.  Used as the base system
        prompt and to generate seed templates.
    test_inputs : list[str]
        User queries to test prompts against.  These need NO labels.
    scorer : Scorer
        Fitness scoring strategy.
    config : NoEvalConfig or None
        Evolution parameters.
    seed_templates : list[str] or None
        Custom seed prompts.  If None, generates from ``task_description``.
    seed : int
        Random seed.
    verbose : bool
        Print progress.

    Example
    -------
    >>> evolver = NoEvalPromptEvolver(
    ...     task_description="You are a helpful SQL assistant.",
    ...     test_inputs=["Show all users", "Count orders by month"],
    ...     scorer=LLMJudge(rubric="Rate SQL correctness 0-10."),
    ... )
    >>> result = evolver.run()
    >>> print(result.best_prompt)
    """

    def __init__(
        self,
        task_description: str,
        test_inputs: list[str],
        scorer: Scorer,
        config: Optional[NoEvalConfig] = None,
        seed_templates: Optional[list[str]] = None,
        seed: int = 42,
        verbose: bool = True,
        extract_category: Optional[Callable[[str, str], Optional[str]]] = None,
        custom_mutations: Optional[list[str]] = None,
    ) -> None:
        self.task_description = task_description
        self.test_inputs = test_inputs
        self.scorer = scorer
        self.config = config or NoEvalConfig()
        self.verbose = verbose
        self._rng = np.random.default_rng(seed)
        self._custom_mutations = custom_mutations

        llm_config = PromptEvolverConfig(
            backend=self.config.backend,
            max_tokens=self.config.max_tokens,
        )
        self._client = LLMClient(llm_config)

        self._seed_templates = seed_templates or self._build_seed_templates()

        self._all_candidates: list[PromptCandidate] = []
        self._history: list[tuple[int, float]] = []
        self._extract_category = extract_category
        self._error_profile = ErrorProfile()
        self._failure_examples: list[tuple[str, str, str]] = []
        self._adaptive_pool: list[str] = []

        # Adaptive penalty scaling: auto-create when scorer has negative
        # weight checks (i.e. penalty checks from seed template config).
        self._penalty_scaler: Optional[PenaltyScaler] = None
        if isinstance(self.scorer, ProxyMetricsScorer):
            neg = [c for c in self.scorer.checks if c.weight < 0]
            if neg:
                self._penalty_scaler = PenaltyScaler(self.scorer.checks)

    def run(self) -> PromptEvolverResult:
        """Run the evolutionary loop.  Returns :class:`PromptEvolverResult`."""
        t0 = time.perf_counter()

        logger.info(
            "Starting no-eval evolution: %d generations, population=%d, "
            "islands=%d, strategy=%s, backend=%s",
            self.config.iterations,
            self.config.population_size,
            self.config.num_islands,
            self.scorer.name(),
            self.config.backend.value,
        )

        if not self._client.is_available():
            logger.warning(
                "LLM backend not available — running in mock mode."
            )

        # Initialise islands with seed templates
        islands: list[list[PromptCandidate]] = [
            [] for _ in range(self.config.num_islands)
        ]

        logger.info("Seeding %d templates across %d islands", len(self._seed_templates), self.config.num_islands)
        if self.verbose:
            print(
                f"  Evaluating {len(self._seed_templates)} seed templates "
                f"({len(self.test_inputs)} test inputs each)...",
                flush=True,
            )
        for i, template in enumerate(self._seed_templates):
            isl_id = i % self.config.num_islands
            candidate = PromptCandidate(
                template=template,
                temperature=float(
                    self._rng.uniform(*self.config.temperature_range)
                ),
                top_p=float(self._rng.uniform(*self.config.top_p_range)),
                generation=0,
                island_id=isl_id,
                operation="seed",
            )
            if self.verbose:
                print(
                    f"  Seed {i + 1}/{len(self._seed_templates)}  "
                    f"island={isl_id}  evaluating...",
                    end="",
                    flush=True,
                )
            candidate.score = self._evaluate(candidate)
            islands[isl_id].append(candidate)
            self._all_candidates.append(candidate)
            if self.verbose:
                print(f"  score={candidate.score:.1f}%", flush=True)
            logger.info(
                "  Seed %d → island %d  score=%.1f%%  template=%.60s...",
                i, isl_id, candidate.score,
                template.replace('\n', ' '),
            )

        best_overall = max(self._all_candidates, key=lambda c: c.score)
        logger.info("Seed evaluation complete — best seed score=%.1f%%", best_overall.score)

        # Warmup: bootstrap error profile from seed evaluation so Gen 1
        # already has adaptive hints available.
        if (
            self.config.warmup_adaptive
            and self.config.adaptive_mutations
            and self._error_profile.worst_categories()
        ):
            self._adaptive_pool = generate_adaptive_mutations(
                self._error_profile,
                self.config.problem_type,
                self._client,
            )
            if self.verbose and self._adaptive_pool:
                print(
                    f"  Warmup: {len(self._adaptive_pool)} adaptive hints "
                    f"bootstrapped from seed evaluation"
                )

        for gen in range(1, self.config.iterations + 1):
            gen_t0 = time.perf_counter()
            logger.info("── Generation %d/%d ──", gen, self.config.iterations)

            # Decay or reset error tracking per generation
            if self.config.adaptive_mutations or self.config.llm_mutation_rate > 0:
                if self.config.error_decay > 0:
                    self._error_profile.decay(self.config.error_decay)
                    # Keep recent failure examples only (last 20)
                    self._failure_examples = self._failure_examples[-20:]
                else:
                    self._error_profile = ErrorProfile()
                    self._failure_examples = []

            for island_id in range(self.config.num_islands):
                island = islands[island_id]
                if not island:
                    continue

                new_candidates: list[PromptCandidate] = []
                for _ in range(self.config.population_size):
                    child = self._breed(island, gen)
                    child.island_id = island_id
                    child.score = self._evaluate(child)
                    new_candidates.append(child)
                    self._all_candidates.append(child)
                    logger.debug(
                        "  Island %d: child op=%-20s score=%.1f%%",
                        island_id, child.operation, child.score,
                    )

                island_best = max(new_candidates, key=lambda c: c.score)
                logger.info(
                    "  Island %d: bred %d candidates, best=%.1f%% (op=%s)",
                    island_id, len(new_candidates),
                    island_best.score, island_best.operation,
                )

                combined = island + new_candidates
                combined.sort(key=_feasibility_key, reverse=True)
                islands[island_id] = combined[: self.config.elite_size]

            # Migration
            if gen % self.config.migration_interval == 0:
                self._migrate(islands)

            # Adaptive penalty scaling: grow weights for frequent penalties
            if self._penalty_scaler is not None:
                self._penalty_scaler.end_generation()

            # Generate adaptive mutations for the next generation
            if self.config.adaptive_mutations and self._error_profile.worst_categories():
                self._adaptive_pool = generate_adaptive_mutations(
                    self._error_profile,
                    self.config.problem_type,
                    self._client,
                )
                logger.info(
                    "  Adaptive mutations: %d hints from %d error categories",
                    len(self._adaptive_pool),
                    len(self._error_profile.worst_categories()),
                )
            else:
                self._adaptive_pool = []

            gen_best = max(
                (c for isl in islands for c in isl),
                key=_feasibility_key,
            )
            improved = (
                _feasibility_key(gen_best) > _feasibility_key(best_overall)
            )
            if improved:
                best_overall = gen_best

            self._history.append((gen, best_overall.score))

            gen_elapsed = time.perf_counter() - gen_t0
            logger.info(
                "  Gen %d summary: best=%.1f%% %stemp=%.3f  "
                "candidates=%d  elapsed=%.1fs  "
                "LLM calls=%d  tokens_in=%d  tokens_out=%d",
                gen, best_overall.score,
                "▲ NEW BEST  " if improved else "",
                best_overall.temperature,
                len(self._all_candidates),
                gen_elapsed,
                self._client.call_count,
                self._client.total_input_tokens,
                self._client.total_output_tokens,
            )

            if self.verbose:
                print(
                    f"  Gen {gen:3d}/{self.config.iterations}  "
                    f"best={best_overall.score:5.1f}%  "
                    f"strategy={self.scorer.name()}  "
                    f"temp={best_overall.temperature:.3f}",
                    flush=True,
                )

        wall_time = time.perf_counter() - t0

        logger.info(
            "Evolution complete: best=%.1f%%  wall_time=%.1fs  "
            "total_candidates=%d  LLM calls=%d  "
            "tokens_in=%d  tokens_out=%d",
            best_overall.score, wall_time,
            len(self._all_candidates),
            self._client.call_count,
            self._client.total_input_tokens,
            self._client.total_output_tokens,
        )

        return PromptEvolverResult(
            best_prompt=best_overall.template,
            best_temperature=best_overall.temperature,
            best_top_p=best_overall.top_p,
            best_accuracy=best_overall.score / 100.0,
            best_score=best_overall.score,
            history=self._history,
            all_candidates=sorted(
                self._all_candidates, key=lambda c: c.score, reverse=True
            ),
            wall_time=wall_time,
            iterations_run=self.config.iterations,
            llm_backend=self.config.backend.value,
        )

    # -- internals -----------------------------------------------------------

    def _build_seed_templates(self) -> list[str]:
        """Build 4 seed templates from the task description."""
        return [
            f"{self.task_description}\n\nRespond accurately and concisely.",
            f"# Role\n{self.task_description}\n\n# Instructions\nBe precise and helpful.",
            (
                f"You are an AI assistant.\n\n## Task\n{self.task_description}\n\n"
                f"## Output\nProvide a clear, direct response."
            ),
            (
                f"System: {self.task_description}\n\n"
                f"Think step by step before responding. Be accurate."
            ),
        ]

    def _evaluate(self, candidate: PromptCandidate) -> float:
        """Evaluate a candidate using the scorer strategy."""
        scores = []
        total_violations = 0
        for test_input in self.test_inputs:
            output = self._client.complete(
                system_prompt=candidate.template,
                user_message=test_input,
                temperature=candidate.temperature,
                top_p=candidate.top_p,
            )

            if output is None:
                scores.append(float(self._rng.uniform(0, 0.5)))
                continue

            # Use score_with_violations when scorer supports it
            if isinstance(self.scorer, ProxyMetricsScorer):
                s, violations = self.scorer.score_with_violations(output)
                total_violations += violations
                if self._penalty_scaler is not None:
                    self._penalty_scaler.record(output)
            else:
                s = self.scorer.score(
                    candidate.template, test_input, output, self._client
                )
            scores.append(s)

            # Track error profile for adaptive mutations / LLM mutation
            if self._extract_category is not None:
                expected_cat = self._extract_category(test_input, "expected")
                predicted_cat = self._extract_category(output, "predicted")
                if expected_cat:
                    correct = s >= 0.5 if predicted_cat is None else (
                        predicted_cat == expected_cat
                    )
                    self._error_profile.record(expected_cat, correct)
                    if not correct:
                        self._failure_examples.append(
                            (test_input, predicted_cat or output[:80], expected_cat)
                        )

        candidate.penalty_violations = total_violations
        return (statistics.mean(scores) * 100.0) if scores else 0.0

    def _breed(
        self, island: list[PromptCandidate], generation: int
    ) -> PromptCandidate:
        parent_a = self._tournament_select(island)
        parents = [parent_a.hash]
        op_parts: list[str] = []
        if self._custom_mutations is not None:
            mutations = list(self._custom_mutations)
        else:
            mutations = list(get_mutations_for_problem_type(self.config.problem_type))

        # Merge adaptive mutations into the pool when enabled
        if self._adaptive_pool:
            mutations = mutations + self._adaptive_pool

        if self._rng.random() < self.config.crossover_rate and len(island) > 1:
            parent_b = self._tournament_select(island)
            child_template = _crossover_templates(
                parent_a.template,
                parent_b.template,
                self._rng,
                require_tool_schemas=False,
            )
            parents.append(parent_b.hash)
            op_parts.append("crossover")
        else:
            child_template = parent_a.template

        # LLM-assisted mutation path: rewrite the prompt using failure cases
        if (
            self.config.llm_mutation_rate > 0
            and self._rng.random() < self.config.llm_mutation_rate
            and self._failure_examples
        ):
            # Sample up to 6 failure examples
            n_fail = min(6, len(self._failure_examples))
            fail_idx = self._rng.choice(
                len(self._failure_examples), size=n_fail, replace=False
            )
            sampled_failures = [self._failure_examples[int(i)] for i in fail_idx]
            child_template = _llm_mutate_template(
                child_template,
                sampled_failures,
                self._client,
                self.config.problem_type,
                require_tool_schemas=False,
            )
            op_parts.append("llm_mutate")
        elif self._rng.random() < self.config.mutation_rate:
            child_template = _mutate_template(
                child_template,
                self._rng,
                self.config.mutation_rate,
                require_tool_schemas=False,
                mutations=mutations,
            )
            op_parts.append("mutate")

        # Describe-entities mutation: randomly expand bare entity names
        if (
            self.config.describe_entities
            and self._client.is_available()
            and self._rng.random() < self.config.mutation_rate * 0.15
            and not _has_entity_descriptions(child_template)
        ):
            child_template = _llm_describe_entities(
                child_template,
                self._client,
                self.config.problem_type,
                require_tool_schemas=False,
            )
            op_parts.append("describe_entities")

        # Optional LLM-based clarity refinement
        if self.config.refine_after_splice and op_parts:
            child_template = _refine_template(
                child_template, self._client, require_tool_schemas=False
            )
            op_parts.append("refine")

        operation = "+".join(op_parts) if op_parts else "clone"

        temp = parent_a.temperature + float(self._rng.normal(0, 0.1))
        temp = float(np.clip(temp, *self.config.temperature_range))
        top_p = parent_a.top_p + float(self._rng.normal(0, 0.05))
        top_p = float(np.clip(top_p, *self.config.top_p_range))

        return PromptCandidate(
            template=child_template,
            temperature=temp,
            top_p=top_p,
            generation=generation,
            operation=operation,
            parent_hashes=parents,
        )

    def _tournament_select(
        self, island: list[PromptCandidate], k: int = 3
    ) -> PromptCandidate:
        k = min(k, len(island))
        indices = self._rng.choice(len(island), size=k, replace=False)
        contestants = [island[int(i)] for i in indices]
        return max(contestants, key=_feasibility_key)

    def _migrate(self, islands: list[list[PromptCandidate]]) -> None:
        n = len(islands)
        if n < 2:
            return
        logger.info("  Migration: sharing elite candidates across %d islands", n)
        for src in range(n):
            if not islands[src]:
                continue
            best = max(islands[src], key=lambda c: c.score)
            dest = (src + 1) % n
            migrant = PromptCandidate(
                template=best.template,
                temperature=best.temperature,
                top_p=best.top_p,
                generation=best.generation,
                score=best.score,
                island_id=dest,
                operation="migrate",
                parent_hashes=[best.hash],
            )
            islands[dest].append(migrant)
