"""Tests for MutaGenAI.strategies — no-eval prompt evolution strategies."""
from __future__ import annotations


import pytest

from MutaGenAI.strategies import (
    CompositeScorer,
    HumanTournament,
    LLMJudge,
    NoEvalConfig,
    NoEvalPromptEvolver,
    PreferencePair,
    PreferenceScorer,
    ProxyCheck,
    ProxyMetricsScorer,
    Scorer,
    SelfConsistencyScorer,
    SyntheticEvalGenerator,
    SyntheticEvalScorer,
    ToolResult,
    ToolSuccessScorer,
    _is_valid_json,
    _feasibility_key,
    PenaltyScaler,
)
from MutaGenAI.prompt_evolver import PromptCandidate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _MockClient:
    """Deterministic mock LLM client for testing."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or []
        self._call_count = 0

    def is_available(self) -> bool:
        return bool(self._responses)

    def complete(self, system_prompt: str, user_message: str,
                 temperature: float = 0.7, top_p: float = 0.95) -> str | None:
        if not self._responses:
            return None
        resp = self._responses[self._call_count % len(self._responses)]
        self._call_count += 1
        return resp


@pytest.fixture
def mock_client():
    return _MockClient


# ---------------------------------------------------------------------------
# Strategy 1: LLM-as-Judge
# ---------------------------------------------------------------------------


class TestLLMJudge:
    def test_parse_json_score(self):
        judge = LLMJudge(rubric="Test rubric", max_score=10.0)
        assert judge._parse_score('{"score": 8, "reason": "good"}') == 0.8

    def test_parse_numeric_fallback(self):
        judge = LLMJudge(rubric="Test rubric", max_score=10.0)
        assert judge._parse_score("I give this a 7 out of 10") == 0.7

    def test_parse_no_score(self):
        judge = LLMJudge(rubric="Test rubric", max_score=10.0)
        assert judge._parse_score("No number here") == 0.0

    def test_score_with_client(self, mock_client):
        judge = LLMJudge(rubric="Rate 0-10.", max_score=10.0)
        client = mock_client(['{"score": 9, "reason": "excellent"}'])
        result = judge.score("You are helpful.", "Hello", "Hi there!", client)
        assert result == pytest.approx(0.9)

    def test_score_none_response(self, mock_client):
        judge = LLMJudge(rubric="Rate 0-10.")
        client = mock_client([])
        result = judge.score("prompt", "input", "output", client)
        assert result == 0.0

    def test_score_clamped_to_one(self):
        judge = LLMJudge(rubric="Test", max_score=10.0)
        assert judge._parse_score('{"score": 15}') == 1.0

    def test_name(self):
        assert LLMJudge(rubric="test").name() == "LLMJudge"


# ---------------------------------------------------------------------------
# Strategy 2: Synthetic Eval
# ---------------------------------------------------------------------------


class TestSyntheticEvalGenerator:
    def test_parse_valid_json(self):
        gen = SyntheticEvalGenerator("Task", num_cases=2)
        raw = '[{"input": "q1", "expected_output": "a1"}, {"input": "q2", "expected_output": "a2"}]'
        cases = gen._parse_cases(raw)
        assert len(cases) == 2
        assert cases[0]["input"] == "q1"

    def test_parse_markdown_fenced(self):
        gen = SyntheticEvalGenerator("Task", num_cases=1)
        raw = '```json\n[{"input": "q", "expected_output": "a"}]\n```'
        cases = gen._parse_cases(raw)
        assert len(cases) == 1

    def test_parse_invalid(self):
        gen = SyntheticEvalGenerator("Task")
        assert gen._parse_cases("not json") == []

    def test_parse_filters_bad_entries(self):
        gen = SyntheticEvalGenerator("Task")
        raw = '[{"input": "q", "expected_output": "a"}, {"bad": "entry"}]'
        cases = gen._parse_cases(raw)
        assert len(cases) == 1

    def test_generate_with_client(self, mock_client):
        gen = SyntheticEvalGenerator("You are an assistant.", num_cases=2)
        client = mock_client(['[{"input": "hi", "expected_output": "hello"}]'])
        cases = gen.generate(client)
        assert len(cases) == 1

    def test_generate_no_client(self, mock_client):
        gen = SyntheticEvalGenerator("Task")
        client = mock_client([])
        assert gen.generate(client) == []


class TestSyntheticEvalScorer:
    def test_score_with_match(self, mock_client):
        cases = [{"input": "hello", "expected_output": "world"}]
        scorer = SyntheticEvalScorer(cases)
        client = mock_client(['{"score": 8}'])
        result = scorer.score("prompt", "hello", "world", client)
        assert result == pytest.approx(0.8)

    def test_score_no_match(self, mock_client):
        cases = [{"input": "hello", "expected_output": "world"}]
        scorer = SyntheticEvalScorer(cases)
        client = mock_client([])
        # Input not in cases
        result = scorer.score("prompt", "unknown", "output", client)
        assert result == 0.5  # Neutral


# ---------------------------------------------------------------------------
# Strategy 3: Tool-Use Success
# ---------------------------------------------------------------------------


class TestToolResult:
    def test_success(self):
        r = ToolResult(success=True, return_code=200, output="OK")
        assert r.success is True

    def test_failure(self):
        r = ToolResult(success=False, return_code=500)
        assert r.success is False


class TestToolSuccessScorer:
    def test_successful_tool_call(self, mock_client):
        def executor(name, params):
            return ToolResult(success=True, return_code=200)

        scorer = ToolSuccessScorer(tool_executor=executor)
        output = '{"tool": "lookup_order", "parameters": {"id": "123"}}'
        result = scorer.score("prompt", "input", output, mock_client([]))
        assert result == 1.0

    def test_failed_tool_call_400(self, mock_client):
        def executor(name, params):
            return ToolResult(success=False, return_code=400)

        scorer = ToolSuccessScorer(tool_executor=executor)
        output = '{"tool": "lookup", "parameters": {"id": "123"}}'
        result = scorer.score("prompt", "input", output, mock_client([]))
        assert result == pytest.approx(0.3)

    def test_unparseable_output(self, mock_client):
        def executor(name, params):
            return ToolResult(success=True)

        scorer = ToolSuccessScorer(tool_executor=executor)
        result = scorer.score("prompt", "input", "not json at all", mock_client([]))
        assert result == 0.0

    def test_default_parse_json(self):
        tool, params = ToolSuccessScorer._default_parse(
            '{"tool": "search", "parameters": {"q": "test"}}'
        )
        assert tool == "search"
        assert params == {"q": "test"}

    def test_default_parse_markdown_fenced(self):
        tool, params = ToolSuccessScorer._default_parse(
            '```json\n{"tool": "search", "params": {"q": "x"}}\n```'
        )
        assert tool == "search"

    def test_default_parse_empty(self):
        tool, params = ToolSuccessScorer._default_parse("")
        assert tool == ""

    def test_custom_parse(self, mock_client):
        def my_parse(output):
            return "custom_tool", {"key": "val"}

        def executor(name, params):
            return ToolResult(success=True)

        scorer = ToolSuccessScorer(tool_executor=executor, parse_fn=my_parse)
        result = scorer.score("prompt", "input", "anything", mock_client([]))
        assert result == 1.0


# ---------------------------------------------------------------------------
# Strategy 4: Self-Consistency
# ---------------------------------------------------------------------------


class TestSelfConsistencyScorer:
    def test_exact_match_similarity(self):
        assert SelfConsistencyScorer._exact_match("hello", "hello") == 1.0

    def test_case_insensitive(self):
        assert SelfConsistencyScorer._exact_match("Hello World", "hello world") == 1.0

    def test_partial_overlap(self):
        score = SelfConsistencyScorer._exact_match("the cat sat", "the dog sat")
        assert 0 < score < 1  # Jaccard: {"the","sat"} / {"the","cat","sat","dog"}

    def test_no_overlap(self):
        assert SelfConsistencyScorer._exact_match("apple", "banana") == 0.0

    def test_both_empty(self):
        assert SelfConsistencyScorer._exact_match("", "") == 1.0

    def test_one_empty(self):
        assert SelfConsistencyScorer._exact_match("hello", "") == 0.0

    def test_score_consistent_outputs(self, mock_client):
        scorer = SelfConsistencyScorer(num_samples=3)
        client = mock_client(["same answer"])
        result = scorer.score("prompt", "input", "same answer", client)
        assert result == 1.0

    def test_score_with_unavailable_client(self, mock_client):
        scorer = SelfConsistencyScorer(num_samples=3)
        client = mock_client([])
        result = scorer.score("prompt", "input", "only one", client)
        assert result == 0.5  # Can't measure with only 1 output

    def test_custom_similarity(self, mock_client):
        def always_half(a, b):
            return 0.5

        scorer = SelfConsistencyScorer(num_samples=3, similarity_fn=always_half)
        client = mock_client(["a", "b"])
        result = scorer.score("prompt", "input", "c", client)
        assert result == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Strategy 5: Proxy Metrics
# ---------------------------------------------------------------------------


class TestProxyCheck:
    def test_basic_check(self):
        check = ProxyCheck("test", lambda o: len(o) > 0)
        assert check.check_fn("hello") is True
        assert check.check_fn("") is False


class TestProxyMetricsScorer:
    def test_all_pass(self, mock_client):
        checks = [
            ProxyCheck("non_empty", lambda o: len(o) > 0),
            ProxyCheck("short", lambda o: len(o) < 100),
        ]
        scorer = ProxyMetricsScorer(checks=checks)
        result = scorer.score("prompt", "input", "hello", mock_client([]))
        assert result == 1.0

    def test_some_pass(self, mock_client):
        checks = [
            ProxyCheck("non_empty", lambda o: len(o) > 0, weight=1.0),
            ProxyCheck("is_json", lambda o: o.startswith("{"), weight=1.0),
        ]
        scorer = ProxyMetricsScorer(checks=checks)
        result = scorer.score("prompt", "input", "plain text", mock_client([]))
        assert result == pytest.approx(0.5)

    def test_none_pass(self, mock_client):
        checks = [ProxyCheck("impossible", lambda o: False)]
        scorer = ProxyMetricsScorer(checks=checks)
        result = scorer.score("prompt", "input", "anything", mock_client([]))
        assert result == 0.0

    def test_weighted_checks(self, mock_client):
        checks = [
            ProxyCheck("always", lambda o: True, weight=3.0),
            ProxyCheck("never", lambda o: False, weight=1.0),
        ]
        scorer = ProxyMetricsScorer(checks=checks)
        result = scorer.score("prompt", "input", "x", mock_client([]))
        assert result == pytest.approx(0.75)

    def test_empty_checks(self, mock_client):
        scorer = ProxyMetricsScorer(checks=[])
        assert scorer.score("p", "i", "o", mock_client([])) == 0.5

    def test_common_checks(self):
        checks = ProxyMetricsScorer.common_checks()
        assert len(checks) >= 6
        # Valid JSON should pass the valid_json check
        valid = '{"tool": "test", "parameters": {}}'
        json_check = next(c for c in checks if c.name == "valid_json")
        assert json_check.check_fn(valid) is True

    def test_negative_weight_bad_output(self, mock_client):
        """Negative-weight check: score drops when check passes (bad output)."""
        checks = [
            ProxyCheck("good", lambda o: True, weight=2.0),
            ProxyCheck("penalty", lambda o: True, weight=-1.0),  # fires
        ]
        scorer = ProxyMetricsScorer(checks=checks)
        result = scorer.score("p", "i", "x", mock_client([]))
        # total_weight = 3.0, earned = 2.0 (good passes) + 0 (penalty fires)
        assert result == pytest.approx(2.0 / 3.0)

    def test_negative_weight_good_output(self, mock_client):
        """Negative-weight check: no deduction when check fails (good output)."""
        checks = [
            ProxyCheck("good", lambda o: True, weight=2.0),
            ProxyCheck("penalty", lambda o: False, weight=-1.0),  # does not fire
        ]
        scorer = ProxyMetricsScorer(checks=checks)
        result = scorer.score("p", "i", "x", mock_client([]))
        # total_weight = 3.0, earned = 2.0 + 1.0 = 3.0
        assert result == pytest.approx(1.0)

    def test_score_clamped_to_zero_one(self, mock_client):
        """Score is always in [0, 1] regardless of weight configuration."""
        checks = [
            ProxyCheck("always", lambda o: True, weight=1.0),
        ]
        scorer = ProxyMetricsScorer(checks=checks)
        result = scorer.score("p", "i", "x", mock_client([]))
        assert 0.0 <= result <= 1.0

    def test_all_negative_all_fire(self, mock_client):
        """All negative-weight checks fire → score 0."""
        checks = [
            ProxyCheck("pen1", lambda o: True, weight=-1.0),
            ProxyCheck("pen2", lambda o: True, weight=-2.0),
        ]
        scorer = ProxyMetricsScorer(checks=checks)
        result = scorer.score("p", "i", "x", mock_client([]))
        assert result == pytest.approx(0.0)

    def test_all_negative_none_fire(self, mock_client):
        """All negative-weight checks don't fire → score 1."""
        checks = [
            ProxyCheck("pen1", lambda o: False, weight=-1.0),
            ProxyCheck("pen2", lambda o: False, weight=-2.0),
        ]
        scorer = ProxyMetricsScorer(checks=checks)
        result = scorer.score("p", "i", "x", mock_client([]))
        assert result == pytest.approx(1.0)


class TestIsValidJson:
    def test_valid(self):
        assert _is_valid_json('{"key": "value"}') is True

    def test_invalid(self):
        assert _is_valid_json("not json") is False

    def test_markdown_fenced(self):
        assert _is_valid_json('```json\n{"a": 1}\n```') is True

    def test_empty(self):
        assert _is_valid_json("") is False

    def test_array(self):
        assert _is_valid_json("[1, 2, 3]") is True


# ---------------------------------------------------------------------------
# Strategy 6: Preference Scoring
# ---------------------------------------------------------------------------


class TestPreferencePair:
    def test_fields(self):
        p = PreferencePair("input", "good", "bad")
        assert p.input_text == "input"
        assert p.good_output == "good"
        assert p.bad_output == "bad"


class TestPreferenceScorer:
    def test_score_with_match(self, mock_client):
        pairs = [PreferencePair("hi", "good", "bad")]
        scorer = PreferenceScorer(pairs)
        client = mock_client(['{"score": 9}'])
        result = scorer.score("prompt", "hi", "good output", client)
        assert result == pytest.approx(0.9)

    def test_score_no_pair(self, mock_client):
        scorer = PreferenceScorer([])
        result = scorer.score("prompt", "unknown", "output", mock_client([]))
        assert result == 0.5

    def test_score_none_response(self, mock_client):
        pairs = [PreferencePair("hi", "good", "bad")]
        scorer = PreferenceScorer(pairs)
        client = mock_client([])
        result = scorer.score("prompt", "hi", "output", client)
        assert result == 0.0


# ---------------------------------------------------------------------------
# Strategy 7: Human Tournament
# ---------------------------------------------------------------------------


class TestHumanTournament:
    def test_custom_prompt_fn(self, mock_client):
        # Always pick the first
        scorer = HumanTournament(prompt_fn=lambda inp, outs: 0)
        client = mock_client([])

        # First call — buffered
        s1 = scorer.score("p", "input1", "output_a", client)
        assert s1 == 0.5  # Only 1 candidate, buffered

        # Second call — triggers batch scoring
        s2 = scorer.score("p", "input1", "output_b", client)
        # The first output (index 0) should be selected as best
        assert s2 >= 0.0

    def test_flush(self, mock_client):
        scorer = HumanTournament(prompt_fn=lambda inp, outs: 0)
        client = mock_client([])
        scorer.score("p", "input1", "only_one", client)
        scorer.flush("input1")  # Should not raise

    def test_already_scored(self, mock_client):
        scorer = HumanTournament(prompt_fn=lambda inp, outs: 0)
        client = mock_client([])
        # Score twice to trigger batch
        scorer.score("p", "q", "output_a", client)
        scorer.score("p", "q", "output_b", client)
        # Score a cached output
        s = scorer.score("p", "q", "output_a", client)
        assert s >= 0.0


# ---------------------------------------------------------------------------
# Composite Scorer
# ---------------------------------------------------------------------------


class TestCompositeScorer:
    def test_weighted_average(self, mock_client):
        class FixedScorer(Scorer):
            def __init__(self, val):
                self._val = val

            def score(self, prompt, test_input, output, client):
                return self._val

        composite = CompositeScorer([
            (FixedScorer(1.0), 0.6),
            (FixedScorer(0.0), 0.4),
        ])
        result = composite.score("p", "i", "o", mock_client([]))
        assert result == pytest.approx(0.6)

    def test_empty_scorers(self, mock_client):
        composite = CompositeScorer([])
        assert composite.score("p", "i", "o", mock_client([])) == 0.0

    def test_name(self):
        class Dummy(Scorer):
            def score(self, *a):
                return 0.0

        composite = CompositeScorer([(Dummy(), 1.0)])
        assert "Composite" in composite.name()


# ---------------------------------------------------------------------------
# NoEvalConfig
# ---------------------------------------------------------------------------


class TestNoEvalConfig:
    def test_defaults(self):
        config = NoEvalConfig()
        assert config.iterations == 5
        assert config.population_size == 4
        assert config.num_islands == 2
        assert config.elite_size == 3
        assert config.migration_interval == 3

    def test_custom_values(self):
        config = NoEvalConfig(iterations=10, population_size=8)
        assert config.iterations == 10
        assert config.population_size == 8


# ---------------------------------------------------------------------------
# NoEvalPromptEvolver
# ---------------------------------------------------------------------------


class TestNoEvalPromptEvolver:
    def test_run_mock_mode(self):
        """Evolver runs in mock mode when LLM is unavailable."""

        class AlwaysHalfScorer(Scorer):
            def score(self, prompt, test_input, output, client):
                return 0.5

        config = NoEvalConfig(iterations=2, population_size=2, num_islands=1)
        evolver = NoEvalPromptEvolver(
            task_description="You are a test assistant.",
            test_inputs=["hello", "world"],
            scorer=AlwaysHalfScorer(),
            config=config,
            seed=42,
            verbose=False,
        )
        result = evolver.run()
        assert result.iterations_run == 2
        assert len(result.all_candidates) > 0
        assert result.best_score >= 0.0
        assert result.best_prompt  # Non-empty

    def test_seed_templates_generation(self):
        config = NoEvalConfig(iterations=1, population_size=1)
        evolver = NoEvalPromptEvolver(
            task_description="You help with math.",
            test_inputs=["2+2"],
            scorer=ProxyMetricsScorer(checks=[]),
            config=config,
            verbose=False,
        )
        assert len(evolver._seed_templates) == 4
        for t in evolver._seed_templates:
            assert "math" in t.lower()

    def test_custom_seed_templates(self):
        config = NoEvalConfig(iterations=1, population_size=1)
        custom = ["Template 1", "Template 2"]
        evolver = NoEvalPromptEvolver(
            task_description="Test",
            test_inputs=["x"],
            scorer=ProxyMetricsScorer(checks=[]),
            config=config,
            seed_templates=custom,
            verbose=False,
        )
        assert evolver._seed_templates == custom

    def test_history_tracking(self):
        class FixedScorer(Scorer):
            def score(self, prompt, test_input, output, client):
                return 0.7

        config = NoEvalConfig(iterations=3, population_size=2, num_islands=1)
        evolver = NoEvalPromptEvolver(
            task_description="Test",
            test_inputs=["a"],
            scorer=FixedScorer(),
            config=config,
            verbose=False,
        )
        result = evolver.run()
        assert len(result.history) == 3
        for gen, score in result.history:
            assert gen >= 1

    def test_candidates_sorted(self):
        class FixedScorer(Scorer):
            def score(self, prompt, test_input, output, client):
                return 0.5

        config = NoEvalConfig(iterations=2, population_size=3, num_islands=2)
        evolver = NoEvalPromptEvolver(
            task_description="Test",
            test_inputs=["a", "b"],
            scorer=FixedScorer(),
            config=config,
            verbose=False,
        )
        result = evolver.run()
        scores = [c.score for c in result.all_candidates]
        assert scores == sorted(scores, reverse=True)

    def test_migration_occurs(self):
        """Verify migration doesn't crash with migration_interval=1."""
        class FixedScorer(Scorer):
            def score(self, *a):
                return 0.5

        config = NoEvalConfig(
            iterations=3, population_size=2,
            num_islands=2, migration_interval=1,
        )
        evolver = NoEvalPromptEvolver(
            task_description="Test",
            test_inputs=["a"],
            scorer=FixedScorer(),
            config=config,
            verbose=False,
        )
        result = evolver.run()
        assert result.iterations_run == 3


# ---------------------------------------------------------------------------
# _feasibility_key tests
# ---------------------------------------------------------------------------


class TestFeasibilityKey:
    """Tests for _feasibility_key in strategies module."""

    def test_zero_violations_beats_any_violations(self):
        feasible = PromptCandidate(template="a", score=50.0, penalty_violations=0)
        infeasible = PromptCandidate(template="b", score=90.0, penalty_violations=3)
        assert _feasibility_key(feasible) > _feasibility_key(infeasible)

    def test_same_feasibility_uses_score(self):
        a = PromptCandidate(template="a", score=80.0, penalty_violations=0)
        b = PromptCandidate(template="b", score=60.0, penalty_violations=0)
        assert _feasibility_key(a) > _feasibility_key(b)

    def test_both_infeasible_uses_score(self):
        a = PromptCandidate(template="a", score=70.0, penalty_violations=1)
        b = PromptCandidate(template="b", score=40.0, penalty_violations=5)
        assert _feasibility_key(a) > _feasibility_key(b)

    def test_max_selects_feasible(self):
        candidates = [
            PromptCandidate(template="a", score=90.0, penalty_violations=2),
            PromptCandidate(template="b", score=30.0, penalty_violations=0),
        ]
        winner = max(candidates, key=_feasibility_key)
        assert winner.template == "b"


# ---------------------------------------------------------------------------
# ProxyMetricsScorer.score_with_violations tests
# ---------------------------------------------------------------------------


class TestScoreWithViolations:
    """Tests for score_with_violations method."""

    def test_no_penalties_zero_violations(self):
        checks = [ProxyCheck("positive", lambda o: True, weight=1.0)]
        scorer = ProxyMetricsScorer(checks)
        score, violations = scorer.score_with_violations("test")
        assert violations == 0
        assert score == 1.0

    def test_penalty_fires_counted(self):
        checks = [
            ProxyCheck("ok", lambda o: True, weight=1.0),
            ProxyCheck("bad", lambda o: True, weight=-0.5),
        ]
        scorer = ProxyMetricsScorer(checks)
        score, violations = scorer.score_with_violations("test")
        assert violations == 1

    def test_penalty_not_fired_zero_violations(self):
        checks = [
            ProxyCheck("ok", lambda o: True, weight=1.0),
            ProxyCheck("bad", lambda o: False, weight=-0.5),
        ]
        scorer = ProxyMetricsScorer(checks)
        score, violations = scorer.score_with_violations("test")
        assert violations == 0

    def test_score_delegates_to_score_with_violations(self):
        checks = [ProxyCheck("c", lambda o: True, weight=2.0)]
        scorer = ProxyMetricsScorer(checks)
        assert scorer.score("p", "i", "x", None) == scorer.score_with_violations("x")[0]


# ---------------------------------------------------------------------------
# PenaltyScaler tests
# ---------------------------------------------------------------------------


class TestPenaltyScaler:
    """Tests for PenaltyScaler adaptive weight scaling."""

    def test_init_validates_threshold(self):
        checks = [ProxyCheck("p", lambda o: True, weight=-1.0)]
        with pytest.raises(ValueError):
            PenaltyScaler(checks, threshold=0.0)
        with pytest.raises(ValueError):
            PenaltyScaler(checks, threshold=1.5)

    def test_init_validates_growth_factor(self):
        checks = [ProxyCheck("p", lambda o: True, weight=-1.0)]
        with pytest.raises(ValueError):
            PenaltyScaler(checks, growth_factor=0.5)

    def test_no_negative_checks_noop(self):
        checks = [ProxyCheck("ok", lambda o: True, weight=1.0)]
        scaler = PenaltyScaler(checks)
        scaler.record("test")
        result = scaler.end_generation()
        assert result == {}

    def test_scales_when_above_threshold(self):
        check = ProxyCheck("bad", lambda o: True, weight=-1.0)
        scaler = PenaltyScaler([check], threshold=0.5, growth_factor=2.0)
        # 3 evals, all fire -> frequency = 1.0 > 0.5
        for _ in range(3):
            scaler.record("x")
        result = scaler.end_generation()
        assert "bad" in result
        assert check.weight == -2.0

    def test_no_scale_when_below_threshold(self):
        fired = False

        def sometimes(o):
            nonlocal fired
            fired = not fired
            return fired

        check = ProxyCheck("maybe", sometimes, weight=-1.0)
        scaler = PenaltyScaler([check], threshold=0.8, growth_factor=2.0)
        # 4 evals, 2 fire -> frequency = 0.5 < 0.8
        for _ in range(4):
            scaler.record("x")
        result = scaler.end_generation()
        assert result == {}
        assert check.weight == -1.0

    def test_resets_after_end_generation(self):
        check = ProxyCheck("p", lambda o: True, weight=-1.0)
        scaler = PenaltyScaler([check], threshold=0.5, growth_factor=1.5)
        scaler.record("x")
        scaler.end_generation()
        # After reset, no evals recorded
        result = scaler.end_generation()
        assert result == {}
