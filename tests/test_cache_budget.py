"""Tests for response caching and cost/budget guardrails."""
from __future__ import annotations

import pytest

from MutaGenAI.prompt_evolver import (
    EvalSample,
    LLMBackend,
    PromptCandidate,
    PromptEvolver,
    PromptEvolverConfig,
    ResponseCache,
    Tool,
    estimate_cost,
)


# ── ResponseCache unit tests ─────────────────────────────────────────────


class TestResponseCache:
    def _req(self, sp="sys", um="hi", t=0.1, p=0.95):
        return {"system_prompt": sp, "user_message": um,
                "temperature": t, "top_p": p}

    def test_miss_then_hit(self):
        c = ResponseCache()
        req = self._req()
        assert c.get(req) is None
        assert c.misses == 1
        c.put(req, "answer")
        assert c.get(req) == "answer"
        assert c.hits == 1

    def test_disabled_never_caches(self):
        c = ResponseCache(enabled=False)
        req = self._req()
        c.put(req, "answer")
        assert c.get(req) is None
        assert len(c) == 0

    def test_none_not_stored(self):
        c = ResponseCache()
        req = self._req()
        c.put(req, None)  # type: ignore[arg-type]
        assert len(c) == 0

    def test_key_rounds_params(self):
        c = ResponseCache()
        c.put(self._req(t=0.10000001), "x")
        # A near-identical temperature should hit the same key.
        assert c.get(self._req(t=0.1)) == "x"

    def test_distinct_inputs_distinct_keys(self):
        c = ResponseCache()
        c.put(self._req(um="a"), "ra")
        c.put(self._req(um="b"), "rb")
        assert c.get(self._req(um="a")) == "ra"
        assert c.get(self._req(um="b")) == "rb"
        assert len(c) == 2


# ── estimate_cost ────────────────────────────────────────────────────────


class TestEstimateCost:
    def test_unknown_pricing_returns_none(self):
        assert estimate_cost(1000, 500, 0.0, 0.0) is None

    def test_computes_blended_cost(self):
        # 1000 in @ $0.005/1k + 500 out @ $0.015/1k = 0.005 + 0.0075
        assert estimate_cost(1000, 500, 0.005, 0.015) == pytest.approx(0.0125)

    def test_only_input_price(self):
        assert estimate_cost(2000, 0, 0.001, 0.0) == pytest.approx(0.002)


# ── Cache integration with the evolver ───────────────────────────────────


class _CountingClient:
    """Fake client that counts dispatched completions."""

    def __init__(self):
        self.calls = 0
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def is_available(self):
        return True

    def complete_many(self, requests):
        self.calls += len(requests)
        return [f"resp:{r['user_message']}" for r in requests]

    def complete(self, **kw):
        self.calls += 1
        return f"resp:{kw['user_message']}"


@pytest.fixture
def tools():
    return [Tool("get_weather", "w", {"location": "string"}),
            Tool("send_email", "e", {"to": "string"})]


@pytest.fixture
def dataset():
    return [EvalSample("weather in London", "get_weather", {"location": "London"}),
            EvalSample("email bob", "send_email", {"to": "bob"})]


class TestEvolverCaching:
    def _evolver(self, tools, dataset, **cfg):
        config = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA, ollama_url="http://localhost:99999",
            **cfg,
        )
        return PromptEvolver(tools, dataset, config=config, seed=1, verbose=False)

    def test_repeat_evaluation_served_from_cache(self, tools, dataset):
        ev = self._evolver(tools, dataset)
        ev._client = _CountingClient()
        cand = PromptCandidate(template="route {tool_schemas}")
        sp = "route schemas"
        ev._score_on_samples(cand, sp, dataset)
        first = ev._client.calls
        assert first == len(dataset)
        ev._score_on_samples(cand, sp, dataset)
        # Second identical pass hits cache → no new calls.
        assert ev._client.calls == first
        assert ev._cache.hits == len(dataset)

    def test_cache_disabled_refetches(self, tools, dataset):
        ev = self._evolver(tools, dataset, use_cache=False)
        ev._client = _CountingClient()
        cand = PromptCandidate(template="route {tool_schemas}")
        ev._score_on_samples(cand, "sp", dataset)
        ev._score_on_samples(cand, "sp", dataset)
        assert ev._client.calls == 2 * len(dataset)
        assert ev._cache.hits == 0


# ── Budget guardrails ────────────────────────────────────────────────────


class TestBudgetGuardrails:
    def _config(self, **kw):
        base = dict(
            iterations=3, population_size=2, num_islands=2, elite_size=2,
            backend=LLMBackend.OLLAMA, ollama_url="http://localhost:99999",
        )
        base.update(kw)
        return PromptEvolverConfig(**base)

    def test_defaults_are_none(self):
        cfg = PromptEvolverConfig()
        assert cfg.budget_usd is None
        assert cfg.max_calls is None

    def test_result_reports_usage(self, tools, dataset):
        ev = PromptEvolver(tools, dataset, config=self._config(), seed=1,
                           verbose=False)
        result = ev.run()
        assert result.stop_reason == "completed"
        assert result.iterations_run == 3
        assert result.llm_calls == 0          # unreachable backend
        assert result.cache_hits >= 0
        assert result.estimated_cost_usd is None  # no pricing

    def test_pricing_yields_cost_estimate(self, tools, dataset):
        ev = PromptEvolver(
            tools, dataset,
            config=self._config(cost_per_1k_input_tokens=0.005),
            seed=1, verbose=False,
        )
        result = ev.run()
        # Pricing configured → cost is a number (0.0 with no tokens spent).
        assert result.estimated_cost_usd == 0.0

    def test_max_calls_zero_stops_immediately(self, tools, dataset):
        ev = PromptEvolver(tools, dataset, config=self._config(max_calls=0),
                           seed=1, verbose=False)
        result = ev.run()
        assert result.stop_reason == "max_calls"
        assert result.iterations_run == 0
        assert result.history == []
        assert result.best_prompt  # seeds were still evaluated

    def test_budget_stop_reason_helper(self, tools, dataset):
        ev = PromptEvolver(
            tools, dataset,
            config=self._config(
                budget_usd=0.01, cost_per_1k_input_tokens=1.0,
            ),
            seed=1, verbose=False,
        )
        # Simulate spend that exceeds the budget.
        ev._client.total_input_tokens = 100  # 100/1000 * $1 = $0.10 > $0.01
        assert ev._budget_stop_reason() == "budget_usd"

    def test_max_calls_helper(self, tools, dataset):
        ev = PromptEvolver(tools, dataset, config=self._config(max_calls=5),
                           seed=1, verbose=False)
        ev._client.call_count = 5
        assert ev._budget_stop_reason() == "max_calls"
