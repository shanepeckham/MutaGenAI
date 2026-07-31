"""Tests for PromptEvolver migration primitives — warm-start seeds and
early stopping (Phase 1) plus baseline evaluation and reporting (Phase 2)."""
from __future__ import annotations

from MutaGenAI.prompt_evolver import (
    EvalSample,
    LLMBackend,
    PromptEvolver,
    PromptEvolverConfig,
    Tool,
    _SEED_TEMPLATES,
)
from MutaGenAI.migration import (
    MigrationReport,
    PromptEvaluation,
    SampleResult,
    evaluate_prompt,
    make_client,
)


class _MockClient:
    """Deterministic client with the usage counters PromptEvolver.run reads."""

    def __init__(self, response: str = "get_weather") -> None:
        self._response = response
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def is_available(self) -> bool:
        return True

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ) -> str:
        self.call_count += 1
        return self._response


def _tools() -> list[Tool]:
    return [Tool("get_weather", "Get the weather", {"city": "str"})]


def _samples() -> list[EvalSample]:
    return [EvalSample("weather in paris", "get_weather", {"city": "paris"})]


def _evolver(config: PromptEvolverConfig, seeds=None) -> PromptEvolver:
    ev = PromptEvolver(
        _tools(), _samples(), config, seed_templates=seeds, verbose=False
    )
    ev._client = _MockClient()  # inject deterministic client
    return ev


class TestWarmStartSeeds:
    def test_custom_seed_templates_used(self):
        seeds = ["SEED A {tool_schemas}", "SEED B {tool_schemas}"]
        ev = _evolver(
            PromptEvolverConfig(iterations=0, num_islands=1), seeds=seeds
        )
        assert ev._seed_templates == seeds
        ev.run()
        seed_cands = [c for c in ev._all_candidates if c.operation == "seed"]
        assert len(seed_cands) == len(seeds)
        assert {c.template for c in seed_cands} == set(seeds)

    def test_defaults_when_no_seeds(self):
        ev = _evolver(PromptEvolverConfig(iterations=0))
        assert ev._seed_templates == list(_SEED_TEMPLATES)

    def test_empty_seed_list_falls_back_to_defaults(self):
        ev = _evolver(PromptEvolverConfig(iterations=0), seeds=[])
        assert ev._seed_templates == list(_SEED_TEMPLATES)


class TestEarlyStop:
    def test_skips_evolution_when_seeds_meet_target(self):
        # Any real score is >= 0.0, so seeds always clear the bar.
        config = PromptEvolverConfig(
            iterations=5,
            num_islands=1,
            population_size=2,
            early_stop_score=0.0,
        )
        result = _evolver(config, seeds=["S {tool_schemas}"]).run()
        assert result.iterations_run == 0

    def test_runs_all_generations_without_early_stop(self):
        config = PromptEvolverConfig(
            iterations=3, num_islands=1, population_size=2
        )
        result = _evolver(config, seeds=["S {tool_schemas}"]).run()
        assert result.iterations_run == 3

    def test_unreachable_target_runs_all_generations(self):
        config = PromptEvolverConfig(
            iterations=2,
            num_islands=1,
            population_size=2,
            early_stop_score=1000.0,  # never reachable (max is 100)
        )
        result = _evolver(config, seeds=["S {tool_schemas}"]).run()
        assert result.iterations_run == 2


# ---------------------------------------------------------------------------
# Phase 2: baseline evaluation + migration report
# ---------------------------------------------------------------------------


def _sr(query: str, correct: bool) -> SampleResult:
    return SampleResult(
        query=query,
        expected_tool="t",
        predicted_tool="t" if correct else None,
        score=1.0 if correct else 0.0,
        correct=correct,
    )


def _eval(prompt: str, flags: dict[str, bool], temp=0.1, top_p=0.95):
    results = [_sr(q, c) for q, c in flags.items()]
    acc = sum(r.score for r in results) / len(results)
    return PromptEvaluation(prompt, acc, results, temp, top_p)


class TestEvaluatePrompt:
    def test_per_sample_and_accuracy(self):
        tools = [
            Tool("get_weather", "Get the weather", {"city": "str"}),
            Tool("other_tool", "Something else"),
        ]
        samples = [
            EvalSample("weather in paris", "get_weather"),
            EvalSample("do the other thing", "other_tool"),
        ]
        client = _MockClient(response="get_weather")
        ev = evaluate_prompt("P {tool_schemas}", tools, samples, client)

        assert ev.total == 2
        assert ev.num_correct == 1
        assert ev.correct_queries == {"weather in paris"}
        assert 0.0 <= ev.accuracy <= 1.0

    def test_unreachable_client_scores_zero(self):
        tools = [Tool("get_weather", "Get the weather")]
        samples = [EvalSample("q", "get_weather")]
        client = _MockClient(response=None)  # simulate unavailable
        ev = evaluate_prompt("P", tools, samples, client)
        assert ev.accuracy == 0.0
        assert ev.num_correct == 0


class TestMakeClient:
    def test_ollama_model_field(self):
        c = make_client("qwen3:8b", LLMBackend.OLLAMA)
        assert c.config.ollama_model == "qwen3:8b"

    def test_openai_model_field(self):
        c = make_client("gpt-4o", LLMBackend.OPENAI)
        assert c.config.openai_model == "gpt-4o"

    def test_azure_deployment_field(self):
        c = make_client("my-deploy", LLMBackend.AZURE_OPENAI)
        assert c.config.azure_deployment == "my-deploy"


class TestMigrationReport:
    def _reports(self, source=True):
        old = _eval("old", {"q1": True, "q2": True, "q3": False})
        transfer = _eval("old", {"q1": True, "q2": False, "q3": False})
        evolved = _eval(
            "evolved", {"q1": True, "q2": True, "q3": False}, temp=0.7
        )
        return MigrationReport.build(
            source_eval=old if source else None,
            transfer_eval=transfer,
            evolved_eval=evolved,
            source_model="old",
            target_model="new",
        )

    def test_regression_sets_with_source(self):
        r = self._reports(source=True)
        assert r.transfer_regressions == ["q2"]  # broke on naive swap
        assert r.recovered == ["q2"]             # evolution fixed it
        assert r.remaining_regressions == []     # nothing left broken

    def test_accuracies_and_deltas(self):
        r = self._reports(source=True)
        assert r.a_old == 2 / 3
        assert r.a_transfer == 1 / 3
        assert r.a_evolved == 2 / 3
        assert r.delta_vs_old == 0.0
        assert r.preserved is True
        assert r.delta_vs_transfer > 0

    def test_decoding_captured(self):
        r = self._reports(source=True)
        assert r.decoding_before == (0.1, 0.95)
        assert r.decoding_after == (0.7, 0.95)

    def test_source_none_falls_back_to_transfer_baseline(self):
        r = self._reports(source=False)
        assert r.a_old is None
        assert r.delta_vs_old is None
        assert r.preserved is True  # no bar to regress against
        # reference is transfer_correct ({q1}); evolved keeps q1 -> no regress
        assert r.remaining_regressions == []

    def test_summary_renders(self):
        r = self._reports(source=True)
        text = r.summary()
        assert "Migration: old -> new" in text
        assert "A_evolved" in text
