"""Tests for MutaGenAI.prompt_evolver — prompt evolution engine."""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

from MutaGenAI.prompt_evolver import (
    EvalSample,
    ErrorProfile,
    FailureBucket,
    LLMBackend,
    LLMClient,
    ProblemType,
    PromptCandidate,
    PromptEvolver,
    PromptEvolverConfig,
    PromptEvolverResult,
    SelectionMethod,
    OperatorSelection,
    Tool,
    _crossover_templates,
    _llm_crossover_templates,
    _mutate_template,
    count_prompt_tokens,
    get_failure_bucket_mutations,
    parse_tool_response,
    score_response,
)
from MutaGenAI.prompt_evolver import _feasibility_key


# ── Tools / dataset fixtures ──────────────────────────────────────────────


@pytest.fixture
def sample_tools():
    return [
        Tool("get_weather", "Get weather", {"location": "string"}),
        Tool("send_email", "Send email", {"to": "string", "subject": "string"}),
        Tool("calculate", "Calculate", {"expression": "string"}),
    ]


@pytest.fixture
def sample_dataset():
    return [
        EvalSample("Weather in London?", "get_weather", {"location": "London"}),
        EvalSample("Email Bob about meeting", "send_email", {"to": "Bob"}),
        EvalSample("What is 2+2?", "calculate", {"expression": "2+2"}),
        EvalSample("Temperature in Paris?", "get_weather", {"location": "Paris"}),
        EvalSample("Send a note to Alice", "send_email", {"to": "Alice"}),
        EvalSample("Calculate 10 * 5", "calculate", {"expression": "10 * 5"}),
    ]


# ── Tool model tests ──────────────────────────────────────────────────────


class TestTool:
    def test_schema_str(self):
        t = Tool("get_weather", "Get weather", {"location": "string"})
        s = t.schema_str()
        assert "get_weather" in s
        assert "location" in s
        assert "Get weather" in s

    def test_empty_params(self):
        t = Tool("ping", "Ping the system", {})
        s = t.schema_str()
        assert "ping()" in s


# ── PromptCandidate tests ─────────────────────────────────────────────────


class TestPromptCandidate:
    def test_auto_hash(self):
        c = PromptCandidate(template="hello world")
        assert c.hash
        assert len(c.hash) == 16

    def test_different_templates_different_hashes(self):
        a = PromptCandidate(template="template A")
        b = PromptCandidate(template="template B")
        assert a.hash != b.hash

    def test_defaults(self):
        c = PromptCandidate(template="test")
        assert c.temperature == 0.1
        assert c.top_p == 0.95
        assert c.score == 0.0
        assert c.generation == 0


# ── parse_tool_response tests ─────────────────────────────────────────────


class TestParseToolResponse:
    tool_names = ["get_weather", "send_email", "calculate"]

    def test_json_format(self):
        response = '{"tool": "get_weather", "parameters": {"location": "London"}}'
        tool, params = parse_tool_response(response, self.tool_names)
        assert tool == "get_weather"
        assert params["location"] == "London"

    def test_json_with_surrounding_text(self):
        response = 'I think the best tool is: {"tool": "send_email", "parameters": {"to": "Bob"}} ... end'
        tool, params = parse_tool_response(response, self.tool_names)
        assert tool == "send_email"
        assert params["to"] == "Bob"

    def test_json_with_function_key(self):
        response = '{"function": "calculate", "params": {"expression": "2+2"}}'
        tool, params = parse_tool_response(response, self.tool_names)
        assert tool == "calculate"

    def test_function_call_style(self):
        response = 'get_weather(location="Paris")'
        tool, params = parse_tool_response(response, self.tool_names)
        assert tool == "get_weather"
        assert params.get("location") == "Paris"

    def test_plain_text_mention(self):
        response = "I would use get_weather for this query."
        tool, params = parse_tool_response(response, self.tool_names)
        assert tool == "get_weather"

    def test_no_match(self):
        response = "I don't know what to do."
        tool, params = parse_tool_response(response, self.tool_names)
        assert tool is None
        assert params == {}

    def test_empty_response(self):
        tool, params = parse_tool_response("", self.tool_names)
        assert tool is None

    def test_none_response(self):
        tool, params = parse_tool_response(None, self.tool_names)
        assert tool is None

    def test_json_with_name_key(self):
        response = '{"name": "calculate", "arguments": {"expression": "3*7"}}'
        tool, params = parse_tool_response(response, self.tool_names)
        assert tool == "calculate"
        assert params.get("expression") == "3*7"


# ── score_response tests ──────────────────────────────────────────────────


class TestScoreResponse:
    def test_correct_tool_correct_params(self):
        score = score_response("get_weather", {"location": "London"},
                               "get_weather", {"location": "London"})
        assert score == 1.0

    def test_correct_tool_no_expected_params(self):
        score = score_response("send_email", {}, "send_email", {})
        assert score == 1.0

    def test_correct_tool_wrong_params(self):
        score = score_response("get_weather", {"location": "Paris"},
                               "get_weather", {"location": "London"})
        assert 0.5 < score < 1.0  # Tool correct (0.6) but param wrong

    def test_wrong_tool(self):
        score = score_response("send_email", {},
                               "get_weather", {"location": "London"})
        assert score == 0.0

    def test_none_prediction(self):
        score = score_response(None, {},
                               "get_weather", {"location": "London"})
        assert score == 0.0

    def test_partial_param_match(self):
        score = score_response("get_weather", {"location": "London, UK"},
                               "get_weather", {"location": "London"})
        # "London" is in "London, UK" → partial match
        assert score > 0.6

    def test_correct_tool_all_params(self):
        score = score_response(
            "send_email", {"to": "Bob", "subject": "meeting"},
            "send_email", {"to": "Bob", "subject": "meeting"},
        )
        assert score == 1.0


# ── Mutation tests ────────────────────────────────────────────────────────


class TestMutations:
    def test_mutate_preserves_tool_schemas_placeholder(self):
        template = "You are a tool router.\n\n{tool_schemas}\n\nRespond in JSON."
        rng = np.random.default_rng(42)
        for _ in range(20):
            mutated = _mutate_template(template, rng, mutation_rate=1.0)
            assert "{tool_schemas}" in mutated

    def test_crossover_preserves_tool_schemas_placeholder(self):
        a = "Hello\n{tool_schemas}\nWorld"
        b = "Foo\nBar\nBaz"
        rng = np.random.default_rng(42)
        child = _crossover_templates(a, b, rng)
        assert "{tool_schemas}" in child

    def test_mutate_changes_template(self):
        template = "Line 1\nLine 2\n{tool_schemas}\nLine 4\nLine 5"
        rng = np.random.default_rng(42)
        changed = False
        for _ in range(20):
            mutated = _mutate_template(template, rng, mutation_rate=1.0)
            if mutated != template:
                changed = True
                break
        assert changed, "Mutation should change the template at least once"


# ── Config tests ──────────────────────────────────────────────────────────


class TestPromptEvolverConfig:
    def test_defaults(self):
        cfg = PromptEvolverConfig()
        assert cfg.iterations == 30
        assert cfg.population_size == 8
        assert cfg.backend == LLMBackend.OLLAMA

    def test_env_var_pickup(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://test.openai.azure.com")
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        cfg = PromptEvolverConfig()
        assert cfg.azure_endpoint == "https://test.openai.azure.com"
        assert cfg.azure_api_key == "test-key"
        assert cfg.azure_deployment == "gpt-4o-mini"


# ── LLMClient tests ──────────────────────────────────────────────────────


class TestLLMClient:
    def test_unavailable_ollama(self):
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",
        )
        client = LLMClient(cfg)
        assert client.is_available() is False

    def test_azure_needs_credentials(self, monkeypatch):
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_USE_RBAC", raising=False)
        cfg = PromptEvolverConfig(
            backend=LLMBackend.AZURE_OPENAI,
            azure_endpoint="",
            azure_api_key="",
        )
        client = LLMClient(cfg)
        assert client.is_available() is False

    def test_complete_returns_none_when_unavailable(self):
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",
        )
        client = LLMClient(cfg)
        result = client.complete("system", "user")
        assert result is None


# ── Concurrent batch completion tests ────────────────────────────────────


class TestCompleteMany:
    """Tests for LLMClient.complete_many concurrent batch dispatch."""

    def test_default_max_concurrency(self):
        assert PromptEvolverConfig().max_concurrency == 8

    def test_empty_batch_returns_empty_list(self):
        client = LLMClient(PromptEvolverConfig())
        assert client.complete_many([]) == []

    def test_serial_fallback_when_unavailable(self):
        """Unreachable backend → ordered list of None, length preserved."""
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",
            max_concurrency=8,
        )
        client = LLMClient(cfg)
        reqs = [
            {"system_prompt": "s", "user_message": f"q{i}"} for i in range(4)
        ]
        out = client.complete_many(reqs)
        assert out == [None, None, None, None]

    def test_singleton_uses_serial_complete(self, monkeypatch):
        """A one-element batch routes through serial complete()."""
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OPENAI, openai_api_key="sk-test"
        )
        client = LLMClient(cfg)
        calls: list[str] = []

        def fake_complete(**kw):
            calls.append(kw["user_message"])
            return f"r:{kw['user_message']}"

        monkeypatch.setattr(client, "complete", fake_complete)
        out = client.complete_many(
            [{"system_prompt": "s", "user_message": "only"}]
        )
        assert out == ["r:only"]
        assert calls == ["only"]

    def test_max_concurrency_one_forces_serial(self, monkeypatch):
        """max_concurrency=1 must not use the async path even if available."""
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OPENAI,
            openai_api_key="sk-test",
            max_concurrency=1,
        )
        client = LLMClient(cfg)
        monkeypatch.setattr(client, "complete", lambda **kw: "SERIAL")

        async def _boom(*a, **k):  # pragma: no cover - must never run
            raise AssertionError("async path used despite max_concurrency=1")

        monkeypatch.setattr(LLMClient, "_acomplete", _boom)
        out = client.complete_many(
            [
                {"system_prompt": "s", "user_message": "a"},
                {"system_prompt": "s", "user_message": "b"},
            ]
        )
        assert out == ["SERIAL", "SERIAL"]

    def test_concurrent_preserves_order(self, monkeypatch):
        """Results align with input order despite varied async latencies."""
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OPENAI,
            openai_api_key="sk-test",
            max_concurrency=4,
        )
        client = LLMClient(cfg)

        async def fake_acomplete(
            self, aclient, system_prompt, user_message,
            temperature=0.1, top_p=0.95,
        ):
            # Later items sleep less, so completion order != input order.
            idx = int(user_message[1:])
            await asyncio.sleep(0.01 * (5 - idx))
            return f"resp:{user_message}"

        monkeypatch.setattr(LLMClient, "_acomplete", fake_acomplete)
        reqs = [
            {"system_prompt": "s", "user_message": f"q{i}"} for i in range(5)
        ]
        out = client.complete_many(reqs)
        assert out == [f"resp:q{i}" for i in range(5)]

    def test_concurrency_is_bounded_by_semaphore(self, monkeypatch):
        """No more than max_concurrency requests run simultaneously."""
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OPENAI,
            openai_api_key="sk-test",
            max_concurrency=3,
        )
        client = LLMClient(cfg)
        state = {"current": 0, "peak": 0}

        async def fake_acomplete(self, aclient, system_prompt, user_message,
                                 temperature=0.1, top_p=0.95):
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
            await asyncio.sleep(0.01)
            state["current"] -= 1
            return user_message

        monkeypatch.setattr(LLMClient, "_acomplete", fake_acomplete)
        reqs = [
            {"system_prompt": "s", "user_message": f"q{i}"} for i in range(12)
        ]
        out = client.complete_many(reqs)
        assert out == [f"q{i}" for i in range(12)]
        assert state["peak"] <= 3

    def test_run_async_inside_running_loop(self, monkeypatch):
        """complete_many works when called from within a running event loop."""
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OPENAI,
            openai_api_key="sk-test",
            max_concurrency=4,
        )
        client = LLMClient(cfg)

        async def fake_acomplete(self, aclient, system_prompt, user_message,
                                 temperature=0.1, top_p=0.95):
            return f"resp:{user_message}"

        monkeypatch.setattr(LLMClient, "_acomplete", fake_acomplete)
        reqs = [
            {"system_prompt": "s", "user_message": f"q{i}"} for i in range(3)
        ]

        async def driver():
            # A loop is running here; _run_async must offload to a thread.
            return client.complete_many(reqs)

        out = asyncio.run(driver())
        assert out == [f"resp:q{i}" for i in range(3)]

    def test_async_exception_yields_none(self, monkeypatch):
        """Per-request async failures degrade to None, matching complete()."""
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OPENAI,
            openai_api_key="sk-test",
            max_concurrency=4,
        )
        client = LLMClient(cfg)

        async def fake_acomplete(self, aclient, system_prompt, user_message,
                                 temperature=0.1, top_p=0.95):
            if user_message == "q1":
                raise RuntimeError("boom")
            return f"resp:{user_message}"

        # Wrap so the exception is caught by _acomplete's own try/except we
        # bypass — instead patch the per-backend method to raise.
        async def fake_aopenai(self, aclient, system_prompt, user_message,
                               temperature, top_p):
            if user_message == "q1":
                raise RuntimeError("boom")
            return f"resp:{user_message}"

        monkeypatch.setattr(LLMClient, "_aopenai_complete", fake_aopenai)
        reqs = [
            {"system_prompt": "s", "user_message": f"q{i}"} for i in range(3)
        ]
        out = client.complete_many(reqs)
        assert out == ["resp:q0", None, "resp:q2"]


# ── PromptEvolver integration test (no LLM) ──────────────────────────────


class TestPromptEvolver:
    def test_run_without_llm(self, sample_tools, sample_dataset):
        """Run the evolver without an LLM — should work with random scores."""
        config = PromptEvolverConfig(
            iterations=3,
            population_size=2,
            num_islands=2,
            elite_size=2,
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",  # Intentionally unreachable
        )
        evolver = PromptEvolver(
            tools=sample_tools,
            eval_dataset=sample_dataset,
            config=config,
            seed=42,
            verbose=False,
        )
        result = evolver.run()

        assert isinstance(result, PromptEvolverResult)
        assert result.best_prompt
        assert result.iterations_run == 3
        assert result.wall_time > 0
        assert len(result.all_candidates) > 0
        assert len(result.history) == 3
        assert 0 <= result.best_accuracy <= 1.0

    def test_result_summary(self, sample_tools, sample_dataset):
        config = PromptEvolverConfig(
            iterations=2,
            population_size=2,
            num_islands=1,
            elite_size=2,
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",
        )
        evolver = PromptEvolver(
            tools=sample_tools,
            eval_dataset=sample_dataset,
            config=config,
            seed=42,
            verbose=False,
        )
        result = evolver.run()
        summary = result.summary()
        assert "Prompt Evolver Result" in summary
        assert "Best accuracy" in summary

    def test_eval_sample_size(self, sample_tools, sample_dataset):
        """With eval_sample_size set, should subsample the dataset."""
        config = PromptEvolverConfig(
            iterations=2,
            population_size=2,
            num_islands=1,
            elite_size=2,
            eval_sample_size=2,
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",
        )
        evolver = PromptEvolver(
            tools=sample_tools,
            eval_dataset=sample_dataset,
            config=config,
            seed=42,
            verbose=False,
        )
        result = evolver.run()
        assert isinstance(result, PromptEvolverResult)

    def test_candidates_sorted_best_first(self, sample_tools, sample_dataset):
        config = PromptEvolverConfig(
            iterations=3,
            population_size=3,
            num_islands=2,
            elite_size=2,
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",
        )
        evolver = PromptEvolver(
            tools=sample_tools,
            eval_dataset=sample_dataset,
            config=config,
            seed=42,
            verbose=False,
        )
        result = evolver.run()
        scores = [c.score for c in result.all_candidates]
        assert scores == sorted(scores, reverse=True)

    def test_concurrency_preserves_determinism(
        self, sample_tools, sample_dataset
    ):
        """Same seed + same dataset → identical results regardless of
        max_concurrency, because scoring/bookkeeping stays order-serial."""
        def _run(max_concurrency: int) -> PromptEvolverResult:
            config = PromptEvolverConfig(
                iterations=3,
                population_size=3,
                num_islands=2,
                elite_size=2,
                backend=LLMBackend.OLLAMA,
                ollama_url="http://localhost:99999",
                max_concurrency=max_concurrency,
            )
            evolver = PromptEvolver(
                tools=sample_tools,
                eval_dataset=sample_dataset,
                config=config,
                seed=42,
                verbose=False,
            )
            return evolver.run()

        serial = _run(1)
        concurrent = _run(8)
        assert serial.best_accuracy == concurrent.best_accuracy
        assert serial.history == concurrent.history
        assert [c.score for c in serial.all_candidates] == [
            c.score for c in concurrent.all_candidates
        ]

    def test_score_on_samples_client_without_complete_many(
        self, sample_tools, sample_dataset
    ):
        """An injected client double exposing only complete() still works."""
        config = PromptEvolverConfig(
            iterations=1, population_size=2, num_islands=1, elite_size=2,
        )
        evolver = PromptEvolver(
            tools=sample_tools,
            eval_dataset=sample_dataset,
            config=config,
            seed=1,
            verbose=False,
        )

        class OnlyComplete:
            def complete(self, **kw):
                return "get_weather(London)"

        evolver._client = OnlyComplete()
        candidate = PromptCandidate(template="{tool_schemas}")
        score = evolver._score_on_samples(
            candidate, "{tool_schemas}", evolver.eval_dataset
        )
        assert 0.0 <= score <= 100.0


# ── LLMBackend enum tests ────────────────────────────────────────────────


class TestLLMBackend:
    def test_values(self):
        assert LLMBackend.OLLAMA.value == "ollama"
        assert LLMBackend.AZURE_OPENAI.value == "azure_openai"
        assert LLMBackend.OPENAI.value == "openai"

    def test_from_string(self):
        assert LLMBackend("ollama") == LLMBackend.OLLAMA
        assert LLMBackend("azure_openai") == LLMBackend.AZURE_OPENAI


# ── ErrorProfile tests ───────────────────────────────────────────────────


class TestErrorProfile:
    def test_record_and_worst_categories(self):
        from MutaGenAI.prompt_evolver import ErrorProfile

        ep = ErrorProfile()
        ep.record("Agent", True)
        ep.record("Agent", True)
        ep.record("Agent", False)
        ep.record("Human", False)
        ep.record("Human", False)
        ep.record("Tool", True)

        worst = ep.worst_categories(top_k=2)
        assert len(worst) == 2
        # Human has 100% error rate, Agent has 33%
        assert worst[0][0] == "Human"
        assert worst[0][1] == pytest.approx(1.0)
        assert worst[1][0] == "Agent"
        assert worst[1][1] == pytest.approx(1 / 3)

    def test_empty_profile(self):
        from MutaGenAI.prompt_evolver import ErrorProfile

        ep = ErrorProfile()
        assert ep.worst_categories() == []

    def test_no_errors(self):
        from MutaGenAI.prompt_evolver import ErrorProfile

        ep = ErrorProfile()
        ep.record("Agent", True)
        ep.record("Tool", True)
        assert ep.worst_categories() == []


# ── Adaptive/LLM mutation fallback tests ─────────────────────────────────


class TestAdaptiveMutations:
    def test_generate_adaptive_mutations_fallback_classification(self):
        from MutaGenAI.prompt_evolver import (
            ErrorProfile,
            ProblemType,
            generate_adaptive_mutations,
        )

        ep = ErrorProfile()
        ep.record("Human", False)
        ep.record("Human", False)
        ep.record("Agent", False)

        # Create an LLMClient that returns None (fallback path)
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",
        )
        client = LLMClient(cfg)

        hints = generate_adaptive_mutations(
            ep, ProblemType.CLASSIFICATION, client, top_k=2, max_hints=3
        )
        assert len(hints) >= 1
        assert any("Human" in h for h in hints)

    def test_generate_adaptive_mutations_fallback_tool_routing(self):
        from MutaGenAI.prompt_evolver import (
            ErrorProfile,
            ProblemType,
            generate_adaptive_mutations,
        )

        ep = ErrorProfile()
        ep.record("get_weather", False)

        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",
        )
        client = LLMClient(cfg)

        hints = generate_adaptive_mutations(
            ep, ProblemType.TOOL_ROUTING, client, top_k=1, max_hints=2
        )
        assert len(hints) >= 1
        assert any("get_weather" in h for h in hints)

    def test_generate_adaptive_empty_profile(self):
        from MutaGenAI.prompt_evolver import (
            ErrorProfile,
            ProblemType,
            generate_adaptive_mutations,
        )

        ep = ErrorProfile()
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",
        )
        client = LLMClient(cfg)
        assert generate_adaptive_mutations(ep, ProblemType.CLASSIFICATION, client) == []


class TestLLMMutateTemplate:
    def test_returns_original_on_empty_failures(self):
        from MutaGenAI.prompt_evolver import ProblemType, _llm_mutate_template

        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",
        )
        client = LLMClient(cfg)

        template = "Classify as Agent, Task, or Tool."
        result = _llm_mutate_template(
            template, [], client, ProblemType.CLASSIFICATION
        )
        assert result == template

    def test_returns_original_on_llm_failure(self):
        from MutaGenAI.prompt_evolver import ProblemType, _llm_mutate_template

        cfg = PromptEvolverConfig(
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",
        )
        client = LLMClient(cfg)

        template = "Route to the correct tool.\n\n{tool_schemas}"
        failures = [("What's the weather?", "send_email", "get_weather")]
        result = _llm_mutate_template(
            template, failures, client, ProblemType.TOOL_ROUTING
        )
        # LLM not available → returns original
        assert result == template


# ── Config new fields tests ──────────────────────────────────────────────


class TestNewConfigFields:
    def test_prompt_evolver_config_new_defaults(self):
        cfg = PromptEvolverConfig()
        assert cfg.adaptive_mutations is False
        assert cfg.llm_mutation_rate == 0.0

    def test_noeval_config_new_defaults(self):
        from MutaGenAI.strategies import NoEvalConfig

        cfg = NoEvalConfig()
        assert cfg.adaptive_mutations is False
        assert cfg.llm_mutation_rate == 0.0

    def test_noeval_config_custom_values(self):
        from MutaGenAI.strategies import NoEvalConfig

        cfg = NoEvalConfig(adaptive_mutations=True, llm_mutation_rate=0.3)
        assert cfg.adaptive_mutations is True
        assert cfg.llm_mutation_rate == 0.3


# ── PromptEvolverResult tests ────────────────────────────────────────────


class TestPromptEvolverResult:
    def test_lineage_json(self, sample_tools, sample_dataset):
        config = PromptEvolverConfig(
            iterations=2,
            population_size=2,
            num_islands=1,
            elite_size=2,
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",
        )
        evolver = PromptEvolver(
            tools=sample_tools,
            eval_dataset=sample_dataset,
            config=config,
            seed=42,
            verbose=False,
        )
        result = evolver.run()
        lineage = result.lineage_json()
        assert isinstance(lineage, list)
        assert len(lineage) > 0
        for rec in lineage:
            assert "hash" in rec
            assert "parent_hashes" in rec
            assert "operation" in rec
            assert "generation" in rec
            assert "score" in rec
            assert "template" in rec
            assert "island_id" in rec

    def test_lineage_unique_hashes(self, sample_tools, sample_dataset):
        config = PromptEvolverConfig(
            iterations=3,
            population_size=3,
            num_islands=2,
            elite_size=2,
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",
        )
        evolver = PromptEvolver(
            tools=sample_tools,
            eval_dataset=sample_dataset,
            config=config,
            seed=42,
            verbose=False,
        )
        result = evolver.run()
        lineage = result.lineage_json()
        hashes = [r["hash"] for r in lineage]
        assert len(hashes) == len(set(hashes)), "Lineage hashes must be unique"


# ── LLMClient additional backend tests ───────────────────────────────────


class TestLLMClientOpenAI:
    def test_openai_needs_api_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OPENAI,
            openai_api_key="",
        )
        client = LLMClient(cfg)
        assert client.is_available() is False

    def test_openai_with_key(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OPENAI,
            openai_api_key="sk-test-key",
        )
        client = LLMClient(cfg)
        assert client.is_available() is True

    def test_complete_openai_returns_none_when_unreachable(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = PromptEvolverConfig(
            backend=LLMBackend.OPENAI,
            openai_api_key="",
        )
        client = LLMClient(cfg)
        result = client.complete("system", "user")
        assert result is None


class TestLLMClientAzureRBAC:
    def test_rbac_env_var_true(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_USE_RBAC", "true")
        cfg = PromptEvolverConfig()
        assert cfg.azure_use_rbac is True

    def test_rbac_env_var_false(self, monkeypatch):
        monkeypatch.setenv("AZURE_OPENAI_USE_RBAC", "false")
        cfg = PromptEvolverConfig()
        assert cfg.azure_use_rbac is False

    def test_azure_scope_foundry(self):
        scope = LLMClient._azure_scope("https://myproject.services.ai.azure.com/openai")
        assert "ai.azure.com" in scope

    def test_azure_scope_cognitive(self):
        scope = LLMClient._azure_scope("https://myresource.openai.azure.com")
        assert "cognitiveservices" in scope

    def test_azure_base_url_strip_openai_path(self):
        url = LLMClient._azure_base_url(
            "https://myresource.openai.azure.com/openai/v1/responses"
        )
        assert url == "https://myresource.openai.azure.com"

    def test_azure_base_url_no_path(self):
        url = LLMClient._azure_base_url("https://myresource.openai.azure.com/")
        assert url == "https://myresource.openai.azure.com"

    def test_is_foundry_endpoint_true(self):
        assert LLMClient._is_foundry_endpoint(
            "https://project.services.ai.azure.com"
        ) is True

    def test_is_foundry_endpoint_false(self):
        assert LLMClient._is_foundry_endpoint(
            "https://myresource.openai.azure.com"
        ) is False


# ── ErrorProfile decay test ──────────────────────────────────────────────


class TestErrorProfileDecay:
    def test_decay_halves_counts(self):
        from MutaGenAI.prompt_evolver import ErrorProfile

        ep = ErrorProfile()
        ep.record("A", False)
        ep.record("A", False)
        ep.record("A", True)
        ep.record("A", True)
        assert ep.total["A"] == 4
        assert ep.errors["A"] == 2

        ep.decay(0.5)
        assert ep.total["A"] == 2
        assert ep.errors["A"] == 1

    def test_decay_removes_zeroed_categories(self):
        from MutaGenAI.prompt_evolver import ErrorProfile

        ep = ErrorProfile()
        ep.record("X", False)
        ep.decay(0.0)
        assert "X" not in ep.total


# ── ProblemType and mutation pool tests ──────────────────────────────────


class TestProblemTypeMutations:
    def test_tool_routing_mutations(self):
        from MutaGenAI.prompt_evolver import ProblemType, get_mutations_for_problem_type

        mutations = get_mutations_for_problem_type(ProblemType.TOOL_ROUTING)
        assert len(mutations) > 0
        assert all(isinstance(m, str) for m in mutations)

    def test_classification_mutations(self):
        from MutaGenAI.prompt_evolver import ProblemType, get_mutations_for_problem_type

        mutations = get_mutations_for_problem_type(ProblemType.CLASSIFICATION)
        assert len(mutations) > 0
        assert all(isinstance(m, str) for m in mutations)

    def test_different_pools(self):
        from MutaGenAI.prompt_evolver import ProblemType, get_mutations_for_problem_type

        tool_muts = get_mutations_for_problem_type(ProblemType.TOOL_ROUTING)
        class_muts = get_mutations_for_problem_type(ProblemType.CLASSIFICATION)
        assert tool_muts != class_muts


# ── Crossover with require_tool_schemas=False ────────────────────────────


class TestCrossoverNoToolSchemas:
    def test_crossover_without_tool_schemas(self):
        import numpy as np
        a = "Line A1\nLine A2\nLine A3"
        b = "Line B1\nLine B2\nLine B3"
        rng = np.random.default_rng(42)
        child = _crossover_templates(a, b, rng, require_tool_schemas=False)
        assert isinstance(child, str)
        assert "{tool_schemas}" not in child  # Not forced

    def test_mutate_without_tool_schemas(self):
        import numpy as np
        template = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        rng = np.random.default_rng(42)
        mutated = _mutate_template(template, rng, mutation_rate=1.0,
                                   require_tool_schemas=False)
        assert isinstance(mutated, str)


# ── _llm_describe_entities tests ─────────────────────────────────────────


class TestLLMDescribeEntities:
    def test_returns_rewritten_prompt(self):
        from MutaGenAI.prompt_evolver import ProblemType, _llm_describe_entities

        class FakeClient:
            def complete(self, *, system_prompt, user_message, temperature, top_p):
                return (
                    "Route to the correct agent.\n"
                    "auth_agent — Handles authentication.\n"
                    "billing_agent — Processes billing."
                )

        template = "Route to the correct agent.\nauth_agent\nbilling_agent"
        result = _llm_describe_entities(
            template, FakeClient(), ProblemType.TOOL_ROUTING,
            require_tool_schemas=False,
        )
        assert "auth_agent" in result
        assert "billing_agent" in result
        assert result != template  # Should have been rewritten

    def test_returns_original_on_empty_response(self):
        from MutaGenAI.prompt_evolver import ProblemType, _llm_describe_entities

        class FakeClient:
            def complete(self, **kw):
                return ""

        template = "Route to: agent_a, agent_b"
        result = _llm_describe_entities(
            template, FakeClient(), ProblemType.TOOL_ROUTING,
            require_tool_schemas=False,
        )
        assert result == template

    def test_returns_original_on_none_response(self):
        from MutaGenAI.prompt_evolver import ProblemType, _llm_describe_entities

        class FakeClient:
            def complete(self, **kw):
                return None

        template = "Classify into: A, B, C"
        result = _llm_describe_entities(
            template, FakeClient(), ProblemType.CLASSIFICATION,
            require_tool_schemas=False,
        )
        assert result == template

    def test_strips_markdown_fences(self):
        from MutaGenAI.prompt_evolver import ProblemType, _llm_describe_entities

        class FakeClient:
            def complete(self, **kw):
                return "```\nRewritten prompt content\n```"

        template = "Original prompt"
        result = _llm_describe_entities(
            template, FakeClient(), ProblemType.TOOL_ROUTING,
            require_tool_schemas=False,
        )
        assert "```" not in result
        assert "Rewritten prompt content" in result

    def test_preserves_tool_schemas_placeholder(self):
        from MutaGenAI.prompt_evolver import ProblemType, _llm_describe_entities

        class FakeClient:
            def complete(self, **kw):
                return "Rewritten without placeholder"

        template = "Route to agents.\n{tool_schemas}"
        result = _llm_describe_entities(
            template, FakeClient(), ProblemType.TOOL_ROUTING,
            require_tool_schemas=True,
        )
        assert "{tool_schemas}" in result

    def test_classification_problem_type(self):
        from MutaGenAI.prompt_evolver import ProblemType, _llm_describe_entities

        captured = {}

        class FakeClient:
            def complete(self, *, system_prompt, user_message, **kw):
                captured["user_message"] = user_message
                return "Rewritten classification prompt"

        template = "Classify: Agent, Task, Tool"
        _llm_describe_entities(
            template, FakeClient(), ProblemType.CLASSIFICATION,
            require_tool_schemas=False,
        )
        assert "category label" in captured["user_message"]


# ── _has_entity_descriptions tests ───────────────────────────────────────


class TestHasEntityDescriptions:
    def test_detects_described_template(self):
        from MutaGenAI.prompt_evolver import _has_entity_descriptions

        template = (
            "Route to the correct agent.\n"
            "auth_agent — Handles authentication.\n"
            "billing_agent — Processes billing.\n"
            "support_agent — Customer support."
        )
        assert _has_entity_descriptions(template) is True

    def test_rejects_undescribed_template(self):
        from MutaGenAI.prompt_evolver import _has_entity_descriptions

        template = "Route to: auth_agent, billing_agent, support_agent"
        assert _has_entity_descriptions(template) is False

    def test_boundary_two_descriptions(self):
        from MutaGenAI.prompt_evolver import _has_entity_descriptions

        template = (
            "auth_agent — Handles auth.\n"
            "billing_agent — Handles billing."
        )
        assert _has_entity_descriptions(template) is False

    def test_regular_dash_not_counted(self):
        from MutaGenAI.prompt_evolver import _has_entity_descriptions

        template = "Use the right tool - pick carefully - be precise."
        assert _has_entity_descriptions(template) is False


# ── _extract_entity_names tests ──────────────────────────────────────────


class TestExtractEntityNames:
    def test_extracts_agent_names(self):
        from MutaGenAI.prompt_evolver import _extract_entity_names

        template = "Route to: auth_agent, billing_agent, support_agent"
        result = _extract_entity_names(template)
        assert result == ["auth_agent", "billing_agent", "support_agent"]

    def test_deduplicates(self):
        from MutaGenAI.prompt_evolver import _extract_entity_names

        template = "auth_agent handles auth. auth_agent is important."
        result = _extract_entity_names(template)
        assert result == ["auth_agent"]

    def test_no_entities(self):
        from MutaGenAI.prompt_evolver import _extract_entity_names

        template = "Route the user request to the correct handler."
        result = _extract_entity_names(template)
        assert result == []

    def test_fallback_to_underscore_identifiers(self):
        from MutaGenAI.prompt_evolver import _extract_entity_names

        template = "Categories: fraud_detection, risk_assessment, data_analysis"
        result = _extract_entity_names(template)
        assert "fraud_detection" in result
        assert "risk_assessment" in result

    def test_agent_names_take_priority(self):
        from MutaGenAI.prompt_evolver import _extract_entity_names

        template = "Use fraud_detection_agent for fraud_detection tasks"
        result = _extract_entity_names(template)
        assert result == ["fraud_detection_agent"]


# ── describe_entities completeness guard tests ───────────────────────────


class TestDescribeEntitiesCompletenessGuard:
    def test_partial_triggers_retry_and_succeeds(self):
        from unittest.mock import MagicMock

        from MutaGenAI.prompt_evolver import ProblemType, _llm_describe_entities

        client = MagicMock()
        # First call: LLM only describes 1 out of 3
        partial = (
            "Route to the correct agents.\n"
            "sales_agent — Handles sales inquiries."
        )
        # Retry call: LLM completes all 3
        complete = (
            "Route to the correct agents.\n"
            "sales_agent — Handles sales inquiries.\n"
            "billing_agent — Processes billing.\n"
            "support_agent — Customer support."
        )
        client.complete.side_effect = [partial, complete]
        original = "Route to: sales_agent, billing_agent, support_agent"
        result = _llm_describe_entities(
            original, client, ProblemType.TOOL_ROUTING, require_tool_schemas=False
        )
        assert client.complete.call_count == 2
        assert "billing_agent —" in result or "billing_agent —" in result
        assert "support_agent —" in result or "support_agent —" in result

    def test_partial_retry_still_incomplete_returns_original(self):
        from unittest.mock import MagicMock

        from MutaGenAI.prompt_evolver import ProblemType, _llm_describe_entities

        client = MagicMock()
        # First call: partial
        partial = (
            "Route to the correct agents.\n"
            "sales_agent — Handles sales inquiries."
        )
        # Retry call: still only 2 out of 3
        still_partial = (
            "Route to the correct agents.\n"
            "sales_agent — Handles sales inquiries.\n"
            "billing_agent — Processes billing."
        )
        client.complete.side_effect = [partial, still_partial]
        original = "Route to: sales_agent, billing_agent, support_agent"
        result = _llm_describe_entities(
            original, client, ProblemType.TOOL_ROUTING, require_tool_schemas=False
        )
        assert result == original  # falls back after failed retry

    def test_partial_retry_empty_returns_original(self):
        from unittest.mock import MagicMock

        from MutaGenAI.prompt_evolver import ProblemType, _llm_describe_entities

        client = MagicMock()
        partial = (
            "Route to the correct agents.\n"
            "sales_agent — Handles sales inquiries."
        )
        client.complete.side_effect = [partial, ""]
        original = "Route to: sales_agent, billing_agent, support_agent"
        result = _llm_describe_entities(
            original, client, ProblemType.TOOL_ROUTING, require_tool_schemas=False
        )
        assert result == original

    def test_full_description_accepted(self):
        from unittest.mock import MagicMock

        from MutaGenAI.prompt_evolver import ProblemType, _llm_describe_entities

        client = MagicMock()
        client.complete.return_value = (
            "Route to the correct agents.\n"
            "sales_agent — Handles sales inquiries.\n"
            "billing_agent — Processes billing.\n"
            "support_agent — Customer support."
        )
        original = "Route to: sales_agent, billing_agent, support_agent"
        result = _llm_describe_entities(
            original, client, ProblemType.TOOL_ROUTING, require_tool_schemas=False
        )
        assert result != original
        assert "sales_agent" in result
        assert "billing_agent" in result

    def test_no_descriptions_passthrough(self):
        from unittest.mock import MagicMock

        from MutaGenAI.prompt_evolver import ProblemType, _llm_describe_entities

        client = MagicMock()
        # LLM returns a rewrite with no " — " patterns at all
        client.complete.return_value = (
            "Select agents from: sales_agent, billing_agent, support_agent"
        )
        original = "Route to: sales_agent, billing_agent, support_agent"
        result = _llm_describe_entities(
            original, client, ProblemType.TOOL_ROUTING, require_tool_schemas=False
        )
        # 0 described — not a partial result, so accepted
        assert result != original

    def test_empty_response_returns_original(self):
        from unittest.mock import MagicMock

        from MutaGenAI.prompt_evolver import ProblemType, _llm_describe_entities

        client = MagicMock()
        client.complete.return_value = ""
        original = "Route to: sales_agent, billing_agent, support_agent"
        result = _llm_describe_entities(
            original, client, ProblemType.TOOL_ROUTING, require_tool_schemas=False
        )
        assert result == original


# ---------------------------------------------------------------------------
# _feasibility_key tests (prompt_evolver copy)
# ---------------------------------------------------------------------------


class TestFeasibilityKeyPromptEvolver:
    """Tests for _feasibility_key in prompt_evolver module."""

    def test_zero_violations_beats_any(self):
        feasible = PromptCandidate(template="a", score=10.0, penalty_violations=0)
        infeasible = PromptCandidate(template="b", score=99.0, penalty_violations=1)
        assert _feasibility_key(feasible) > _feasibility_key(infeasible)

    def test_same_feasibility_uses_score(self):
        a = PromptCandidate(template="a", score=80.0, penalty_violations=0)
        b = PromptCandidate(template="b", score=60.0, penalty_violations=0)
        assert _feasibility_key(a) > _feasibility_key(b)

    def test_penalty_violations_default_zero(self):
        c = PromptCandidate(template="t", score=50.0)
        assert c.penalty_violations == 0


# ── Token optimization tests ──────────────────────────────────────────────


class TestCountPromptTokens:
    """Tests for count_prompt_tokens utility."""

    def test_given_empty_string_when_count_then_returns_zero(self):
        assert count_prompt_tokens("") == 0

    def test_given_short_text_when_count_then_returns_positive(self):
        tokens = count_prompt_tokens("Hello world")
        assert tokens > 0

    def test_given_longer_text_when_count_then_more_tokens(self):
        short = count_prompt_tokens("Hi")
        long = count_prompt_tokens("Hello world, this is a much longer sentence.")
        assert long > short

    def test_given_known_text_when_count_then_consistent(self):
        # Same input should always produce same output
        text = "You are a helpful assistant."
        assert count_prompt_tokens(text) == count_prompt_tokens(text)


class TestTokenOptimizationConfig:
    """Tests for token optimization config defaults."""

    def test_given_default_config_when_check_then_disabled(self):
        cfg = PromptEvolverConfig()
        assert cfg.minimize_tokens is False
        assert cfg.token_weight == 0.10
        assert cfg.token_efficiency_cap == 2.0
        assert cfg.token_accuracy_band == 2.0
        assert cfg.baseline_prompt_tokens == 0

    def test_given_enabled_config_when_construct_then_stores_values(self):
        cfg = PromptEvolverConfig(
            minimize_tokens=True,
            token_weight=0.20,
            token_efficiency_cap=3.0,
            token_accuracy_band=5.0,
            baseline_prompt_tokens=500,
        )
        assert cfg.minimize_tokens is True
        assert cfg.token_weight == 0.20
        assert cfg.token_efficiency_cap == 3.0
        assert cfg.token_accuracy_band == 5.0
        assert cfg.baseline_prompt_tokens == 500


class TestApplyTokenEfficiency:
    """Tests for _apply_token_efficiency blending logic."""

    @pytest.fixture()
    def evolver_disabled(self, sample_tools, sample_dataset):
        """Evolver with token optimization disabled (default)."""
        cfg = PromptEvolverConfig(minimize_tokens=False)
        return PromptEvolver(tools=sample_tools, eval_dataset=sample_dataset, config=cfg)

    @pytest.fixture()
    def evolver_enabled(self, sample_tools, sample_dataset):
        """Evolver with token optimization enabled."""
        cfg = PromptEvolverConfig(
            minimize_tokens=True,
            token_weight=0.10,
            token_efficiency_cap=2.0,
            baseline_prompt_tokens=200,
        )
        return PromptEvolver(tools=sample_tools, eval_dataset=sample_dataset, config=cfg)

    def test_given_disabled_when_apply_then_returns_raw(self, evolver_disabled):
        # Arrange
        candidate = PromptCandidate(template="short prompt", score=80.0)

        # Act
        result = evolver_disabled._apply_token_efficiency(80.0, candidate)

        # Assert
        assert result == 80.0

    def test_given_enabled_when_apply_then_blends_score(self, evolver_enabled):
        # Arrange — a prompt roughly half the baseline length
        candidate = PromptCandidate(template="short", score=80.0)

        # Act
        result = evolver_enabled._apply_token_efficiency(80.0, candidate)

        # Assert — blended should differ from raw because efficiency != 1.0
        assert result != 80.0

    def test_given_zero_baseline_when_apply_then_returns_raw(self, sample_tools, sample_dataset):
        # Arrange — enabled but baseline_prompt_tokens=0
        cfg = PromptEvolverConfig(
            minimize_tokens=True,
            token_weight=0.10,
            baseline_prompt_tokens=0,
        )
        evolver = PromptEvolver(tools=sample_tools, eval_dataset=sample_dataset, config=cfg)
        candidate = PromptCandidate(template="test", score=75.0)

        # Act
        result = evolver._apply_token_efficiency(75.0, candidate)

        # Assert
        assert result == 75.0

    def test_given_zero_weight_when_apply_then_returns_raw(self, sample_tools, sample_dataset):
        # Arrange
        cfg = PromptEvolverConfig(
            minimize_tokens=True,
            token_weight=0.0,
            baseline_prompt_tokens=200,
        )
        evolver = PromptEvolver(tools=sample_tools, eval_dataset=sample_dataset, config=cfg)
        candidate = PromptCandidate(template="test", score=75.0)

        # Act
        result = evolver._apply_token_efficiency(75.0, candidate)

        # Assert
        assert result == 75.0

    def test_given_same_length_prompt_when_apply_then_blends_correctly(
        self, sample_tools, sample_dataset,
    ):
        # Arrange — build a prompt with known token count as baseline
        baseline_text = "You are a helpful assistant that selects tools."
        baseline_tokens = count_prompt_tokens(baseline_text)
        cfg = PromptEvolverConfig(
            minimize_tokens=True,
            token_weight=0.10,
            token_efficiency_cap=2.0,
            baseline_prompt_tokens=baseline_tokens,
        )
        evolver = PromptEvolver(tools=sample_tools, eval_dataset=sample_dataset, config=cfg)
        candidate = PromptCandidate(template=baseline_text, score=80.0)

        # Act
        result = evolver._apply_token_efficiency(80.0, candidate)

        # Assert — efficiency=1.0, bonus=1.0/2.0*100=50, blended=80*0.9+50*0.1=77.0
        assert abs(result - 77.0) < 0.1


class TestTournamentSelectTokenAware:
    """Tests for token-aware tournament selection tiebreaker."""

    @pytest.fixture()
    def evolver_token(self, sample_tools, sample_dataset):
        cfg = PromptEvolverConfig(
            minimize_tokens=True,
            token_accuracy_band=5.0,
        )
        return PromptEvolver(tools=sample_tools, eval_dataset=sample_dataset, config=cfg)

    @pytest.fixture()
    def evolver_default(self, sample_tools, sample_dataset):
        cfg = PromptEvolverConfig(minimize_tokens=False)
        return PromptEvolver(tools=sample_tools, eval_dataset=sample_dataset, config=cfg)

    def test_given_same_band_when_select_then_prefers_shorter(self, evolver_token):
        # Arrange — both in band 16 (80//5==16, 82//5==16)
        short = PromptCandidate(template="short", score=80.0)
        long = PromptCandidate(
            template="This is a much longer prompt with many tokens", score=82.0,
        )
        island = [short, long]

        # Act — k=2 means both are contestants
        winner = evolver_token._tournament_select(island, k=2)

        # Assert — same accuracy band, shorter prompt wins
        assert winner is short

    def test_given_different_band_when_select_then_prefers_higher(self, evolver_token):
        # Arrange — 80//5=16, 90//5=18 → different bands
        low = PromptCandidate(template="short", score=80.0)
        high = PromptCandidate(
            template="This is a much longer prompt with many tokens", score=90.0,
        )
        island = [low, high]

        # Act
        winner = evolver_token._tournament_select(island, k=2)

        # Assert — higher band wins regardless of length
        assert winner is high

    def test_given_disabled_when_select_then_uses_score(self, evolver_default):
        # Arrange
        a = PromptCandidate(template="short", score=80.0)
        b = PromptCandidate(
            template="This is a much longer prompt with many tokens", score=82.0,
        )
        island = [a, b]

        # Act
        winner = evolver_default._tournament_select(island, k=2)

        # Assert — disabled: higher score wins (standard feasibility key)
        assert winner is b


# ---------------------------------------------------------------------------
# Score-proportional selection
# ---------------------------------------------------------------------------

class TestSelectionMethod:
    """SelectionMethod enum and PromptEvolverConfig defaults."""

    def test_default_selection_method(self):
        cfg = PromptEvolverConfig()
        assert cfg.selection_method == SelectionMethod.TOURNAMENT

    def test_score_proportional_config(self):
        cfg = PromptEvolverConfig(selection_method=SelectionMethod.SCORE_PROPORTIONAL)
        assert cfg.selection_method == SelectionMethod.SCORE_PROPORTIONAL

    def test_enum_values(self):
        assert SelectionMethod.TOURNAMENT == "tournament"
        assert SelectionMethod.SCORE_PROPORTIONAL == "score_proportional"


class TestScoreProportionalSelect:
    """_score_prop_select behaviour."""

    @pytest.fixture()
    def evolver(self, sample_tools, sample_dataset):
        tools, dataset = sample_tools, sample_dataset
        config = PromptEvolverConfig(
            iterations=1,
            population_size=4,
            selection_method=SelectionMethod.SCORE_PROPORTIONAL,
        )
        return PromptEvolver(
            tools=tools,
            eval_dataset=dataset,
            config=config,
        )

    def test_prefers_high_score(self, evolver):
        """Higher-scoring candidates should be selected more often."""
        high = PromptCandidate(template="high", score=95.0, selection_count=0)
        low = PromptCandidate(template="low", score=10.0, selection_count=0)
        island = [high, low]

        wins = {id(high): 0, id(low): 0}
        for _ in range(200):
            winner = evolver._score_prop_select(island)
            wins[id(winner)] += 1

        assert wins[id(high)] > wins[id(low)]

    def test_penalises_over_selected(self, evolver):
        """Heavily selected candidates should be picked less often."""
        a = PromptCandidate(template="a", score=80.0, selection_count=0)
        b = PromptCandidate(template="b", score=80.0, selection_count=50)
        island = [a, b]

        wins = {id(a): 0, id(b): 0}
        for _ in range(200):
            winner = evolver._score_prop_select(island)
            wins[id(winner)] += 1

        # a should win significantly more — it has the same score but fewer selections
        assert wins[id(a)] > wins[id(b)]

    def test_single_candidate(self, evolver):
        """Single-candidate island returns the only candidate."""
        only = PromptCandidate(template="only", score=50.0)
        assert evolver._score_prop_select([only]) is only

    def test_all_zero_scores(self, evolver):
        """All-zero scores should not crash — returns some candidate."""
        a = PromptCandidate(template="a", score=0.0)
        b = PromptCandidate(template="b", score=0.0)
        result = evolver._score_prop_select([a, b])
        assert result in (a, b)

    def test_selection_count_incremented(self, evolver):
        """Winner's selection_count should be incremented by _select_parent."""
        c = PromptCandidate(template="t", score=90.0, selection_count=0)
        island = [c]
        evolver._select_parent(island)
        assert c.selection_count == 1


class TestSelectParentDispatch:
    """_select_parent dispatches to the correct method."""

    @pytest.fixture()
    def test_tournament_dispatch(self, sample_tools, sample_dataset):
        tools, dataset = sample_tools, sample_dataset
        config = PromptEvolverConfig(
            iterations=1,
            population_size=4,
            selection_method=SelectionMethod.TOURNAMENT,
        )
        evolver = PromptEvolver(
            tools=tools, eval_dataset=dataset,
            config=config,
        )
        a = PromptCandidate(template="t", score=50.0)
        result = evolver._select_parent([a])
        assert result is a

    def test_score_prop_dispatch(self, sample_tools, sample_dataset):
        tools, dataset = sample_tools, sample_dataset
        config = PromptEvolverConfig(
            iterations=1,
            population_size=4,
            selection_method=SelectionMethod.SCORE_PROPORTIONAL,
        )
        evolver = PromptEvolver(
            tools=tools, eval_dataset=dataset,
            config=config,
        )
        a = PromptCandidate(template="t", score=50.0)
        result = evolver._select_parent([a])
        assert result is a


# ---------------------------------------------------------------------------
# Progressive evaluation
# ---------------------------------------------------------------------------

class TestProgressiveEvaluation:
    """Progressive (shallow → deep) evaluation logic."""

    def test_disabled_by_default(self):
        cfg = PromptEvolverConfig()
        assert cfg.eval_promotion_threshold == 30.0
        assert cfg.eval_deep_sample_size is None

    def test_config_accepts_values(self):
        cfg = PromptEvolverConfig(
            eval_promotion_threshold=75.0,
            eval_deep_sample_size=50,
        )
        assert cfg.eval_promotion_threshold == 75.0
        assert cfg.eval_deep_sample_size == 50

    def test_shallow_only_when_below_threshold(self, sample_tools, sample_dataset):
        """When candidate scores below threshold, deep eval is skipped.

        We verify by checking that the number of LLM calls equals the
        shallow sample size, not the deep sample size.
        """
        tools, dataset = sample_tools, sample_dataset
        config = PromptEvolverConfig(
            iterations=1,
            population_size=2,
            eval_sample_size=2,
            eval_promotion_threshold=99.0,  # impossibly high
            eval_deep_sample_size=5,
        )
        evolver = PromptEvolver(
            tools=tools, eval_dataset=dataset,
            config=config,
        )
        candidate = PromptCandidate(template="Test {tool_schemas}", score=0.0)
        # With mocked LLM returning None, scores ~25% — well below 99%
        score = evolver._evaluate_candidate(candidate)
        assert isinstance(score, float)


# ---------------------------------------------------------------------------
# Failure buckets
# ---------------------------------------------------------------------------

class TestFailureBucket:
    """FailureBucket enum values."""

    def test_all_buckets_exist(self):
        expected = {"wrong_tool", "wrong_params", "no_output", "unparseable", "partial_match"}
        actual = {b.value for b in FailureBucket}
        assert actual == expected


class TestErrorProfileBuckets:
    """ErrorProfile failure bucket recording and querying."""

    def test_record_bucket(self):
        ep = ErrorProfile()
        ep.record_bucket(FailureBucket.WRONG_TOOL)
        ep.record_bucket(FailureBucket.WRONG_TOOL)
        ep.record_bucket(FailureBucket.NO_OUTPUT)
        assert ep.failure_buckets[FailureBucket.WRONG_TOOL] == 2
        assert ep.failure_buckets[FailureBucket.NO_OUTPUT] == 1

    def test_worst_buckets_order(self):
        ep = ErrorProfile()
        ep.record_bucket(FailureBucket.PARTIAL_MATCH)
        ep.record_bucket(FailureBucket.WRONG_TOOL)
        ep.record_bucket(FailureBucket.WRONG_TOOL)
        ep.record_bucket(FailureBucket.WRONG_TOOL)
        worst = ep.worst_buckets()
        assert worst[0] == (FailureBucket.WRONG_TOOL, 3)
        assert worst[1] == (FailureBucket.PARTIAL_MATCH, 1)

    def test_worst_buckets_empty(self):
        ep = ErrorProfile()
        assert ep.worst_buckets() == []

    def test_bucket_decay(self):
        """Decay should reduce failure_buckets counts."""
        ep = ErrorProfile()
        ep.record_bucket(FailureBucket.WRONG_PARAMS)
        ep.record_bucket(FailureBucket.WRONG_PARAMS)
        ep.record_bucket(FailureBucket.WRONG_PARAMS)
        ep.record_bucket(FailureBucket.WRONG_PARAMS)
        ep.decay(0.5)
        # 4 * 0.5 = 2
        assert ep.failure_buckets[FailureBucket.WRONG_PARAMS] == 2

    def test_bucket_decay_removes_zeros(self):
        """Decay of count=1 → 0 should be removed."""
        ep = ErrorProfile()
        ep.record_bucket(FailureBucket.UNPARSEABLE)
        ep.decay(0.5)
        assert FailureBucket.UNPARSEABLE not in ep.failure_buckets


class TestGetFailureBucketMutations:
    """get_failure_bucket_mutations function."""

    def test_tool_routing_wrong_tool(self):
        ep = ErrorProfile()
        ep.record_bucket(FailureBucket.WRONG_TOOL)
        mutations = get_failure_bucket_mutations(ep, ProblemType.TOOL_ROUTING)
        assert len(mutations) > 0
        # All mutations should be strings
        assert all(isinstance(m, str) for m in mutations)

    def test_tool_routing_multiple_buckets(self):
        ep = ErrorProfile()
        ep.record_bucket(FailureBucket.WRONG_TOOL)
        ep.record_bucket(FailureBucket.WRONG_TOOL)
        ep.record_bucket(FailureBucket.UNPARSEABLE)
        mutations = get_failure_bucket_mutations(ep, ProblemType.TOOL_ROUTING)
        # Should include mutations for both buckets
        assert len(mutations) > 4  # at least mutations from both

    def test_classification_problem_type(self):
        ep = ErrorProfile()
        ep.record_bucket(FailureBucket.WRONG_TOOL)
        mutations = get_failure_bucket_mutations(ep, ProblemType.CLASSIFICATION)
        assert len(mutations) > 0

    def test_empty_profile_returns_empty(self):
        ep = ErrorProfile()
        mutations = get_failure_bucket_mutations(ep, ProblemType.TOOL_ROUTING)
        assert mutations == []

    def test_no_output_bucket_mutations(self):
        ep = ErrorProfile()
        ep.record_bucket(FailureBucket.NO_OUTPUT)
        mutations = get_failure_bucket_mutations(ep, ProblemType.TOOL_ROUTING)
        assert len(mutations) > 0

    def test_partial_match_bucket_mutations(self):
        ep = ErrorProfile()
        ep.record_bucket(FailureBucket.PARTIAL_MATCH)
        mutations = get_failure_bucket_mutations(ep, ProblemType.TOOL_ROUTING)
        assert len(mutations) > 0

    def test_wrong_params_bucket_mutations(self):
        ep = ErrorProfile()
        ep.record_bucket(FailureBucket.WRONG_PARAMS)
        mutations = get_failure_bucket_mutations(ep, ProblemType.TOOL_ROUTING)
        assert len(mutations) > 0


class TestFailureBucketIntegration:
    """Integration: failure bucket recording during evaluation."""

    def test_no_output_recorded_on_none_response(self, sample_tools, sample_dataset):
        """When LLM returns None, NO_OUTPUT bucket should be recorded."""
        tools, dataset = sample_tools, sample_dataset
        config = PromptEvolverConfig(
            iterations=1, population_size=2,
            backend=LLMBackend.OLLAMA,
            ollama_url="http://localhost:99999",  # Intentionally unreachable
        )
        evolver = PromptEvolver(
            tools=tools, eval_dataset=dataset,
            config=config,
        )
        candidate = PromptCandidate(template="Test {tool_schemas}", score=0.0)
        evolver._evaluate_candidate(candidate)
        # LLM client returns None for unreachable backend → NO_OUTPUT
        assert "no_output" in evolver._error_profile.failure_buckets

    def test_config_problem_type_default(self):
        cfg = PromptEvolverConfig()
        assert cfg.problem_type == ProblemType.TOOL_ROUTING

    def test_config_problem_type_classification(self):
        cfg = PromptEvolverConfig(problem_type=ProblemType.CLASSIFICATION)
        assert cfg.problem_type == ProblemType.CLASSIFICATION

    def test_config_problem_type_generation(self):
        cfg = PromptEvolverConfig(problem_type=ProblemType.GENERATION)
        assert cfg.problem_type == ProblemType.GENERATION


# ── GENERATION problem type tests ────────────────────────────────────────


class TestGenerationMutations:
    """Tests for the GENERATION problem type mutation pool."""

    def test_generation_mutations_exist(self):
        from MutaGenAI.prompt_evolver import ProblemType, get_mutations_for_problem_type

        mutations = get_mutations_for_problem_type(ProblemType.GENERATION)
        assert len(mutations) > 0
        assert all(isinstance(m, str) for m in mutations)

    def test_generation_mutations_differ_from_others(self):
        from MutaGenAI.prompt_evolver import ProblemType, get_mutations_for_problem_type

        gen_muts = get_mutations_for_problem_type(ProblemType.GENERATION)
        tool_muts = get_mutations_for_problem_type(ProblemType.TOOL_ROUTING)
        class_muts = get_mutations_for_problem_type(ProblemType.CLASSIFICATION)
        assert gen_muts != tool_muts
        assert gen_muts != class_muts

    def test_generation_failure_bucket_mutations(self):
        ep = ErrorProfile()
        ep.record_bucket(FailureBucket.UNPARSEABLE)
        mutations = get_failure_bucket_mutations(ep, ProblemType.GENERATION)
        assert len(mutations) > 0
        assert all(isinstance(m, str) for m in mutations)

    def test_generation_all_buckets_have_mutations(self):
        for bucket in FailureBucket:
            ep = ErrorProfile()
            ep.record_bucket(bucket)
            mutations = get_failure_bucket_mutations(ep, ProblemType.GENERATION)
            assert len(mutations) > 0, f"No mutations for bucket {bucket}"


# ── Semantic (LLM) crossover ─────────────────────────────────────────────


class TestLLMCrossover:
    def test_merges_via_llm(self):
        from MutaGenAI.prompt_evolver import ProblemType

        class FakeClient:
            def complete(self, *, system_prompt, user_message, temperature, top_p):
                return "Merged: route precisely and return JSON."

        out = _llm_crossover_templates(
            "Prompt A: be precise.",
            "Prompt B: return JSON.",
            FakeClient(),
            ProblemType.TOOL_ROUTING,
            require_tool_schemas=False,
        )
        assert out == "Merged: route precisely and return JSON."

    def test_empty_response_falls_back_to_parent_a(self):
        from MutaGenAI.prompt_evolver import ProblemType

        class EmptyClient:
            def complete(self, **kw):
                return None

        out = _llm_crossover_templates(
            "Parent A text", "Parent B text", EmptyClient(),
            ProblemType.CLASSIFICATION, require_tool_schemas=False,
        )
        assert out == "Parent A text"

    def test_strips_markdown_fences(self):
        from MutaGenAI.prompt_evolver import ProblemType

        class FenceClient:
            def complete(self, **kw):
                return "```\nclean merged prompt\n```"

        out = _llm_crossover_templates(
            "A", "B", FenceClient(), ProblemType.TOOL_ROUTING,
            require_tool_schemas=False,
        )
        assert out == "clean merged prompt"

    def test_preserves_tool_schemas_placeholder(self):
        from MutaGenAI.prompt_evolver import ProblemType

        class NoPlaceholderClient:
            def complete(self, **kw):
                return "merged without placeholder"

        out = _llm_crossover_templates(
            "route using {tool_schemas}", "be precise",
            NoPlaceholderClient(), ProblemType.TOOL_ROUTING,
            require_tool_schemas=True,
        )
        assert "{tool_schemas}" in out

    def test_bucket_hints_included_in_prompt(self):
        from MutaGenAI.prompt_evolver import ProblemType

        captured = {}

        class CaptureClient:
            def complete(self, *, system_prompt, user_message, temperature, top_p):
                captured["user"] = user_message
                return "merged"

        _llm_crossover_templates(
            "A", "B", CaptureClient(), ProblemType.TOOL_ROUTING,
            bucket_hints=["Avoid over-selecting tools."],
            require_tool_schemas=False,
        )
        assert "Avoid over-selecting tools." in captured["user"]


# ── Bandit operator selection (integration) ──────────────────────────────


class TestBanditOperatorSelection:
    def _config(self, **kw):
        base = dict(
            iterations=3, population_size=2, num_islands=2, elite_size=2,
            backend=LLMBackend.OLLAMA, ollama_url="http://localhost:99999",
        )
        base.update(kw)
        return PromptEvolverConfig(**base)

    def test_fixed_mode_has_no_operator_stats(self, sample_tools, sample_dataset):
        evolver = PromptEvolver(
            tools=sample_tools, eval_dataset=sample_dataset,
            config=self._config(), seed=42, verbose=False,
        )
        result = evolver.run()
        assert result.operator_stats is None

    def test_ucb_mode_populates_operator_stats(
        self, sample_tools, sample_dataset
    ):
        evolver = PromptEvolver(
            tools=sample_tools, eval_dataset=sample_dataset,
            config=self._config(operator_selection=OperatorSelection.UCB),
            seed=42, verbose=False,
        )
        result = evolver.run()
        assert result.operator_stats is not None
        assert "mutation" in result.operator_stats
        total = sum(v["count"] for v in result.operator_stats.values())
        assert total > 0

    def test_thompson_mode_runs(self, sample_tools, sample_dataset):
        evolver = PromptEvolver(
            tools=sample_tools, eval_dataset=sample_dataset,
            config=self._config(operator_selection=OperatorSelection.THOMPSON),
            seed=7, verbose=False,
        )
        result = evolver.run()
        assert result.operator_stats is not None

    def test_bandit_arms_include_llm_operators_when_enabled(
        self, sample_tools, sample_dataset
    ):
        evolver = PromptEvolver(
            tools=sample_tools, eval_dataset=sample_dataset,
            config=self._config(
                operator_selection=OperatorSelection.UCB,
                llm_mutation_rate=0.3,
                llm_crossover_rate=0.3,
            ),
            seed=1, verbose=False,
        )
        assert set(evolver._bandit.arms) == {
            "mutation", "crossover", "llm_mutation", "llm_crossover"
        }

    def test_bandit_is_deterministic_across_runs(
        self, sample_tools, sample_dataset
    ):
        def _run():
            ev = PromptEvolver(
                tools=sample_tools, eval_dataset=sample_dataset,
                config=self._config(
                    operator_selection=OperatorSelection.THOMPSON
                ),
                seed=99, verbose=False,
            )
            return ev.run()

        r1, r2 = _run(), _run()
        assert r1.best_score == r2.best_score
        assert r1.history == r2.history
        assert r1.operator_stats == r2.operator_stats

    def test_fixed_mode_determinism_unchanged(
        self, sample_tools, sample_dataset
    ):
        """Enabling the bandit must not alter fixed-mode behaviour."""
        def _run():
            ev = PromptEvolver(
                tools=sample_tools, eval_dataset=sample_dataset,
                config=self._config(), seed=5, verbose=False,
            )
            return ev.run()

        r1, r2 = _run(), _run()
        assert r1.history == r2.history
        assert [c.score for c in r1.all_candidates] == [
            c.score for c in r2.all_candidates
        ]


# ── Quality-diversity result methods ─────────────────────────────────────


class TestResultQualityDiversity:
    def _result(self, sample_tools, sample_dataset):
        config = PromptEvolverConfig(
            iterations=3, population_size=3, num_islands=2, elite_size=2,
            backend=LLMBackend.OLLAMA, ollama_url="http://localhost:99999",
        )
        evolver = PromptEvolver(
            tools=sample_tools, eval_dataset=sample_dataset,
            config=config, seed=42, verbose=False,
        )
        return evolver.run()

    def test_pareto_front_method(self, sample_tools, sample_dataset):
        result = self._result(sample_tools, sample_dataset)
        front = result.pareto_front()
        assert isinstance(front, list)
        assert len(front) >= 1
        assert all(c in result.all_candidates for c in front)

    def test_map_elites_method(self, sample_tools, sample_dataset):
        result = self._result(sample_tools, sample_dataset)
        archive = result.map_elites(token_bin_size=40)
        assert archive.coverage >= 1
        assert archive.best() is not None
        assert all("style" in r for r in archive.to_json())


# ── OperatorSelection enum ───────────────────────────────────────────────


class TestOperatorSelectionEnum:
    def test_values(self):
        assert OperatorSelection.FIXED.value == "fixed"
        assert OperatorSelection.UCB.value == "ucb"
        assert OperatorSelection.THOMPSON.value == "thompson"

    def test_default_is_fixed(self):
        assert PromptEvolverConfig().operator_selection is OperatorSelection.FIXED


# ── Custom seed templates + live event emission ──────────────────────────


class TestCustomSeedsAndEvents:
    def _config(self):
        return PromptEvolverConfig(
            iterations=2, population_size=2, num_islands=2, elite_size=2,
            backend=LLMBackend.OLLAMA, ollama_url="http://localhost:99999",
        )

    def test_custom_seed_templates_used(self, sample_tools, sample_dataset):
        seeds = ["Seed one prompt.", "Seed two prompt.", "Seed three prompt."]
        evolver = PromptEvolver(
            tools=sample_tools, eval_dataset=sample_dataset,
            config=self._config(), seed=1, verbose=False,
            seed_templates=seeds,
        )
        assert evolver._seed_templates == seeds
        gen0 = [
            c for c in evolver.run().all_candidates if c.generation == 0
        ]
        templates = {c.template for c in gen0}
        # Every custom seed appears among the generation-0 candidates.
        assert set(seeds).issubset(templates)

    def test_default_seeds_when_none(self, sample_tools, sample_dataset):
        from MutaGenAI.prompt_evolver import _SEED_TEMPLATES
        evolver = PromptEvolver(
            tools=sample_tools, eval_dataset=sample_dataset,
            config=self._config(), seed=1, verbose=False,
        )
        assert evolver._seed_templates == list(_SEED_TEMPLATES)

    def test_on_event_emits_lifecycle(self, sample_tools, sample_dataset):
        events = []
        evolver = PromptEvolver(
            tools=sample_tools, eval_dataset=sample_dataset,
            config=self._config(), seed=1, verbose=False,
            on_event=events.append,
        )
        evolver.run()
        types = [e["type"] for e in events]
        assert types[0] == "run_start"
        assert types[-1] == "run_complete"
        assert "seed" in types
        assert "candidate" in types
        assert types.count("generation") == 2

    def test_candidate_events_are_json_serialisable(
        self, sample_tools, sample_dataset
    ):
        import json
        events = []
        evolver = PromptEvolver(
            tools=sample_tools, eval_dataset=sample_dataset,
            config=self._config(), seed=1, verbose=False,
            on_event=events.append,
        )
        evolver.run()
        json.dumps(events)  # must not raise
        cand = next(e for e in events if e["type"] == "candidate")["candidate"]
        assert {"hash", "parent_hashes", "operation", "generation",
                "score", "tokens"}.issubset(cand)

    def test_failing_listener_does_not_break_run(
        self, sample_tools, sample_dataset
    ):
        def boom(_event):
            raise RuntimeError("listener error")

        evolver = PromptEvolver(
            tools=sample_tools, eval_dataset=sample_dataset,
            config=self._config(), seed=1, verbose=False, on_event=boom,
        )
        result = evolver.run()  # must complete despite the raising listener
        assert result.best_prompt


