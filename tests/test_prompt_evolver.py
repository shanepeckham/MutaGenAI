"""Tests for MutaGenAI.prompt_evolver — prompt evolution engine."""
from __future__ import annotations


import numpy as np
import pytest

from MutaGenAI.prompt_evolver import (
    EvalSample,
    LLMBackend,
    LLMClient,
    PromptCandidate,
    PromptEvolver,
    PromptEvolverConfig,
    PromptEvolverResult,
    Tool,
    _crossover_templates,
    _mutate_template,
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


from MutaGenAI.prompt_evolver import count_prompt_tokens


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
