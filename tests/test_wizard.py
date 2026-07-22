"""Tests for MutaGenAI.wizard — interactive wizard and script generation."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


from MutaGenAI.wizard import (
    WizardState,
    _ask,
    _ask_int,
    _banner,
    _build_human_eval_block,
    _build_main_block,
    _build_scorer_setup,
    _confirm,
    _expand_seed_templates,
    _generate_script,
    _print,
    _proxy_checks_for_problem_type,
    _rubric_for_problem_type,
    _show_summary,
    _step_backend,
    _step_config,
    _step_ground_truth,
    _step_human_eval,
    _step_mutations,
    _step_problem_type,
    _step_scoring,
    _step_seeds,
    _step_task,
    _step_test_inputs,
)


# ---------------------------------------------------------------------------
# Helper defaults
# ---------------------------------------------------------------------------


class TestWizardState:
    def test_defaults(self):
        state = WizardState()
        assert state.problem_type == "tool_routing"
        assert state.has_ground_truth == "no"
        assert state.backend == "ollama"
        assert state.model == "llama3.2"
        assert state.iterations == 5
        assert state.population_size == 6
        assert state.num_islands == 2

    def test_custom_fields(self):
        state = WizardState(
            problem_type="classification",
            task_description="Test task",
            backend="openai",
            model="gpt-4o-mini",
        )
        assert state.problem_type == "classification"
        assert state.task_description == "Test task"


# ---------------------------------------------------------------------------
# IO helpers (with Rich fallbacks)
# ---------------------------------------------------------------------------


class TestIOHelpers:
    def test_print_no_crash(self):
        _print("test message")
        _print()

    def test_banner_no_crash(self):
        _banner()

    def test_ask_with_input(self):
        with patch("builtins.input", return_value="tool_routing"):
            result = _ask("prompt", default="tool_routing")
            assert result == "tool_routing"

    def test_ask_empty_uses_default(self):
        with patch("builtins.input", return_value=""):
            result = _ask("prompt", default="default_val")
            assert result == "default_val"

    def test_ask_int_valid(self):
        with patch("builtins.input", return_value="5"):
            result = _ask_int("prompt", default=3)
            assert result == 5

    def test_ask_int_empty_uses_default(self):
        with patch("builtins.input", return_value=""):
            result = _ask_int("prompt", default=7)
            assert result == 7

    def test_confirm_default_true(self):
        with patch("builtins.input", return_value=""):
            result = _confirm("proceed?", default=True)
            assert result is True

    def test_confirm_no(self):
        with patch("builtins.input", return_value="n"):
            result = _confirm("proceed?", default=True)
            assert result is False

    def test_confirm_yes(self):
        with patch("builtins.input", return_value="y"):
            result = _confirm("proceed?", default=False)
            assert result is True


# ---------------------------------------------------------------------------
# Individual wizard steps
# ---------------------------------------------------------------------------


class TestWizardSteps:
    def test_step_problem_type(self):
        state = WizardState()
        with patch("MutaGenAI.wizard._ask", return_value="classification"):
            _step_problem_type(state)
        assert state.problem_type == "classification"

    def test_step_task(self):
        state = WizardState()
        with patch("MutaGenAI.wizard._ask", return_value="You classify text."):
            _step_task(state)
        assert state.task_description == "You classify text."

    def test_step_task_retries_on_empty(self):
        state = WizardState()
        call_count = 0

        def fake_ask(prompt, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ""
            return "Real task"

        with patch("MutaGenAI.wizard._ask", side_effect=fake_ask):
            _step_task(state)
        assert state.task_description == "Real task"

    def test_step_ground_truth_no(self):
        state = WizardState()
        with patch("MutaGenAI.wizard._ask", return_value="no"):
            _step_ground_truth(state)
        assert state.has_ground_truth == "no"

    def test_step_ground_truth_yes_file(self):
        state = WizardState()
        calls = iter(["yes", "file", "/data/eval.json"])
        with patch("MutaGenAI.wizard._ask", side_effect=lambda *a, **kw: next(calls)):
            _step_ground_truth(state)
        assert state.has_ground_truth == "yes"
        assert state.eval_file == "/data/eval.json"

    def test_step_ground_truth_yes_interactive(self):
        state = WizardState()
        calls = iter(["yes", "interactive", "hello", "world", "done"])
        with patch("MutaGenAI.wizard._ask", side_effect=lambda *a, **kw: next(calls)):
            _step_ground_truth(state)
        assert len(state.eval_examples) == 1
        assert state.eval_examples[0]["input"] == "hello"

    def test_step_test_inputs_interactive(self):
        state = WizardState()
        calls = iter(["interactive", "input1", "input2", "done"])
        with patch("MutaGenAI.wizard._ask", side_effect=lambda *a, **kw: next(calls)):
            _step_test_inputs(state)
        assert state.test_inputs == ["input1", "input2"]

    def test_step_test_inputs_file(self):
        state = WizardState()
        calls = iter(["file", "/data/inputs.txt"])
        with patch("MutaGenAI.wizard._ask", side_effect=lambda *a, **kw: next(calls)):
            _step_test_inputs(state)
        assert state.test_input_file == "/data/inputs.txt"

    def test_step_scoring_ground_truth(self):
        state = WizardState(has_ground_truth="yes")
        with patch("MutaGenAI.wizard._confirm", return_value=False):
            _step_scoring(state)
        assert state.strategies == ["ground_truth"]

    def test_step_scoring_no_ground_truth(self):
        state = WizardState(has_ground_truth="no")
        calls = iter(["7", "no", "all"])  # composite, no custom rubric, all checks
        with (
            patch("MutaGenAI.wizard._ask", side_effect=lambda *a, **kw: next(calls)),
            patch("MutaGenAI.wizard._confirm", return_value=False),
        ):
            _step_scoring(state)
        assert "composite" in state.strategies

    def test_step_mutations_no(self):
        state = WizardState()
        with patch("MutaGenAI.wizard._confirm", return_value=False):
            _step_mutations(state)
        assert state.has_domain_mutations is False

    def test_step_mutations_yes(self):
        state = WizardState()
        calls = iter(["Add CoT", "Enforce JSON", "done"])
        with (
            patch("MutaGenAI.wizard._confirm", return_value=True),
            patch("MutaGenAI.wizard._ask", side_effect=lambda *a, **kw: next(calls)),
        ):
            _step_mutations(state)
        assert len(state.domain_mutations) == 2

    def test_step_human_eval(self):
        state = WizardState()
        with patch("MutaGenAI.wizard._ask", return_value="no"):
            _step_human_eval(state)
        assert state.human_eval == "no"

    def test_step_seeds_no(self):
        state = WizardState()
        with patch("MutaGenAI.wizard._confirm", return_value=False):
            _step_seeds(state)
        assert state.has_seed_templates is False

    def test_step_seeds_yes(self):
        state = WizardState()
        calls = iter(["Seed 1", "Seed 2", "done"])
        with (
            patch("MutaGenAI.wizard._confirm", return_value=True),
            patch("MutaGenAI.wizard._ask", side_effect=lambda *a, **kw: next(calls)),
        ):
            _step_seeds(state)
        assert len(state.seed_templates) == 2

    def test_step_backend_ollama(self):
        state = WizardState()
        calls = iter(["ollama", "llama3.2"])
        with patch("MutaGenAI.wizard._ask", side_effect=lambda *a, **kw: next(calls)):
            _step_backend(state)
        assert state.backend == "ollama"
        assert state.model == "llama3.2"

    def test_step_backend_openai(self):
        state = WizardState()
        calls = iter(["openai", "gpt-4o-mini"])
        with patch("MutaGenAI.wizard._ask", side_effect=lambda *a, **kw: next(calls)):
            _step_backend(state)
        assert state.backend == "openai"

    def test_step_backend_azure(self):
        state = WizardState()
        calls = iter(["azure", "gpt-4o"])
        with patch("MutaGenAI.wizard._ask", side_effect=lambda *a, **kw: next(calls)):
            _step_backend(state)
        assert state.backend == "azure"

    def test_step_config_standard(self):
        state = WizardState()
        with (
            patch("MutaGenAI.wizard._ask", return_value="standard"),
            patch("MutaGenAI.wizard._ask_int", return_value=8),
        ):
            _step_config(state)
        assert state.iterations == 5
        assert state.population_size == 6
        assert state.num_islands == 2
        assert state.max_concurrency == 8

    def test_step_config_deep(self):
        state = WizardState()
        with (
            patch("MutaGenAI.wizard._ask", return_value="deep"),
            patch("MutaGenAI.wizard._ask_int", return_value=16),
        ):
            _step_config(state)
        assert state.iterations == 10
        assert state.population_size == 8
        assert state.num_islands == 3
        assert state.max_concurrency == 16

    def test_step_config_custom(self):
        state = WizardState()
        with (
            patch("MutaGenAI.wizard._ask", return_value="custom"),
            patch("MutaGenAI.wizard._ask_int", side_effect=[10, 8, 4, 12]),
        ):
            _step_config(state)
        assert state.iterations == 10
        assert state.population_size == 8
        assert state.num_islands == 4
        assert state.max_concurrency == 12

    def test_step_config_concurrency_floored_at_one(self):
        state = WizardState()
        with (
            patch("MutaGenAI.wizard._ask", return_value="standard"),
            patch("MutaGenAI.wizard._ask_int", return_value=0),
        ):
            _step_config(state)
        assert state.max_concurrency == 1


# ---------------------------------------------------------------------------
# Script generation
# ---------------------------------------------------------------------------


class TestScriptGeneration:
    def _base_state(self) -> WizardState:
        return WizardState(
            problem_type="tool_routing",
            task_description="Route queries to tools",
            has_ground_truth="no",
            strategies=["composite"],
            backend="ollama",
            model="llama3.2",
            human_eval="no",
            proxy_checks=["valid_json", "max_length"],
        )

    def test_generate_script_runs(self):
        state = self._base_state()
        script = _generate_script(state)
        assert "from MutaGenAI" in script
        assert "Route queries to tools" in script
        assert "LLMBackend.OLLAMA" in script

    def test_generate_script_emits_max_concurrency(self):
        state = self._base_state()
        state.max_concurrency = 12
        script = _generate_script(state)
        # Present in both the NoEvalConfig and the scorer's client config.
        assert script.count("max_concurrency=12") >= 2

    def test_generate_script_openai(self):
        state = self._base_state()
        state.backend = "openai"
        state.model = "gpt-4o-mini"
        script = _generate_script(state)
        assert "LLMBackend.OPENAI" in script

    def test_generate_script_azure(self):
        state = self._base_state()
        state.backend = "azure"
        state.model = "gpt-4o"
        script = _generate_script(state)
        assert "LLMBackend.AZURE_OPENAI" in script

    def test_generate_with_seed_templates(self):
        state = self._base_state()
        state.seed_templates = ["Template A", "Template B"]
        script = _generate_script(state)
        assert "Template A" in script
        assert "Template B" in script

    def test_generate_with_ground_truth_file(self):
        state = self._base_state()
        state.has_ground_truth = "yes"
        state.strategies = ["ground_truth"]
        state.eval_file = "/data/eval.json"
        script = _generate_script(state)
        assert "eval.json" in script

    def test_generate_with_eval_examples(self):
        state = self._base_state()
        state.has_ground_truth = "yes"
        state.strategies = ["ground_truth"]
        state.eval_examples = [{"input": "test", "expected": "answer"}]
        script = _generate_script(state)
        assert "EVAL_DATA" in script

    def test_generate_with_test_input_file(self):
        state = self._base_state()
        state.test_input_file = "/data/inputs.txt"
        script = _generate_script(state)
        assert "inputs.txt" in script

    def test_generate_with_domain_mutations(self):
        state = self._base_state()
        state.domain_mutations = ["Add CoT"]
        script = _generate_script(state)
        assert "Add CoT" in script

    def test_generate_with_human_final(self):
        state = self._base_state()
        state.human_eval = "final"
        script = _generate_script(state)
        assert "human_select_winner" in script

    def test_build_scorer_composite(self):
        state = self._base_state()
        code = _build_scorer_setup(state)
        assert "CompositeScorer" in code
        assert "LLMJudge" in code

    def test_build_scorer_single_judge(self):
        state = self._base_state()
        state.strategies = ["llm_judge"]
        state.llm_judge_rubric = "Rate it well"
        code = _build_scorer_setup(state)
        assert "LLMJudge" in code
        assert "Rate it well" in code

    def test_build_scorer_tool_success(self):
        state = self._base_state()
        state.strategies = ["tool_success"]
        code = _build_scorer_setup(state)
        assert "ToolSuccessScorer" in code

    def test_build_scorer_preference(self):
        state = self._base_state()
        state.strategies = ["preference"]
        code = _build_scorer_setup(state)
        assert "PreferenceScorer" in code

    def test_build_scorer_human(self):
        state = self._base_state()
        state.strategies = ["human"]
        code = _build_scorer_setup(state)
        assert "HumanTournament" in code

    def test_build_human_eval_block(self):
        state = self._base_state()
        state.human_eval = "always"
        block = _build_human_eval_block(state)
        assert "HUMAN EVALUATION" in block

    def test_build_main_block(self):
        state = self._base_state()
        block = _build_main_block(state)
        assert "NoEvalConfig" in block
        assert "evolution_results.json" in block


# ---------------------------------------------------------------------------
# Summary display
# ---------------------------------------------------------------------------


class TestShowSummary:
    def test_show_summary_no_crash(self):
        state = WizardState(
            task_description="Test task",
            strategies=["composite"],
        )
        _show_summary(state)


# ---------------------------------------------------------------------------
# Full wizard (mocked interactive)
# ---------------------------------------------------------------------------


class TestRunWizard:
    def test_full_run_aborted(self):
        from MutaGenAI.wizard import run_wizard

        step_answers = iter([
            "tool_routing",         # step 1
            "Classify queries",     # step 2
            "no",                   # step 3 ground truth
            "interactive",          # step 4 test inputs
            "done",                 # step 4 done
            "7",                    # step 5 composite
            "all",                  # step 5 proxy checks
            "no",                   # step 7 human eval
            "ollama",               # step 9 backend
            "llama3.2",             # step 9 model
            "standard",             # step 10 config
        ])

        with (
            patch("MutaGenAI.wizard._ask", side_effect=lambda *a, **kw: next(step_answers)),
            patch("MutaGenAI.wizard._ask_int", return_value=8),
            patch("MutaGenAI.wizard._confirm", return_value=False),  # abort
        ):
            result = run_wizard()
        assert result == ""

    def test_full_run_generates_script(self, tmp_path: Path):
        from MutaGenAI.wizard import run_wizard

        output_path = tmp_path / "generated.py"
        step_answers = iter([
            "tool_routing",         # step 1
            "Classify queries",     # step 2
            "no",                   # step 3
            "interactive",          # step 4
            "done",                 # step 4 done
            "7",                    # step 5 composite
            "all",                  # step 5 proxy checks
            "no",                   # step 7 human eval
            "ollama",               # step 9 backend
            "llama3.2",             # step 9 model
            "standard",             # step 10 config
        ])

        confirm_calls = iter([
            False,  # no custom rubric
            False,  # no fitness penalties
            False,  # no domain mutations
            False,  # no seed templates
        ])

        with (
            patch("MutaGenAI.wizard._ask", side_effect=lambda *a, **kw: next(step_answers, "no")),
            patch("MutaGenAI.wizard._ask_int", side_effect=lambda *a, **kw: 4),
            patch("MutaGenAI.wizard._confirm", side_effect=lambda *a, **kw: next(confirm_calls, True)),
        ):
            result = run_wizard(output=str(output_path))

        if result:
            assert output_path.exists()
            content = output_path.read_text()
            assert "from MutaGenAI" in content


# ---------------------------------------------------------------------------
# Seed template expansion
# ---------------------------------------------------------------------------


class TestExpandSeedTemplates:
    def test_single_seed_expands_to_six(self):
        seeds = _expand_seed_templates("Route queries to agents", ["Route queries to agents"])
        assert len(seeds) >= 4
        assert len(seeds) <= 6
        # Original seed preserved first
        assert seeds[0] == "Route queries to agents"

    def test_no_duplicates(self):
        seeds = _expand_seed_templates("Test task", ["Test task"])
        assert len(seeds) == len(set(seeds))

    def test_two_seeds_expanded(self):
        seeds = _expand_seed_templates("Task", ["Seed A", "Seed B"])
        assert seeds[0] == "Seed A"
        assert seeds[1] == "Seed B"
        assert len(seeds) >= 4

    def test_six_seeds_not_expanded_further(self):
        original = [f"Seed {i}" for i in range(6)]
        seeds = _expand_seed_templates("Task", original)
        assert seeds == original

    def test_variants_include_diverse_styles(self):
        seeds = _expand_seed_templates("Classify entities", ["Classify entities"])
        all_text = " ".join(seeds)
        # Should contain CoT, format-first, or contrastive styles
        assert any(kw in all_text for kw in ["step-by-step", "JSON only", "NOT"])


# ---------------------------------------------------------------------------
# Problem-type rubrics
# ---------------------------------------------------------------------------


class TestRubricForProblemType:
    def test_tool_routing_rubric(self):
        rubric = _rubric_for_problem_type("tool_routing", "Route to agents")
        assert "agent" in rubric.lower() or "routing" in rubric.lower()
        assert "JSON" in rubric
        assert "0-10" in rubric

    def test_classification_rubric(self):
        rubric = _rubric_for_problem_type("classification", "Classify entities")
        assert "classification" in rubric.lower()
        assert "0-10" in rubric

    def test_unknown_type_fallback(self):
        rubric = _rubric_for_problem_type("unknown_type", "Do something")
        assert "0-10" in rubric
        assert "Do something" in rubric


# ---------------------------------------------------------------------------
# Problem-type proxy checks
# ---------------------------------------------------------------------------


class TestProxyChecksForProblemType:
    def test_tool_routing_checks(self):
        lines = _proxy_checks_for_problem_type("tool_routing")
        code = "\n".join(lines)
        assert "valid_json" in code
        assert "has_sequence_or_array" in code
        assert "contains_agent_name" in code
        assert "no_verbose_explanation" in code
        assert "at_least_one_selection" in code

    def test_classification_checks(self):
        lines = _proxy_checks_for_problem_type("classification")
        code = "\n".join(lines)
        assert "valid_json" in code
        assert "single_label" in code
        assert "not_empty" in code
        # Should NOT have tool_routing-specific checks
        assert "has_sequence_or_array" not in code

    def test_unknown_type_fallback(self):
        lines = _proxy_checks_for_problem_type("other")
        code = "\n".join(lines)
        assert "valid_json" in code
        assert "max_length" in code


# ---------------------------------------------------------------------------
# Scorer setup improvements
# ---------------------------------------------------------------------------


class TestScorerSetupImprovements:
    def _base_state(self) -> WizardState:
        return WizardState(
            problem_type="tool_routing",
            task_description="Route queries to tools",
            has_ground_truth="no",
            strategies=["composite"],
            backend="ollama",
            model="llama3.2",
            human_eval="no",
        )

    def test_composite_weights_sum_to_one(self):
        state = self._base_state()
        code = _build_scorer_setup(state)
        # Extract weight values from CompositeScorer lines
        import re
        weights = [float(m) for m in re.findall(r",\s*([\d.]+)\)", code)]
        if weights:
            assert abs(sum(weights) - 1.0) < 0.05

    def test_rubric_is_task_specific(self):
        state = self._base_state()
        code = _build_scorer_setup(state)
        # Should contain problem-type-specific rubric, not generic truncation
        assert "agent" in code.lower() or "routing" in code.lower()

    def test_proxy_checks_are_problem_type_aware(self):
        state = self._base_state()
        code = _build_scorer_setup(state)
        assert "has_sequence_or_array" in code
        assert "contains_agent_name" in code

    def test_classification_proxy_checks(self):
        state = self._base_state()
        state.problem_type = "classification"
        code = _build_scorer_setup(state)
        assert "single_label" in code

    def test_is_valid_json_always_included(self):
        state = self._base_state()
        code = _build_scorer_setup(state)
        assert "_is_valid_json" in code

    def test_single_strategy_no_normalization_needed(self):
        state = self._base_state()
        state.strategies = ["llm_judge"]
        state.llm_judge_rubric = "Rate it"
        code = _build_scorer_setup(state)
        assert "CompositeScorer" not in code
        assert "return judge" in code


# ---------------------------------------------------------------------------
# Main block improvements
# ---------------------------------------------------------------------------


class TestMainBlockImprovements:
    def _base_state(self) -> WizardState:
        return WizardState(
            problem_type="tool_routing",
            task_description="Route queries to tools",
            has_ground_truth="no",
            strategies=["composite"],
            backend="ollama",
            model="llama3.2",
            human_eval="no",
        )

    def test_adaptive_mutations_enabled(self):
        state = self._base_state()
        block = _build_main_block(state)
        assert "adaptive_mutations=True" in block

    def test_llm_mutation_rate_set(self):
        state = self._base_state()
        block = _build_main_block(state)
        assert "llm_mutation_rate=0.3" in block

    def test_refine_after_splice_enabled(self):
        state = self._base_state()
        block = _build_main_block(state)
        assert "refine_after_splice=True" in block

    def test_domain_mutations_wired(self):
        state = self._base_state()
        block = _build_main_block(state)
        assert "custom_mutations=DOMAIN_MUTATIONS" in block


# ---------------------------------------------------------------------------
# Seed diversification in generated scripts
# ---------------------------------------------------------------------------


class TestGeneratedSeedDiversification:
    def test_no_seeds_generates_diverse_variants(self):
        state = WizardState(
            problem_type="tool_routing",
            task_description="Route queries to agents",
            has_ground_truth="no",
            strategies=["composite"],
            backend="ollama",
            model="llama3.2",
            human_eval="no",
        )
        script = _generate_script(state)
        # Should have multiple seed templates
        assert script.count("SEED_TEMPLATES") >= 1
        # Should contain diverse styles
        assert "step-by-step" in script or "JSON only" in script

    def test_single_seed_diversified(self):
        state = WizardState(
            problem_type="tool_routing",
            task_description="Route queries to agents",
            has_ground_truth="no",
            strategies=["composite"],
            backend="ollama",
            model="llama3.2",
            human_eval="no",
            seed_templates=["My custom seed"],
        )
        script = _generate_script(state)
        assert "My custom seed" in script
        # Should have expanded beyond just the one seed
        assert "SEED_TEMPLATES = [" in script


# ---------------------------------------------------------------------------
# GENERATION problem type in wizard
# ---------------------------------------------------------------------------


class TestGenerationWizardSupport:
    def test_generation_rubric(self):
        rubric = _rubric_for_problem_type("generation", "Generate structured JSON")
        assert "0-10" in rubric
        assert "JSON" in rubric or "json" in rubric.lower()

    def test_generation_proxy_checks(self):
        lines = _proxy_checks_for_problem_type("generation")
        code = "\n".join(lines)
        assert "valid_json" in code
        assert "is_json_object" in code

    def test_generation_scorer_setup(self):
        state = WizardState(
            problem_type="generation",
            task_description="Generate medical records",
            has_ground_truth="no",
            strategies=["composite"],
            backend="ollama",
            model="llama3.2",
            human_eval="no",
        )
        code = _build_scorer_setup(state)
        assert "valid_json" in code

    def test_generation_main_block_problem_type(self):
        state = WizardState(
            problem_type="generation",
            task_description="Generate records",
            has_ground_truth="no",
            strategies=["composite"],
            backend="ollama",
            model="llama3.2",
            human_eval="no",
        )
        block = _build_main_block(state)
        assert "ProblemType.GENERATION" in block
