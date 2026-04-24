"""Prompture — Evolutionary prompt optimisation for LLM agents.

Standalone library for evolving system prompts, tool-calling instructions,
and agent configurations using evolutionary strategies.

Quick start::

    from prompture import PromptEvolver, PromptEvolverConfig, LLMBackend

    config = PromptEvolverConfig(
        problem_type="tool_calling",
        llm_backend=LLMBackend.OLLAMA,
    )
    evolver = PromptEvolver(config)
    result = evolver.evolve(seed_prompt, eval_samples)

No-eval strategies (when you don't have ground-truth labels)::

    from prompture import NoEvalPromptEvolver, NoEvalConfig, LLMJudge

    scorer = LLMJudge(model="gpt-4.1")
    config = NoEvalConfig(scorer=scorer)
    evolver = NoEvalPromptEvolver(config)
    result = evolver.evolve(seed_prompt, eval_samples)

Interactive wizard::

    prompture init
"""

from __future__ import annotations

__version__ = "0.1.0"

from prompture.prompt_evolver import (
    ErrorProfile,
    PromptEvolver,
    PromptEvolverConfig,
    PromptEvolverResult,
    PromptCandidate,
    ProblemType,
    Tool,
    EvalSample,
    LLMBackend,
    LLMClient,
    evolve_prompt_with_cmaes,
    generate_adaptive_mutations,
    get_mutations_for_problem_type,
    _feasibility_key,
)
from prompture.strategies import (
    NoEvalPromptEvolver,
    NoEvalConfig,
    Scorer,
    LLMJudge,
    SyntheticEvalGenerator,
    SyntheticEvalScorer,
    ToolSuccessScorer,
    ToolResult,
    SelfConsistencyScorer,
    ProxyMetricsScorer,
    ProxyCheck,
    PreferenceScorer,
    PreferencePair,
    HumanTournament,
    CompositeScorer,
    PenaltyScaler,
)
from prompture.seed_loader import (
    Penalty,
    SeedTemplateConfig,
    load_seed_templates,
    load_seed_template_config,
    list_seed_templates,
    penalties_to_proxy_checks,
    register_penalty_condition,
)
from prompture.wizard import run_wizard

__all__ = [
    # Core engine
    "ErrorProfile",
    "PromptEvolver",
    "PromptEvolverConfig",
    "PromptEvolverResult",
    "PromptCandidate",
    "ProblemType",
    "Tool",
    "EvalSample",
    "LLMBackend",
    "LLMClient",
    "evolve_prompt_with_cmaes",
    "generate_adaptive_mutations",
    "get_mutations_for_problem_type",
    # No-eval strategies
    "NoEvalPromptEvolver",
    "NoEvalConfig",
    "Scorer",
    "LLMJudge",
    "SyntheticEvalGenerator",
    "SyntheticEvalScorer",
    "ToolSuccessScorer",
    "ToolResult",
    "SelfConsistencyScorer",
    "ProxyMetricsScorer",
    "ProxyCheck",
    "PreferenceScorer",
    "PreferencePair",
    "HumanTournament",
    "CompositeScorer",
    "PenaltyScaler",
    # Seed templates
    "Penalty",
    "SeedTemplateConfig",
    "load_seed_templates",
    "load_seed_template_config",
    "list_seed_templates",
    "penalties_to_proxy_checks",
    "register_penalty_condition",
    # Wizard
    "run_wizard",
]
