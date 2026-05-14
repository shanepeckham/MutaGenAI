"""
EvoSim Prompt Evolver — evolutionary optimisation of LLM prompt templates.

Uses a hybrid approach:
  • FunSearch-style evolution of prompt template *functions* (discrete structure)
  • CMA-ES / DE for continuous LLM parameters (temperature, top-p)

Supports Ollama (local) and Azure OpenAI backends.

Typical usage::

    from MutaGenAI.prompt_evolver import (
        PromptEvolver, PromptEvolverConfig,
        Tool, EvalSample, LLMBackend,
    )

    tools = [
        Tool("get_weather", "Get current weather", {"location": "string"}),
        Tool("send_email", "Send an email", {"to": "string", "body": "string"}),
    ]
    dataset = [
        EvalSample("What's the weather in London?", "get_weather", {"location": "London"}),
        EvalSample("Email Bob about the meeting", "send_email", {"to": "Bob", "body": "meeting"}),
    ]

    evolver = PromptEvolver(
        tools=tools,
        eval_dataset=dataset,
        backend=LLMBackend.OLLAMA,
    )
    result = evolver.run()
    print(result.best_prompt)
    print(result.best_accuracy)
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class LLMBackend(Enum):
    """Supported LLM backends."""

    OLLAMA = "ollama"
    AZURE_OPENAI = "azure_openai"
    OPENAI = "openai"


@dataclass
class Tool:
    """A tool available to the agent.

    Attributes:
        name:        Unique tool identifier (e.g. ``"get_weather"``).
        description: Human-readable description of what the tool does.
        parameters:  Mapping of parameter name → type hint string.
    """

    name: str
    description: str
    parameters: dict[str, str] = field(default_factory=dict)

    def schema_str(self) -> str:
        """Return a compact schema string for embedding in prompts."""
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"{self.name}({params}) — {self.description}"


@dataclass
class EvalSample:
    """A single evaluation sample: user query → expected tool + params.

    Attributes:
        query:           The user's natural-language utterance.
        expected_tool:   The correct tool name to invoke.
        expected_params: Expected parameter values (for partial-match scoring).
    """

    query: str
    expected_tool: str
    expected_params: dict[str, str] = field(default_factory=dict)


_candidate_counter: int = 0


@dataclass
class PromptCandidate:
    """A scored prompt template candidate."""

    template: str
    temperature: float = 0.1
    top_p: float = 0.95
    score: float = 0.0
    generation: int = 0
    hash: str = ""
    island_id: int = -1
    operation: str = "seed"
    parent_hashes: list[str] = field(default_factory=list)
    penalty_violations: int = 0

    def __post_init__(self) -> None:
        if not self.hash:
            global _candidate_counter  # noqa: PLW0603
            _candidate_counter += 1
            # Include counter to guarantee uniqueness even when templates match
            blob = f"{self.template}|{self.generation}|{self.island_id}|{_candidate_counter}"
            self.hash = hashlib.sha256(blob.encode()).hexdigest()[:16]


@dataclass
class PromptEvolverConfig:
    """Configuration for the prompt evolution loop.

    Attributes:
        iterations:         Number of evolutionary generations.
        population_size:    Candidates evaluated per generation.
        num_islands:        Island-model diversity pools.
        elite_size:         Top-k survivors per island.
        mutation_rate:       Probability of mutating a template section.
        crossover_rate:      Probability of crossing two parent templates.
        eval_sample_size:   Samples drawn per evaluation (None = use all).
        temperature_range:  (min, max) for temperature search.
        top_p_range:        (min, max) for top-p search.
        backend:            LLM backend to use.
        ollama_url:         Ollama REST endpoint.
        ollama_model:       Model name for Ollama.
        azure_endpoint:     Azure OpenAI endpoint URL.
        azure_api_key:      Azure OpenAI API key (optional if using RBAC).
        azure_deployment:   Azure OpenAI deployment name.
        azure_api_version:  Azure OpenAI API version.
        azure_use_rbac:     Use Entra ID (DefaultAzureCredential) instead of
                            API key.  Requires ``azure-identity``.
        openai_api_key:     OpenAI API key (for direct OpenAI).
        openai_model:       OpenAI model name.
        timeout:            HTTP timeout for LLM calls.
        max_retries:        Max retries on LLM call failure.
        max_tokens:         Max tokens to generate (``num_predict`` for
                            Ollama, ``max_tokens`` for OpenAI/Azure).
                            ``None`` means no limit (model default).
        refine_after_splice: If ``True``, run an LLM call after each
                            mutation/crossover to remove duplicate lines,
                            fix incoherent phrasing, and tighten the
                            prompt while preserving its meaning.
                            Costs one extra LLM call per breed operation.
        adaptive_mutations: If ``True``, after each generation the evolver
                            analyses which tools/classes have the highest
                            error rate and uses an LLM to generate targeted
                            mutation snippets.  Default ``False``.
        llm_mutation_rate:  Probability of applying an LLM-assisted rewrite
                            instead of random snippet mutation during
                            breeding.  ``0.0`` disables.  Default ``0.0``.
        describe_entities:  If ``True``, the describe-entities LLM rewrite
                            is added to the mutation pool.  During breeding,
                            a candidate may be randomly selected for
                            expansion of bare agent/category names into
                            ``name — description`` format.  This lets
                            evolution discover whether descriptions help.
                            Default ``False``.
        minimize_tokens:    Enable token-length optimisation.  When ``True``,
                            two mechanisms activate:
                            (A) a baseline-relative efficiency bonus is
                            blended into the fitness score, and
                            (B) tournament selection uses a lexicographic
                            tiebreaker that prefers shorter prompts within
                            the same accuracy band.
                            Default ``False``.
        token_weight:       Blend weight for the efficiency bonus (A).
                            ``0.0`` disables blending, ``1.0`` scores
                            purely on token efficiency.  Default ``0.10``.
        token_efficiency_cap: Maximum efficiency ratio (baseline /
                            candidate tokens) before capping.  Prevents
                            degenerate ultra-short prompts from scoring
                            disproportionately.  Default ``2.0``.
        token_accuracy_band: Width of the accuracy band for tiebreaker
                            (B).  Candidates whose scores fall in the
                            same ``score // band`` bucket are compared
                            by prompt length.  Default ``2.0``.
        baseline_prompt_tokens: Baseline prompt token count used to
                            compute the efficiency ratio.  If ``0``
                            (default), token optimisation scoring is
                            skipped even when ``minimize_tokens=True``.
    """

    iterations: int = 30
    population_size: int = 8
    num_islands: int = 3
    elite_size: int = 4
    mutation_rate: float = 0.6
    crossover_rate: float = 0.3
    eval_sample_size: Optional[int] = None
    temperature_range: tuple[float, float] = (0.0, 1.0)
    top_p_range: tuple[float, float] = (0.5, 1.0)
    backend: LLMBackend = LLMBackend.OLLAMA
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_deployment: str = ""
    azure_api_version: str = "2024-12-01-preview"
    azure_use_rbac: bool = True
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    timeout: float = 30.0
    max_retries: int = 2
    max_tokens: Optional[int] = None
    refine_after_splice: bool = False
    adaptive_mutations: bool = False
    llm_mutation_rate: float = 0.0
    describe_entities: bool = False
    minimize_tokens: bool = False
    token_weight: float = 0.10
    token_efficiency_cap: float = 2.0
    token_accuracy_band: float = 2.0
    baseline_prompt_tokens: int = 0

    def __post_init__(self) -> None:
        # Auto-detect from environment variables
        if not self.azure_endpoint:
            self.azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        if not self.azure_api_key:
            self.azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
        if not self.azure_deployment:
            self.azure_deployment = os.environ.get(
                "AZURE_OPENAI_DEPLOYMENT", ""
            )
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "")
        # RBAC override from env
        rbac_env = os.environ.get("AZURE_OPENAI_USE_RBAC", "").lower()
        if rbac_env in ("1", "true", "yes"):
            self.azure_use_rbac = True
        elif rbac_env in ("0", "false", "no"):
            self.azure_use_rbac = False


# ---------------------------------------------------------------------------
# LLM Client — unified Ollama / Azure OpenAI / OpenAI
# ---------------------------------------------------------------------------


class LLMClient:
    """Unified LLM client supporting Ollama and Azure OpenAI."""

    # Azure token scopes — AI Foundry (*.services.ai.azure.com) requires
    # https://ai.azure.com/.default while classic Azure OpenAI
    # (*.openai.azure.com) uses https://cognitiveservices.azure.com/.default.
    _SCOPE_AI_FOUNDRY = "https://ai.azure.com/.default"
    _SCOPE_COGNITIVE = "https://cognitiveservices.azure.com/.default"

    @staticmethod
    def _azure_scope(endpoint: str) -> str:
        """Return the correct OAuth scope for the Azure endpoint."""
        if ".services.ai.azure.com" in endpoint:
            return LLMClient._SCOPE_AI_FOUNDRY
        return LLMClient._SCOPE_COGNITIVE

    def __init__(self, config: PromptEvolverConfig) -> None:
        self.config = config
        self._available: Optional[bool] = None
        self._azure_credential: Any = None
        self._azure_token: Optional[str] = None
        self._azure_token_expires: float = 0.0
        # Usage tracking
        self.call_count: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def _record_usage(self, response_data: dict[str, Any]) -> None:
        """Accumulate token usage from an OpenAI-compatible response."""
        self.call_count += 1
        usage = response_data.get("usage", {})
        self.total_input_tokens += usage.get("prompt_tokens", 0)
        self.total_output_tokens += usage.get("completion_tokens", 0)

    def is_available(self) -> bool:
        """Check if the configured backend is reachable."""
        if self._available is not None:
            return self._available
        try:
            import httpx

            if self.config.backend == LLMBackend.OLLAMA:
                resp = httpx.get(
                    f"{self.config.ollama_url}/api/tags", timeout=5.0
                )
                self._available = resp.status_code == 200
            elif self.config.backend == LLMBackend.AZURE_OPENAI:
                if not self.config.azure_endpoint or not self.config.azure_deployment:
                    self._available = False
                elif self.config.azure_use_rbac:
                    # RBAC: need azure-identity + valid credential
                    try:
                        from azure.identity import DefaultAzureCredential  # type: ignore[import-not-found]

                        self._azure_credential = DefaultAzureCredential()
                        scope = self._azure_scope(self.config.azure_endpoint)
                        token = self._azure_credential.get_token(scope)
                        self._azure_token = token.token
                        self._azure_token_expires = token.expires_on
                        self._available = True
                    except Exception as exc:
                        logger.debug("Azure RBAC auth failed: %s", exc)
                        self._available = False
                else:
                    # API-key auth
                    self._available = bool(self.config.azure_api_key)
            elif self.config.backend == LLMBackend.OPENAI:
                self._available = bool(self.config.openai_api_key)
            else:
                self._available = False
        except Exception:
            self._available = False
        return self._available

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ) -> Optional[str]:
        """Send a chat completion request and return the assistant response."""
        # Fast-path: skip HTTP call if backend is known unavailable
        if not self.is_available():
            return None

        try:
            import httpx
        except ImportError:
            logger.error("httpx required: pip install httpx")
            return None

        try:
            if self.config.backend == LLMBackend.OLLAMA:
                return self._ollama_complete(
                    httpx, system_prompt, user_message, temperature, top_p
                )
            elif self.config.backend == LLMBackend.AZURE_OPENAI:
                return self._azure_complete(
                    httpx, system_prompt, user_message, temperature, top_p
                )
            elif self.config.backend == LLMBackend.OPENAI:
                return self._openai_complete(
                    httpx, system_prompt, user_message, temperature, top_p
                )
        except Exception as exc:
            logger.debug("LLM call failed: %s", exc)
        return None

    def _ollama_complete(
        self,
        httpx: Any,
        system_prompt: str,
        user_message: str,
        temperature: float,
        top_p: float,
    ) -> Optional[str]:
        options: dict[str, Any] = {"temperature": temperature, "top_p": top_p}
        if self.config.max_tokens is not None:
            options["num_predict"] = self.config.max_tokens
        resp = httpx.post(
            f"{self.config.ollama_url}/api/chat",
            json={
                "model": self.config.ollama_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "stream": False,
                "options": options,
            },
            timeout=self.config.timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            # Ollama returns eval_count / prompt_eval_count
            self.call_count += 1
            self.total_input_tokens += data.get("prompt_eval_count", 0)
            self.total_output_tokens += data.get("eval_count", 0)
            content: str = data.get("message", {}).get("content", "")
            return content
        return None

    def _get_azure_headers(self) -> dict[str, str]:
        """Return auth headers for Azure OpenAI (RBAC bearer or API key)."""
        if self.config.azure_use_rbac and self._azure_credential is not None:
            # Refresh token if within 5 minutes of expiry
            if time.time() > self._azure_token_expires - 300:
                scope = self._azure_scope(self.config.azure_endpoint)
                token = self._azure_credential.get_token(scope)
                self._azure_token = token.token
                self._azure_token_expires = token.expires_on
            return {
                "Authorization": f"Bearer {self._azure_token}",
                "Content-Type": "application/json",
            }
        return {
            "api-key": self.config.azure_api_key,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _azure_base_url(endpoint: str) -> str:
        """Derive the Azure OpenAI base URL from a user-supplied endpoint.

        Azure AI Foundry endpoints often include a path like
        ``/openai/v1/responses``.  We strip everything from ``/openai/``
        onwards so we can append the standard chat-completions path.
        """
        idx = endpoint.find("/openai/")
        if idx != -1:
            return endpoint[:idx]
        return endpoint.rstrip("/")

    @staticmethod
    def _is_foundry_endpoint(endpoint: str) -> bool:
        """Return True when the endpoint is an Azure AI Foundry host."""
        return ".services.ai.azure.com" in endpoint

    def _azure_complete(
        self,
        httpx: Any,
        system_prompt: str,
        user_message: str,
        temperature: float,
        top_p: float,
    ) -> Optional[str]:
        base = self._azure_base_url(self.config.azure_endpoint)
        foundry = self._is_foundry_endpoint(self.config.azure_endpoint)
        if foundry:
            # AI Foundry uses OpenAI-compatible paths with model in body
            url = f"{base}/openai/v1/chat/completions"
        else:
            # Classic Azure OpenAI uses deployment path + api-version
            url = (
                f"{base}"
                f"/openai/deployments/{self.config.azure_deployment}"
                f"/chat/completions?api-version={self.config.azure_api_version}"
            )
        body: dict[str, Any] = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 256,
        }
        if foundry:
            body["model"] = self.config.azure_deployment
        logger.debug("Azure POST %s", url)
        resp = httpx.post(
            url,
            headers=self._get_azure_headers(),
            json=body,
            timeout=self.config.timeout,
        )
        if resp.status_code != 200:
            logger.debug("Azure status=%d body=%s", resp.status_code, resp.text[:500])
        if resp.status_code == 200:
            data = resp.json()
            self._record_usage(data)
            choices = data.get("choices", [])
            if choices:
                content: str = choices[0].get("message", {}).get("content", "")
                return content
        return None

    def _openai_complete(
        self,
        httpx: Any,
        system_prompt: str,
        user_message: str,
        temperature: float,
        top_p: float,
    ) -> Optional[str]:
        resp = httpx.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.openai_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.config.openai_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": 256,
            },
            timeout=self.config.timeout,
        )
        if resp.status_code == 200:
            data = resp.json()
            self._record_usage(data)
            choices = data.get("choices", [])
            if choices:
                content: str = choices[0].get("message", {}).get("content", "")
                return content
        return None


# ---------------------------------------------------------------------------
# Response parser — extract tool name + params from LLM output
# ---------------------------------------------------------------------------


def parse_tool_response(
    response: str, tool_names: list[str]
) -> tuple[Optional[str], dict[str, str]]:
    """Extract tool name and parameters from an LLM response.

    Handles multiple output formats:
      - JSON: ``{"tool": "name", "parameters": {...}}``
      - Function call: ``tool_name(param="value")``
      - Plain text with tool name mention

    Returns (tool_name, params_dict) or (None, {}) on failure.
    """
    if not response:
        return None, {}

    # Try JSON parse first
    try:
        # Find JSON object in the response (handles nested braces)
        start = response.find("{")
        if start != -1:
            depth = 0
            end = start
            for i in range(start, len(response)):
                if response[i] == "{":
                    depth += 1
                elif response[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            json_str = response[start:end]
            data = json.loads(json_str)
            tool = data.get("tool") or data.get("function") or data.get("name", "")
            params = data.get("parameters") or data.get("params") or data.get("arguments", {})
            if isinstance(params, str):
                try:
                    params = json.loads(params)
                except (json.JSONDecodeError, ValueError):
                    params = {}
            if tool in tool_names:
                return tool, {str(k): str(v) for k, v in params.items()}
    except (json.JSONDecodeError, ValueError):
        pass

    # Try function-call style: tool_name(...)
    for name in tool_names:
        pattern = rf"\b{re.escape(name)}\s*\("
        match = re.search(pattern, response)
        if match:
            # Extract params from parentheses
            start = match.end()
            depth = 1
            end = start
            for i in range(start, len(response)):
                if response[i] == "(":
                    depth += 1
                elif response[i] == ")":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            param_str = response[start:end]
            kv_params: dict[str, str] = {}
            for kv in re.findall(r'(\w+)\s*=\s*["\']([^"\']*)["\']', param_str):
                kv_params[kv[0]] = kv[1]
            return name, kv_params

    # Fallback: find the first tool name mentioned in the response
    response_lower = response.lower()
    for name in tool_names:
        if name.lower() in response_lower:
            return name, {}

    return None, {}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_response(
    predicted_tool: Optional[str],
    predicted_params: dict[str, str],
    expected_tool: str,
    expected_params: dict[str, str],
) -> float:
    """Score a single prediction against ground truth.

    Returns a float in [0, 1]:
      - 0.0  if tool name is wrong
      - 0.6  if tool name is correct but no params match
      - 0.6 + 0.4 * (fraction of matching params) if tool is correct
    """
    if predicted_tool != expected_tool:
        return 0.0

    if not expected_params:
        return 1.0

    tool_score = 0.6
    matches: float = 0
    for key, expected_val in expected_params.items():
        predicted_val = predicted_params.get(key, "")
        if predicted_val.lower().strip() == expected_val.lower().strip():
            matches += 1
        elif expected_val.lower() in predicted_val.lower():
            matches += 0.5

    param_score = matches / len(expected_params) if expected_params else 1.0
    return tool_score + 0.4 * param_score


# ---------------------------------------------------------------------------
# Seed prompt templates
# ---------------------------------------------------------------------------


_SEED_TEMPLATES = [
    # Template 0: Minimal JSON-output
    textwrap.dedent("""\
        You are a tool-routing assistant. Given a user query, decide which tool
        to call and with what parameters.

        Available tools:
        {tool_schemas}

        Respond with ONLY a JSON object:
        {{"tool": "<tool_name>", "parameters": {{<param_name>: <value>}}}}
    """),

    # Template 1: Chain-of-thought
    textwrap.dedent("""\
        You are an intent classifier for an agentic system. Analyse the user's
        request step by step:
        1. Identify the user's intent.
        2. Match it to exactly ONE of the available tools.
        3. Extract the required parameters from the query.

        Available tools:
        {tool_schemas}

        Think step by step, then respond with a JSON object on the LAST line:
        {{"tool": "<tool_name>", "parameters": {{<param_name>: <value>}}}}
    """),

    # Template 2: Few-shot style (no examples, but structured)
    textwrap.dedent("""\
        SYSTEM: You are a zero-shot tool router. Your job is to classify the
        user's intent and route to the correct tool.

        TOOLS:
        {tool_schemas}

        RULES:
        - Pick exactly one tool.
        - Extract parameter values from the user message.
        - If a parameter is not mentioned, use a reasonable default.
        - Output ONLY valid JSON: {{"tool": "...", "parameters": {{...}}}}
    """),

    # Template 3: Structured with role emphasis
    textwrap.dedent("""\
        # Role
        You are a precise intent classification agent. You must route every
        user request to exactly one tool from your toolbox.

        # Toolbox
        {tool_schemas}

        # Output Format
        Return a single JSON object with these exact keys:
        - "tool": the name of the tool to invoke
        - "parameters": an object mapping parameter names to extracted values

        Do not include any explanation, markdown, or extra text.
    """),
]


# ---------------------------------------------------------------------------
# Problem types
# ---------------------------------------------------------------------------


class ProblemType(str, Enum):
    """The nature of the task being optimised.

    Determines which built-in mutation snippets are injected during
    evolution. Users select this in the wizard or via the CLI.
    """

    TOOL_ROUTING = "tool_routing"
    CLASSIFICATION = "classification"


# ---------------------------------------------------------------------------
# Prompt mutations – per problem type
# ---------------------------------------------------------------------------


_TOOL_ROUTING_MUTATIONS: list[str] = [
    "Respond concisely in JSON format only.",
    "You MUST pick exactly one tool. Do not refuse.",
    "Consider the user's full intent before choosing a tool.",
    "If multiple tools could apply, choose the most specific one.",
    "Pay close attention to parameter extraction from the query.",
    "Always try to fill in parameter values from the user's message.",
    "Think about what the user actually needs, not just keywords.",
    "Match the user's intent to the tool purpose, not the tool name.",
    "Output format: {\"tool\": \"<name>\", \"parameters\": {<key>: <value>}}",
    "Do not include any explanation outside the JSON object.",
    "Extract entity values (names, locations, numbers) as parameters.",
    "If a required parameter is unclear, infer it from context.",
    "Be precise — ambiguous queries should still resolve to one tool.",
    "The parameters object must contain string values only.",
    "For each agent/tool, include a brief description of its purpose and when to use it.",
    "Describe what each agent does so the routing decision is grounded in capability, not name.",
    "Annotate each agent with its responsibility to disambiguate similar-sounding agents.",
]

_CLASSIFICATION_MUTATIONS: list[str] = [
    "Return exactly one category label — no extra text.",
    "Consider the semantic meaning of the input, not just keywords.",
    "Think step-by-step before choosing a category.",
    "Consider every possible category before deciding.",
    "Choose the single most specific category that fits.",
    "If the input is ambiguous, pick the category with strongest evidence.",
    "Focus on the distinguishing features between similar categories.",
    "Do not add explanations, reasoning, or confidence scores.",
    "Consider the full context of the input before classifying.",
    "Recall the definition of each category and match against it.",
    "Map the input to categories based on purpose and function, not surface words.",
    "When in doubt, re-read the input and look for decisive cues.",
    "Be consistent — similar inputs should always get the same label.",
    "Output must be a single word or short phrase matching a valid category.",
    "For each category, include a brief description of its meaning and when it applies.",
    "Describe what each category represents so the classification is grounded in semantics, not label names.",
    "Annotate each category with its definition to disambiguate overlapping labels.",
]

# Legacy alias for backward compatibility
_SECTION_MUTATIONS = _TOOL_ROUTING_MUTATIONS


def get_mutations_for_problem_type(
    problem_type: ProblemType,
) -> list[str]:
    """Return the built-in mutation snippets for a given problem type."""
    if problem_type is ProblemType.CLASSIFICATION:
        return _CLASSIFICATION_MUTATIONS
    return _TOOL_ROUTING_MUTATIONS


# ---------------------------------------------------------------------------
# Error-guided adaptive mutations
# ---------------------------------------------------------------------------


@dataclass
class ErrorProfile:
    """Per-category error counts collected during evaluation.

    For classification, categories are class labels.
    For tool routing, categories are tool names.
    """

    total: dict[str, int] = field(default_factory=dict)
    errors: dict[str, int] = field(default_factory=dict)

    def record(self, category: str, correct: bool) -> None:
        self.total[category] = self.total.get(category, 0) + 1
        if not correct:
            self.errors[category] = self.errors.get(category, 0) + 1

    def worst_categories(self, top_k: int = 3) -> list[tuple[str, float]]:
        """Return up to *top_k* categories sorted by error rate (desc).

        Only includes categories with at least one sample.
        """
        rates: list[tuple[str, float]] = []
        for cat, tot in self.total.items():
            if tot > 0:
                err = self.errors.get(cat, 0)
                rates.append((cat, err / tot))
        rates.sort(key=lambda x: x[1], reverse=True)
        return [(c, r) for c, r in rates[:top_k] if r > 0]

    def decay(self, factor: float) -> None:
        """Multiply all counts by *factor* so older data fades out.

        A factor of 0.5 halves every count, giving recent generations
        more influence while retaining a memory of past errors.
        """
        for cat in list(self.total):
            self.total[cat] = int(self.total[cat] * factor)
            if cat in self.errors:
                self.errors[cat] = int(self.errors[cat] * factor)
            # Prune zeroed-out categories
            if self.total[cat] <= 0:
                del self.total[cat]
                self.errors.pop(cat, None)


def generate_adaptive_mutations(
    error_profile: ErrorProfile,
    problem_type: ProblemType,
    client: "LLMClient",
    top_k: int = 3,
    max_hints: int = 4,
) -> list[str]:
    """Generate mutation snippets targeting the worst-performing categories.

    Uses an LLM call to produce domain-appropriate hints that address
    specific confusion patterns.  Falls back to template-based hints when
    the LLM is unavailable.

    Works for both ``CLASSIFICATION`` and ``TOOL_ROUTING`` problem types.
    """
    worst = error_profile.worst_categories(top_k)
    if not worst:
        return []

    hints: list[str] = []

    # Build a problem-type-aware prompt for the LLM
    if problem_type is ProblemType.CLASSIFICATION:
        task_noun = "classes"
        action = "classifying inputs into the correct category"
    else:
        task_noun = "tools"
        action = "routing user queries to the correct tool"

    category_lines = "\n".join(
        f"  - {cat} (error rate {rate:.0%})" for cat, rate in worst
    )

    system_prompt = textwrap.dedent(f"""\
        You are a prompt-engineering expert improving a system prompt for {action}.

        The following {task_noun} have the highest error rates:
        {category_lines}

        Write exactly {max_hints} short instruction lines (one sentence each)
        that could be added to the system prompt to reduce errors on these
        {task_noun}.  Each line should be a direct instruction to the model.

        Rules:
        - Each line must be self-contained and ≤ 120 characters.
        - Do NOT repeat existing instructions.
        - Reference specific {task_noun} by name where helpful.
        - Return ONLY the lines, one per line. No numbering, no bullets.
    """)

    response = client.complete(
        system_prompt=system_prompt,
        user_message=f"Worst {task_noun}: {category_lines}",
        temperature=0.7,
        top_p=0.95,
    )

    if response and response.strip():
        for line in response.strip().splitlines():
            cleaned = line.strip().lstrip("-•·0123456789.) ")
            if cleaned and len(cleaned) <= 150:
                hints.append(cleaned)
                if len(hints) >= max_hints:
                    break

    # Fallback: template-based hints when LLM returns nothing
    if not hints:
        for cat, rate in worst:
            if problem_type is ProblemType.CLASSIFICATION:
                hints.append(
                    f"Pay extra attention when the input might be '{cat}' "
                    f"— it is frequently misclassified."
                )
            else:
                hints.append(
                    f"Double-check queries that might need the '{cat}' tool "
                    f"— it is often missed."
                )
            if len(hints) >= max_hints:
                break

    return hints


# ---------------------------------------------------------------------------
# LLM-assisted mutation
# ---------------------------------------------------------------------------


_LLM_MUTATE_SYSTEM_CLASSIFICATION = textwrap.dedent("""\
    You are a prompt-engineering expert.  You will receive a system prompt used
    for classification and a few example inputs that it got WRONG.

    Rewrite the system prompt to fix these errors.  You may:
    - Add, remove, or rephrase instructions.
    - Reorder lines for clarity.
    - Add disambiguation hints for confused categories.

    Rules:
    - Keep the prompt short (≤ 15 lines).
    - Do NOT change the set of valid categories.
    - Return ONLY the rewritten prompt — no commentary, no markdown fences.
""")

_LLM_MUTATE_SYSTEM_TOOL_ROUTING = textwrap.dedent("""\
    You are a prompt-engineering expert.  You will receive a system prompt used
    for tool routing and a few example queries that it routed INCORRECTLY.

    Rewrite the system prompt to fix these routing errors.  You may:
    - Add, remove, or rephrase instructions.
    - Reorder lines for clarity.
    - Add hints for distinguishing similar tools.

    Rules:
    - Keep the prompt short (≤ 15 lines).
    - Preserve the {{tool_schemas}} placeholder if present.
    - Return ONLY the rewritten prompt — no commentary, no markdown fences.
""")


# ---------------------------------------------------------------------------
# LLM-assisted "describe entities" mutation
# ---------------------------------------------------------------------------


_DESCRIBE_ENTITIES_SYSTEM = textwrap.dedent("""\
    You are a prompt-engineering expert.  You will receive a system prompt that
    contains a list of agent names (or category labels).  Your job is to rewrite
    the prompt so that each agent/category is accompanied by a concise one-line
    description of its purpose and when it should be selected.

    Guidelines:
    - Keep each description to ONE sentence (≤ 20 words).
    - Ground descriptions in the agent/category NAME — infer the purpose from
      the name itself (e.g. ``fraud_detection_agent`` → "Detects and flags
      potentially fraudulent transactions or account activity.").
    - Format each entry as: ``agent_name — Description of what it does.``
    - Preserve ALL existing instructions and structure.  Only expand the
      agent/category list section.
    - Do NOT add or remove agents/categories.
    - Do NOT change the overall prompt intent or add new instructions.
    - Return ONLY the rewritten prompt — no commentary, no markdown fences.
""")


def _has_entity_descriptions(template: str) -> bool:
    """Heuristic: return True if the template already contains entity descriptions.

    Detects the ``name \u2014 description`` pattern produced by
    :func:`_llm_describe_entities`.  If 3+ such patterns are found the
    template is considered already described.
    """
    return template.count(" \u2014 ") >= 3


def _extract_entity_names(template: str) -> list[str]:
    """Extract entity names (agent names or category labels) from a prompt.

    Looks for ``word_word_agent`` patterns first.  Falls back to
    underscore-joined identifiers of 2+ parts (e.g. ``fraud_detection``).
    Returns a deduplicated list in order of first appearance.
    """
    # Agent-style names: foo_bar_agent
    agents = re.findall(r"\b(\w+_agent)\b", template)
    if agents:
        return list(dict.fromkeys(agents))  # dedupe, preserve order

    # Fallback: any multi-part underscore identifier (≥2 segments)
    identifiers = re.findall(r"\b([a-z][a-z0-9]*(?:_[a-z][a-z0-9]*)+)\b", template)
    if identifiers:
        return list(dict.fromkeys(identifiers))

    return []


def _llm_describe_entities(
    template: str,
    client: "LLMClient",
    problem_type: ProblemType,
    require_tool_schemas: bool = True,
) -> str:
    """Rewrite a prompt to add verbose descriptions to each agent/category.

    Uses an LLM call to expand bare entity names (agents or class labels)
    into ``name — description`` format.  This gives the routing/classification
    model better semantic grounding for its decisions.

    A completeness guard ensures that ALL entities receive descriptions.
    If the LLM only describes a subset, a follow-up call is made asking
    it to complete the missing descriptions.  If the retry still leaves
    gaps, the original template is returned unchanged.

    Returns the rewritten prompt, or the original on failure.
    """
    if problem_type is ProblemType.CLASSIFICATION:
        entity_kind = "category labels"
    else:
        entity_kind = "agent names"

    user_message = (
        f"The following system prompt contains {entity_kind} that need "
        f"descriptions added.  Rewrite the prompt so each {entity_kind.rstrip('s')} "
        f"has a concise purpose description.\n\n"
        f"```\n{template}\n```"
    )

    rewritten = client.complete(
        system_prompt=_DESCRIBE_ENTITIES_SYSTEM,
        user_message=user_message,
        temperature=0.4,
        top_p=0.95,
    )

    if not rewritten or not rewritten.strip():
        return template

    cleaned = rewritten.strip()
    # Strip markdown fences if the LLM added them
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    if require_tool_schemas and "{tool_schemas}" in template and "{tool_schemas}" not in cleaned:
        cleaned += "\n\nAvailable tools:\n{tool_schemas}"

    if not cleaned:
        return template

    # ── Completeness guard ──────────────────────────────────
    # If some entities were described but not all, retry with an
    # explicit list of the missing ones so every entity gets a
    # description or the mutation is discarded.
    entities = _extract_entity_names(template)
    if entities:
        described = [e for e in entities if f"{e} —" in cleaned or f"{e} —" in cleaned]
        missing = [e for e in entities if e not in described]
        if missing and described:
            logger.info(
                "describe_entities: partial result (%d/%d described) "
                "— retrying for %d missing entities",
                len(described), len(entities), len(missing),
            )
            missing_list = ", ".join(missing)
            retry_message = (
                f"The following prompt has descriptions for some "
                f"{entity_kind} but is missing descriptions for: "
                f"{missing_list}\n\n"
                f"Add a concise one-line description for each missing "
                f"entity using the same ``name — description`` format.  "
                f"Keep all existing content and descriptions unchanged.  "
                f"Return ONLY the complete rewritten prompt.\n\n"
                f"```\n{cleaned}\n```"
            )
            retry_result = client.complete(
                system_prompt=_DESCRIBE_ENTITIES_SYSTEM,
                user_message=retry_message,
                temperature=0.3,
                top_p=0.95,
            )
            if retry_result and retry_result.strip():
                retry_cleaned = retry_result.strip()
                if retry_cleaned.startswith("```"):
                    rlines = retry_cleaned.splitlines()
                    rlines = [l for l in rlines if not l.strip().startswith("```")]
                    retry_cleaned = "\n".join(rlines).strip()

                if require_tool_schemas and "{tool_schemas}" in template and "{tool_schemas}" not in retry_cleaned:
                    retry_cleaned += "\n\nAvailable tools:\n{tool_schemas}"

                # Verify the retry actually covered the gaps
                still_missing = [
                    e for e in entities
                    if f"{e} —" not in retry_cleaned and f"{e} —" not in retry_cleaned
                ]
                if not still_missing:
                    logger.info(
                        "describe_entities: retry succeeded — all %d entities described",
                        len(entities),
                    )
                    return retry_cleaned
                else:
                    logger.info(
                        "describe_entities: retry still missing %d/%d entities "
                        "— returning original",
                        len(still_missing), len(entities),
                    )
                    return template
            else:
                logger.info(
                    "describe_entities: retry returned empty — returning original"
                )
                return template

    return cleaned

def _llm_mutate_template(
    template: str,
    failure_examples: list[tuple[str, str, str]],
    client: "LLMClient",
    problem_type: ProblemType,
    require_tool_schemas: bool = True,
) -> str:
    """Use an LLM to rewrite a prompt based on failure cases.

    Parameters
    ----------
    template :
        The current system prompt.
    failure_examples :
        List of ``(input_text, predicted, expected)`` triples for
        misclassified / misrouted samples.
    client :
        LLM client to use for the rewrite.
    problem_type :
        Determines which system prompt to use for the rewriter.
    require_tool_schemas :
        If ``True``, ensure ``{tool_schemas}`` placeholder is preserved.

    Returns the rewritten prompt, or the original if the LLM fails.
    """
    if not failure_examples:
        return template

    if problem_type is ProblemType.CLASSIFICATION:
        system = _LLM_MUTATE_SYSTEM_CLASSIFICATION
        example_label = "Expected class"
        pred_label = "Predicted"
    else:
        system = _LLM_MUTATE_SYSTEM_TOOL_ROUTING
        example_label = "Expected tool"
        pred_label = "Predicted"

    # Format failure examples (limit to 6 to control token usage)
    examples_str = "\n".join(
        f"  Input: {inp[:120]}\n  {pred_label}: {pred}\n  {example_label}: {exp}"
        for inp, pred, exp in failure_examples[:6]
    )

    user_message = (
        f"Current prompt:\n```\n{template}\n```\n\n"
        f"Failure examples:\n{examples_str}"
    )

    rewritten = client.complete(
        system_prompt=system,
        user_message=user_message,
        temperature=0.7,
        top_p=0.95,
    )

    if not rewritten or not rewritten.strip():
        return template

    cleaned = rewritten.strip()
    # Strip markdown fences if the LLM added them despite instructions
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    if require_tool_schemas and "{tool_schemas}" in template and "{tool_schemas}" not in cleaned:
        cleaned += "\n\nAvailable tools:\n{tool_schemas}"

    return cleaned if cleaned else template


def _mutate_template(
    template: str,
    rng: np.random.Generator,
    mutation_rate: float = 0.5,
    require_tool_schemas: bool = True,
    mutations: list[str] | None = None,
) -> str:
    """Apply random mutations to a prompt template.

    Parameters
    ----------
    mutations :
        Snippet pool to draw insertions from.  Defaults to the
        tool-routing mutations for backward compatibility.
    """
    pool = mutations if mutations is not None else _TOOL_ROUTING_MUTATIONS
    lines = template.strip().split("\n")

    if rng.random() < mutation_rate:
        # Insert a new instruction line at a random position
        instruction = str(rng.choice(pool))
        pos = int(rng.integers(1, max(2, len(lines))))
        lines.insert(pos, instruction)

    if rng.random() < mutation_rate * 0.5 and len(lines) > 3:
        # Remove a random non-essential line
        removable = [
            i
            for i in range(len(lines))
            if "{tool_schemas}" not in lines[i] and lines[i].strip()
        ]
        if removable:
            idx = int(rng.choice(removable))
            lines.pop(idx)

    if rng.random() < mutation_rate * 0.3 and len(lines) > 2:
        # Swap two adjacent lines
        idx = int(rng.integers(0, len(lines) - 1))
        lines[idx], lines[idx + 1] = lines[idx + 1], lines[idx]

    result = "\n".join(lines)

    # Ensure {tool_schemas} placeholder is preserved (tool-routing only)
    if require_tool_schemas and "{tool_schemas}" not in result:
        result += "\n\nAvailable tools:\n{tool_schemas}"

    return result


def _crossover_templates(
    parent_a: str,
    parent_b: str,
    rng: np.random.Generator,
    require_tool_schemas: bool = True,
) -> str:
    """Create a child template by crossing two parents at a random line."""
    lines_a = parent_a.strip().split("\n")
    lines_b = parent_b.strip().split("\n")

    cut_a = int(rng.integers(1, max(2, len(lines_a))))
    cut_b = int(rng.integers(1, max(2, len(lines_b))))

    child_lines = lines_a[:cut_a] + lines_b[cut_b:]
    result = "\n".join(child_lines)

    if require_tool_schemas and "{tool_schemas}" not in result:
        result += "\n\nAvailable tools:\n{tool_schemas}"

    return result


_REFINE_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a prompt editor.  You will receive a system prompt that was
    produced by splicing two parent prompts together.

    Your ONLY job:
    1. Remove duplicate or near-duplicate lines/sentences.
    2. Fix any incoherent transitions caused by the splice.
    3. Remove contradictory instructions.
    4. Keep the prompt as short as possible.

    Rules:
    - Do NOT change the task or intent.
    - Do NOT add new instructions.
    - Do NOT remove the placeholder {tool_schemas} if it is present.
    - Return ONLY the cleaned prompt — no commentary, no markdown fences.
""")


def _refine_template(
    template: str, client: "LLMClient", require_tool_schemas: bool = True
) -> str:
    """Use an LLM call to deduplicate and clean a spliced template.

    If the LLM is unreachable or the response is empty, falls back to
    returning the original template unchanged.
    """
    refined = client.complete(
        system_prompt=_REFINE_SYSTEM_PROMPT,
        user_message=template,
        temperature=0.0,
        top_p=1.0,
    )
    if not refined or not refined.strip():
        return template

    cleaned = refined.strip()

    # Preserve the {tool_schemas} placeholder if the original had it
    if (
        require_tool_schemas
        and "{tool_schemas}" in template
        and "{tool_schemas}" not in cleaned
    ):
        cleaned += "\n\nAvailable tools:\n{tool_schemas}"

    return cleaned


# ---------------------------------------------------------------------------
# PromptEvolver result
# ---------------------------------------------------------------------------


@dataclass
class PromptEvolverResult:
    """Outcome of a prompt evolution run.

    Attributes:
        best_prompt:      The highest-scoring prompt template.
        best_temperature: Optimal temperature setting.
        best_top_p:       Optimal top-p setting.
        best_accuracy:    Accuracy on the evaluation dataset.
        best_score:       Raw fitness score (accuracy * 100).
        history:          Per-generation ``(gen, best_score)`` trace.
        all_candidates:   Every candidate evaluated, sorted best-first.
        wall_time:        Total elapsed seconds.
        iterations_run:   Generations completed.
        llm_backend:      Which backend was used.
    """

    best_prompt: str
    best_temperature: float
    best_top_p: float
    best_accuracy: float
    best_score: float
    history: list[tuple[int, float]]
    all_candidates: list[PromptCandidate]
    wall_time: float
    iterations_run: int
    llm_backend: str

    def summary(self) -> str:
        lines = [
            "Prompt Evolver Result",
            "=" * 50,
            f"  Iterations:       {self.iterations_run}",
            f"  Best accuracy:    {self.best_accuracy:.1%}",
            f"  Best temperature: {self.best_temperature:.3f}",
            f"  Best top-p:       {self.best_top_p:.3f}",
            f"  Candidates tried: {len(self.all_candidates)}",
            f"  Wall time:        {self.wall_time:.1f}s",
            f"  LLM backend:      {self.llm_backend}",
            "",
            "Best prompt template:",
            "-" * 50,
            self.best_prompt,
        ]
        return "\n".join(lines)

    def lineage_json(self) -> list[dict]:
        """Return lineage records for all candidates as a list of dicts.

        Each dict contains the candidate's hash, parent_hashes, operation,
        generation, island_id, score, temperature, top_p, and full template.
        Suitable for JSON serialisation and the lineage tree visualiser.
        """
        return [
            {
                "hash": c.hash,
                "parent_hashes": c.parent_hashes,
                "operation": c.operation,
                "generation": c.generation,
                "island_id": c.island_id,
                "score": round(c.score, 2),
                "temperature": round(c.temperature, 4),
                "top_p": round(c.top_p, 4),
                "template": c.template,
            }
            for c in self.all_candidates
        ]


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def count_prompt_tokens(text: str) -> int:
    """Count tokens in a prompt string.

    Uses *tiktoken* with the ``cl100k_base`` encoding (GPT-4 family).
    Falls back to ``len(text) // 4`` when tiktoken is not installed.
    """
    try:
        import tiktoken  # type: ignore[import-untyped]

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except (ImportError, KeyError):
        return len(text) // 4


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


def _feasibility_key(candidate: PromptCandidate) -> tuple[int, float]:
    """Sort key implementing feasibility-first ordering.

    Candidates with zero ``penalty_violations`` always rank above those
    with any violations.  Within the same feasibility tier, higher
    ``score`` wins.  Returns a tuple suitable for ``max(..., key=)``
    and ``sorted(..., reverse=True)``.
    """
    feasible = 1 if candidate.penalty_violations == 0 else 0
    return (feasible, candidate.score)


class PromptEvolver:
    """Evolutionary prompt optimiser for zero-shot tool classification.

    Combines:
    - Island-model evolution of prompt template text (mutation + crossover)
    - Continuous optimisation of temperature and top-p via differential
      evolution within each candidate

    Parameters
    ----------
    tools :
        List of :class:`Tool` objects defining the agent's capabilities.
    eval_dataset :
        Evaluation samples mapping user queries to expected tool calls.
    config :
        Tuning knobs for the evolution loop and LLM client.
    seed :
        Random seed for reproducibility.
    verbose :
        Print progress to stderr.
    """

    def __init__(
        self,
        tools: list[Tool],
        eval_dataset: list[EvalSample],
        config: Optional[PromptEvolverConfig] = None,
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        self.tools = tools
        self.eval_dataset = eval_dataset
        self.config = config or PromptEvolverConfig()
        self.verbose = verbose
        self._rng = np.random.default_rng(seed)
        self._client = LLMClient(self.config)
        self._tool_names = [t.name for t in tools]
        self._tool_schemas_str = "\n".join(
            f"  - {t.schema_str()}" for t in tools
        )
        self._all_candidates: list[PromptCandidate] = []
        self._history: list[tuple[int, float]] = []
        self._require_tool_schemas = bool(tools)
        self._error_profile = ErrorProfile()
        self._failure_examples: list[tuple[str, str, str]] = []
        self._adaptive_pool: list[str] = []

    def run(self) -> PromptEvolverResult:
        """Run the evolutionary prompt optimisation loop."""
        t0 = time.perf_counter()

        logger.info(
            "Starting evolution: %d generations, population=%d, "
            "islands=%d, backend=%s",
            self.config.iterations,
            self.config.population_size,
            self.config.num_islands,
            self.config.backend.value,
        )

        if not self._client.is_available():
            logger.warning(
                "LLM backend %s not available — running in mock mode "
                "(scores will be random). Start Ollama or configure Azure OpenAI.",
                self.config.backend.value,
            )

        # Initialise islands with seed templates
        islands: list[list[PromptCandidate]] = [
            [] for _ in range(self.config.num_islands)
        ]

        logger.info("Seeding %d templates across %d islands", len(_SEED_TEMPLATES), self.config.num_islands)
        for i, template in enumerate(_SEED_TEMPLATES):
            assigned_island = i % self.config.num_islands
            candidate = PromptCandidate(
                template=template,
                temperature=float(
                    self._rng.uniform(*self.config.temperature_range)
                ),
                top_p=float(self._rng.uniform(*self.config.top_p_range)),
                generation=0,
                island_id=assigned_island,
                operation="seed",
            )
            candidate.score = self._evaluate_candidate(candidate)
            islands[assigned_island].append(candidate)
            self._all_candidates.append(candidate)
            logger.info(
                "  Seed %d → island %d  score=%.1f%%  template=%.60s...",
                i, assigned_island, candidate.score,
                template.replace('\n', ' '),
            )

        best_overall = max(self._all_candidates, key=lambda c: c.score)
        logger.info("Seed evaluation complete — best seed score=%.1f%%", best_overall.score)

        for gen in range(1, self.config.iterations + 1):
            gen_t0 = time.perf_counter()
            logger.info("── Generation %d/%d ──", gen, self.config.iterations)

            # Reset error tracking per generation for adaptive mutations
            if self.config.adaptive_mutations or self.config.llm_mutation_rate > 0:
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
                    child.score = self._evaluate_candidate(child)
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

                # Merge and select elite (feasibility-first)
                combined = island + new_candidates
                combined.sort(key=_feasibility_key, reverse=True)
                islands[island_id] = combined[: self.config.elite_size]

            # Migration: share best across islands every 5 generations
            if gen % 5 == 0:
                self._migrate(islands)

            # Generate adaptive mutations for the next generation
            if self.config.adaptive_mutations and self._error_profile.worst_categories():
                self._adaptive_pool = generate_adaptive_mutations(
                    self._error_profile,
                    ProblemType.TOOL_ROUTING,
                    self._client,
                )
                logger.info(
                    "  Adaptive mutations: %d hints from %d error categories",
                    len(self._adaptive_pool),
                    len(self._error_profile.worst_categories()),
                )
            else:
                self._adaptive_pool = []

            # Track best (feasibility-first)
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
                "top_p=%.3f  candidates=%d  elapsed=%.1fs  "
                "LLM calls=%d  tokens_in=%d  tokens_out=%d",
                gen, best_overall.score,
                "▲ NEW BEST  " if improved else "",
                best_overall.temperature,
                best_overall.top_p,
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
                    f"temp={best_overall.temperature:.3f}  "
                    f"top_p={best_overall.top_p:.3f}"
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
            best_prompt=best_overall.template.replace(
                "{tool_schemas}", self._tool_schemas_str
            ),
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

    # -- internal ------------------------------------------------------------

    def _breed(
        self, island: list[PromptCandidate], generation: int
    ) -> PromptCandidate:
        """Create a new candidate from island parents via mutation/crossover."""
        # Tournament selection
        parent_a = self._tournament_select(island)

        # Track lineage
        parent_hashes = [parent_a.hash]
        operation = "mutation"

        if self._rng.random() < self.config.crossover_rate and len(island) > 1:
            parent_b = self._tournament_select(island)
            child_template = _crossover_templates(
                parent_a.template,
                parent_b.template,
                self._rng,
                require_tool_schemas=self._require_tool_schemas,
            )
            parent_hashes = [parent_a.hash, parent_b.hash]
            operation = "crossover"
        else:
            child_template = parent_a.template

        # Build mutation pool (static + adaptive)
        mutations = list(get_mutations_for_problem_type(ProblemType.TOOL_ROUTING))
        if self._adaptive_pool:
            mutations = mutations + self._adaptive_pool

        # LLM-assisted mutation path: rewrite using failure cases
        if (
            self.config.llm_mutation_rate > 0
            and self._rng.random() < self.config.llm_mutation_rate
            and self._failure_examples
        ):
            n_fail = min(6, len(self._failure_examples))
            fail_idx = self._rng.choice(
                len(self._failure_examples), size=n_fail, replace=False
            )
            sampled_failures = [self._failure_examples[int(i)] for i in fail_idx]
            child_template = _llm_mutate_template(
                child_template,
                sampled_failures,
                self._client,
                ProblemType.TOOL_ROUTING,
                require_tool_schemas=self._require_tool_schemas,
            )
            operation = "llm_mutation"
        elif self._rng.random() < self.config.mutation_rate:
            # Mutate template
            child_template = _mutate_template(
                child_template,
                self._rng,
                self.config.mutation_rate,
                require_tool_schemas=self._require_tool_schemas,
                mutations=mutations,
            )

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
                ProblemType.TOOL_ROUTING,
                require_tool_schemas=self._require_tool_schemas,
            )
            operation = "describe_entities"
            logger.debug("Breed: applied describe_entities mutation")

        # Optional LLM-based clarity refinement
        if self.config.refine_after_splice:
            child_template = _refine_template(
                child_template,
                self._client,
                require_tool_schemas=self._require_tool_schemas,
            )

        # Mutate continuous params with DE-style perturbation
        temp = parent_a.temperature + float(self._rng.normal(0, 0.1))
        temp = float(
            np.clip(temp, *self.config.temperature_range)
        )
        top_p = parent_a.top_p + float(self._rng.normal(0, 0.05))
        top_p = float(np.clip(top_p, *self.config.top_p_range))

        return PromptCandidate(
            template=child_template,
            temperature=temp,
            top_p=top_p,
            generation=generation,
            parent_hashes=parent_hashes,
            operation=operation,
        )

    def _tournament_select(
        self, island: list[PromptCandidate], k: int = 3
    ) -> PromptCandidate:
        """Tournament selection: feasibility-first, then best score.

        A candidate with zero penalty violations always beats one with
        any violations, regardless of raw fitness.

        When ``minimize_tokens`` is enabled with a positive
        ``token_accuracy_band``, candidates in the same accuracy band
        are compared by prompt length (shorter wins).
        """
        k = min(k, len(island))
        indices = self._rng.choice(len(island), size=k, replace=False)
        contestants = [island[int(i)] for i in indices]

        cfg = self.config
        if cfg.minimize_tokens and cfg.token_accuracy_band > 0:
            band = cfg.token_accuracy_band

            def _token_aware_key(c: PromptCandidate) -> tuple:
                feasible = 1 if c.penalty_violations == 0 else 0
                bucket = int(c.score // band)
                tokens = count_prompt_tokens(c.template)
                return (feasible, bucket, -tokens)

            return max(contestants, key=_token_aware_key)

        return max(contestants, key=_feasibility_key)

    def _migrate(self, islands: list[list[PromptCandidate]]) -> None:
        """Copy the best candidate from each island to a random neighbour."""
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
                operation="migration",
                parent_hashes=[best.hash],
            )
            islands[dest].append(migrant)
            self._all_candidates.append(migrant)

    def _apply_token_efficiency(
        self, raw_score: float, candidate: PromptCandidate,
    ) -> float:
        """Blend a baseline-relative efficiency bonus into the raw score.

        When ``minimize_tokens`` is enabled, computes
        ``efficiency = baseline_tokens / candidate_tokens`` (capped at
        ``token_efficiency_cap``), converts it to a 0–100 bonus, and
        blends it with the raw accuracy score using ``token_weight``.

        Returns the raw score unchanged when token optimisation is
        disabled or no baseline is set.
        """
        cfg = self.config
        if not (cfg.minimize_tokens and cfg.token_weight > 0
                and cfg.baseline_prompt_tokens > 0):
            return raw_score

        prompt_tokens = count_prompt_tokens(candidate.template)
        efficiency = cfg.baseline_prompt_tokens / max(prompt_tokens, 1)
        efficiency_bonus = (
            min(efficiency, cfg.token_efficiency_cap)
            / cfg.token_efficiency_cap
            * 100.0
        )
        return (
            raw_score * (1 - cfg.token_weight)
            + efficiency_bonus * cfg.token_weight
        )

    def _evaluate_candidate(self, candidate: PromptCandidate) -> float:
        """Evaluate a candidate prompt on the dataset. Returns accuracy * 100."""
        system_prompt = candidate.template.replace(
            "{tool_schemas}", self._tool_schemas_str
        )

        # Optionally subsample the evaluation set
        if (
            self.config.eval_sample_size
            and self.config.eval_sample_size < len(self.eval_dataset)
        ):
            indices = self._rng.choice(
                len(self.eval_dataset),
                size=self.config.eval_sample_size,
                replace=False,
            )
            samples = [self.eval_dataset[int(i)] for i in indices]
        else:
            samples = self.eval_dataset

        total_score = 0.0
        for sample in samples:
            response = self._client.complete(
                system_prompt=system_prompt,
                user_message=sample.query,
                temperature=candidate.temperature,
                top_p=candidate.top_p,
            )

            if response is None:
                # LLM not available — random score for testing
                total_score += float(self._rng.uniform(0, 0.5))
                continue

            predicted_tool, predicted_params = parse_tool_response(
                response, self._tool_names
            )
            sample_score = score_response(
                predicted_tool,
                predicted_params,
                sample.expected_tool,
                sample.expected_params,
            )
            total_score += sample_score

            # Track error profile for adaptive mutations / LLM mutation
            expected_tool = sample.expected_tool
            correct = sample_score >= 0.5
            self._error_profile.record(expected_tool, correct)
            if not correct:
                self._failure_examples.append(
                    (sample.query, predicted_tool or response[:80], expected_tool)
                )

        accuracy = total_score / len(samples) if samples else 0.0
        raw_score = accuracy * 100.0
        return self._apply_token_efficiency(raw_score, candidate)


# ---------------------------------------------------------------------------
# Convenience: run with EvoSim CMA-ES for continuous param tuning
# ---------------------------------------------------------------------------


def evolve_prompt_with_cmaes(
    tools: list[Tool],
    eval_dataset: list[EvalSample],
    prompt_template: str,
    config: Optional[PromptEvolverConfig] = None,
    max_generations: int = 50,
    seed: int = 42,
    verbose: bool = True,
) -> PromptEvolverResult:
    """Tune temperature and top-p for a fixed prompt template using CMA-ES.

    This is useful when you already have a good prompt template and want
    to find the optimal continuous parameters.

    Uses EvoSim's CMA-ES algorithm internally.
    """
    from evosim import Problem, Variable
    from MutaGenAI.algorithms.cmaes import CMAES
    from MutaGenAI.problem import Direction

    cfg = config or PromptEvolverConfig()
    client = LLMClient(cfg)
    tool_names = [t.name for t in tools]
    tool_schemas_str = "\n".join(f"  - {t.schema_str()}" for t in tools)
    rng = np.random.default_rng(seed)

    best_found: dict[str, Any] = {"score": -1.0, "temp": 0.1, "top_p": 0.95}

    def fitness_fn(X: np.ndarray) -> np.ndarray:
        results = np.zeros(len(X))
        for i in range(len(X)):
            temperature = float(np.clip(X[i, 0], *cfg.temperature_range))
            top_p = float(np.clip(X[i, 1], *cfg.top_p_range))
            system_prompt = prompt_template.replace("{tool_schemas}", tool_schemas_str)

            total_score = 0.0
            for sample in eval_dataset:
                response = client.complete(
                    system_prompt, sample.query, temperature, top_p
                )
                if response is None:
                    total_score += float(rng.uniform(0, 0.5))
                    continue
                pred_tool, pred_params = parse_tool_response(
                    response, tool_names
                )
                total_score += score_response(
                    pred_tool, pred_params,
                    sample.expected_tool, sample.expected_params,
                )

            accuracy = total_score / len(eval_dataset) if eval_dataset else 0.0
            # CMA-ES minimises, so negate
            results[i] = -accuracy

            if accuracy > best_found["score"]:
                best_found["score"] = accuracy
                best_found["temp"] = temperature
                best_found["top_p"] = top_p

        return results

    variables = [
        Variable.continuous("temperature", cfg.temperature_range[0], cfg.temperature_range[1]),
        Variable.continuous("top_p", cfg.top_p_range[0], cfg.top_p_range[1]),
    ]
    problem = Problem(
        fitness_fn=fitness_fn,
        variables=variables,
        direction=Direction.MINIMIZE,
        name="Prompt Param Tuning",
    )

    t0 = time.perf_counter()
    cmaes = CMAES(problem, seed=seed)
    result = cmaes.run(max_generations=max_generations)
    wall_time = time.perf_counter() - t0

    return PromptEvolverResult(
        best_prompt=prompt_template.replace("{tool_schemas}", tool_schemas_str),
        best_temperature=best_found["temp"],
        best_top_p=best_found["top_p"],
        best_accuracy=best_found["score"],
        best_score=best_found["score"] * 100,
        history=[],
        all_candidates=[],
        wall_time=wall_time,
        iterations_run=max_generations,
        llm_backend=cfg.backend.value,
    )
