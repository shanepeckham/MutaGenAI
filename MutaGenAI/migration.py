"""Model migration — baseline evaluation and three-way comparison reporting.

Phase 2 of model-migration support. Provides side-effect-free utilities to
*measure* a prompt's accuracy on a given model and to contrast three
configurations when swapping models:

* **A_old**      — old model + old prompt (the accuracy bar to preserve)
* **A_transfer** — new model + old prompt (naive-swap baseline)
* **A_evolved**  — new model + evolved prompt (the migrated result)

:class:`MigrationReport` also surfaces the *regression set* — samples that the
old model got right but the new model breaks — which is the real migration
risk and the natural focus for evolution.

These helpers reuse the existing tool-response parser and scorer, so accuracy
here is computed exactly as during evolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from MutaGenAI.prompt_evolver import (
    EvalSample,
    LLMBackend,
    LLMClient,
    PromptEvolverConfig,
    Tool,
    parse_tool_response,
    score_response,
)


# ---------------------------------------------------------------------------
# Per-sample and per-prompt evaluation
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    """Outcome of evaluating one sample under one prompt/model."""

    query: str
    expected_tool: str
    predicted_tool: Optional[str]
    score: float
    correct: bool


@dataclass
class PromptEvaluation:
    """Result of evaluating a prompt across an eval set on one model."""

    prompt: str
    accuracy: float  # mean sample score in [0, 1]
    results: list[SampleResult]
    temperature: float
    top_p: float

    @property
    def correct_queries(self) -> set[str]:
        return {r.query for r in self.results if r.correct}

    @property
    def num_correct(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def total(self) -> int:
        return len(self.results)


def evaluate_prompt(
    prompt: str,
    tools: list[Tool],
    samples: list[EvalSample],
    client: LLMClient,
    *,
    temperature: float = 0.1,
    top_p: float = 0.95,
) -> PromptEvaluation:
    """Measure *prompt* on *client* over *samples* (no evolution, no side effects).

    Substitutes ``{tool_schemas}`` into the prompt, sends each sample query,
    parses the tool response, and scores it with the same logic used during
    evolution. An unreachable client (``complete`` returns ``None``) counts as
    an incorrect, zero-score sample so baselines stay deterministic.
    """
    tool_names = [t.name for t in tools]
    tool_schemas_str = "\n".join(f"  - {t.schema_str()}" for t in tools)
    system_prompt = prompt.replace("{tool_schemas}", tool_schemas_str)

    results: list[SampleResult] = []
    for sample in samples:
        response = client.complete(
            system_prompt=system_prompt,
            user_message=sample.query,
            temperature=temperature,
            top_p=top_p,
        )
        if response is None:
            results.append(
                SampleResult(sample.query, sample.expected_tool, None, 0.0, False)
            )
            continue
        predicted_tool, predicted_params = parse_tool_response(
            response, tool_names
        )
        sample_score = score_response(
            predicted_tool,
            predicted_params,
            sample.expected_tool,
            sample.expected_params,
        )
        results.append(
            SampleResult(
                sample.query,
                sample.expected_tool,
                predicted_tool,
                sample_score,
                sample_score >= 0.5,
            )
        )

    accuracy = (
        sum(r.score for r in results) / len(results) if results else 0.0
    )
    return PromptEvaluation(prompt, accuracy, results, temperature, top_p)


def make_client(
    model: str,
    backend: LLMBackend = LLMBackend.OLLAMA,
    **config_kwargs,
) -> LLMClient:
    """Build an :class:`LLMClient` for *model* on *backend* (migration convenience).

    Routes *model* to the correct config field per backend so the same call
    works for Ollama, OpenAI, and Azure OpenAI targets.
    """
    kwargs = dict(config_kwargs)
    if backend == LLMBackend.OLLAMA:
        kwargs["ollama_model"] = model
    elif backend == LLMBackend.OPENAI:
        kwargs["openai_model"] = model
    elif backend == LLMBackend.AZURE_OPENAI:
        kwargs["azure_deployment"] = model
    return LLMClient(PromptEvolverConfig(backend=backend, **kwargs))


# ---------------------------------------------------------------------------
# Three-way migration report
# ---------------------------------------------------------------------------


@dataclass
class MigrationReport:
    """Three-way comparison for a model swap, plus the regression set."""

    source_model: str
    target_model: str
    a_old: Optional[float]  # old model + old prompt (None if source not eval'd)
    a_transfer: float       # new model + old prompt
    a_evolved: float        # new model + evolved prompt
    evolved_prompt: str
    transfer_regressions: list[str] = field(default_factory=list)
    remaining_regressions: list[str] = field(default_factory=list)
    recovered: list[str] = field(default_factory=list)
    decoding_before: tuple[float, float] = (0.0, 0.0)
    decoding_after: tuple[float, float] = (0.0, 0.0)

    @property
    def delta_vs_transfer(self) -> float:
        """Accuracy gained by evolution over the naive swap."""
        return self.a_evolved - self.a_transfer

    @property
    def delta_vs_old(self) -> Optional[float]:
        """Accuracy of the migrated prompt relative to the old-model bar."""
        return None if self.a_old is None else self.a_evolved - self.a_old

    @property
    def preserved(self) -> bool:
        """True if the migrated prompt meets or beats the old-model bar."""
        return self.a_old is None or self.a_evolved >= self.a_old

    @classmethod
    def build(
        cls,
        *,
        transfer_eval: PromptEvaluation,
        evolved_eval: PromptEvaluation,
        source_model: str,
        target_model: str,
        source_eval: Optional[PromptEvaluation] = None,
    ) -> "MigrationReport":
        """Assemble a report from the three evaluations.

        When *source_eval* is ``None`` (old model unavailable), regressions are
        measured against the naive-transfer baseline instead of the old model.
        """
        reference_correct = (
            source_eval.correct_queries
            if source_eval is not None
            else transfer_eval.correct_queries
        )
        transfer_correct = transfer_eval.correct_queries
        evolved_correct = evolved_eval.correct_queries

        transfer_regressions = sorted(reference_correct - transfer_correct)
        remaining_regressions = sorted(reference_correct - evolved_correct)
        recovered = sorted(
            (reference_correct - transfer_correct)
            - (reference_correct - evolved_correct)
        )
        return cls(
            source_model=source_model,
            target_model=target_model,
            a_old=source_eval.accuracy if source_eval is not None else None,
            a_transfer=transfer_eval.accuracy,
            a_evolved=evolved_eval.accuracy,
            evolved_prompt=evolved_eval.prompt,
            transfer_regressions=transfer_regressions,
            remaining_regressions=remaining_regressions,
            recovered=recovered,
            decoding_before=(transfer_eval.temperature, transfer_eval.top_p),
            decoding_after=(evolved_eval.temperature, evolved_eval.top_p),
        )

    def summary(self) -> str:
        a_old = "n/a" if self.a_old is None else f"{self.a_old:.1%}"
        dv_old = self.delta_vs_old
        dv_old_s = "n/a" if dv_old is None else f"{dv_old:+.1%}"
        lines = [
            f"Migration: {self.source_model} -> {self.target_model}",
            "=" * 56,
            f"  A_old      (old model, old prompt)      {a_old}",
            f"  A_transfer (new model, old prompt)      {self.a_transfer:.1%}",
            f"  A_evolved  (new model, evolved prompt)  {self.a_evolved:.1%}",
            f"  Gain vs naive swap:   {self.delta_vs_transfer:+.1%}",
            f"  Delta vs old-model:   {dv_old_s}"
            f"   ({'preserved' if self.preserved else 'REGRESSED'})",
            f"  Transfer regressions: {len(self.transfer_regressions)}",
            f"  Recovered by evolve:  {len(self.recovered)}",
            f"  Remaining regressions:{len(self.remaining_regressions)}",
            f"  Decoding: temp {self.decoding_before[0]:.2f}->"
            f"{self.decoding_after[0]:.2f}  "
            f"top_p {self.decoding_before[1]:.2f}->{self.decoding_after[1]:.2f}",
        ]
        if self.remaining_regressions:
            lines.append("  Still failing:")
            lines += [f"    - {q}" for q in self.remaining_regressions[:10]]
        return "\n".join(lines)
