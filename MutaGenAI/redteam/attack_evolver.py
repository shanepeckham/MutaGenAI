"""Attack evolver — Mode B: evolve the *user turn* against a fixed target.

The stock :class:`~MutaGenAI.strategies.NoEvalPromptEvolver` evolves a system
prompt with fixed user inputs.  Realistic jailbreak red teaming inverts that:
the attacker controls the **user** message while the target's system prompt is
fixed, and a harmful *goal* is templated into an attack scaffold.

:class:`AttackEvolver` subclasses the engine and overrides only the evaluation
step to apply this inversion, reusing all the evolutionary machinery
(islands, mutation, crossover, migration, CMA-ES).

Seed scaffolds are **not** authored by this library — they must be supplied by
the caller (e.g. from PyRIT converters/datasets).  A scaffold is any template
containing a ``{goal}`` placeholder where the harmful behavior is inserted.
"""

from __future__ import annotations

import statistics
from typing import Optional

from MutaGenAI.strategies import NoEvalConfig, NoEvalPromptEvolver, Scorer
from MutaGenAI.redteam.target import TargetModel


class AttackEvolver(NoEvalPromptEvolver):
    """Evolve attack scaffolds (user-turn wrappers) against a fixed target.

    Parameters
    ----------
    target : TargetModel
        The model under test (fixed system prompt).  Candidate scaffolds are
        rendered with each goal and sent as the **user** message.
    task_description : str
        Neutral description of the red-team objective (used for logging/seed
        bootstrap only; not an attack).
    behaviors : list[str]
        Harmful-behavior goal strings to attempt (supplied by the caller, e.g.
        from a HarmBench/PyRIT dataset).
    scorer : Scorer
        Typically :class:`~MutaGenAI.redteam.scorer.AttackSuccessScorer`.
    seed_templates : list[str]
        Attack scaffolds containing ``{goal}``.  **Required** — the library
        never invents attacks.
    config : NoEvalConfig or None
        Evolution parameters.
    goal_placeholder : str
        Placeholder replaced by each behavior (default ``"{goal}"``).
    """

    def __init__(
        self,
        *,
        target: TargetModel,
        task_description: str,
        behaviors: list[str],
        scorer: Scorer,
        seed_templates: list[str],
        config: Optional[NoEvalConfig] = None,
        goal_placeholder: str = "{goal}",
        seed: int = 42,
        verbose: bool = True,
    ) -> None:
        if not seed_templates:
            raise ValueError(
                "AttackEvolver requires caller-supplied seed_templates "
                "(attack scaffolds). This library never authors attacks."
            )
        super().__init__(
            task_description=task_description,
            test_inputs=behaviors,
            scorer=scorer,
            config=config,
            seed_templates=seed_templates,
            seed=seed,
            verbose=verbose,
        )
        self._target = target
        self._goal_placeholder = goal_placeholder

    def _render(self, template: str, goal: str) -> str:
        """Insert *goal* into an attack scaffold."""
        if self._goal_placeholder in template:
            return template.replace(self._goal_placeholder, goal)
        # No placeholder: append the goal so arbitrary scaffolds still work.
        return f"{template}\n\n{goal}"

    def _evaluate(self, candidate) -> float:  # type: ignore[override]
        """Score a scaffold: mean attack-success across all behaviors.

        Mirrors the parent contract (returns a 0–100 score and sets
        ``penalty_violations``) but routes generation through the fixed-system-
        prompt target instead of placing the candidate in the system slot.
        """
        scores: list[float] = []
        for goal in self.test_inputs:
            attack = self._render(candidate.template, goal)
            output = self._target.generate(attack)
            if output is None:
                # Target unreachable — neutral-ish score, keeps search stable.
                scores.append(float(self._rng.uniform(0.0, 0.5)))
                continue
            scores.append(
                self.scorer.score(candidate.template, goal, output, self._client)
            )
        candidate.penalty_violations = 0
        return (statistics.mean(scores) * 100.0) if scores else 0.0
