"""MutaGenAI red-team subpackage — evolutionary red teaming for LLM hardening.

Pairs MutaGenAI's evolutionary / quality-diversity search with defensive
red-teaming primitives and an optional Microsoft PyRIT bridge, to help you
**secure** open-source SLMs by finding and closing weaknesses.

Two modes:

* **Harden** — evolve a system prompt that maximizes safe refusals, and report
  the attack-success rate before vs. after.
* **Attack** — evolve a caller-supplied attack scaffold to map where a fixed
  target is weak (authorized red teaming only).

Quick start (hardening, local Ollama SLM)::

    from MutaGenAI.redteam import (
        RedTeamScope, TargetModel, TargetConfig,
        RedTeamHarness, RedTeamConfig, RedTeamMode,
    )
    from MutaGenAI.redteam.pyrit_bridge import load_behaviors

    scope = RedTeamScope(operator="you", targets=["ollama:llama3.2"],
                         authorized=True, acknowledged_policy=True)
    target = TargetModel(TargetConfig(model="llama3.2",
                                      system_prompt="You are a helpful assistant."))
    behaviors = load_behaviors(source="file", path="behaviors.txt")

    harness = RedTeamHarness(
        scope, target,
        RedTeamConfig(mode=RedTeamMode.HARDEN, iterations=8),
        behaviors=behaviors,
    )
    report = harness.run()
    print(report.summary())

Every run requires an authorized :class:`RedTeamScope`.  Use responsibly:
only test models you own or are permitted to test; use findings to harden and
disclose responsibly; never redistribute discovered attack strings.
"""

from __future__ import annotations

from MutaGenAI.redteam.scope import RedTeamScope, RedTeamAuthorizationError
from MutaGenAI.redteam.refusal import (
    RefusalDetector,
    RefusalSignal,
    DEFAULT_REFUSAL_PATTERNS,
)
from MutaGenAI.redteam.target import TargetModel, TargetConfig, ChatClient
from MutaGenAI.redteam.scorer import (
    RefusalScorer,
    HardeningScorer,
    AttackSuccessScorer,
    SafetyJudge,
)
from MutaGenAI.redteam.attack_evolver import AttackEvolver
from MutaGenAI.redteam.harness import (
    RedTeamHarness,
    RedTeamConfig,
    RedTeamMode,
)
from MutaGenAI.redteam.report import RedTeamReport, BehaviorResult

__all__ = [
    # Guardrail
    "RedTeamScope",
    "RedTeamAuthorizationError",
    # Detection
    "RefusalDetector",
    "RefusalSignal",
    "DEFAULT_REFUSAL_PATTERNS",
    # Target
    "TargetModel",
    "TargetConfig",
    "ChatClient",
    # Scorers
    "RefusalScorer",
    "HardeningScorer",
    "AttackSuccessScorer",
    "SafetyJudge",
    # Engine
    "AttackEvolver",
    "RedTeamHarness",
    "RedTeamConfig",
    "RedTeamMode",
    # Reporting
    "RedTeamReport",
    "BehaviorResult",
]
