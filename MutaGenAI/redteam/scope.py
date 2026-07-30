"""Authorization scope — the mandatory guardrail for red-team runs.

Red teaming must only ever be performed against models you own or are
explicitly authorized to test.  Every :class:`~MutaGenAI.redteam.harness.
RedTeamHarness` requires an authorized :class:`RedTeamScope` before it will
run.  This module deliberately fails *closed*: an unconfigured scope raises.

The scope is intentionally simple and self-documenting so it can be committed
to a private tracker (never with secrets) as an audit record of *who*
authorized *what*.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


class RedTeamAuthorizationError(RuntimeError):
    """Raised when a red-team run is attempted without valid authorization."""


@dataclass
class RedTeamScope:
    """Declares who is authorized to test which targets, and for what purpose.

    Parameters
    ----------
    operator : str
        Identifier of the person/team running the assessment.
    targets : list[str]
        Target model identifiers that are in scope (e.g.
        ``["ollama:llama3.2", "ollama:qwen2.5"]``).  A run is refused unless
        the target being tested is listed here.
    authorized : bool
        Must be explicitly set to ``True``.  Records that authorization was
        granted for this assessment.
    acknowledged_policy : bool
        Must be explicitly set to ``True``.  Records that the operator
        acknowledges the responsible-use policy: findings are used only to
        harden models and are disclosed responsibly, and discovered attack
        strings are never redistributed.
    purpose : str
        Free-text description of the engagement's defensive purpose.
    notes : str
        Optional audit notes (ticket ref, authorizer, expiry, etc.).
    """

    operator: str
    targets: list[str] = field(default_factory=list)
    authorized: bool = False
    acknowledged_policy: bool = False
    purpose: str = "Defensive red teaming / model hardening"
    notes: str = ""

    # -- construction ------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> "RedTeamScope":
        """Load a scope from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_file(self, path: str | Path) -> Path:
        """Write the scope to a JSON file and return the path."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return p

    # -- enforcement -------------------------------------------------------

    def require(self, target: Optional[str] = None) -> None:
        """Assert this scope authorizes testing *target*.  Raise otherwise.

        Called automatically by the harness.  Fails closed on every missing
        precondition so an empty or default scope can never run.
        """
        if not self.authorized:
            raise RedTeamAuthorizationError(
                "RedTeamScope.authorized is False. Set it to True only for "
                "targets you own or are explicitly permitted to test."
            )
        if not self.acknowledged_policy:
            raise RedTeamAuthorizationError(
                "RedTeamScope.acknowledged_policy is False. You must "
                "acknowledge the responsible-use policy before running: "
                "findings are for hardening and responsible disclosure only."
            )
        if not self.operator.strip():
            raise RedTeamAuthorizationError("RedTeamScope.operator is empty.")
        if not self.targets:
            raise RedTeamAuthorizationError(
                "RedTeamScope.targets is empty — declare the models in scope."
            )
        if target is not None and target not in self.targets:
            raise RedTeamAuthorizationError(
                f"Target {target!r} is not in the authorized scope "
                f"{self.targets!r}."
            )

    def authorizes(self, target: str) -> bool:
        """Return ``True`` if *target* may be tested under this scope."""
        try:
            self.require(target)
            return True
        except RedTeamAuthorizationError:
            return False
