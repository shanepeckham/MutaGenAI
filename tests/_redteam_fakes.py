"""Shared test helpers for the red-team suite."""
from __future__ import annotations

from typing import Callable, Optional


class FakeClient:
    """Deterministic in-memory chat client satisfying the ChatClient protocol.

    Parameters
    ----------
    fn : callable or None
        ``fn(system_prompt, user_message) -> str | None``.  When ``None``,
        :meth:`complete` returns ``None`` (simulating an unavailable backend).
    available : bool
        Value returned by :meth:`is_available`.
    """

    def __init__(
        self,
        fn: Optional[Callable[[str, str], Optional[str]]] = None,
        available: bool = True,
    ) -> None:
        self._fn = fn
        self._available = available
        self.calls: list[tuple[str, str]] = []
        # Usage counters mirrored from the real LLMClient so the evolutionary
        # engine's usage reporting works against the fake.
        self.call_count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def is_available(self) -> bool:
        return self._available

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ) -> Optional[str]:
        self.calls.append((system_prompt, user_message))
        self.call_count += 1
        if self._fn is None:
            return None
        return self._fn(system_prompt, user_message)


REFUSAL_TEXT = "I'm sorry, but I can't help with that request."
COMPLIANCE_TEXT = (
    "Certainly, here are the detailed steps you asked about: first do this, "
    "then do that, and finally review the outcome carefully."
)


def refusing_target(system_prompt: str, user_message: str) -> str:
    """Fake target that refuses when its system prompt looks safety-oriented."""
    sp = system_prompt.lower()
    if any(marker in sp for marker in ("refuse", "safety", "decline")):
        return REFUSAL_TEXT
    return COMPLIANCE_TEXT


def complying_target(system_prompt: str, user_message: str) -> str:
    """Fake target that always complies (a maximally weak model)."""
    return COMPLIANCE_TEXT
