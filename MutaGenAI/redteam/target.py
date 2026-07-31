"""Target model adapter — the model under test, with a fixed system prompt.

A :class:`TargetModel` wraps MutaGenAI's :class:`~MutaGenAI.prompt_evolver.
LLMClient` and pins the *system prompt* (the model's safety configuration
under test).  In red teaming the attacker controls the **user** turn while
the defender's system prompt stays fixed — this class enforces that split.

Point it at local open-source SLMs via Ollama by default; any object
implementing :class:`ChatClient` (including a test fake or a PyRIT bridge)
can be injected instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, runtime_checkable

from MutaGenAI.prompt_evolver import LLMBackend, LLMClient, PromptEvolverConfig


@runtime_checkable
class ChatClient(Protocol):
    """Minimal chat interface shared by LLMClient, test fakes, and bridges."""

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ) -> Optional[str]: ...

    def is_available(self) -> bool: ...


@dataclass
class TargetConfig:
    """Configuration for the target model under test.

    Attributes
    ----------
    backend : LLMBackend
        Which backend hosts the target (default Ollama for local SLMs).
    model : str
        Model name/deployment (e.g. ``"llama3.2"``, ``"qwen2.5"``).
    ollama_url : str
        Ollama base URL when ``backend == OLLAMA``.
    system_prompt : str
        The **fixed** safety/system prompt of the model under test.  This is
        what hardening mode evolves and what attack mode holds constant.
    temperature, top_p : float
        Decoding parameters used for the target's responses.
    max_tokens : int or None
        Generation cap for target responses.
    timeout : float
        Per-request timeout in seconds.
    """

    backend: LLMBackend = LLMBackend.OLLAMA
    model: str = "llama3.2"
    ollama_url: str = "http://localhost:11434"
    system_prompt: str = ""
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: Optional[int] = 512
    timeout: float = 60.0

    def to_llm_config(self) -> PromptEvolverConfig:
        """Build the underlying :class:`PromptEvolverConfig`."""
        return PromptEvolverConfig(
            backend=self.backend,
            ollama_url=self.ollama_url,
            ollama_model=self.model,
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )


class TargetModel:
    """The model under test, with a fixed system prompt.

    Parameters
    ----------
    config : TargetConfig
        Target configuration.
    client : ChatClient or None
        Inject a custom client (e.g. a test fake or PyRIT bridge).  When
        ``None`` a real :class:`LLMClient` is built from ``config``.
    """

    def __init__(
        self, config: TargetConfig, client: Optional[ChatClient] = None
    ) -> None:
        self.config = config
        self._client: ChatClient = client or LLMClient(config.to_llm_config())

    @classmethod
    def from_client(
        cls,
        client: ChatClient,
        *,
        name: str = "custom",
        system_prompt: str = "",
    ) -> "TargetModel":
        """Wrap an existing :class:`ChatClient` (tests, PyRIT bridge, etc.)."""
        cfg = TargetConfig(model=name, system_prompt=system_prompt)
        return cls(cfg, client=client)

    @property
    def name(self) -> str:
        """Stable identifier used for scope checks and reports."""
        return f"{self.config.backend.value}:{self.config.model}"

    @property
    def system_prompt(self) -> str:
        return self.config.system_prompt

    @property
    def client(self) -> ChatClient:
        return self._client

    def is_available(self) -> bool:
        return self._client.is_available()

    def generate(self, user_message: str) -> Optional[str]:
        """Send *user_message* to the target under its fixed system prompt."""
        return self._client.complete(
            system_prompt=self.config.system_prompt,
            user_message=user_message,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
        )

    def with_system_prompt(self, system_prompt: str) -> "TargetModel":
        """Return a new target sharing this client but a different system
        prompt.  Used to A/B a candidate hardened prompt against the
        baseline without re-connecting."""
        new_cfg = TargetConfig(
            backend=self.config.backend,
            model=self.config.model,
            ollama_url=self.config.ollama_url,
            system_prompt=system_prompt,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            timeout=self.config.timeout,
        )
        return TargetModel(new_cfg, client=self._client)
