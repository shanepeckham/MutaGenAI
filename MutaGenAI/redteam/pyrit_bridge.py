"""Microsoft PyRIT bridge — datasets, converters, scorers, and targets.

This module pairs MutaGenAI's evolutionary search with `Microsoft PyRIT
<https://github.com/microsoft/PyRIT>`_ (``pip install pyrit``).  PyRIT brings
the curated harmful-behavior datasets, prompt converters, safety scorers, and
broad target coverage; MutaGenAI brings the evolutionary/quality-diversity
search that PyRIT lacks.

Everything here is **lazy** — importing this module never imports PyRIT.  Each
function imports PyRIT on demand and raises a clear, actionable error if it is
missing, so the rest of the harness runs fine without it.

The bridge is validated against ``pyrit>=0.14``.  Because PyRIT's async APIs
evolve between releases, the wrappers are defensive (``getattr`` probing,
``asyncio`` shims) and are marked experimental.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from MutaGenAI.strategies import Scorer
from MutaGenAI.redteam.target import ChatClient


PYRIT_INSTALL_HINT = (
    "Microsoft PyRIT is not installed. Install it with:\n"
    "    pip install pyrit\n"
    "or add the optional extra:\n"
    "    pip install 'MutaGenAI[redteam]'\n"
    "See https://github.com/microsoft/PyRIT"
)


def pyrit_available() -> bool:
    """Return ``True`` if PyRIT can be imported."""
    try:
        import pyrit  # noqa: F401  (import probe)

        return True
    except Exception:
        return False


def require_pyrit() -> None:
    """Raise :class:`ImportError` with install guidance if PyRIT is missing."""
    if not pyrit_available():
        raise ImportError(PYRIT_INSTALL_HINT)


def _run_async(coro: Any) -> Any:
    """Run *coro* to completion, tolerating an already-running loop."""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        # A loop is already running (e.g. Jupyter) — use a fresh one.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# ---------------------------------------------------------------------------
# Behaviors / datasets
# ---------------------------------------------------------------------------


def load_behaviors(
    source: str = "file",
    *,
    path: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[str]:
    """Load harmful-behavior goal strings.

    Parameters
    ----------
    source : str
        ``"file"`` (default) loads from a local ``path`` you provide — the
        reliable, offline route.  ``"jailbreakbench"`` loads the ungated
        JailbreakBench JBB-Behaviors harmful split directly from HuggingFace
        (needs the ``datasets`` package + network, no PyRIT).  Any other value
        names a PyRIT dataset fetcher (e.g. ``"harmbench"``, ``"adv_bench"``),
        which requires PyRIT and network access.
    path : str or None
        For ``source="file"``: a ``.json`` list (of strings, or of objects
        with a ``prompt``/``behavior``/``value`` field) or a ``.txt`` file
        with one behavior per line.  In ``.txt`` files, blank lines and lines
        starting with ``#`` (comments) are ignored.
    limit : int or None
        Cap the number of behaviors returned.

    Notes
    -----
    This library ships **no** harmful content.  You supply your own behavior
    set (e.g. JailbreakBench, HarmBench, AdvBench, or PyRIT) for your
    authorized assessment.
    """
    if source == "file":
        if not path:
            raise ValueError("source='file' requires a path=... argument.")
        behaviors = _load_behaviors_from_file(path)
    elif source in _HF_BEHAVIOR_LOADERS:
        behaviors = _HF_BEHAVIOR_LOADERS[source]()
    else:
        behaviors = _load_behaviors_from_pyrit(source)
    return behaviors[:limit] if limit else behaviors


def _load_jailbreakbench() -> list[str]:
    """Load the JailbreakBench JBB-Behaviors harmful split (ungated, MIT)."""
    from datasets import load_dataset

    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")
    goals: list[str] = []
    for row in ds:
        goal = row.get("Goal") or row.get("goal")
        if goal:
            goals.append(str(goal))
    return goals


# HuggingFace-native behavior loaders (no PyRIT required).
_HF_BEHAVIOR_LOADERS = {"jailbreakbench": _load_jailbreakbench}


def _load_behaviors_from_file(path: str) -> list[str]:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        data = json.loads(text)
        out: list[str] = []
        for item in data:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for key in ("prompt", "behavior", "value", "text", "goal"):
                    if key in item:
                        out.append(str(item[key]))
                        break
        return out
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _load_behaviors_from_pyrit(source: str) -> list[str]:
    require_pyrit()
    from pyrit import datasets as pyrit_datasets  # type: ignore

    fetcher_name = f"fetch_{source}_dataset"
    fetcher = getattr(pyrit_datasets, fetcher_name, None)
    if fetcher is None:
        available = [n for n in dir(pyrit_datasets) if n.startswith("fetch_")]
        raise ValueError(
            f"Unknown PyRIT dataset {source!r} ({fetcher_name} not found). "
            f"Available fetchers: {available}"
        )
    dataset = fetcher()
    prompts = getattr(dataset, "prompts", dataset)
    return [str(getattr(p, "value", p)) for p in prompts]


# ---------------------------------------------------------------------------
# Converters → seed diversification
# ---------------------------------------------------------------------------


def expand_seeds_with_converters(
    seeds: list[str],
    converter_names: list[str],
    *,
    keep_originals: bool = True,
) -> list[str]:
    """Diversify attack scaffolds using PyRIT prompt converters.

    Applies each named PyRIT converter (e.g. ``"Base64Converter"``,
    ``"ROT13Converter"``, ``"CaesarConverter"``) to every seed scaffold,
    producing transformed variants that seed a more diverse initial
    population.  The transformation is applied to caller-supplied scaffolds;
    this function does not generate attack content of its own.
    """
    require_pyrit()
    from pyrit import prompt_converter as pc  # type: ignore

    expanded: list[str] = list(seeds) if keep_originals else []
    for name in converter_names:
        cls = getattr(pc, name, None)
        if cls is None:
            raise ValueError(f"PyRIT converter {name!r} not found.")
        converter = cls()
        for seed in seeds:
            try:
                result = _run_async(converter.convert_async(prompt=seed))
                expanded.append(str(getattr(result, "output_text", result)))
            except Exception:
                # A converter that cannot handle a template is skipped rather
                # than aborting the whole expansion.
                continue
    return expanded


# ---------------------------------------------------------------------------
# Scorer adapter
# ---------------------------------------------------------------------------


class PyRITScorerAdapter(Scorer):
    """Adapt a PyRIT scorer to MutaGenAI's :class:`Scorer` interface.

    Wrap any PyRIT scorer whose ``score_text_async`` returns objects exposing
    a truthy ``get_value()`` / ``score_value``.  ``invert`` flips the mapping
    for refusal-style scorers (where "refused" should map to attack-failure).

    Example
    -------
    >>> from pyrit.score import SelfAskRefusalScorer
    >>> adapter = PyRITScorerAdapter(SelfAskRefusalScorer(chat_target=t),
    ...                              invert=True)  # refused -> 0.0
    """

    def __init__(self, pyrit_scorer: Any, *, invert: bool = False) -> None:
        require_pyrit()
        self._scorer = pyrit_scorer
        self._invert = invert

    def score(
        self, prompt: str, test_input: str, output: str, client
    ) -> float:
        try:
            result = _run_async(self._scorer.score_text_async(text=output))
        except Exception:
            return 0.0
        score_obj = result[0] if isinstance(result, (list, tuple)) else result
        value = getattr(score_obj, "get_value", lambda: None)()
        if value is None:
            value = getattr(score_obj, "score_value", 0.0)
        num = 1.0 if value in (True, "True", "true") else 0.0 if value in (
            False,
            "False",
            "false",
        ) else float(value)
        return (1.0 - num) if self._invert else num


# ---------------------------------------------------------------------------
# Target adapter
# ---------------------------------------------------------------------------


class PyRITTargetClient:
    """Adapt a PyRIT ``PromptChatTarget`` to the :class:`ChatClient` protocol.

    Lets the harness drive any PyRIT-supported target (Azure OpenAI, Ollama,
    HTTP, etc.) through MutaGenAI's evaluation loop.  System-prompt handling is
    best-effort: PyRIT targets take a single prompt, so the system prompt is
    prepended.  Experimental; validated against ``pyrit>=0.14``.
    """

    def __init__(self, pyrit_target: Any) -> None:
        require_pyrit()
        self._target = pyrit_target

    def is_available(self) -> bool:
        return True

    def complete(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.1,
        top_p: float = 0.95,
    ) -> Optional[str]:
        prompt = (
            f"{system_prompt}\n\n{user_message}"
            if system_prompt
            else user_message
        )
        try:
            from pyrit.orchestrator import (  # type: ignore
                PromptSendingOrchestrator,
            )

            orchestrator = PromptSendingOrchestrator(objective_target=self._target)
            responses = _run_async(
                orchestrator.send_prompts_async(prompt_list=[prompt])
            )
            return _extract_pyrit_text(responses)
        except Exception:
            return None


def _extract_pyrit_text(responses: Any) -> Optional[str]:
    """Best-effort extraction of assistant text from a PyRIT response."""
    if not responses:
        return None
    item = responses[0] if isinstance(responses, (list, tuple)) else responses
    for attr in ("get_value", "get_response_text"):
        fn = getattr(item, attr, None)
        if callable(fn):
            try:
                return str(fn())
            except Exception:
                pass
    pieces = getattr(item, "request_pieces", None)
    if pieces:
        return str(getattr(pieces[-1], "converted_value", pieces[-1]))
    return str(item)


def make_target_from_pyrit(
    pyrit_target: Any, *, name: str = "pyrit", system_prompt: str = ""
):
    """Convenience: wrap a PyRIT target as a MutaGenAI :class:`TargetModel`."""
    from MutaGenAI.redteam.target import TargetModel

    return TargetModel.from_client(
        PyRITTargetClient(pyrit_target), name=name, system_prompt=system_prompt
    )
