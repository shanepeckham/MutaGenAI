"""Load seed templates and penalty definitions from external JSON files."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from MutaGenAI.strategies import ProxyCheck

logger = logging.getLogger(__name__)

_SEED_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "seed_templates"


# ── Condition registry ────────────────────────────────────────────────────

_CONDITION_REGISTRY: dict[str, Callable] = {}


def register_penalty_condition(
    name: str,
    factory: Callable[[Penalty], Callable[[str], bool]],
) -> None:
    """Register a custom penalty condition type.

    Parameters
    ----------
    name :
        Condition identifier (e.g. ``"word_count_gt"``).
    factory :
        ``(Penalty) -> (str -> bool)`` — receives the penalty definition
        and returns a check function that fires (returns ``True``) when
        the output violates the constraint.

    Example
    -------
    >>> def _word_count_gt(p):
    ...     t = int(p.threshold or 0)
    ...     return lambda output: len(output.split()) > t
    >>> register_penalty_condition("word_count_gt", _word_count_gt)
    """
    _CONDITION_REGISTRY[name] = factory


# ── Penalty schema ───────────────────────────────────────────────────────

@dataclass
class Penalty:
    """A declarative fitness penalty loaded from a seed template file.

    Each penalty specifies a *condition type*, a *threshold* or
    *pattern*, and a weight.  When converted to a ``ProxyCheck`` via
    :func:`penalties_to_proxy_checks`, the check function is inverted
    so it returns ``True`` for **acceptable** output, and the weight
    is made positive.

    Supported condition types
    -------------------------
    ``json_array_length_gt``
        Fires when the output is a JSON array with more than *threshold*
        elements.  Penalises over-selection.
    ``json_array_length_lt``
        Fires when the output is a JSON array with fewer than *threshold*
        elements.  Penalises under-selection.
    ``output_length_gt``
        Fires when ``len(output)`` exceeds *threshold* characters.
    ``output_length_lt``
        Fires when ``len(output)`` is below *threshold* characters.
    ``contains``
        Fires when *pattern* (plain substring) appears in the output.
    ``not_contains``
        Fires when *pattern* (plain substring) is absent from the output.
    ``regex_match``
        Fires when *pattern* (regex) matches anywhere in the output.
    ``regex_no_match``
        Fires when *pattern* (regex) does NOT match anywhere in the
        output.
    ``json_array_items_not_subset``
        Fires when any item in a JSON array is **not** in the allowed
        set.  *pattern* is a pipe-separated list of valid values
        (e.g. ``"a|b|c"``).  If the output is a JSON object, the
        first list-valued field is used.  Non-JSON output does not
        fire the penalty (use a separate ``valid_json`` check).

    Custom conditions can be added via :func:`register_penalty_condition`.

    Raises
    ------
    ValueError
        If *condition* is not a registered condition type.
    """

    name: str
    description: str
    condition: str
    weight: float = -2.0
    threshold: int | float | None = None
    pattern: str | None = None

    def __post_init__(self) -> None:
        if self.condition not in _CONDITION_REGISTRY:
            raise ValueError(
                f"Unknown penalty condition: {self.condition!r}. "
                f"Valid conditions: {sorted(_CONDITION_REGISTRY)}"
            )


@dataclass
class SeedTemplateConfig:
    """Full parsed content of a seed template JSON file."""

    name: str
    description: str
    seeds: list[str]
    penalties: list[Penalty] = field(default_factory=list)


def load_seed_templates(name: str, *, directory: Path | None = None) -> list[str]:
    """Load seed templates from ``seed_templates/<name>.json``.

    Parameters
    ----------
    name : str
        File stem inside the seed templates directory (without ``.json``).
    directory : Path or None
        Override the default ``seed_templates/`` directory.

    Returns
    -------
    list[str]
        The seed prompt strings.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    ValueError
        If the file is malformed or contains no seeds.
    """
    return load_seed_template_config(name, directory=directory).seeds


def load_seed_template_config(
    name: str,
    *,
    directory: Path | None = None,
) -> SeedTemplateConfig:
    """Load the full seed template config including penalties.

    Parameters
    ----------
    name : str
        File stem inside the seed templates directory (without ``.json``).
    directory : Path or None
        Override the default ``seed_templates/`` directory.

    Returns
    -------
    SeedTemplateConfig
        Parsed config containing seeds, penalties, and metadata.
    """
    base = directory or _SEED_TEMPLATES_DIR
    path = base / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Seed template file not found: {path}\n"
            f"Available templates: {list_seed_templates(directory=base)}"
        )

    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    seeds = data.get("seeds")
    if not seeds or not isinstance(seeds, list):
        raise ValueError(
            f"Seed template file must contain a non-empty 'seeds' list: {path}"
        )

    for i, s in enumerate(seeds):
        if not isinstance(s, str) or not s.strip():
            raise ValueError(f"Seed {i} in {path} must be a non-empty string.")

    # Parse optional penalties
    penalties: list[Penalty] = []
    raw_penalties = data.get("penalties", [])
    if isinstance(raw_penalties, list):
        for j, rp in enumerate(raw_penalties):
            if not isinstance(rp, dict):
                raise ValueError(
                    f"Penalty {j} in {path} must be a JSON object."
                )
            if "name" not in rp or "condition" not in rp:
                raise ValueError(
                    f"Penalty {j} in {path} must have 'name' and 'condition' keys."
                )
            penalties.append(Penalty(
                name=rp["name"],
                description=rp.get("description", ""),
                condition=rp["condition"],
                weight=float(rp.get("weight", -2.0)),
                threshold=rp.get("threshold"),
                pattern=rp.get("pattern"),
            ))
        if penalties:
            logger.info(
                "Loaded %d penalties from %s", len(penalties), path.name,
            )

    return SeedTemplateConfig(
        name=data.get("name", name),
        description=data.get("description", ""),
        seeds=seeds,
        penalties=penalties,
    )


def list_seed_templates(*, directory: Path | None = None) -> list[str]:
    """Return names (stems) of all available seed template files."""
    base = directory or _SEED_TEMPLATES_DIR
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.json"))


# ── Built-in condition factories ──────────────────────────────────────────


def _cond_json_array_length_gt(penalty: Penalty) -> Callable[[str], bool]:
    threshold = int(penalty.threshold or 0)
    def _check(output: str, _t: int = threshold) -> bool:
        try:
            parsed = json.loads(output)
            return isinstance(parsed, list) and len(parsed) > _t
        except (json.JSONDecodeError, ValueError):
            return False
    return _check


def _cond_json_array_length_lt(penalty: Penalty) -> Callable[[str], bool]:
    threshold = int(penalty.threshold or 0)
    def _check(output: str, _t: int = threshold) -> bool:
        try:
            parsed = json.loads(output)
            return isinstance(parsed, list) and len(parsed) < _t
        except (json.JSONDecodeError, ValueError):
            return False
    return _check


def _cond_output_length_gt(penalty: Penalty) -> Callable[[str], bool]:
    threshold = int(penalty.threshold or 0)
    return lambda output, _t=threshold: len(output) > _t


def _cond_output_length_lt(penalty: Penalty) -> Callable[[str], bool]:
    threshold = int(penalty.threshold or 0)
    return lambda output, _t=threshold: len(output) < _t


def _cond_contains(penalty: Penalty) -> Callable[[str], bool]:
    pat = penalty.pattern or ""
    return lambda output, _p=pat: _p in output


def _cond_not_contains(penalty: Penalty) -> Callable[[str], bool]:
    pat = penalty.pattern or ""
    return lambda output, _p=pat: _p not in output


def _cond_regex_match(penalty: Penalty) -> Callable[[str], bool]:
    compiled = re.compile(penalty.pattern or "")
    return lambda output, _r=compiled: bool(_r.search(output))


def _cond_regex_no_match(penalty: Penalty) -> Callable[[str], bool]:
    compiled = re.compile(penalty.pattern or "")
    return lambda output, _r=compiled: not _r.search(output)


def _cond_json_array_items_not_subset(penalty: Penalty) -> Callable[[str], bool]:
    """Fire when any array item is not in the pipe-separated allowed set."""
    allowed = set((penalty.pattern or "").split("|"))

    def _check(output: str, _allowed: set[str] = allowed) -> bool:
        try:
            parsed = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return False

        items: list[Any] | None = None
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    items = v
                    break

        if items is None:
            return False

        return any(str(item) not in _allowed for item in items)

    return _check


# Register all built-in conditions
for _name, _factory in [
    ("json_array_length_gt", _cond_json_array_length_gt),
    ("json_array_length_lt", _cond_json_array_length_lt),
    ("output_length_gt", _cond_output_length_gt),
    ("output_length_lt", _cond_output_length_lt),
    ("contains", _cond_contains),
    ("not_contains", _cond_not_contains),
    ("regex_match", _cond_regex_match),
    ("regex_no_match", _cond_regex_no_match),
    ("json_array_items_not_subset", _cond_json_array_items_not_subset),
]:
    register_penalty_condition(_name, _factory)


def _build_check_fn(penalty: Penalty) -> Callable[[str], bool]:
    """Build a ``(str) -> bool`` that returns True when the penalty fires.

    Uses the condition registry to find the appropriate factory function.
    """
    factory = _CONDITION_REGISTRY.get(penalty.condition)
    if factory is None:
        raise ValueError(
            f"Unknown penalty condition: {penalty.condition!r}. "
            f"Valid conditions: {sorted(_CONDITION_REGISTRY)}"
        )
    return factory(penalty)


def penalties_to_proxy_checks(penalties: list[Penalty]) -> list[ProxyCheck]:
    """Convert declarative penalties into ``ProxyCheck`` objects.

    Each penalty's check function is **inverted** so that it returns
    ``True`` when the output is **acceptable** (the penalty does NOT
    fire).  The weight is converted to a positive value.  This means
    penalty checks follow the same convention as regular ``ProxyCheck``
    objects and can be mixed directly into a ``ProxyMetricsScorer``
    check list.

    Parameters
    ----------
    penalties :
        Penalty definitions to convert.

    Returns
    -------
    list[ProxyCheck]
        One ``ProxyCheck`` per penalty, ready to append to a scorer's
        check list.

    Example
    -------
    >>> from MutaGenAI import load_seed_template_config, penalties_to_proxy_checks
    >>> config = load_seed_template_config("agent_routing")
    >>> penalty_checks = penalties_to_proxy_checks(config.penalties)
    >>> all_checks = positive_checks + penalty_checks
    >>> scorer = ProxyMetricsScorer(checks=all_checks)
    """
    result: list[ProxyCheck] = []
    for p in penalties:
        raw_fn = _build_check_fn(p)
        # Invert: True when output is acceptable (penalty does NOT fire)
        inverted_fn = lambda output, _f=raw_fn: not _f(output)
        weight = abs(p.weight) if p.weight != 0 else 1.0
        result.append(ProxyCheck(
            name=p.name,
            check_fn=inverted_fn,
            weight=weight,
        ))
    return result
