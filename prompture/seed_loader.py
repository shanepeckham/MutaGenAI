"""Load seed templates from external JSON configuration files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SEED_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "seed_templates"


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

    return seeds


def list_seed_templates(*, directory: Path | None = None) -> list[str]:
    """Return names (stems) of all available seed template files."""
    base = directory or _SEED_TEMPLATES_DIR
    if not base.exists():
        return []
    return sorted(p.stem for p in base.glob("*.json"))
