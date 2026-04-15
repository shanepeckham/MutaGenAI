"""Tests for prompture.seed_loader — seed template loading."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from prompture.seed_loader import list_seed_templates, load_seed_templates


@pytest.fixture
def tmp_seed_dir(tmp_path: Path) -> Path:
    """Create a temporary seed templates directory."""
    return tmp_path / "seeds"


class TestLoadSeedTemplates:
    def test_load_builtin_entity_classification(self):
        seeds = load_seed_templates("entity_classification_minimal")
        assert isinstance(seeds, list)
        assert len(seeds) >= 1
        for s in seeds:
            assert isinstance(s, str)
            assert s.strip()

    def test_load_full_entity_classification(self):
        seeds = load_seed_templates("entity_classification")
        assert isinstance(seeds, list)
        assert len(seeds) >= 1

    def test_load_from_custom_directory(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {"name": "test", "seeds": ["Seed A", "Seed B"]}
        (tmp_seed_dir / "test.json").write_text(json.dumps(data))
        seeds = load_seed_templates("test", directory=tmp_seed_dir)
        assert seeds == ["Seed A", "Seed B"]

    def test_file_not_found(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Seed template file not found"):
            load_seed_templates("nonexistent", directory=tmp_seed_dir)

    def test_empty_seeds_raises(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {"name": "bad", "seeds": []}
        (tmp_seed_dir / "bad.json").write_text(json.dumps(data))
        with pytest.raises(ValueError, match="non-empty 'seeds' list"):
            load_seed_templates("bad", directory=tmp_seed_dir)

    def test_missing_seeds_key(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {"name": "no_seeds"}
        (tmp_seed_dir / "no_seeds.json").write_text(json.dumps(data))
        with pytest.raises(ValueError, match="non-empty 'seeds' list"):
            load_seed_templates("no_seeds", directory=tmp_seed_dir)

    def test_non_string_seed_raises(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {"seeds": ["valid", 123]}
        (tmp_seed_dir / "mixed.json").write_text(json.dumps(data))
        with pytest.raises(ValueError, match="non-empty string"):
            load_seed_templates("mixed", directory=tmp_seed_dir)

    def test_empty_string_seed_raises(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {"seeds": ["valid", "  "]}
        (tmp_seed_dir / "blank.json").write_text(json.dumps(data))
        with pytest.raises(ValueError, match="non-empty string"):
            load_seed_templates("blank", directory=tmp_seed_dir)


class TestListSeedTemplates:
    def test_list_builtin(self):
        names = list_seed_templates()
        assert "entity_classification" in names
        assert "entity_classification_minimal" in names

    def test_list_custom_directory(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        (tmp_seed_dir / "alpha.json").write_text("{}")
        (tmp_seed_dir / "beta.json").write_text("{}")
        (tmp_seed_dir / "not_json.txt").write_text("")
        names = list_seed_templates(directory=tmp_seed_dir)
        assert names == ["alpha", "beta"]

    def test_list_nonexistent_directory(self, tmp_path: Path):
        names = list_seed_templates(directory=tmp_path / "nonexistent")
        assert names == []
