"""Tests for MutaGenAI.redteam.pyrit_bridge (works with or without PyRIT)."""
from __future__ import annotations

import json

import pytest

from MutaGenAI.redteam import pyrit_bridge as bridge


class TestAvailability:
    def test_pyrit_available_returns_bool(self):
        assert isinstance(bridge.pyrit_available(), bool)

    def test_require_pyrit_matches_availability(self):
        if bridge.pyrit_available():
            bridge.require_pyrit()  # should not raise
        else:
            with pytest.raises(ImportError):
                bridge.require_pyrit()


class TestLoadBehaviorsFromFile:
    def test_txt_file(self, tmp_path):
        p = tmp_path / "behaviors.txt"
        p.write_text("one\ntwo\n\nthree\n", encoding="utf-8")
        assert bridge.load_behaviors(source="file", path=str(p)) == [
            "one",
            "two",
            "three",
        ]

    def test_json_list_of_strings(self, tmp_path):
        p = tmp_path / "behaviors.json"
        p.write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
        assert bridge.load_behaviors(source="file", path=str(p)) == ["a", "b", "c"]

    def test_json_list_of_objects(self, tmp_path):
        p = tmp_path / "behaviors.json"
        p.write_text(
            json.dumps([{"behavior": "x"}, {"prompt": "y"}, {"value": "z"}]),
            encoding="utf-8",
        )
        assert bridge.load_behaviors(source="file", path=str(p)) == ["x", "y", "z"]

    def test_limit(self, tmp_path):
        p = tmp_path / "behaviors.txt"
        p.write_text("a\nb\nc\nd\n", encoding="utf-8")
        assert bridge.load_behaviors(source="file", path=str(p), limit=2) == [
            "a",
            "b",
        ]

    def test_txt_skips_comments_and_blanks(self, tmp_path):
        p = tmp_path / "behaviors.txt"
        p.write_text(
            "# header comment\n\none\n  # indented comment\ntwo\n",
            encoding="utf-8",
        )
        assert bridge.load_behaviors(source="file", path=str(p)) == ["one", "two"]

    def test_bundled_example_file_loads(self):
        behaviors = bridge.load_behaviors(
            source="file", path="examples/redteam/behaviors.txt"
        )
        assert len(behaviors) >= 5
        assert all(not b.startswith("#") for b in behaviors)

    def test_file_source_requires_path(self):
        with pytest.raises(ValueError):
            bridge.load_behaviors(source="file")


class TestPyritOnlyPaths:
    def test_dataset_source_without_pyrit_raises(self):
        if bridge.pyrit_available():
            pytest.skip("PyRIT installed; offline behavior not applicable")
        with pytest.raises(ImportError):
            bridge.load_behaviors(source="harmbench")

    def test_expand_seeds_without_pyrit_raises(self):
        if bridge.pyrit_available():
            pytest.skip("PyRIT installed; offline behavior not applicable")
        with pytest.raises(ImportError):
            bridge.expand_seeds_with_converters(["{goal}"], ["Base64Converter"])
