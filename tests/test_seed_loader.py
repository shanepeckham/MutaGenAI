"""Tests for MutaGenAI.seed_loader — seed template loading."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pytest

from MutaGenAI.seed_loader import (
    Penalty,
    SeedTemplateConfig,
    _build_check_fn,
    _CONDITION_REGISTRY,
    list_seed_templates,
    load_seed_template_config,
    load_seed_templates,
    penalties_to_proxy_checks,
    register_penalty_condition,
)
from MutaGenAI.strategies import ProxyCheck


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


class TestLoadSeedTemplateConfig:
    """Tests for load_seed_template_config — full config with penalties."""

    def test_returns_seed_template_config(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {"name": "test", "description": "desc", "seeds": ["A"]}
        (tmp_seed_dir / "test.json").write_text(json.dumps(data))
        config = load_seed_template_config("test", directory=tmp_seed_dir)
        assert isinstance(config, SeedTemplateConfig)
        assert config.name == "test"
        assert config.description == "desc"
        assert config.seeds == ["A"]
        assert config.penalties == []

    def test_loads_penalties(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {
            "name": "penalised",
            "seeds": ["Prompt A"],
            "penalties": [
                {
                    "name": "over_sel",
                    "description": "Too many",
                    "condition": "json_array_length_gt",
                    "threshold": 6,
                    "weight": -3.0,
                }
            ],
        }
        (tmp_seed_dir / "penalised.json").write_text(json.dumps(data))
        config = load_seed_template_config("penalised", directory=tmp_seed_dir)
        assert len(config.penalties) == 1
        p = config.penalties[0]
        assert isinstance(p, Penalty)
        assert p.name == "over_sel"
        assert p.condition == "json_array_length_gt"
        assert p.threshold == 6
        assert p.weight == -3.0

    def test_loads_penalty_with_pattern(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {
            "seeds": ["Seed"],
            "penalties": [
                {
                    "name": "no_apology",
                    "condition": "contains",
                    "pattern": "sorry",
                    "weight": -1.5,
                }
            ],
        }
        (tmp_seed_dir / "pat.json").write_text(json.dumps(data))
        config = load_seed_template_config("pat", directory=tmp_seed_dir)
        assert config.penalties[0].pattern == "sorry"

    def test_no_penalties_key_gives_empty_list(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {"seeds": ["Seed"]}
        (tmp_seed_dir / "nopenalty.json").write_text(json.dumps(data))
        config = load_seed_template_config("nopenalty", directory=tmp_seed_dir)
        assert config.penalties == []

    def test_penalty_missing_name_raises(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {"seeds": ["S"], "penalties": [{"condition": "contains"}]}
        (tmp_seed_dir / "bad.json").write_text(json.dumps(data))
        with pytest.raises(ValueError, match="'name' and 'condition'"):
            load_seed_template_config("bad", directory=tmp_seed_dir)

    def test_penalty_missing_condition_raises(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {"seeds": ["S"], "penalties": [{"name": "x"}]}
        (tmp_seed_dir / "bad2.json").write_text(json.dumps(data))
        with pytest.raises(ValueError, match="'name' and 'condition'"):
            load_seed_template_config("bad2", directory=tmp_seed_dir)

    def test_penalty_non_dict_raises(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {"seeds": ["S"], "penalties": ["not a dict"]}
        (tmp_seed_dir / "bad3.json").write_text(json.dumps(data))
        with pytest.raises(ValueError, match="must be a JSON object"):
            load_seed_template_config("bad3", directory=tmp_seed_dir)

    def test_default_weight(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {
            "seeds": ["S"],
            "penalties": [{"name": "x", "condition": "contains", "pattern": "hi"}],
        }
        (tmp_seed_dir / "dw.json").write_text(json.dumps(data))
        config = load_seed_template_config("dw", directory=tmp_seed_dir)
        assert config.penalties[0].weight == -2.0

    def test_builtin_agent_routing_has_penalties(self):
        config = load_seed_template_config("agent_routing")
        assert len(config.penalties) >= 1
        assert any(p.name == "over_selection" for p in config.penalties)

    def test_load_seed_templates_backward_compat(self, tmp_seed_dir: Path):
        """load_seed_templates still returns list[str]."""
        tmp_seed_dir.mkdir(parents=True)
        data = {
            "seeds": ["A", "B"],
            "penalties": [
                {"name": "x", "condition": "contains", "pattern": "z"}
            ],
        }
        (tmp_seed_dir / "compat.json").write_text(json.dumps(data))
        result = load_seed_templates("compat", directory=tmp_seed_dir)
        assert result == ["A", "B"]
        assert isinstance(result, list)


class TestPenaltiesToProxyChecks:
    """Tests for penalties_to_proxy_checks — returns ProxyCheck with inverted semantics."""

    def test_json_array_length_gt_inverted(self):
        p = Penalty(
            name="over", description="", condition="json_array_length_gt",
            threshold=3, weight=-2.0,
        )
        checks = penalties_to_proxy_checks([p])
        assert len(checks) == 1
        assert isinstance(checks[0], ProxyCheck)
        assert checks[0].name == "over"
        assert checks[0].weight == 2.0  # abs of -2.0
        # Inverted: True when acceptable (array <= 3)
        assert checks[0].check_fn('["a","b","c","d"]') is False
        assert checks[0].check_fn('["a","b"]') is True

    def test_json_array_length_lt_inverted(self):
        p = Penalty(
            name="under", description="", condition="json_array_length_lt",
            threshold=2, weight=-1.0,
        )
        checks = penalties_to_proxy_checks([p])
        # Inverted: True when acceptable (array >= 2)
        assert checks[0].check_fn('["a"]') is False
        assert checks[0].check_fn('["a","b","c"]') is True

    def test_output_length_gt_inverted(self):
        p = Penalty(
            name="long", description="", condition="output_length_gt",
            threshold=10, weight=-1.0,
        )
        checks = penalties_to_proxy_checks([p])
        # Inverted: True when acceptable (length <= 10)
        assert checks[0].check_fn("x" * 11) is False
        assert checks[0].check_fn("short") is True

    def test_output_length_lt_inverted(self):
        p = Penalty(
            name="short", description="", condition="output_length_lt",
            threshold=5, weight=-1.0,
        )
        checks = penalties_to_proxy_checks([p])
        # Inverted: True when acceptable (length >= 5)
        assert checks[0].check_fn("hi") is False
        assert checks[0].check_fn("hello world") is True

    def test_contains_inverted(self):
        p = Penalty(
            name="has_sorry", description="", condition="contains",
            pattern="sorry", weight=-1.0,
        )
        checks = penalties_to_proxy_checks([p])
        # Inverted: True when acceptable (no "sorry")
        assert checks[0].check_fn("I'm sorry") is False
        assert checks[0].check_fn("All good") is True

    def test_not_contains_inverted(self):
        p = Penalty(
            name="no_json", description="", condition="not_contains",
            pattern="[", weight=-1.0,
        )
        checks = penalties_to_proxy_checks([p])
        # Inverted: True when acceptable ("[" IS present)
        assert checks[0].check_fn("plain text") is False
        assert checks[0].check_fn("[1,2,3]") is True

    def test_regex_match_inverted(self):
        p = Penalty(
            name="has_digits", description="", condition="regex_match",
            pattern=r"\d{3,}", weight=-1.0,
        )
        checks = penalties_to_proxy_checks([p])
        # Inverted: True when acceptable (no 3+ digit sequences)
        assert checks[0].check_fn("error code 404") is False
        assert checks[0].check_fn("no digits here") is True

    def test_regex_no_match_inverted(self):
        p = Penalty(
            name="must_have_agent", description="", condition="regex_no_match",
            pattern=r"_agent\b", weight=-1.0,
        )
        checks = penalties_to_proxy_checks([p])
        # Inverted: True when acceptable ("_agent" IS present)
        assert checks[0].check_fn("just text") is False
        assert checks[0].check_fn("use auth_agent") is True

    def test_unknown_condition_raises_on_construction(self):
        with pytest.raises(ValueError, match="Unknown penalty condition"):
            Penalty(
                name="bad", description="", condition="nonexistent",
                weight=-1.0,
            )

    def test_weight_made_positive(self):
        p = Penalty(
            name="pos", description="", condition="output_length_gt",
            threshold=10, weight=-2.0,
        )
        checks = penalties_to_proxy_checks([p])
        assert checks[0].weight == 2.0

    def test_positive_weight_kept_positive(self):
        p = Penalty(
            name="pos", description="", condition="output_length_gt",
            threshold=10, weight=2.0,
        )
        checks = penalties_to_proxy_checks([p])
        assert checks[0].weight == 2.0

    def test_json_parse_failure_returns_true(self):
        """Non-JSON output is acceptable (penalty does not fire)."""
        p = Penalty(
            name="over", description="", condition="json_array_length_gt",
            threshold=3, weight=-2.0,
        )
        checks = penalties_to_proxy_checks([p])
        assert checks[0].check_fn("not json at all") is True

    def test_json_non_array_returns_true(self):
        """Non-array JSON is acceptable (penalty does not fire)."""
        p = Penalty(
            name="over", description="", condition="json_array_length_gt",
            threshold=3, weight=-2.0,
        )
        checks = penalties_to_proxy_checks([p])
        assert checks[0].check_fn('{"key": "value"}') is True

    def test_multiple_penalties(self):
        penalties = [
            Penalty(name="a", description="", condition="output_length_gt",
                    threshold=100, weight=-1.0),
            Penalty(name="b", description="", condition="contains",
                    pattern="error", weight=-2.0),
        ]
        checks = penalties_to_proxy_checks(penalties)
        assert len(checks) == 2
        assert checks[0].name == "a"
        assert checks[1].name == "b"
        assert isinstance(checks[0], ProxyCheck)
        assert isinstance(checks[1], ProxyCheck)


class TestPenaltyPostInit:
    """Tests for Penalty.__post_init__ validation."""

    def test_valid_condition_accepted(self):
        p = Penalty(name="ok", description="", condition="contains", pattern="x")
        assert p.condition == "contains"

    def test_invalid_condition_raises(self):
        with pytest.raises(ValueError, match="Unknown penalty condition"):
            Penalty(name="bad", description="", condition="totally_fake")

    def test_error_lists_valid_conditions(self):
        with pytest.raises(ValueError, match="json_array_length_gt"):
            Penalty(name="bad", description="", condition="nope")


class TestConditionRegistry:
    """Tests for the extensible condition registry."""

    def test_builtin_conditions_registered(self):
        expected = {
            "json_array_length_gt", "json_array_length_lt",
            "output_length_gt", "output_length_lt",
            "contains", "not_contains",
            "regex_match", "regex_no_match",
        }
        assert expected.issubset(set(_CONDITION_REGISTRY))

    def test_register_custom_condition(self):
        def _word_count_gt(p: Penalty) -> Callable[[str], bool]:
            t = int(p.threshold or 0)
            return lambda output, _t=t: len(output.split()) > _t

        register_penalty_condition("word_count_gt", _word_count_gt)
        try:
            p = Penalty(
                name="wordy", description="", condition="word_count_gt",
                threshold=3, weight=-1.0,
            )
            checks = penalties_to_proxy_checks([p])
            # Inverted: True when acceptable (word count <= 3)
            assert checks[0].check_fn("one two three") is True
            assert checks[0].check_fn("one two three four") is False
        finally:
            # Clean up so other tests are not affected
            _CONDITION_REGISTRY.pop("word_count_gt", None)

    def test_build_check_fn_uses_registry(self):
        p = Penalty(name="t", description="", condition="contains", pattern="hi")
        fn = _build_check_fn(p)
        assert fn("hi there") is True
        assert fn("hello") is False


# ---------------------------------------------------------------------------
# schema_to_proxy_checks tests
# ---------------------------------------------------------------------------


class TestSchemaToProxyChecks:
    """Tests for schema_to_proxy_checks utility."""

    def test_basic_schema(self):
        from MutaGenAI.seed_loader import schema_to_proxy_checks

        schema = {"name": "string", "score": "number"}
        checks = schema_to_proxy_checks(schema)
        names = [c.name for c in checks]
        assert "valid_json" in names
        assert "has_name" in names
        assert "has_score" in names
        assert "name_non_empty" in names

    def test_array_field(self):
        from MutaGenAI.seed_loader import schema_to_proxy_checks

        schema = {"items": []}
        checks = schema_to_proxy_checks(schema)
        names = [c.name for c in checks]
        assert "items_is_list" in names

    def test_nested_object(self):
        from MutaGenAI.seed_loader import schema_to_proxy_checks

        schema = {"details": {"reasoning": "string", "evidence": "string"}}
        checks = schema_to_proxy_checks(schema)
        names = [c.name for c in checks]
        assert "details_has_reasoning" in names
        assert "details_has_evidence" in names

    def test_valid_json_check_works(self):
        from MutaGenAI.seed_loader import schema_to_proxy_checks

        checks = schema_to_proxy_checks({"x": "string"})
        valid_json_check = [c for c in checks if c.name == "valid_json"][0]
        assert valid_json_check.check_fn('{"x": "hello"}') is True
        assert valid_json_check.check_fn("not json") is False

    def test_has_key_check_works(self):
        from MutaGenAI.seed_loader import schema_to_proxy_checks

        checks = schema_to_proxy_checks({"name": "string"})
        has_name = [c for c in checks if c.name == "has_name"][0]
        assert has_name.check_fn('{"name": "Alice"}') is True
        assert has_name.check_fn('{"other": "value"}') is False

    def test_non_empty_check_works(self):
        from MutaGenAI.seed_loader import schema_to_proxy_checks

        checks = schema_to_proxy_checks({"name": "string"})
        non_empty = [c for c in checks if c.name == "name_non_empty"][0]
        assert non_empty.check_fn('{"name": "Alice"}') is True
        assert non_empty.check_fn('{"name": ""}') is False
        assert non_empty.check_fn('{"name": "  "}') is False

    def test_list_check_works(self):
        from MutaGenAI.seed_loader import schema_to_proxy_checks

        checks = schema_to_proxy_checks({"tags": []})
        is_list = [c for c in checks if c.name == "tags_is_list"][0]
        assert is_list.check_fn('{"tags": ["a", "b"]}') is True
        assert is_list.check_fn('{"tags": []}') is False
        assert is_list.check_fn('{"tags": "not a list"}') is False

    def test_custom_weight(self):
        from MutaGenAI.seed_loader import schema_to_proxy_checks

        checks = schema_to_proxy_checks({"x": "string"}, weight=2.5)
        assert all(c.weight == 2.5 for c in checks)

    def test_empty_schema(self):
        from MutaGenAI.seed_loader import schema_to_proxy_checks

        checks = schema_to_proxy_checks({})
        assert len(checks) == 1  # only valid_json
        assert checks[0].name == "valid_json"


# ---------------------------------------------------------------------------
# output_schema in seed template config
# ---------------------------------------------------------------------------


class TestOutputSchema:
    """Tests for output_schema field in seed templates."""

    def test_output_schema_loaded(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {
            "name": "test",
            "seeds": ["Generate: {output_schema}"],
            "output_schema": {"diagnosis": "string", "confidence": "number"},
        }
        (tmp_seed_dir / "test.json").write_text(json.dumps(data))
        config = load_seed_template_config("test", directory=tmp_seed_dir)
        assert config.output_schema == {"diagnosis": "string", "confidence": "number"}

    def test_output_schema_substituted_in_seeds(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {
            "name": "test",
            "seeds": ["Output JSON matching: {output_schema}"],
            "output_schema": {"name": "string"},
        }
        (tmp_seed_dir / "test.json").write_text(json.dumps(data))
        config = load_seed_template_config("test", directory=tmp_seed_dir)
        assert "{output_schema}" not in config.seeds[0]
        assert '"name"' in config.seeds[0]

    def test_no_output_schema_is_none(self, tmp_seed_dir: Path):
        tmp_seed_dir.mkdir(parents=True)
        data = {"name": "test", "seeds": ["Hello"]}
        (tmp_seed_dir / "test.json").write_text(json.dumps(data))
        config = load_seed_template_config("test", directory=tmp_seed_dir)
        assert config.output_schema is None
