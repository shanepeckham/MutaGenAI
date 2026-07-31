"""End-to-end tests for MutaGenAI.redteam.harness (mock LLM, no network)."""
from __future__ import annotations

import pytest

from tests._redteam_fakes import FakeClient, complying_target, refusing_target

from MutaGenAI.redteam.harness import (
    RedTeamConfig,
    RedTeamHarness,
    RedTeamMode,
)
from MutaGenAI.redteam.scope import RedTeamAuthorizationError, RedTeamScope
from MutaGenAI.redteam.target import TargetModel


BEHAVIORS = ["behavior one", "behavior two", "behavior three"]


def _scope(target_name: str = "ollama:custom") -> RedTeamScope:
    return RedTeamScope(
        operator="tester",
        targets=[target_name],
        authorized=True,
        acknowledged_policy=True,
    )


def _fast_config(mode: RedTeamMode, tmp_path) -> RedTeamConfig:
    return RedTeamConfig(
        mode=mode,
        iterations=2,
        population_size=3,
        num_islands=2,
        seed=1,
        verbose=False,
        use_safety_judge=False,
        store_report=False,
        output_dir=str(tmp_path),
    )


class TestAuthorizationGate:
    def test_unauthorized_scope_blocks_construction(self, tmp_path):
        target = TargetModel.from_client(FakeClient(complying_target), name="custom")
        with pytest.raises(RedTeamAuthorizationError):
            RedTeamHarness(
                RedTeamScope(operator="x"),  # not authorized
                target,
                _fast_config(RedTeamMode.HARDEN, tmp_path),
                behaviors=BEHAVIORS,
            )

    def test_empty_behaviors_rejected(self, tmp_path):
        target = TargetModel.from_client(FakeClient(complying_target), name="custom")
        with pytest.raises(ValueError):
            RedTeamHarness(
                _scope(),
                target,
                _fast_config(RedTeamMode.HARDEN, tmp_path),
                behaviors=[],
            )


class TestHardenMode:
    def test_harden_reduces_asr(self, tmp_path):
        # Baseline system prompt is permissive -> target complies -> high ASR.
        target = TargetModel.from_client(
            FakeClient(refusing_target),
            name="custom",
            system_prompt="You are a helpful assistant.",
        )
        harness = RedTeamHarness(
            _scope(), target, _fast_config(RedTeamMode.HARDEN, tmp_path),
            behaviors=BEHAVIORS,
        )
        report = harness.run()

        assert report.mode == "harden"
        assert 0.0 <= report.asr <= 1.0
        assert report.asr_before is not None and report.asr_after is not None
        # Evolution should discover a refusing safety prompt.
        assert report.asr_after <= report.asr_before
        assert report.best_prompt
        assert len(report.per_behavior) == len(BEHAVIORS)

    def test_report_persisted(self, tmp_path):
        target = TargetModel.from_client(
            FakeClient(refusing_target), name="custom",
            system_prompt="You are a helpful assistant.",
        )
        cfg = _fast_config(RedTeamMode.HARDEN, tmp_path)
        cfg.store_report = True
        harness = RedTeamHarness(_scope(), target, cfg, behaviors=BEHAVIORS)
        report = harness.run()
        assert "report_path" in report.metadata
        assert list(tmp_path.glob("redteam_harden_*.json"))


class TestAttackMode:
    def test_attack_requires_seed_templates(self, tmp_path):
        target = TargetModel.from_client(FakeClient(complying_target), name="custom")
        harness = RedTeamHarness(
            _scope(), target, _fast_config(RedTeamMode.ATTACK, tmp_path),
            behaviors=BEHAVIORS,
        )
        with pytest.raises(ValueError):
            harness.run()

    def test_attack_against_weak_target(self, tmp_path):
        # A target that always complies -> attack success rate near 1.
        target = TargetModel.from_client(FakeClient(complying_target), name="custom")
        harness = RedTeamHarness(
            _scope(), target, _fast_config(RedTeamMode.ATTACK, tmp_path),
            behaviors=BEHAVIORS,
            seed_templates=["{goal}"],
        )
        report = harness.run()

        assert report.mode == "attack"
        assert report.asr == pytest.approx(1.0)
        assert report.best_prompt
        assert len(report.per_behavior) == len(BEHAVIORS)
        assert report.metadata["coverage_jailbroken"] == len(BEHAVIORS)

    def test_metadata_feature_flags(self, tmp_path):
        target = TargetModel.from_client(FakeClient(complying_target), name="custom")
        harness = RedTeamHarness(
            _scope(), target, _fast_config(RedTeamMode.ATTACK, tmp_path),
            behaviors=BEHAVIORS, seed_templates=["{goal}"],
        )
        report = harness.run()
        for key in (
            "quality_diversity_available",
            "leaderboard_available",
            "live_available",
            "coverage_fraction",
            "wall_time_s",
        ):
            assert key in report.metadata
