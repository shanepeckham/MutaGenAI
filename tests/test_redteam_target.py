"""Tests for MutaGenAI.redteam.target — the target model adapter."""
from __future__ import annotations

from tests._redteam_fakes import FakeClient, COMPLIANCE_TEXT

from MutaGenAI.redteam.target import TargetModel, TargetConfig
from MutaGenAI.prompt_evolver import LLMBackend


class TestTargetModel:
    def test_generate_uses_fixed_system_prompt(self):
        fake = FakeClient(lambda s, u: f"sys={s!r} user={u!r}")
        target = TargetModel.from_client(
            fake, name="fake", system_prompt="SAFE"
        )
        out = target.generate("hello")
        assert "sys='SAFE'" in out
        assert "user='hello'" in out
        assert fake.calls == [("SAFE", "hello")]

    def test_name_property(self):
        target = TargetModel(TargetConfig(backend=LLMBackend.OLLAMA, model="qwen2.5"))
        assert target.name == "ollama:qwen2.5"

    def test_from_client_name(self):
        target = TargetModel.from_client(FakeClient(), name="custom")
        assert target.name == "ollama:custom"

    def test_is_available_reflects_client(self):
        assert TargetModel.from_client(FakeClient(available=True)).is_available()
        assert not TargetModel.from_client(
            FakeClient(available=False)
        ).is_available()

    def test_with_system_prompt_shares_client(self):
        fake = FakeClient(lambda s, u: COMPLIANCE_TEXT)
        base = TargetModel.from_client(fake, name="fake", system_prompt="A")
        swapped = base.with_system_prompt("B")
        assert swapped.system_prompt == "B"
        assert swapped.client is fake  # shares the connection
        assert base.system_prompt == "A"  # original unchanged

    def test_unavailable_generate_returns_none(self):
        target = TargetModel.from_client(FakeClient(available=False))
        # FakeClient with fn=None returns None regardless
        assert target.generate("x") is None
