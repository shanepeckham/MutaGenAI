"""Tests for MutaGenAI.redteam.scope — the authorization guardrail."""
from __future__ import annotations

import pytest

from MutaGenAI.redteam.scope import RedTeamScope, RedTeamAuthorizationError


class TestRequire:
    def test_default_scope_is_refused(self):
        scope = RedTeamScope(operator="tester")
        with pytest.raises(RedTeamAuthorizationError):
            scope.require()

    def test_unauthorized_flag_refused(self):
        scope = RedTeamScope(
            operator="tester",
            targets=["ollama:llama3.2"],
            authorized=False,
            acknowledged_policy=True,
        )
        with pytest.raises(RedTeamAuthorizationError):
            scope.require("ollama:llama3.2")

    def test_missing_policy_ack_refused(self):
        scope = RedTeamScope(
            operator="tester",
            targets=["ollama:llama3.2"],
            authorized=True,
            acknowledged_policy=False,
        )
        with pytest.raises(RedTeamAuthorizationError):
            scope.require("ollama:llama3.2")

    def test_empty_targets_refused(self):
        scope = RedTeamScope(
            operator="tester", authorized=True, acknowledged_policy=True
        )
        with pytest.raises(RedTeamAuthorizationError):
            scope.require()

    def test_target_out_of_scope_refused(self):
        scope = self._authorized()
        with pytest.raises(RedTeamAuthorizationError):
            scope.require("ollama:other-model")

    def test_authorized_passes(self):
        scope = self._authorized()
        scope.require("ollama:llama3.2")  # should not raise

    def test_authorizes_helper(self):
        scope = self._authorized()
        assert scope.authorizes("ollama:llama3.2") is True
        assert scope.authorizes("ollama:not-listed") is False

    @staticmethod
    def _authorized() -> RedTeamScope:
        return RedTeamScope(
            operator="tester",
            targets=["ollama:llama3.2"],
            authorized=True,
            acknowledged_policy=True,
        )


class TestRoundTrip:
    def test_file_round_trip(self, tmp_path):
        scope = RedTeamScope(
            operator="tester",
            targets=["ollama:llama3.2"],
            authorized=True,
            acknowledged_policy=True,
            notes="ticket-123",
        )
        path = tmp_path / "scope.json"
        scope.to_file(path)
        loaded = RedTeamScope.from_file(path)
        assert loaded == scope
