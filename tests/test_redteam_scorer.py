"""Tests for MutaGenAI.redteam.scorer — red-team fitness scorers."""
from __future__ import annotations

from tests._redteam_fakes import FakeClient, COMPLIANCE_TEXT, REFUSAL_TEXT

from MutaGenAI.redteam.scorer import (
    AttackSuccessScorer,
    RefusalScorer,
    SafetyJudge,
)


class TestRefusalScorer:
    def test_refusal_scores_one(self):
        scorer = RefusalScorer()
        assert scorer.score("p", "goal", REFUSAL_TEXT, None) == 1.0

    def test_compliance_scores_zero(self):
        scorer = RefusalScorer()
        assert scorer.score("p", "goal", COMPLIANCE_TEXT, None) == 0.0


class TestAttackSuccessScorer:
    def test_refusal_scores_zero(self):
        scorer = AttackSuccessScorer()
        assert scorer.score("p", "goal", REFUSAL_TEXT, None) == 0.0

    def test_compliance_no_judge_scores_one(self):
        scorer = AttackSuccessScorer()
        assert scorer.score("p", "goal", COMPLIANCE_TEXT, None) == 1.0

    def test_empty_output_scores_zero(self):
        scorer = AttackSuccessScorer()
        assert scorer.score("p", "goal", "", None) == 0.0

    def test_judge_blend_full_weight(self):
        # Judge says "not complied" -> with weight 1.0 the score collapses to 0
        judge = SafetyJudge()
        client = FakeClient(lambda s, u: '{"complied": 0, "reason": "off"}')
        scorer = AttackSuccessScorer(judge=judge, judge_weight=1.0)
        assert scorer.score("p", "goal", COMPLIANCE_TEXT, client) == 0.0

    def test_judge_blend_complied(self):
        judge = SafetyJudge()
        client = FakeClient(lambda s, u: '{"complied": 1, "reason": "yes"}')
        scorer = AttackSuccessScorer(judge=judge, judge_weight=0.5)
        # not_refused (1.0)*0.5 + complied (1.0)*0.5 == 1.0
        assert scorer.score("p", "goal", COMPLIANCE_TEXT, client) == 1.0


class TestSafetyJudge:
    def test_parse_json_complied(self):
        judge = SafetyJudge()
        client = FakeClient(lambda s, u: '{"complied": 1}')
        assert judge.judge_compliance("goal", "output", client) == 1.0

    def test_parse_json_safe(self):
        judge = SafetyJudge()
        client = FakeClient(lambda s, u: '{"complied": 0}')
        assert judge.judge_compliance("goal", "output", client) == 0.0

    def test_unavailable_client_returns_zero(self):
        judge = SafetyJudge()
        client = FakeClient(available=False)
        assert judge.judge_compliance("goal", "output", client) == 0.0

    def test_none_client_returns_zero(self):
        judge = SafetyJudge()
        assert judge.judge_compliance("goal", "output", None) == 0.0

    def test_regex_fallback(self):
        judge = SafetyJudge()
        client = FakeClient(lambda s, u: "verdict: complied = 1 (bad)")
        assert judge.judge_compliance("goal", "output", client) == 1.0
