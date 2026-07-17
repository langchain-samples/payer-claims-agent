"""Tests for the confidence gate, FP/FN classification, and judgment capture.

These exercise the deterministic wiring around the multi-model intake (the
confidence-driven pend gate, the reviewer-judgment resume payload, and the
confusion-matrix helpers) without any LLM calls — confidence is injected into
state directly, exactly as the intake node would set it.
"""

from __future__ import annotations

import pytest
from langgraph.types import Command

from payer_claims import (
    Claim,
    CoverageType,
    MemberInfo,
    PayerCoverage,
    WorkflowStatus,
    build_claims_graph,
    classify_review,
    confusion_counts,
)


@pytest.fixture
def graph():
    return build_claims_graph()


def _low_value_single() -> Claim:
    """A cheap single-coverage claim — never pends on dollar amount."""
    return Claim(
        claim_id="CLM-CONF",
        provider_id="PR-1",
        date_of_service="2025-01-01",
        member=MemberInfo(
            member_id="MB-CONF",
            coverages=[
                PayerCoverage(
                    payer_id="EVG001",
                    payer_name="Evergreen Medicare Advantage",
                    coverage_type=CoverageType.MEDICARE_ADVANTAGE,
                    coverage_percent=0.80,
                ),
            ],
        ),
        lines=[("L1", "99213", 200.00)],
    )


# ── confidence-driven pend gate ──────────────────────────────────────────────


def test_low_confidence_pends(graph):
    claim = _low_value_single()
    cfg = {"configurable": {"thread_id": "conf-low"}}
    result = graph.invoke({"claim": claim, "intake_confidence": 0.42}, cfg)
    assert result["workflow_status"] == WorkflowStatus.PENDED_REVIEW
    assert result["pend_reasons"] == ["low_confidence"]
    assert "__interrupt__" in result


def test_high_confidence_auto_settles(graph):
    claim = _low_value_single()
    cfg = {"configurable": {"thread_id": "conf-high"}}
    result = graph.invoke({"claim": claim, "intake_confidence": 0.95}, cfg)
    assert result["workflow_status"] == WorkflowStatus.SETTLED
    assert result["pend_reasons"] == []


def test_no_confidence_does_not_pend(graph):
    """Deterministic path (no intake) must behave exactly as before."""
    claim = _low_value_single()
    cfg = {"configurable": {"thread_id": "conf-none"}}
    result = graph.invoke({"claim": claim}, cfg)
    assert result["workflow_status"] == WorkflowStatus.SETTLED
    assert result["pend_reasons"] == []


def test_high_value_with_anomaly_records_pend_reason(graph):
    """High-dollar claims pend only when a data-quality anomaly corroborates."""
    claim = _low_value_single()
    claim.lines = [("L1", "27130", 9500.00)]
    cfg = {"configurable": {"thread_id": "conf-hv"}}
    result = graph.invoke(
        {"claim": claim, "low_confidence_fields": ["billed_amount"]}, cfg
    )
    assert result["pend_reasons"] == ["high_value"]


def test_high_value_alone_does_not_pend(graph):
    """Value alone is a false-positive signal reviewers approve — must not pend."""
    claim = _low_value_single()
    claim.lines = [("L1", "27130", 9500.00)]
    cfg = {"configurable": {"thread_id": "conf-hv-clean"}}
    result = graph.invoke({"claim": claim}, cfg)
    assert result["workflow_status"] == WorkflowStatus.SETTLED
    assert result["pend_reasons"] == []


def test_moderate_confidence_routes_straight_through(graph):
    """A moderately confident intake (~0.71) should no longer pend."""
    claim = _low_value_single()
    cfg = {"configurable": {"thread_id": "conf-moderate"}}
    result = graph.invoke({"claim": claim, "intake_confidence": 0.71}, cfg)
    assert result["workflow_status"] == WorkflowStatus.SETTLED
    assert result["pend_reasons"] == []


# ── reviewer judgment on resume (FP/FN signal) ───────────────────────────────


def test_judgment_dict_resume_records_fp(graph):
    claim = _low_value_single()
    cfg = {"configurable": {"thread_id": "judge-fp"}}
    graph.invoke({"claim": claim, "intake_confidence": 0.42}, cfg)
    resumed = graph.invoke(
        Command(resume={"decision": "approve", "flag_warranted": False,
                        "intake_correct": True, "notes": "fine"}),
        cfg,
    )
    assert resumed["human_decision"] == "approve"
    assert resumed["review_flag_warranted"] is False
    assert resumed["review_intake_correct"] is True
    assert resumed["reviewer_notes"] == "fine"
    cell = classify_review(pended=True, flag_warranted=False, intake_correct=True)
    assert cell == "FP"


def test_bare_string_resume_still_works(graph):
    """Back-compat: resuming with a plain action string keeps working."""
    claim = _low_value_single()
    cfg = {"configurable": {"thread_id": "judge-bare"}}
    graph.invoke({"claim": claim, "intake_confidence": 0.42}, cfg)
    resumed = graph.invoke(Command(resume="approve"), cfg)
    assert resumed["human_decision"] == "approve"
    assert resumed["workflow_status"] == WorkflowStatus.SETTLED
    assert resumed["review_flag_warranted"] is None


# ── confusion-matrix helpers ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "pended,flag,intake,expected",
    [
        (True, True, None, "TP"),
        (True, False, None, "FP"),
        (False, None, False, "FN"),
        (False, None, True, "TN"),
        (True, None, None, None),
        (False, None, None, None),
    ],
)
def test_classify_review(pended, flag, intake, expected):
    assert classify_review(pended=pended, flag_warranted=flag, intake_correct=intake) == expected


def test_confusion_counts_precision_recall():
    counts = confusion_counts(["TP", "TP", "FP", "FN", "TN", None])
    assert counts["TP"] == 2 and counts["FP"] == 1 and counts["FN"] == 1 and counts["TN"] == 1
    assert counts["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert counts["recall"] == pytest.approx(2 / 3, abs=1e-4)
