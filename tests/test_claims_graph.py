"""Tests for COB-centric payer-to-payer claims processing."""

from __future__ import annotations

import pytest
from langgraph.types import Command

from payer_claims import (
    COBRule,
    CoverageType,
    Claim,
    MemberInfo,
    PayerCoverage,
    WorkflowStatus,
    build_claims_graph,
)


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def graph():
    return build_claims_graph()


@pytest.fixture
def dual_eligible_claim():
    """Evergreen Medicare Advantage + Kentucky Medicaid."""
    return Claim(
        claim_id="CLM-DE",
        provider_id="PR-001",
        date_of_service="2025-01-15",
        member=MemberInfo(
            member_id="MB-DE",
            date_of_birth="1958-04-12",
            is_employed=False,
            coverages=[
                PayerCoverage(
                    payer_id="EVG001",
                    payer_name="Evergreen Medicare Advantage",
                    coverage_type=CoverageType.MEDICARE_ADVANTAGE,
                    coverage_percent=0.80,
                ),
                PayerCoverage(
                    payer_id="MCD001",
                    payer_name="Kentucky Medicaid",
                    coverage_type=CoverageType.MEDICAID,
                    coverage_percent=0.80,
                ),
            ],
        ),
        lines=[("L1", "99213", 450.00), ("L2", "85025", 125.00)],
    )


@pytest.fixture
def single_coverage_claim():
    """Single coverage — no COB needed."""
    return Claim(
        claim_id="CLM-SC",
        provider_id="PR-001",
        date_of_service="2025-01-15",
        member=MemberInfo(
            member_id="MB-SC",
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


@pytest.fixture
def high_value_dual():
    """High-value dual-eligible that will pend for review."""
    return Claim(
        claim_id="CLM-HV",
        provider_id="PR-001",
        date_of_service="2025-01-15",
        member=MemberInfo(
            member_id="MB-HV",
            date_of_birth="1958-04-12",
            is_employed=False,
            coverages=[
                PayerCoverage(
                    payer_id="EVG001",
                    payer_name="Evergreen Medicare Advantage",
                    coverage_type=CoverageType.MEDICARE_ADVANTAGE,
                    coverage_percent=0.80,
                ),
                PayerCoverage(
                    payer_id="MCD001",
                    payer_name="Kentucky Medicaid",
                    coverage_type=CoverageType.MEDICAID,
                    coverage_percent=0.80,
                ),
            ],
        ),
        lines=[("L1", "27130", 9500.00)],
    )


# ── validation ───────────────────────────────────────────────────────────────


def test_validation_pass(graph, dual_eligible_claim):
    config = _config(dual_eligible_claim)
    result = graph.invoke({"claim": dual_eligible_claim}, config)
    assert result["is_valid"] is True
    assert result["validation_errors"] == []


def test_validation_no_claim(graph):
    config = {"configurable": {"thread_id": "no-claim"}}
    result = graph.invoke({"claim": None}, config)
    assert result["is_valid"] is False
    assert any("No claim" in e for e in result["validation_errors"])


def test_validation_no_member(graph):
    claim = Claim(claim_id="C1", provider_id="P1", date_of_service="2025-01-01", lines=[("L1", "99213", 100.0)])
    result = graph.invoke({"claim": claim}, _config(claim))
    assert result["is_valid"] is False
    assert any("Member" in e for e in result["validation_errors"])


def test_validation_no_coverages(graph):
    claim = Claim(
        claim_id="C1", provider_id="P1", date_of_service="2025-01-01",
        member=MemberInfo(member_id="M1", coverages=[]),
        lines=[("L1", "99213", 100.0)],
    )
    result = graph.invoke({"claim": claim}, _config(claim))
    assert result["is_valid"] is False
    assert any("coverages" in e for e in result["validation_errors"])


def test_validation_no_lines(graph, single_coverage_claim):
    single_coverage_claim.lines = []
    result = graph.invoke({"claim": single_coverage_claim}, _config(single_coverage_claim))
    assert result["is_valid"] is False
    assert any("no line" in e for e in result["validation_errors"])


def test_validation_negative_charge(graph, single_coverage_claim):
    single_coverage_claim.lines = [("L1", "99213", -50.0)]
    result = graph.invoke({"claim": single_coverage_claim}, _config(single_coverage_claim))
    assert result["is_valid"] is False
    assert any("positive" in e for e in result["validation_errors"])


# ── COB determination ───────────────────────────────────────────────────────


def test_cob_dual_eligible(graph, dual_eligible_claim):
    result = graph.invoke({"claim": dual_eligible_claim}, _config(dual_eligible_claim))
    assert result["cob_rule_applied"] == COBRule.MEDICARE_PRIMARY_MEDICAID_SECONDARY
    assert result["is_dual_coverage"] is True


def test_cob_single_coverage(graph, single_coverage_claim):
    result = graph.invoke({"claim": single_coverage_claim}, _config(single_coverage_claim))
    assert result["cob_rule_applied"] == COBRule.SINGLE_COVERAGE
    assert result["is_dual_coverage"] is False
    assert result["secondary_payer"] is None


def test_cob_working_aged(graph):
    """Working aged: commercial primary, Medicare secondary."""
    claim = Claim(
        claim_id="CLM-WA",
        provider_id="PR-1",
        date_of_service="2025-01-01",
        member=MemberInfo(
            member_id="MB-WA", date_of_birth="1955-03-22",
            is_employed=True, employer_name="Acme",
            coverages=[
                PayerCoverage(payer_id="BSB", payer_name="Blue Shield", coverage_type=CoverageType.COMMERCIAL, coverage_percent=0.80),
                PayerCoverage(payer_id="HUM", payer_name="Evergreen Medicare", coverage_type=CoverageType.MEDICARE, coverage_percent=0.80),
            ],
        ),
        lines=[("L1", "99214", 350.0)],
    )
    result = graph.invoke({"claim": claim}, _config(claim))
    assert result["cob_rule_applied"] == COBRule.COMMERCIAL_PRIMARY_MEDICARE_SECONDARY


def test_cob_two_commercials(graph):
    """Birthday rule between two commercial plans."""
    claim = Claim(
        claim_id="CLM-TC",
        provider_id="PR-1",
        date_of_service="2025-01-01",
        member=MemberInfo(
            member_id="MB-TC",
            coverages=[
                PayerCoverage(payer_id="UHG", payer_name="UnitedHealth", coverage_type=CoverageType.COMMERCIAL, coverage_percent=0.80),
                PayerCoverage(payer_id="CIG", payer_name="Cigna", coverage_type=CoverageType.COMMERCIAL, coverage_percent=0.70),
            ],
        ),
        lines=[("L1", "99213", 300.0)],
    )
    result = graph.invoke({"claim": claim}, _config(claim))
    assert result["cob_rule_applied"] == COBRule.BIRTHDAY_RULE


# ── single-coverage path (no secondary adjudication) ─────────────────────────


def test_single_coverage_skips_secondary(graph, single_coverage_claim):
    result = graph.invoke({"claim": single_coverage_claim}, _config(single_coverage_claim))
    assert result["is_dual_coverage"] is False
    assert result.get("secondary_eob") is None
    assert result["secondary_total_paid"] == 0.0


# ── dual-coverage path ───────────────────────────────────────────────────────


def test_dual_coverage_has_both_eobs(graph, dual_eligible_claim):
    result = graph.invoke({"claim": dual_eligible_claim}, _config(dual_eligible_claim))
    assert result["primary_eob"] is not None
    assert result["secondary_eob"] is not None
    assert result["secondary_total_paid"] > 0


# ── payment math ─────────────────────────────────────────────────────────────


def test_dual_coverage_payment_split(graph, dual_eligible_claim):
    result = graph.invoke({"claim": dual_eligible_claim}, _config(dual_eligible_claim))
    # 450 + 125 = 575 billed; allowed = 575*0.8 = 460
    # Primary pays 460*0.8 = 368; patient = 92
    # Secondary billed = 92; allowed = 92*0.8 = 73.6; pays 73.6*0.8 = 58.88
    # Patient final = 73.6 - 58.88 = 14.72
    assert result["primary_total_paid"] == pytest.approx(368.00)
    assert result["secondary_total_paid"] == pytest.approx(58.88)
    assert result["patient_responsibility"] == pytest.approx(14.72)


def test_single_coverage_payment(graph, single_coverage_claim):
    result = graph.invoke({"claim": single_coverage_claim}, _config(single_coverage_claim))
    # 200 billed; allowed = 160; payer pays 128; patient owes 32
    assert result["primary_total_paid"] == pytest.approx(128.00)
    assert result["secondary_total_paid"] == 0.0
    assert result["patient_responsibility"] == pytest.approx(32.00)


# ── HITL: high-value pends → approve ────────────────────────────────────────


def test_high_value_pends_for_review(graph, high_value_dual):
    config = _config(high_value_dual)
    result = graph.invoke({"claim": high_value_dual}, config)
    assert result["workflow_status"] == WorkflowStatus.PENDED_REVIEW
    assert "__interrupt__" in result


def test_hitl_approve(graph, high_value_dual):
    config = _config(high_value_dual)
    result = graph.invoke({"claim": high_value_dual}, config)
    assert "__interrupt__" in result

    resumed = graph.invoke(Command(resume="approve"), config)
    assert resumed["human_decision"] == "approve"
    assert resumed["workflow_status"] == WorkflowStatus.SETTLED


def test_hitl_deny(graph, high_value_dual):
    config = _config(high_value_dual)
    result = graph.invoke({"claim": high_value_dual}, config)
    assert "__interrupt__" in result

    resumed = graph.invoke(Command(resume="deny"), config)
    assert resumed["human_decision"] == "deny"
    assert resumed["workflow_status"] == WorkflowStatus.DENIED


def test_hitl_return(graph, high_value_dual):
    config = _config(high_value_dual)
    result = graph.invoke({"claim": high_value_dual}, config)
    assert "__interrupt__" in result

    resumed = graph.invoke(Command(resume="return"), config)
    assert resumed["human_decision"] == "return"


# ── audit log ────────────────────────────────────────────────────────────────


def test_audit_log_covers_all_nodes(graph, dual_eligible_claim):
    config = _config(dual_eligible_claim)
    result = graph.invoke({"claim": dual_eligible_claim}, config)

    nodes = {"validate_claim", "determine_cob", "adjudicate_primary",
             "adjudicate_secondary", "finalize_settlement"}
    for entry in result["audit_log"]:
        node_part = entry.split(" | ")[1]
        nodes.discard(node_part)
    assert nodes == set(), f"Missing audit entries for: {nodes}"


# ── PHI redaction at the pipeline boundary ──────────────────────────────────

import re

_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_ISO_DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def _phi_claim() -> Claim:
    """A claim carrying raw PHI: DOB, SSN-shaped id, member/group/policy ids."""
    return Claim(
        claim_id="CLM-PHI",
        provider_id="PR-1",
        date_of_service="2025-01-15",
        member=MemberInfo(
            member_id="MB-123-45-6789",
            date_of_birth="1958-04-12",
            is_employed=False,
            coverages=[
                PayerCoverage(
                    payer_id="EVG001", payer_name="Evergreen Medicare Advantage",
                    coverage_type=CoverageType.MEDICARE_ADVANTAGE, coverage_percent=0.80,
                    group_id="GRP-999-88-7777", policy_id="POL-2020-01-01",
                ),
                PayerCoverage(
                    payer_id="MCD001", payer_name="Kentucky Medicaid",
                    coverage_type=CoverageType.MEDICAID, coverage_percent=0.80,
                ),
            ],
        ),
        lines=[("L1", "99213", 450.0)],
    )


def test_settlement_output_has_no_raw_phi(graph):
    """Root settlement output must not contain a raw SSN or ISO date-of-birth."""
    claim = _phi_claim()
    result = graph.invoke({"claim": claim}, _config(claim))
    assert result["workflow_status"] == WorkflowStatus.SETTLED

    out = result["claim"]
    assert _ISO_DATE_RE.search(out.date_of_service or "") is None
    assert _ISO_DATE_RE.search(out.member.date_of_birth or "") is None
    assert _SSN_RE.search(out.member.member_id or "") is None
    for cov in out.member.coverages:
        assert _ISO_DATE_RE.search(cov.policy_id or "") is None
        assert _SSN_RE.search(cov.group_id or "") is None


def test_redaction_unconditional_regardless_of_masking(graph, monkeypatch):
    """Redaction is not gated on the masking mode — the unmasked opt-out is gone."""
    monkeypatch.setenv("LANGSMITH_UNMASK_PHI", "true")
    claim = _phi_claim()
    result = graph.invoke({"claim": claim}, _config(claim))
    out = result["claim"]
    assert _ISO_DATE_RE.search(out.member.date_of_birth or "") is None
    assert _SSN_RE.search(out.member.member_id or "") is None


# ── helpers ──────────────────────────────────────────────────────────────────


def _config(claim: Claim) -> dict:
    return {"configurable": {"thread_id": claim.claim_id}}
