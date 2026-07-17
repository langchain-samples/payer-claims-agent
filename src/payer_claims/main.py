"""Entry point — demonstrates COB scenarios.

Covers the key dual-coverage patterns:
  1. Dual-eligible (Medicare + Medicaid) — auto-settle low-value
  2. Dual-eligible high-value — pends for HITL review
  3. Medicare + Commercial (working aged) — commercial primary
  4. Single coverage — no COB needed
  5. Two commercials — birthday rule
"""

from __future__ import annotations

from langgraph.types import Command

from payer_claims import (
    Claim,
    COBRule,
    CoverageType,
    MemberInfo,
    PayerCoverage,
    build_claims_graph,
    run_config,
)

_DEMO_ADJUSTER = "demo-adjuster"


# ── helpers ──────────────────────────────────────────────────────────────────


def _dual_eligible(
    claim_id: str,
    lines: list[tuple[str, str, float]],
) -> Claim:
    """Medicare + Medicaid (dual-eligible) — the main COB use case."""
    return Claim(
        claim_id=claim_id,
        provider_id="PR-1001",
        date_of_service="2025-01-15",
        member=MemberInfo(
            member_id=f"MB-{claim_id}",
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
        lines=list(lines),
    )


def _medicare_commercial_working(
    claim_id: str,
    lines: list[tuple[str, str, float]],
) -> Claim:
    """Working aged: commercial primary, Medicare secondary."""
    return Claim(
        claim_id=claim_id,
        provider_id="PR-1001",
        date_of_service="2025-01-15",
        member=MemberInfo(
            member_id=f"MB-{claim_id}",
            date_of_birth="1955-03-22",
            is_employed=True,
            employer_name="Acme Corp",
            coverages=[
                PayerCoverage(
                    payer_id="BSB001",
                    payer_name="Blue Shield Commercial",
                    coverage_type=CoverageType.COMMERCIAL,
                    coverage_percent=0.80,
                ),
                PayerCoverage(
                    payer_id="EVG002",
                    payer_name="Evergreen Medicare",
                    coverage_type=CoverageType.MEDICARE,
                    coverage_percent=0.80,
                ),
            ],
        ),
        lines=list(lines),
    )


def _two_commercials(
    claim_id: str,
    lines: list[tuple[str, str, float]],
) -> Claim:
    """Birthday rule: two commercial plans."""
    return Claim(
        claim_id=claim_id,
        provider_id="PR-1001",
        date_of_service="2025-01-15",
        member=MemberInfo(
            member_id=f"MB-{claim_id}",
            coverages=[
                PayerCoverage(
                    payer_id="UHG001",
                    payer_name="UnitedHealth Commercial",
                    coverage_type=CoverageType.COMMERCIAL,
                    coverage_percent=0.80,
                ),
                PayerCoverage(
                    payer_id="CIG001",
                    payer_name="Cigna Commercial",
                    coverage_type=CoverageType.COMMERCIAL,
                    coverage_percent=0.70,
                ),
            ],
        ),
        lines=list(lines),
    )


def _single_coverage(
    claim_id: str,
    lines: list[tuple[str, str, float]],
) -> Claim:
    """No COB needed — single payer only."""
    return Claim(
        claim_id=claim_id,
        provider_id="PR-1001",
        date_of_service="2025-01-15",
        member=MemberInfo(
            member_id=f"MB-{claim_id}",
            coverages=[
                PayerCoverage(
                    payer_id="EVG001",
                    payer_name="Evergreen Medicare Advantage",
                    coverage_type=CoverageType.MEDICARE_ADVANTAGE,
                    coverage_percent=0.80,
                ),
            ],
        ),
        lines=list(lines),
    )


def _print_audit(result: dict) -> None:
    for entry in result.get("audit_log", []):
        print(f"  {entry}")


def _print_settlement(result: dict) -> None:
    print(f"  COB Rule:      {result.get('cob_rule_applied', '?')}")
    print(f"  Primary paid:  ${result.get('primary_total_paid', 0):,.2f}")
    print(f"  Secondary paid: ${result.get('secondary_total_paid', 0):,.2f}")
    print(f"  Patient owes:  ${result.get('patient_responsibility', 0):,.2f}")
    print(f"  Status:        {result.get('workflow_status', '?')}")


# ── demos ────────────────────────────────────────────────────────────────────


def demo_dual_eligible_small():
    """Dual-eligible low-value — auto-settles; Medicare primary, Medicaid secondary."""
    print("=== 1. Dual-Eligible (Medicare + Medicaid) — Low-Value Auto-Settle ===\n")
    graph = build_claims_graph()

    claim = _dual_eligible("CLM-DE-001", [
        ("L1", "99213", 450.00),
        ("L2", "85025", 125.00),
    ])
    config = run_config(thread_id=claim.claim_id, user_id=_DEMO_ADJUSTER)

    result = graph.invoke({"claim": claim}, config)

    _print_settlement(result)
    _print_audit(result)
    print()


def demo_dual_eligible_high_value():
    """Dual-eligible high-value — pends for HITL review, then approved."""
    print("=== 2. Dual-Eligible High-Value — Pends → HITL Approve ===\n")
    graph = build_claims_graph()

    claim = _dual_eligible("CLM-DE-002", [
        ("L1", "27130", 9500.00),
    ])
    config = run_config(thread_id=claim.claim_id, user_id=_DEMO_ADJUSTER)

    # Step 1: invoke → hits interrupt at finalize_settlement
    result = graph.invoke({"claim": claim}, config)

    print("  [INTERRUPT triggered for high-value review]")
    if "__interrupt__" in result:
        for i in result["__interrupt__"]:
            data = i.value
            print(f"  Member:   {data['member_id']}")
            print(f"  Primary:  {data['primary']['payer']} — paid ${data['primary']['paid']:,.2f}")
            print(f"  Secondary: {data['secondary']['payer']} — paid ${data['secondary']['paid']:,.2f}")
            print(f"  Patient:  ${data['patient_responsibility']:,.2f}")
            print()

    # Step 2: reviewer approves
    resumed = graph.invoke(Command(resume="approve"), config)
    print("  [Reviewer APPROVED]")
    _print_settlement(resumed)
    _print_audit(resumed)
    print()


def demo_working_aged():
    """Working aged: commercial primary, Medicare secondary."""
    print("=== 3. Working Aged (Commercial + Medicare) — Commercial Primary ===\n")
    graph = build_claims_graph()

    claim = _medicare_commercial_working("CLM-WA-001", [
        ("L1", "99214", 350.00),
    ])
    config = run_config(thread_id=claim.claim_id, user_id=_DEMO_ADJUSTER)

    result = graph.invoke({"claim": claim}, config)

    print(f"  COB Rule: {result['cob_rule_applied']}")
    print(f"  Primary:  {result['primary_payer'].payer_name}")
    print(f"  Secondary: {result['secondary_payer'].payer_name}")
    _print_settlement(result)
    _print_audit(result)
    print()


def demo_two_commercials():
    """Two commercial plans — birthday rule."""
    print("=== 4. Two Commercials — Birthday Rule ===\n")
    graph = build_claims_graph()

    claim = _two_commercials("CLM-TC-001", [
        ("L1", "99213", 300.00),
        ("L2", "85025", 80.00),
    ])
    config = run_config(thread_id=claim.claim_id, user_id=_DEMO_ADJUSTER)

    result = graph.invoke({"claim": claim}, config)

    _print_settlement(result)
    _print_audit(result)
    print()


def demo_single_coverage():
    """Single coverage — no COB needed."""
    print("=== 5. Single Coverage — No COB ===\n")
    graph = build_claims_graph()

    claim = _single_coverage("CLM-SC-001", [
        ("L1", "99213", 200.00),
    ])
    config = run_config(thread_id=claim.claim_id, user_id=_DEMO_ADJUSTER)

    result = graph.invoke({"claim": claim}, config)

    _print_settlement(result)
    _print_audit(result)
    print()


if __name__ == "__main__":
    demo_dual_eligible_small()
    demo_dual_eligible_high_value()
    demo_working_aged()
    demo_two_commercials()
    demo_single_coverage()
