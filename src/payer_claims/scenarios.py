"""Canonical COB demo scenarios — the single source of truth for fixtures.

Both the Streamlit demo UI and the evaluation dataset build on these five
claims, so the UI presets and the eval regression oracle can never drift apart.
"""

from __future__ import annotations

from .state import Claim, CoverageType, MemberInfo, PayerCoverage


def _cov(pid: str, name: str, ctype: CoverageType, pct: float) -> PayerCoverage:
    return PayerCoverage(payer_id=pid, payer_name=name, coverage_type=ctype, coverage_percent=pct)


def dual_eligible(claim_id: str, lines: list[tuple[str, str, float]]) -> Claim:
    """Medicare + Medicaid (dual-eligible) — the main COB use case."""
    return Claim(
        claim_id=claim_id, provider_id="PR-1001", date_of_service="2025-01-15",
        member=MemberInfo(
            member_id=f"MB-{claim_id}", date_of_birth="1958-04-12", is_employed=False,
            coverages=[
                _cov("EVG001", "Evergreen Medicare Advantage", CoverageType.MEDICARE_ADVANTAGE, 0.80),
                _cov("MCD001", "Kentucky Medicaid", CoverageType.MEDICAID, 0.80),
            ],
        ),
        lines=list(lines),
    )


def working_aged(claim_id: str, lines: list[tuple[str, str, float]]) -> Claim:
    """Working aged: commercial primary, Medicare secondary."""
    return Claim(
        claim_id=claim_id, provider_id="PR-1001", date_of_service="2025-01-15",
        member=MemberInfo(
            member_id=f"MB-{claim_id}", date_of_birth="1955-03-22",
            is_employed=True, employer_name="Acme Corp",
            coverages=[
                _cov("BSB001", "Blue Shield Commercial", CoverageType.COMMERCIAL, 0.80),
                _cov("EVG002", "Evergreen Medicare", CoverageType.MEDICARE, 0.80),
            ],
        ),
        lines=list(lines),
    )


def two_commercials(claim_id: str, lines: list[tuple[str, str, float]]) -> Claim:
    """Birthday rule: two commercial plans."""
    return Claim(
        claim_id=claim_id, provider_id="PR-1001", date_of_service="2025-01-15",
        member=MemberInfo(
            member_id=f"MB-{claim_id}",
            coverages=[
                _cov("UHG001", "UnitedHealth Commercial", CoverageType.COMMERCIAL, 0.80),
                _cov("CIG001", "Cigna Commercial", CoverageType.COMMERCIAL, 0.70),
            ],
        ),
        lines=list(lines),
    )


def single_coverage(claim_id: str, lines: list[tuple[str, str, float]]) -> Claim:
    """No COB needed — single payer only."""
    return Claim(
        claim_id=claim_id, provider_id="PR-1001", date_of_service="2025-01-15",
        member=MemberInfo(
            member_id=f"MB-{claim_id}",
            coverages=[_cov("EVG001", "Evergreen Medicare Advantage", CoverageType.MEDICARE_ADVANTAGE, 0.80)],
        ),
        lines=list(lines),
    )


# Ordered canonical claims (id -> claim), reused by the UI and the eval dataset.
CANONICAL_CLAIMS: list[Claim] = [
    dual_eligible("CLM-DE-001", [("L1", "99213", 450.00), ("L2", "85025", 125.00)]),
    dual_eligible("CLM-DE-002", [("L1", "27130", 9500.00)]),
    working_aged("CLM-WA-001", [("L1", "99214", 350.00)]),
    two_commercials("CLM-TC-001", [("L1", "99213", 300.00), ("L2", "85025", 80.00)]),
    single_coverage("CLM-SC-001", [("L1", "99213", 200.00)]),
]

# Human-readable labels for the UI dropdown.
LABELS: dict[str, str] = {
    "CLM-DE-001": "Dual-eligible (Medicare + Medicaid) — low value, auto-settles",
    "CLM-DE-002": "Dual-eligible — high value, pends for review",
    "CLM-WA-001": "Working aged (Commercial primary + Medicare)",
    "CLM-TC-001": "Two commercials (birthday rule)",
    "CLM-SC-001": "Single coverage (no COB)",
}

# label -> Claim (what the UI selectbox iterates).
SCENARIOS: dict[str, Claim] = {
    LABELS.get(c.claim_id, c.claim_id): c for c in CANONICAL_CLAIMS
}
