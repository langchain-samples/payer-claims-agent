"""Canonical COB evaluation dataset + LangSmith upload.

Five reference scenarios (mirroring the demos in ``main.py``) with the exact,
hand-verified COB rule / payment split / settlement status each should produce.
Because the adjudication core is deterministic, these double as a regression
oracle: any code change that shifts a number here is a real regression, and
LangSmith Engine can recommend new examples to append over time.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from payer_claims import Claim, CoverageType, MemberInfo, PayerCoverage
from payer_claims.scenarios import CANONICAL_CLAIMS

CORE_DATASET_NAME = "payer-claims: COB core"


# ── (de)serialization: Claim <-> JSON-safe dict ────────────────────────────────


def serialize_claim(claim: Claim) -> dict[str, Any]:
    """Claim -> JSON-safe dict (enum values flattened to strings)."""
    data = asdict(claim)
    if claim.member:
        for cov in data["member"]["coverages"]:
            ct = cov["coverage_type"]
            cov["coverage_type"] = ct.value if isinstance(ct, CoverageType) else ct
    # tuples -> lists for clean JSON round-trips
    data["lines"] = [list(line) for line in claim.lines]
    return data


def deserialize_claim(data: dict[str, Any]) -> Claim:
    """JSON-safe dict -> Claim (inverse of :func:`serialize_claim`)."""
    member_d = data.get("member")
    member = None
    if member_d:
        member = MemberInfo(
            member_id=member_d.get("member_id", ""),
            date_of_birth=member_d.get("date_of_birth", ""),
            is_employed=member_d.get("is_employed", False),
            employer_name=member_d.get("employer_name", ""),
            coverages=[
                PayerCoverage(
                    payer_id=c.get("payer_id", ""),
                    payer_name=c["payer_name"],
                    coverage_type=CoverageType(c["coverage_type"]),
                    coverage_percent=c.get("coverage_percent", 0.0),
                    group_id=c.get("group_id", ""),
                    policy_id=c.get("policy_id", ""),
                )
                for c in member_d.get("coverages", [])
            ],
        )
    return Claim(
        claim_id=data["claim_id"],
        provider_id=data.get("provider_id", ""),
        date_of_service=data.get("date_of_service", ""),
        member=member,
        lines=[tuple(line) for line in data.get("lines", [])],
    )


# ── hand-verified reference outputs, keyed by claim id ──────────────────────────

# Each row is the exact expected result of the deterministic core for the
# matching scenario in payer_claims.scenarios.CANONICAL_CLAIMS.
# ``should_pend`` is the ground-truth label for the pend-gate classifier: does
# this claim genuinely need a human? It lets the routing evaluator score the
# gate's confusion matrix (TP/FP/FN/TN). For the deterministic core only the
# high-value claim should pend; low-confidence pends are exercised by intake.
REFERENCE_OUTPUTS: dict[str, dict[str, Any]] = {
    "CLM-DE-001": {
        "cob_rule": "medicare_primary_medicaid_secondary",
        "primary_paid": 368.00,
        "secondary_paid": 58.88,
        "patient_responsibility": 14.72,
        "workflow_status": "settled",
        "should_pend": False,
        "is_valid": True,
    },
    "CLM-DE-002": {
        "cob_rule": "medicare_primary_medicaid_secondary",
        "primary_paid": 6080.00,
        "secondary_paid": 972.80,
        "patient_responsibility": 243.20,
        "workflow_status": "pended_review",  # high-value → HITL
        "should_pend": True,
        "is_valid": True,
    },
    "CLM-WA-001": {
        "cob_rule": "commercial_primary_medicare_secondary",
        "primary_paid": 224.00,
        "secondary_paid": 35.84,
        "patient_responsibility": 8.96,
        "workflow_status": "settled",
        "should_pend": False,
        "is_valid": True,
    },
    "CLM-TC-001": {
        "cob_rule": "birthday_rule",
        "primary_paid": 243.20,
        "secondary_paid": 34.048,
        "patient_responsibility": 14.592,
        "workflow_status": "settled",
        "should_pend": False,
        "is_valid": True,
    },
    "CLM-SC-001": {
        "cob_rule": "single_coverage",
        "primary_paid": 128.00,
        "secondary_paid": 0.00,
        "patient_responsibility": 32.00,
        "workflow_status": "settled",
        "should_pend": False,
        "is_valid": True,
    },
}

# Input claim + reference outputs, sourced from the shared scenario fixtures.
CANONICAL: list[tuple[Claim, dict[str, Any]]] = [
    (claim, REFERENCE_OUTPUTS[claim.claim_id]) for claim in CANONICAL_CLAIMS
]


def examples() -> list[dict[str, Any]]:
    """Return dataset rows as ``{"inputs": ..., "outputs": ...}`` dicts."""
    return [
        {"inputs": {"claim": serialize_claim(claim)}, "outputs": dict(ref)}
        for claim, ref in CANONICAL
    ]


def build_dataset(client=None, *, name: str = CORE_DATASET_NAME):
    """Create the LangSmith dataset (idempotent) and populate it once.

    Uses a PHI-anonymized client by default so member identifiers are redacted
    in the uploaded examples' trace linkage.
    """
    if client is None:
        from payer_claims import make_client

        client = make_client()

    try:
        dataset = client.create_dataset(
            dataset_name=name,
            description="Canonical payer-to-payer COB scenarios with hand-verified "
            "reference outputs (regression oracle for the deterministic core).",
        )
    except Exception:
        dataset = client.read_dataset(dataset_name=name)

    existing = list(client.list_examples(dataset_id=dataset.id))
    if not existing:
        rows = examples()
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[r["inputs"] for r in rows],
            outputs=[r["outputs"] for r in rows],
        )
        return dataset

    # Self-heal: sync reference outputs into pre-existing examples so label
    # additions (e.g. `should_pend`) propagate to a dataset created by an older
    # version of this file, instead of silently scoring against stale labels.
    by_claim = {claim.claim_id: ref for claim, ref in CANONICAL}
    for ex in existing:
        cid = (ex.inputs or {}).get("claim", {}).get("claim_id")
        want = by_claim.get(cid)
        if want is not None and dict(ex.outputs or {}) != dict(want):
            client.update_example(ex.id, outputs=dict(want))
    return dataset
