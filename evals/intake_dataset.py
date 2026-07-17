"""Intake-extraction eval dataset — the fuzzy step where the agent improves.

Six hand-labeled claim documents with reference structured extractions. Four
name a **Medicare Advantage** plan *without* the literal phrase "Medicare
Advantage" (Imperial Health Plan, Brand New Day, Clover Health, Zing Health).
A naive first-draft prompt classifies those as ``commercial``; the
improved prompt (which knows the carrier taxonomy) gets them right. Two control
rows (an explicit-MA row and a true-commercial pair) should score equally under
both prompts.

This dataset is what makes the "eval-driven improvement" experiment show a real
before/after lift. All rows are static in-repo literals.
"""

from __future__ import annotations

from typing import Any

INTAKE_DATASET_NAME = "payer-claims: intake extraction"


# Each row: raw document (input) + the hand-verified extraction (reference).
# `coverage_type` values are graded by index against the model's output.
INTAKE_EXAMPLES: list[dict[str, Any]] = [
    {
        "inputs": {
            "raw_claim_text": (
                "CLAIM CLM-EV-001  Provider PR-9001  DOS 2025-04-02  Member MB-EV-001\n"
                "Coverages:\n"
                "  1) Evergreen Medicare Advantage (EVG100) — pays 80%\n"
                "  2) Kentucky Medicaid (MCD100) — pays 80%, payer of last resort\n"
                "Lines:\n"
                "  L1  99213  office visit ...... $500.00\n"
                "  L2  80053  metabolic panel ... $120.00\n"
            )
        },
        "outputs": {
            "member_id": "MB-EV-001",
            "coverages": [
                {"payer_name": "Evergreen Medicare Advantage", "coverage_type": "medicare_advantage"},
                {"payer_name": "Kentucky Medicaid", "coverage_type": "medicaid"},
            ],
            "lines": [
                {"procedure_code": "99213", "billed_amount": 500.00},
                {"procedure_code": "80053", "billed_amount": 120.00},
            ],
        },
    },
    {
        "inputs": {
            "raw_claim_text": (
                "CLAIM CLM-EV-002  Provider PR-9002  DOS 2025-04-03  Member MB-EV-002\n"
                "Coverages:\n"
                "  1) Imperial Health Plan HMO (IMP200) — pays 80%\n"
                "  2) Kentucky Medicaid (MCD200) — pays 80%\n"
                "Lines:\n"
                "  L1  99214  office visit ...... $650.00\n"
            )
        },
        "outputs": {
            "member_id": "MB-EV-002",
            "coverages": [
                {"payer_name": "Imperial Health Plan HMO", "coverage_type": "medicare_advantage"},
                {"payer_name": "Kentucky Medicaid", "coverage_type": "medicaid"},
            ],
            "lines": [{"procedure_code": "99214", "billed_amount": 650.00}],
        },
    },
    {
        "inputs": {
            "raw_claim_text": (
                "CLAIM CLM-EV-003  Provider PR-9003  DOS 2025-04-04  Member MB-EV-003\n"
                "Coverages:\n"
                "  1) Brand New Day HMO (BND300) — pays 80%\n"
                "Lines:\n"
                "  L1  99215  office visit ...... $300.00\n"
            )
        },
        "outputs": {
            "member_id": "MB-EV-003",
            "coverages": [
                {"payer_name": "Brand New Day HMO", "coverage_type": "medicare_advantage"},
            ],
            "lines": [{"procedure_code": "99215", "billed_amount": 300.00}],
        },
    },
    {
        "inputs": {
            "raw_claim_text": (
                "CLAIM CLM-EV-004  Provider PR-9004  DOS 2025-04-05  Member MB-EV-004\n"
                "Coverages:\n"
                "  1) Blue Cross Blue Shield PPO — employer group plan (BCB400) — pays 80%\n"
                "  2) Cigna PPO commercial (CIG400) — pays 70%\n"
                "Lines:\n"
                "  L1  99213  office visit ...... $300.00\n"
                "  L2  85025  CBC panel ......... $80.00\n"
            )
        },
        "outputs": {
            "member_id": "MB-EV-004",
            "coverages": [
                {"payer_name": "Blue Cross Blue Shield PPO", "coverage_type": "commercial"},
                {"payer_name": "Cigna PPO", "coverage_type": "commercial"},
            ],
            "lines": [
                {"procedure_code": "99213", "billed_amount": 300.00},
                {"procedure_code": "85025", "billed_amount": 80.00},
            ],
        },
    },
    {
        "inputs": {
            "raw_claim_text": (
                "CLAIM CLM-EV-005  Provider PR-9005  DOS 2025-04-06  Member MB-EV-005\n"
                "Coverages:\n"
                "  1) Clover Health PPO (CLV500) — pays 80%\n"
                "  2) Kentucky Medicaid (MCD500) — pays 80%\n"
                "Lines:\n"
                "  L1  27447  total knee arthroplasty ...... $12,000.00\n"
            )
        },
        "outputs": {
            "member_id": "MB-EV-005",
            "coverages": [
                {"payer_name": "Clover Health PPO", "coverage_type": "medicare_advantage"},
                {"payer_name": "Kentucky Medicaid", "coverage_type": "medicaid"},
            ],
            "lines": [{"procedure_code": "27447", "billed_amount": 12000.00}],
        },
    },
    {
        "inputs": {
            "raw_claim_text": (
                "CLAIM CLM-EV-006  Provider PR-9006  DOS 2025-04-07  Member MB-EV-006\n"
                "Coverages:\n"
                "  1) Zing Health HMO (ZNG600) — pays 80%\n"
                "  2) Ohio Medicaid (MCD600) — pays 80%\n"
                "Lines:\n"
                "  L1  99214  office visit ...... $420.00\n"
            )
        },
        "outputs": {
            "member_id": "MB-EV-006",
            "coverages": [
                {"payer_name": "Zing Health HMO", "coverage_type": "medicare_advantage"},
                {"payer_name": "Ohio Medicaid", "coverage_type": "medicaid"},
            ],
            "lines": [{"procedure_code": "99214", "billed_amount": 420.00}],
        },
    },
]


def build_intake_dataset(client=None, *, name: str = INTAKE_DATASET_NAME):
    """Create the intake dataset (idempotent) and sync reference outputs."""
    if client is None:
        from payer_claims import make_client

        client = make_client()

    try:
        dataset = client.create_dataset(
            dataset_name=name,
            description="Hand-labeled claim documents for grading intake extraction "
            "(coverage-type classification + amounts). Drives the eval-driven "
            "prompt/model improvement experiment.",
        )
    except Exception:
        dataset = client.read_dataset(dataset_name=name)

    existing = list(client.list_examples(dataset_id=dataset.id))
    if not existing:
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[r["inputs"] for r in INTAKE_EXAMPLES],
            outputs=[r["outputs"] for r in INTAKE_EXAMPLES],
        )
    else:
        # Self-heal: keep references in sync if this file changes.
        by_member = {r["outputs"]["member_id"]: r["outputs"] for r in INTAKE_EXAMPLES}
        for ex in existing:
            mid = (ex.outputs or {}).get("member_id")
            want = by_member.get(mid)
            if want is not None and dict(ex.outputs or {}) != dict(want):
                client.update_example(ex.id, outputs=dict(want))
    return dataset
