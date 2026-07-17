"""State schema for COB-centric payer-to-payer claims processing.

Models a dual-coverage / dual-eligible workflow:
determine primary → adjudicate primary → submit remainder to secondary →
calculate payment split → human review for exceptions.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated

from typing_extensions import TypedDict


# ---------------------------------------------------------------------------
# Domain enums
# ---------------------------------------------------------------------------


class CoverageType(str, Enum):
    MEDICARE = "medicare"
    MEDICAID = "medicaid"
    COMMERCIAL = "commercial"
    MEDICARE_ADVANTAGE = "medicare_advantage"


class COBRule(str, Enum):
    """The COB rule that determined ordering."""

    MEDICARE_PRIMARY_MEDICAID_SECONDARY = "medicare_primary_medicaid_secondary"
    MEDICAID_LAST_RESORT = "medicaid_last_resort"
    COMMERCIAL_PRIMARY_MEDICARE_SECONDARY = "commercial_primary_medicare_secondary"  # working aged
    MEDICARE_PRIMARY_COMMERCIAL_SECONDARY = "medicare_primary_commercial_secondary"  # retired
    EMPLOYMENT_RULE = "employment_rule"
    BIRTHDAY_RULE = "birthday_rule"
    SINGLE_COVERAGE = "single_coverage"


class LineStatus(str, Enum):
    PAID = "paid"
    DENIED = "denied"
    PARTIAL = "partial"
    PENDED = "pended"


class WorkflowStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    SETTLED = "settled"
    PENDED_REVIEW = "pended_review"
    DENIED = "denied"


# ---------------------------------------------------------------------------
# Domain data classes
# ---------------------------------------------------------------------------


@dataclass
class PayerCoverage:
    """Single payer's coverage details on a member."""
    payer_id: str
    payer_name: str
    coverage_type: CoverageType
    coverage_percent: float = 0.0  # e.g. 0.80 = 80% coinsurance
    group_id: str = ""
    policy_id: str = ""


@dataclass
class MemberInfo:
    member_id: str
    date_of_birth: str = ""          # YYYY-MM-DD
    is_employed: bool = False
    employer_name: str = ""
    coverages: list[PayerCoverage] = field(default_factory=list)


@dataclass
class LineAdjudication:
    """Result of adjudicating a single claim line against a payer."""
    line_id: str
    procedure_code: str
    billed_amount: float
    allowed_amount: float = 0.0
    paid_amount: float = 0.0
    patient_portion: float = 0.0      # copay + deductible + coinsurance
    status: LineStatus = LineStatus.PENDED
    denial_reason: str = ""


@dataclass
class PayerEOB:
    """Explanation of Benefits from a single payer."""
    payer_id: str
    payer_name: str
    is_primary: bool
    lines: list[LineAdjudication] = field(default_factory=list)
    total_billed: float = 0.0
    total_allowed: float = 0.0
    total_paid: float = 0.0
    total_patient_portion: float = 0.0


@dataclass
class Claim:
    claim_id: str
    provider_id: str
    date_of_service: str
    member: MemberInfo | None = None
    lines: list[tuple[str, str, float]] = field(default_factory=list)
    # (line_id, procedure_code, billed_amount)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class ClaimsState(TypedDict):
    """COB claims processing state."""

    # --- Input ---
    claim: Claim | None
    # Optional unstructured claim document; when present and the intake node is
    # enabled, the LLM intake step normalizes it into `claim`.
    raw_claim_text: str | None

    # --- Audit trail ---
    audit_log: Annotated[list[str], operator.add]

    # --- Validation ---
    validation_errors: list[str]
    is_valid: bool

    # --- COB determination ---
    primary_payer: PayerCoverage | None
    secondary_payer: PayerCoverage | None
    cob_rule_applied: COBRule | None
    is_dual_coverage: bool
    # True when COB ordering fell through to the generic fallback rule — a
    # signal the extracted coverage fields are internally inconsistent, which
    # forces the claim to human_review rather than auto-settle.
    cob_ambiguous: bool

    # --- Per-payer adjudication ---
    primary_eob: PayerEOB | None
    secondary_eob: PayerEOB | None

    # --- Intake confidence & model routing ---
    # Populated by the multi-step, multi-model intake node. `intake_confidence`
    # is the strong verifier model's overall 0-1 confidence in the extraction;
    # `intake_model_path` records which model handled which task (task-based
    # routing), e.g. ["extract:claude-haiku-4-5", "verify:claude-sonnet-5"].
    intake_confidence: float | None
    low_confidence_fields: list[str]
    intake_model_path: list[str]
    intake_escalated: bool

    # --- Final settlement ---
    primary_total_paid: float
    secondary_total_paid: float
    patient_responsibility: float
    workflow_status: WorkflowStatus
    # Why the claim pended (empty when auto-settled). Values: "high_value",
    # "low_confidence", "cob_ambiguous". Drives the human-review gate and the
    # FP/FN framing.
    pend_reasons: list[str]

    # --- Human review ---
    human_decision: str | None
    reviewer_notes: str
    # Reviewer judgment captured for the FP/FN loop (also written to LangSmith
    # as feedback). `review_flag_warranted`: did this claim genuinely need
    # review? (distinguishes a true positive pend from a false positive).
    # `review_intake_correct`: was the LLM intake transcription correct?
    review_flag_warranted: bool | None
    review_intake_correct: bool | None
