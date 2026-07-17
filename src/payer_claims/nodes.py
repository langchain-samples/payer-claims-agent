"""COB-centric node implementations for payer-to-payer claims processing.

Models a dual-coverage workflow:
  1. Validate claim + member coverages
  2. Determine COB ordering (Medicare/Medicaid/commercial rules)
  3. Adjudicate against primary payer
  4. Submit remainder to secondary payer
  5. Calculate final payment split
  6. Human review for exceptions

All nodes return partial state updates (dict). The human_review node uses
Command for combined state-update + routing.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from langgraph.types import Command, interrupt

from .state import (
    COBRule,
    CoverageType,
    LineAdjudication,
    LineStatus,
    PayerCoverage,
    PayerEOB,
    ClaimsState,
    WorkflowStatus,
)


# =============================================================================
# Node: validate_claim
# =============================================================================


def validate_claim(state: ClaimsState) -> dict:
    claim = state["claim"]
    errors: list[str] = []

    if not claim:
        errors.append("No claim provided")
        return {"validation_errors": errors, "is_valid": False}

    if not claim.member:
        errors.append("Member info is required for COB determination")
    elif not claim.member.coverages:
        errors.append("Member has no coverages — cannot determine COB")

    if not claim.lines:
        errors.append("Claim has no line items")

    for lid, code, amt in claim.lines:
        if amt <= 0:
            errors.append(f"Line {lid}: billed_amount must be positive")

    is_valid = len(errors) == 0
    log_entry = _log(
        "validate_claim",
        "PASSED" if is_valid else f"FAILED: {'; '.join(errors)}",
    )
    return {
        "validation_errors": errors,
        "is_valid": is_valid,
        "audit_log": [log_entry],
    }


# =============================================================================
# Route: after validation
# =============================================================================


def route_after_validation(
    state: ClaimsState,
) -> Literal["determine_cob", "notify_denial"]:
    return "determine_cob" if state["is_valid"] else "notify_denial"


# =============================================================================
# Node: determine_cob — figure out who is primary, who is secondary
# =============================================================================

# COB ordering rules per CMS / NAIC model:
#   Medicare is primary when member is 65+ and not actively employed with
#     employer group health plan covering 20+ employees
#   Medicaid is ALWAYS payer of last resort
#   For dual-eligible (Medicare + Medicaid): Medicare primary, Medicaid secondary
#   Commercial + Medicare (working aged): commercial primary if employed
#   Between two commercial plans: birthday rule (subscriber w/ earlier birthday
#     in calendar year is primary)


def determine_cob(state: ClaimsState) -> dict:
    claim = state["claim"]
    member = claim.member
    coverages = member.coverages

    # ── classify each coverage ──────────────────────────────────────────────
    medicares = [c for c in coverages if c.coverage_type in (CoverageType.MEDICARE, CoverageType.MEDICARE_ADVANTAGE)]
    medicaids = [c for c in coverages if c.coverage_type == CoverageType.MEDICAID]
    commercials = [c for c in coverages if c.coverage_type == CoverageType.COMMERCIAL]

    primary: PayerCoverage | None = None
    secondary: PayerCoverage | None = None
    rule: COBRule | None = None
    notes: list[str] = []

    # ── case 1: single coverage (no COB) ────────────────────────────────────
    if len(coverages) == 1:
        primary = coverages[0]
        secondary = None
        rule = COBRule.SINGLE_COVERAGE
        notes.append(f"Single coverage: {primary.payer_name} ({primary.coverage_type.value})")

    # ── case 2: Medicare + Medicaid (dual-eligible) ─────────────────────────
    elif medicares and medicaids:
        primary = medicares[0]
        secondary = medicaids[0]
        rule = COBRule.MEDICARE_PRIMARY_MEDICAID_SECONDARY
        notes.append(
            f"Dual-eligible: Medicare ({primary.payer_name}) primary, "
            f"Medicaid ({secondary.payer_name}) secondary (last resort)"
        )

    # ── case 3: Medicare + Commercial ────────────────────────────────────────
    elif medicares and commercials:
        if member.is_employed:
            primary = commercials[0]
            secondary = medicares[0]
            rule = COBRule.COMMERCIAL_PRIMARY_MEDICARE_SECONDARY
            notes.append("Working aged: commercial primary, Medicare secondary")
        else:
            primary = medicares[0]
            secondary = commercials[0]
            rule = COBRule.MEDICARE_PRIMARY_COMMERCIAL_SECONDARY
            notes.append("Retired: Medicare primary, commercial secondary")

    # ── case 4: two commercials ──────────────────────────────────────────────
    elif len(commercials) >= 2:
        primary = commercials[0]
        secondary = commercials[1]
        rule = COBRule.BIRTHDAY_RULE
        notes.append(
            f"Two commercials: {primary.payer_name} primary (birthday rule), "
            f"{secondary.payer_name} secondary"
        )

    # ── case 5: Medicaid + anything else (always last resort) ─────────────────
    elif medicaids:
        non_medicaid = [c for c in coverages if c.coverage_type != CoverageType.MEDICAID]
        if non_medicaid:
            primary = non_medicaid[0]
            secondary = medicaids[0]
            rule = COBRule.MEDICAID_LAST_RESORT
            notes.append(
                f"Medicaid ({secondary.payer_name}) is last resort; "
                f"{primary.payer_name} primary"
            )
        else:
            primary = medicaids[0]
            rule = COBRule.SINGLE_COVERAGE
            notes.append(f"Medicaid only: {primary.payer_name}")

    # ── fallback ──────────────────────────────────────────────────────────────
    else:
        primary = coverages[0]
        secondary = coverages[1] if len(coverages) > 1 else None
        rule = COBRule.EMPLOYMENT_RULE
        notes.append("Fallback: employment-based ordering")

    is_dual = secondary is not None

    log_entry = _log(
        "determine_cob",
        f"rule={rule.value} | primary={primary.payer_name} ({primary.coverage_type.value})"
        + (f" | secondary={secondary.payer_name} ({secondary.coverage_type.value})"
           if secondary else ""),
    )

    return {
        "primary_payer": primary,
        "secondary_payer": secondary,
        "cob_rule_applied": rule,
        "is_dual_coverage": is_dual,
        "audit_log": [log_entry],
    }


# =============================================================================
# Node: adjudicate_primary — run claim against primary payer
# =============================================================================


def adjudicate_primary(state: ClaimsState) -> dict:
    claim = state["claim"]
    primary = state["primary_payer"]

    lines = []
    total_billed = 0.0
    total_allowed = 0.0
    total_paid = 0.0
    total_patient = 0.0

    for lid, code, billed in claim.lines:
        allowed = billed * 0.80                 # 80% of billed is allowed
        payer_share = allowed * primary.coverage_percent
        patient_share = allowed - payer_share
        status = LineStatus.PAID

        lines.append(LineAdjudication(
            line_id=lid,
            procedure_code=code,
            billed_amount=billed,
            allowed_amount=allowed,
            paid_amount=payer_share,
            patient_portion=patient_share,
            status=status,
        ))

        total_billed += billed
        total_allowed += allowed
        total_paid += payer_share
        total_patient += patient_share

    eob = PayerEOB(
        payer_id=primary.payer_id,
        payer_name=primary.payer_name,
        is_primary=True,
        lines=lines,
        total_billed=total_billed,
        total_allowed=total_allowed,
        total_paid=total_paid,
        total_patient_portion=total_patient,
    )

    log_entry = _log(
        "adjudicate_primary",
        f"payer={primary.payer_name} | paid=${total_paid:.2f} | "
        f"patient_portion=${total_patient:.2f}",
    )
    return {"primary_eob": eob, "audit_log": [log_entry]}


# =============================================================================
# Route: after primary adjudication — go to secondary or skip?
# =============================================================================


def route_after_primary(
    state: ClaimsState,
) -> Literal["adjudicate_secondary", "finalize_settlement"]:
    if state["is_dual_coverage"]:
        return "adjudicate_secondary"
    return "finalize_settlement"


# =============================================================================
# Node: adjudicate_secondary — submit remaining balance to secondary payer
# =============================================================================


def adjudicate_secondary(state: ClaimsState) -> dict:
    claim = state["claim"]
    secondary = state["secondary_payer"]
    primary_eob = state["primary_eob"]

    lines = []
    total_billed = 0.0
    total_allowed = 0.0
    total_paid = 0.0
    total_patient = 0.0

    for p_line in primary_eob.lines:
        # The "billed" to secondary is the patient portion from primary
        # (what primary didn't cover)
        billed_to_secondary = p_line.patient_portion
        allowed = billed_to_secondary * 0.80
        payer_share = allowed * secondary.coverage_percent
        patient_share = allowed - payer_share

        lines.append(LineAdjudication(
            line_id=p_line.line_id,
            procedure_code=p_line.procedure_code,
            billed_amount=billed_to_secondary,        # remainder from primary
            allowed_amount=allowed,
            paid_amount=payer_share,
            patient_portion=patient_share,
            status=LineStatus.PAID,
        ))

        total_billed += billed_to_secondary
        total_allowed += allowed
        total_paid += payer_share
        total_patient += patient_share

    eob = PayerEOB(
        payer_id=secondary.payer_id,
        payer_name=secondary.payer_name,
        is_primary=False,
        lines=lines,
        total_billed=total_billed,
        total_allowed=total_allowed,
        total_paid=total_paid,
        total_patient_portion=total_patient,
    )

    log_entry = _log(
        "adjudicate_secondary",
        f"payer={secondary.payer_name} | remainder=${total_billed:.2f} | "
        f"paid=${total_paid:.2f} | patient=${total_patient:.2f}",
    )
    return {"secondary_eob": eob, "audit_log": [log_entry]}


# =============================================================================
# Node: finalize_settlement — compute who pays what
# =============================================================================


def finalize_settlement(state: ClaimsState) -> dict:
    primary_eob = state["primary_eob"]
    secondary_eob = state.get("secondary_eob")  # absent for single coverage

    primary_paid = primary_eob.total_paid
    secondary_paid = secondary_eob.total_paid if secondary_eob else 0.0

    # Patient responsibility = what remains after both payers
    if secondary_eob:
        patient = secondary_eob.total_patient_portion
    else:
        patient = primary_eob.total_patient_portion

    # ── pend gate: two independent reasons to route to a human ─────────────
    # 1. high-dollar claims (money risk)
    # 2. low intake confidence (data-quality risk) — the multi-model intake
    #    verifier wasn't sure it transcribed the claim faithfully.
    total_claim = primary_eob.total_billed
    HIGH_THRESHOLD = 5_000.0
    LOW_CONFIDENCE_THRESHOLD = 0.80

    pend_reasons: list[str] = []
    if total_claim >= HIGH_THRESHOLD:
        pend_reasons.append("high_value")

    confidence = state.get("intake_confidence")
    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
        pend_reasons.append("low_confidence")

    if pend_reasons:
        status = WorkflowStatus.PENDED_REVIEW
    else:
        status = WorkflowStatus.SETTLED

    conf_str = f"{confidence:.2f}" if confidence is not None else "n/a"
    log_entry = _log(
        "finalize_settlement",
        f"status={status.value} | primary_paid=${primary_paid:.2f} | "
        f"secondary_paid=${secondary_paid:.2f} | "
        f"patient_responsibility=${patient:.2f} | "
        f"intake_confidence={conf_str} | "
        f"pend_reasons={pend_reasons or 'none'}",
    )
    return {
        "primary_total_paid": primary_paid,
        "secondary_total_paid": secondary_paid,
        "patient_responsibility": patient,
        "workflow_status": status,
        "pend_reasons": pend_reasons,
        "audit_log": [log_entry],
    }


# =============================================================================
# Route: after settlement
# =============================================================================


def route_after_settlement(
    state: ClaimsState,
) -> Literal["human_review", "__end__"]:
    if state["workflow_status"] == WorkflowStatus.PENDED_REVIEW:
        return "human_review"
    return "__end__"


# =============================================================================
# Node: human_review — interrupt for exceptions / high-value review
# =============================================================================


def human_review(
    state: ClaimsState,
) -> Command[Literal["__end__", "notify_denial"]]:
    claim = state["claim"]
    primary = state["primary_payer"]
    secondary = state["secondary_payer"]

    # PHI never enters the interrupt value: it is serialized verbatim into the
    # run's GraphInterrupt error string, which LangSmith ingests and surfaces in
    # error-filtered views. Stash the full review context in a secured store and
    # interrupt with only an opaque reference token; the reviewer UI rehydrates
    # the claim from the store using that token.
    claim_ref = stash_review_context(
        {
            "claim_id": claim.claim_id,
            "provider_id": claim.provider_id,
            "dos": claim.date_of_service,
            "member_id": claim.member.member_id,
            "cob_rule": state["cob_rule_applied"].value if state["cob_rule_applied"] else "unknown",
            "primary": {
                "payer": primary.payer_name,
                "type": primary.coverage_type.value,
                "paid": state["primary_total_paid"],
            },
            "secondary": {
                "payer": secondary.payer_name,
                "type": secondary.coverage_type.value,
                "paid": state["secondary_total_paid"],
            } if secondary else None,
            "lines": [
                {"id": lid, "code": code, "billed": amt}
                for lid, code, amt in claim.lines
            ],
            "patient_responsibility": state["patient_responsibility"],
            "pend_reasons": state.get("pend_reasons", []),
            "intake_confidence": state.get("intake_confidence"),
            "low_confidence_fields": state.get("low_confidence_fields", []),
        }
    )

    payload = {
        "claim_ref": claim_ref,
        "action": (
            "Review this COB settlement. Look up the claim in the reviewer "
            "console via claim_ref. Respond with 'approve' | 'deny' | 'return', "
            "or a dict {decision, flag_warranted, intake_correct, notes}."
        ),
    }

    decision = interrupt(payload)

    # Accept either a bare action string (back-compat) or a judgment dict that
    # also records the FP/FN signal (was the flag warranted? was intake right?).
    if isinstance(decision, dict):
        action = decision.get("decision", "approve")
        flag_warranted = decision.get("flag_warranted")
        intake_correct = decision.get("intake_correct")
        notes = decision.get("notes", "")
    else:
        action = decision
        flag_warranted = None
        intake_correct = None
        notes = ""

    judgment = {
        "review_flag_warranted": flag_warranted,
        "review_intake_correct": intake_correct,
    }
    judged = (
        f" | flag_warranted={flag_warranted} intake_correct={intake_correct}"
        if flag_warranted is not None or intake_correct is not None
        else ""
    )

    if action == "approve":
        return Command(
            update={
                "human_decision": "approve",
                "workflow_status": WorkflowStatus.SETTLED,
                "reviewer_notes": notes,
                **judgment,
                "audit_log": [_log("human_review", f"APPROVED — settlement confirmed{judged}")],
            },
            goto="__end__",
        )
    elif action == "deny":
        return Command(
            update={
                "human_decision": "deny",
                "workflow_status": WorkflowStatus.DENIED,
                "reviewer_notes": notes or "Denied by reviewer",
                **judgment,
                "audit_log": [_log("human_review", f"DENIED{judged}")],
            },
            goto="notify_denial",
        )
    else:
        # "return" — send back for correction
        return Command(
            update={
                "human_decision": "return",
                "reviewer_notes": notes or "Returned to operations for correction",
                **judgment,
                "audit_log": [_log("human_review", f"RETURNED for correction{judged}")],
            },
            goto="__end__",
        )


# =============================================================================
# Node: notify_denial
# =============================================================================


def notify_denial(state: ClaimsState) -> dict:
    claim = state["claim"]
    claim_id = claim.claim_id if claim else "<no-claim>"

    reasons = state.get("validation_errors", [])
    if not reasons:
        reasons = [state.get("reviewer_notes", "Denied")]

    log_entry = _log(
        "notify_denial",
        f"claim={claim_id} | reasons={reasons}",
    )
    return {"audit_log": [log_entry]}


# =============================================================================
# helpers
# =============================================================================

# Secured, out-of-band store mapping an opaque reference token → the full
# review context. Keeping the PHI here (rather than in the interrupt value)
# means it never reaches the serialized GraphInterrupt error string. A
# production deployment should back this with an access-controlled store; the
# reviewer UI resolves a token via ``resolve_review_context``.
_REVIEW_STORE: dict[str, dict[str, Any]] = {}


def stash_review_context(context: dict[str, Any]) -> str:
    """Store review context out-of-band and return an opaque reference token."""
    token = f"claim-ref-{uuid.uuid4().hex}"
    _REVIEW_STORE[token] = context
    return token


def resolve_review_context(token: str) -> dict[str, Any] | None:
    """Look up review context previously stashed under an opaque token."""
    return _REVIEW_STORE.get(token)


def _log(node: str, detail: str) -> str:
    return f"{datetime.now().isoformat(timespec='seconds')} | {node} | {detail}"
