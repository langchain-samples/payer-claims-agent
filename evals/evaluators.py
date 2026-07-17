"""Evaluators for the COB agent.

Design principle: **check what you can compute exactly with code; only judge
what is genuinely subjective.**

* The COB rule chosen, the payment split, and the settlement status are exact,
  auditable facts — graded with deterministic code evaluators (no LLM, no
  cost, no flakiness).
* Intake normalization quality (did the LLM faithfully transcribe messy text?)
  is fuzzy — graded with a single LLM-as-judge.

All code evaluators use the ``(inputs, outputs, reference_outputs)`` signature
consumed by ``Client.evaluate``.
"""

from __future__ import annotations

from typing import Any

# Dollar tolerance for floating-point payment comparisons.
_MONEY_TOL = 0.01


# ── deterministic code evaluators (the auditable core) ─────────────────────────


def cob_rule_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Did the graph select the correct coordination-of-benefits rule?"""
    expected = reference_outputs.get("cob_rule")
    actual = outputs.get("cob_rule")
    match = actual == expected
    return {
        "key": "cob_rule_correct",
        "score": int(match),
        "comment": f"expected={expected} actual={actual}",
    }


def payment_split_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Are primary/secondary payments and patient responsibility all exact?"""
    fields = ("primary_paid", "secondary_paid", "patient_responsibility")
    deltas = {}
    ok = True
    for f in fields:
        exp = float(reference_outputs.get(f, 0.0))
        act = float(outputs.get(f, 0.0))
        delta = abs(exp - act)
        deltas[f] = round(delta, 4)
        if delta > _MONEY_TOL:
            ok = False
    return {
        "key": "payment_split_correct",
        "score": int(ok),
        "comment": f"deltas={deltas} (tol={_MONEY_TOL})",
    }


def settlement_status_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Did the claim reach the expected terminal/pended status?"""
    expected = reference_outputs.get("workflow_status")
    actual = outputs.get("workflow_status")
    match = actual == expected
    return {
        "key": "settlement_status_correct",
        "score": int(match),
        "comment": f"expected={expected} actual={actual}",
    }


def routing_decision_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Score the pend-gate classifier and record its confusion cell.

    The gate predicts *pend* (positive) vs *auto-settle* (negative); the
    ``should_pend`` reference label is ground truth. ``score`` is 1 when the
    prediction matches, and ``value`` carries the confusion cell (TP/FP/FN/TN)
    so FP/FN rates can be charted and filtered directly in LangSmith.
    """
    expected_pend = bool(reference_outputs.get("should_pend"))
    actual_pend = outputs.get("pended")
    if actual_pend is None:  # fall back to status if the target didn't set it
        actual_pend = outputs.get("workflow_status") == "pended_review"
    actual_pend = bool(actual_pend)

    if actual_pend and expected_pend:
        cell = "TP"
    elif actual_pend and not expected_pend:
        cell = "FP"
    elif not actual_pend and expected_pend:
        cell = "FN"
    else:
        cell = "TN"

    return {
        "key": "routing_decision_correct",
        "score": int(actual_pend == expected_pend),
        "value": cell,
        "comment": (
            f"cell={cell} | expected_pend={expected_pend} actual_pend={actual_pend} "
            f"| pend_reasons={outputs.get('pend_reasons')}"
        ),
    }


CORE_EVALUATORS = [
    cob_rule_correct,
    payment_split_correct,
    settlement_status_correct,
    routing_decision_correct,
]


# ── intake-extraction evaluators (grade the LLM step deterministically) ────────
# These compare the extracted structured fields to hand-labeled references — no
# LLM, so the before/after improvement they measure is stable and cheap. Expects
# outputs/reference shaped like evals.intake_dataset: {coverages:[{coverage_type}],
# lines:[{procedure_code, billed_amount}], member_id}.


def coverage_type_accuracy(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Fraction of coverages whose coverage_type matches the reference (by order)."""
    ref = reference_outputs.get("coverages", []) or []
    got = outputs.get("coverages", []) or []
    total = len(ref)
    if total == 0:
        return {"key": "coverage_type_accuracy", "score": None, "comment": "no reference coverages"}
    correct = sum(
        1
        for i, r in enumerate(ref)
        if i < len(got) and got[i].get("coverage_type") == r.get("coverage_type")
    )
    got_types = [c.get("coverage_type") for c in got]
    exp_types = [c.get("coverage_type") for c in ref]
    return {
        "key": "coverage_type_accuracy",
        "score": correct / total,
        "comment": f"{correct}/{total} | expected={exp_types} got={got_types}",
    }


def amount_accuracy(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Fraction of line billed_amounts that match the reference (by order, ±$0.01)."""
    ref = reference_outputs.get("lines", []) or []
    got = outputs.get("lines", []) or []
    total = len(ref)
    if total == 0:
        return {"key": "amount_accuracy", "score": None, "comment": "no reference lines"}
    correct = sum(
        1
        for i, r in enumerate(ref)
        if i < len(got)
        and abs(float(got[i].get("billed_amount", 0.0)) - float(r.get("billed_amount", 0.0))) <= _MONEY_TOL
    )
    return {
        "key": "amount_accuracy",
        "score": correct / total,
        "comment": f"{correct}/{total} line amounts within ${_MONEY_TOL}",
    }


def intake_extraction_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """Exact-ish overall: every coverage_type right, every amount right, counts match."""
    cov = coverage_type_accuracy(inputs, outputs, reference_outputs)
    amt = amount_accuracy(inputs, outputs, reference_outputs)
    counts_ok = len(outputs.get("coverages", []) or []) == len(reference_outputs.get("coverages", []) or []) \
        and len(outputs.get("lines", []) or []) == len(reference_outputs.get("lines", []) or [])
    perfect = counts_ok and cov.get("score") == 1.0 and amt.get("score") in (1.0, None)
    return {
        "key": "intake_extraction_correct",
        "score": int(bool(perfect)),
        "comment": f"counts_ok={counts_ok} coverage={cov.get('score')} amount={amt.get('score')}",
    }


INTAKE_EVALUATORS = [
    coverage_type_accuracy,
    amount_accuracy,
    intake_extraction_correct,
]


# ── LLM-as-judge (only for the fuzzy intake step) ──────────────────────────────


def _judge():
    """Lazy judge model so importing this module needs no API key."""
    from typing import Annotated, TypedDict

    from langchain_anthropic import ChatAnthropic

    class IntakeGrade(TypedDict):
        reasoning: Annotated[str, ..., "Brief justification"]
        faithful: Annotated[bool, ..., "True if every extracted field is supported by the source text with no invented values"]
        confidence: Annotated[float, ..., "0.0-1.0"]

    # NB: claude-sonnet-5 rejects the `temperature` param — do not set it.
    model = ChatAnthropic(model="claude-sonnet-5")
    return model.with_structured_output(IntakeGrade)


def intake_normalization_quality(inputs: dict, outputs: dict, reference_outputs: dict | None = None) -> dict:
    """LLM-judge: did intake transcribe the raw claim text faithfully?

    Expects ``inputs['raw_claim_text']`` and ``outputs['parsed_claim']`` (dict).
    Grades fidelity only — it must NOT reward or penalize any downstream COB or
    payment reasoning, which the LLM never performs.
    """
    raw = inputs.get("raw_claim_text")
    parsed = outputs.get("parsed_claim")
    if not raw or parsed is None:
        return {
            "key": "intake_normalization_quality",
            "score": None,
            "comment": "not an intake example (no raw_claim_text/parsed_claim) — skipped",
        }

    judge = _judge()
    prompt = (
        "You are auditing a claims-intake transcription. Compare the STRUCTURED "
        "extraction against the SOURCE document. Mark it faithful only if every "
        "extracted value (payer names, coverage types, amounts, codes, ids) is "
        "directly supported by the source and nothing was invented.\n\n"
        f"SOURCE DOCUMENT:\n{raw}\n\n"
        f"STRUCTURED EXTRACTION:\n{parsed}\n"
    )
    grade = judge.invoke([{"role": "user", "content": prompt}])
    return {
        "key": "intake_normalization_quality",
        "score": 1 if grade["faithful"] else 0,
        "comment": f"{grade['reasoning']} (confidence={grade['confidence']})",
    }
