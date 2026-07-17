"""Deterministic tests for the intake-extraction evaluators (no LLM calls).

These grade synthetic extraction outputs against references, so they lock in the
scoring logic that the eval-driven improvement experiment relies on.
"""

from __future__ import annotations

from evals.evaluators import (
    amount_accuracy,
    coverage_type_accuracy,
    intake_extraction_correct,
)

_REF = {
    "member_id": "MB-1",
    "coverages": [
        {"payer_name": "Imperial Health Plan", "coverage_type": "medicare_advantage"},
        {"payer_name": "Kentucky Medicaid", "coverage_type": "medicaid"},
    ],
    "lines": [{"procedure_code": "99214", "billed_amount": 650.00}],
}


def _out(cov_types, amount=650.00):
    return {
        "coverages": [{"coverage_type": t} for t in cov_types],
        "lines": [{"procedure_code": "99214", "billed_amount": amount}],
    }


def test_coverage_type_all_correct():
    r = coverage_type_accuracy({}, _out(["medicare_advantage", "medicaid"]), _REF)
    assert r["score"] == 1.0


def test_coverage_type_baseline_miss():
    # Baseline mislabels the MA carrier as commercial → 1/2.
    r = coverage_type_accuracy({}, _out(["commercial", "medicaid"]), _REF)
    assert r["score"] == 0.5


def test_coverage_type_missing_coverage():
    r = coverage_type_accuracy({}, _out(["medicare_advantage"]), _REF)
    assert r["score"] == 0.5  # dropped the second coverage


def test_amount_accuracy_within_tolerance():
    assert amount_accuracy({}, _out(["medicare_advantage", "medicaid"], 650.004), _REF)["score"] == 1.0


def test_amount_accuracy_wrong():
    assert amount_accuracy({}, _out(["medicare_advantage", "medicaid"], 999.0), _REF)["score"] == 0.0


def test_extraction_correct_perfect():
    r = intake_extraction_correct({}, _out(["medicare_advantage", "medicaid"]), _REF)
    assert r["score"] == 1


def test_extraction_correct_fails_on_type_miss():
    r = intake_extraction_correct({}, _out(["commercial", "medicaid"]), _REF)
    assert r["score"] == 0


def test_empty_output_scores_zero_not_crash():
    r = coverage_type_accuracy({}, {"coverages": [], "lines": []}, _REF)
    assert r["score"] == 0.0
    assert intake_extraction_correct({}, {"coverages": [], "lines": []}, _REF)["score"] == 0
