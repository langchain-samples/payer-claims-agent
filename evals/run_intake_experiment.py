"""Eval-driven improvement: baseline vs improved intake, as two experiments.

Runs the intake step over the hand-labeled dataset twice — a naive **baseline**
(literal prompt, no model escalation) and the **improved** agent (carrier-aware
prompt + confidence-driven escalation to a stronger model) — and prints the
score lift. Both land in LangSmith as comparable experiments.

Usage::

    # needs LANGSMITH_API_KEY + ANTHROPIC_API_KEY
    uv run python -m evals.run_intake_experiment              # PHI redacted (default)
    uv run python -m evals.run_intake_experiment --unmasked   # inputs/outputs visible
                                                              # (synthetic data only)

This is the concrete "the eval made the agent better" artifact: open the two
experiments side-by-side in LangSmith and watch coverage_type_accuracy jump.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv(override=bool(os.environ.get("PAYER_CLAIMS_ENV_OVERRIDE")))
except ModuleNotFoundError:
    pass

from payer_claims import configure_tracing, make_client
from payer_claims.intake import EXTRACT_PROMPT_BASELINE, normalize_claim

from .evaluators import INTAKE_EVALUATORS
from .intake_dataset import INTAKE_DATASET_NAME, build_intake_dataset

_METRICS = ["coverage_type_accuracy", "amount_accuracy", "intake_extraction_correct"]


def _flatten_claim(claim) -> dict:
    """Reduce a Claim to the structured fields the intake evaluators compare."""
    covs = claim.member.coverages if claim.member else []
    return {
        "member_id": claim.member.member_id if claim.member else "",
        "coverages": [
            {"payer_name": c.payer_name, "coverage_type": c.coverage_type.value} for c in covs
        ],
        "lines": [
            {"procedure_code": code, "billed_amount": amt} for _lid, code, amt in claim.lines
        ],
    }


def baseline_target(inputs: dict) -> dict:
    """v1 — naive first draft: literal prompt, single model, no escalation."""
    try:
        claim, _ = normalize_claim(
            inputs["raw_claim_text"],
            extract_prompt=EXTRACT_PROMPT_BASELINE,
            escalate=False,
        )
        return _flatten_claim(claim)
    except Exception as exc:
        return {"coverages": [], "lines": [], "error": f"{type(exc).__name__}: {exc}"}


def improved_target(inputs: dict) -> dict:
    """v2 — improved: carrier-aware prompt + confidence-driven escalation."""
    try:
        claim, _ = normalize_claim(inputs["raw_claim_text"])  # defaults = improved
        return _flatten_claim(claim)
    except Exception as exc:
        return {"coverages": [], "lines": [], "error": f"{type(exc).__name__}: {exc}"}


def _means(results) -> dict[str, float]:
    df = results.to_pandas()
    out: dict[str, float] = {}
    for m in _METRICS:
        col = next((c for c in df.columns if c == f"feedback.{m}"), None)
        if col is not None:
            out[m] = float(df[col].mean())
    return out


def _print_delta(v1: dict, v2: dict) -> None:
    print("\n  metric                      baseline(v1)   improved(v2)   delta")
    print("  " + "-" * 62)
    for m in _METRICS:
        a = v1.get(m)
        b = v2.get(m)
        if a is None or b is None:
            continue
        arrow = "▲" if b > a + 1e-9 else ("▼" if b < a - 1e-9 else "=")
        print(f"  {m:<26} {a:>10.2f}   {b:>12.2f}   {arrow} {b - a:+.2f}")


def main(unmasked: bool = False) -> None:
    # Default masks/redacts PHI; `unmasked` shows full inputs/outputs and must
    # ONLY be used with synthetic / non-PHI data.
    if unmasked:
        os.environ.pop("LANGSMITH_HIDE_INPUTS", None)
        os.environ.pop("LANGSMITH_HIDE_OUTPUTS", None)
        traced = configure_tracing(project="payer-claims-agent", mask_phi=False)
        client = make_client(anonymizer=None)  # full inputs/outputs visible
        suffix = "-unmasked"
        print(f"tracing enabled: {traced} | PHI masking OFF — inputs/outputs VISIBLE")
        print("  NOTE: unmasked is for SYNTHETIC data only; never run on real PHI.")
    else:
        traced = configure_tracing(project="payer-claims-agent")
        client = make_client()  # PHI-redacted
        suffix = ""
        print(f"tracing enabled: {traced} | PHI redacted (default)")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  ERROR: ANTHROPIC_API_KEY required (intake calls the LLM). Aborting.")
        return

    build_intake_dataset(client)

    print("\nRunning BASELINE (v1: literal prompt, no escalation)…")
    v1 = client.evaluate(
        baseline_target,
        data=INTAKE_DATASET_NAME,
        evaluators=INTAKE_EVALUATORS,
        experiment_prefix=f"intake-v1-baseline{suffix}",
        max_concurrency=4,
    )
    print("Running IMPROVED (v2: carrier-aware prompt + escalation)…")
    v2 = client.evaluate(
        improved_target,
        data=INTAKE_DATASET_NAME,
        evaluators=INTAKE_EVALUATORS,
        experiment_prefix=f"intake-v2-improved{suffix}",
        max_concurrency=4,
    )

    try:
        m1, m2 = _means(v1), _means(v2)
        _print_delta(m1, m2)
        cov1, cov2 = m1.get("coverage_type_accuracy"), m2.get("coverage_type_accuracy")
        if cov1 is not None and cov2 is not None:
            print(
                f"\n  ➜ Coverage-type accuracy improved {cov1:.0%} → {cov2:.0%} "
                f"after the prompt + model-routing fix."
            )
    except Exception:
        print("  (install pandas for the inline delta; both experiments are in LangSmith)")

    print(f"\nDone. Compare 'intake-v1-baseline{suffix}' vs "
          f"'intake-v2-improved{suffix}' in LangSmith.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--unmasked",
        action="store_true",
        help="run with PHI masking OFF so trace inputs/outputs are visible "
        "(synthetic data only — never real PHI)",
    )
    args = parser.parse_args()
    main(unmasked=args.unmasked)
