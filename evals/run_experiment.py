"""Run a LangSmith experiment over the COB core dataset.

Usage::

    # env: LANGSMITH_API_KEY (+ ANTHROPIC_API_KEY only if you use intake evals)
    uv run python -m evals.run_experiment

This closes the "test" loop: it evaluates the current graph against the
canonical dataset and records an experiment in LangSmith. Regressions show up
as dropped scores; LangSmith Engine watches these experiments (and production
traces) to drive the "improve" loop.
"""

from __future__ import annotations

import os

# Load .env so this entrypoint uses the project's configured credentials, the
# same way the Streamlit app does. Note: os.environ still wins over .env — a
# stale exported LANGSMITH_API_KEY in your shell will shadow .env and cause a
# 401. `unset LANGSMITH_API_KEY` if that happens. Pass override=True below to
# force .env to win.
try:
    from dotenv import load_dotenv

    load_dotenv(override=bool(os.environ.get("PAYER_CLAIMS_ENV_OVERRIDE")))
except ModuleNotFoundError:  # python-dotenv not installed → rely on real env
    pass

from payer_claims import build_claims_graph, configure_tracing, make_client

from .dataset import CORE_DATASET_NAME, build_dataset, deserialize_claim
from .evaluators import CORE_EVALUATORS


def _flatten(result: dict) -> dict:
    """Reduce raw graph state to the flat fields the evaluators compare."""
    cob = result.get("cob_rule_applied")
    status = result.get("workflow_status")
    status_val = status.value if status is not None else None
    return {
        "cob_rule": cob.value if cob is not None else None,
        "primary_paid": result.get("primary_total_paid", 0.0),
        "secondary_paid": result.get("secondary_total_paid", 0.0),
        "patient_responsibility": result.get("patient_responsibility", 0.0),
        "workflow_status": status_val,
        "pended": status_val == "pended_review",
        "pend_reasons": result.get("pend_reasons", []),
        "intake_confidence": result.get("intake_confidence"),
        "is_valid": result.get("is_valid"),
    }


def cob_core_target(inputs: dict) -> dict:
    """Eval target: run the deterministic graph on a structured claim."""
    claim = deserialize_claim(inputs["claim"])
    graph = build_claims_graph()  # deterministic path (no intake)
    config = {"configurable": {"thread_id": f"eval-{claim.claim_id}"}}
    result = graph.invoke({"claim": claim}, config)
    return _flatten(result)


def main() -> None:
    # Observe: turn on tracing (PHI-masked by default). Safe no-op without a key.
    traced = configure_tracing(project="payer-claims-agent")
    print(f"tracing enabled: {traced}")

    client = make_client()  # PHI-anonymized
    build_dataset(client)   # idempotent

    results = client.evaluate(
        cob_core_target,
        data=CORE_DATASET_NAME,
        evaluators=CORE_EVALUATORS,
        experiment_prefix="cob-core",
        max_concurrency=4,
    )

    # Print a compact pass/fail summary + the pend-gate confusion matrix.
    print(f"\nExperiment: {getattr(results, 'experiment_name', '(see LangSmith)')}")
    try:
        df = results.to_pandas()
        score_cols = [c for c in df.columns if c.startswith("feedback.")]
        for col in score_cols:
            print(f"  {col.removeprefix('feedback.')}: mean={df[col].mean():.2f}")
        _print_confusion(df)
    except Exception:
        print("  (install pandas for an inline summary; full results in LangSmith)")


def _print_confusion(df) -> None:
    """Tally the pend-gate confusion matrix from predicted vs expected pend."""
    from payer_claims import classify_review, confusion_counts

    if "outputs.pended" not in df.columns or "reference.should_pend" not in df.columns:
        return
    cells: list[str] = []
    for _, row in df.iterrows():
        pended = bool(row["outputs.pended"])
        should = bool(row["reference.should_pend"])
        # In the offline experiment ground truth is `should_pend`, so a pend is
        # a TP iff it should have pended; an auto-settle is a FN iff it should.
        cells.append(
            classify_review(
                pended=pended,
                flag_warranted=should if pended else None,
                intake_correct=(not should) if not pended else None,
            )
        )
    counts = confusion_counts(cells)
    print(
        f"  pend-gate confusion: TP={counts['TP']} FP={counts['FP']} "
        f"FN={counts['FN']} TN={counts['TN']} "
        f"(precision={counts['precision']} recall={counts['recall']})"
    )


if __name__ == "__main__":
    main()
