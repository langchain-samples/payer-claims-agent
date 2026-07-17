"""Populate LangSmith with useful demo traces, feedback, and an experiment.

Runs a representative mix of claims through the graph with tracing on, then
attaches reviewer feedback so the project shows a realistic **confusion
matrix** (TP/FP/FN/TN) — the story the Approvals Queue tells live. Finally it
runs the eval experiment so there is an experiment to open alongside the
traces.

Usage::

    # needs LANGSMITH_API_KEY (+ ANTHROPIC_API_KEY for the LLM-intake traces)
    uv run python -m evals.generate_demo_data
    uv run python -m evals.generate_demo_data --skip-experiment
    uv run python -m evals.generate_demo_data --skip-traces   # experiment only

Only two runs call the LLM (one clean intake, one messy intake that escalates);
everything else is deterministic, so token cost is small.
"""

from __future__ import annotations

import argparse
import os

try:
    from dotenv import load_dotenv

    load_dotenv(override=bool(os.environ.get("PAYER_CLAIMS_ENV_OVERRIDE")))
except ModuleNotFoundError:
    pass

from langgraph.types import Command

from payer_claims import (
    build_claims_graph,
    configure_tracing,
    confusion_counts,
    invoke_with_run_id,
    record_review_feedback,
)
from payer_claims.scenarios import (
    dual_eligible,
    single_coverage,
    two_commercials,
    working_aged,
)

# Reuse the clean/messy sample docs the UI ships.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
try:
    from demo_common import SAMPLE_RAW_TEXT_CLEAN, SAMPLE_RAW_TEXT_MESSY
except Exception:  # pragma: no cover - app dir layout guard
    SAMPLE_RAW_TEXT_CLEAN = "Claim CLM-CLEAN; Evergreen Medicare Advantage 80%; L1 99213 $450"
    SAMPLE_RAW_TEXT_MESSY = "clm CLM-MESSY ...EVERGREEN? medicare?? ...L1 9921? $4?0"


def _run(graph, state: dict, tid: str):
    return invoke_with_run_id(graph, state, {"configurable": {"thread_id": tid}})


def _resume(graph, tid: str, decision: dict):
    return graph.invoke(Command(resume=decision), {"configurable": {"thread_id": tid}})


def generate_traces() -> list[str]:
    """Run the flows and attach feedback. Returns the confusion cells written."""
    graph = build_claims_graph(with_intake=True)
    has_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    cells: list[str] = []

    # ── LLM-intake traces (the multi-model pipeline in action) ───────────────
    if has_llm:
        # Clean fax → high confidence, auto-settles → spot-checked correct (TN).
        res, rid = _run(graph, {"raw_claim_text": SAMPLE_RAW_TEXT_CLEAN}, "demo-clean")
        fb = record_review_feedback(
            rid, review_decision="approve", intake_correct=True, pended=False,
            notes="demo: clean intake, auto-settled, spot-checked correct",
        )
        cells.append(fb.get("cell"))
        print(f"  clean intake  → confidence={res.get('intake_confidence')} cell={fb.get('cell')}")

        # Messy scan → low confidence → escalates → pends → TP on approve.
        res, rid = _run(graph, {"raw_claim_text": SAMPLE_RAW_TEXT_MESSY}, "demo-messy")
        if res.get("__interrupt__"):
            _resume(graph, "demo-messy", {"decision": "approve", "flag_warranted": True, "intake_correct": False})
        fb = record_review_feedback(
            rid, review_decision="approve", flag_warranted=True, intake_correct=False,
            pended=True, notes="demo: garbled scan correctly pended",
        )
        cells.append(fb.get("cell"))
        print(f"  messy intake  → confidence={res.get('intake_confidence')} escalated={res.get('intake_escalated')} cell={fb.get('cell')}")
    else:
        print("  (no ANTHROPIC_API_KEY — skipping the two LLM-intake traces)")

    # ── deterministic pended claims with injected confidence ─────────────────
    # High-value dual → pended → reviewer confirms (TP), then denies.
    res, rid = _run(graph, {"claim": dual_eligible("CLM-GEN-HV", [("L1", "27130", 9500.0)])}, "demo-hv")
    _resume(graph, "demo-hv", {"decision": "deny", "flag_warranted": True, "intake_correct": True})
    fb = record_review_feedback(rid, review_decision="deny", flag_warranted=True, intake_correct=True, pended=True)
    cells.append(fb.get("cell"))
    print(f"  high-value    → cell={fb.get('cell')}")

    # Low-confidence pend that was actually fine → FALSE POSITIVE.
    res, rid = _run(
        graph,
        {"claim": dual_eligible("CLM-GEN-FP", [("L1", "99213", 300.0)]), "intake_confidence": 0.6},
        "demo-fp",
    )
    _resume(graph, "demo-fp", {"decision": "approve", "flag_warranted": False, "intake_correct": True})
    fb = record_review_feedback(rid, review_decision="approve", flag_warranted=False, intake_correct=True, pended=True)
    cells.append(fb.get("cell"))
    print(f"  false positive→ cell={fb.get('cell')}")

    # Auto-settled claim later found wrong on spot-check → FALSE NEGATIVE.
    res, rid = _run(graph, {"claim": single_coverage("CLM-GEN-FN", [("L1", "99213", 200.0)])}, "demo-fn")
    fb = record_review_feedback(
        rid, intake_correct=False, pended=False,
        notes="demo: auto-settled but spot-check found a mis-parsed coverage",
    )
    cells.append(fb.get("cell"))
    print(f"  false negative→ cell={fb.get('cell')}")

    # ── extra deterministic traces for variety (no feedback) ─────────────────
    _run(graph, {"claim": working_aged("CLM-GEN-WA", [("L1", "99214", 350.0)])}, "demo-wa")
    _run(graph, {"claim": two_commercials("CLM-GEN-TC", [("L1", "99213", 300.0)])}, "demo-tc")

    return [c for c in cells if c]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-traces", action="store_true", help="don't generate traces + feedback")
    parser.add_argument("--skip-experiment", action="store_true", help="don't run the COB core experiment")
    parser.add_argument("--skip-improvement", action="store_true", help="don't run the intake v1-vs-v2 comparison")
    parser.add_argument("--skip-masking", action="store_true", help="don't emit the PHI-masking demo traces")
    args = parser.parse_args()

    traced = configure_tracing(project="payer-claims-agent")
    print(f"tracing enabled: {traced}")
    if not traced:
        print("  WARNING: no LANGSMITH_API_KEY — traces/feedback will not reach LangSmith.")

    if not args.skip_traces:
        print("\nGenerating demo traces + feedback…")
        cells = generate_traces()
        counts = confusion_counts(cells)
        print(
            f"\nFeedback written — confusion: TP={counts['TP']} FP={counts['FP']} "
            f"FN={counts['FN']} TN={counts['TN']} "
            f"(precision={counts['precision']} recall={counts['recall']})"
        )

    if not args.skip_experiment:
        print("\nRunning COB core eval experiment…")
        from evals.run_experiment import main as run_experiment

        run_experiment()

    if not args.skip_improvement and os.environ.get("ANTHROPIC_API_KEY"):
        print("\nRunning intake improvement comparison (baseline v1 vs improved v2)…")
        from evals.run_intake_experiment import main as run_intake

        run_intake()

    if not args.skip_masking:
        print("\nEmitting PHI-masking demo traces (unmasked / redacted / hidden)…")
        from evals.generate_masking_demo import main as run_masking

        run_masking()

    print("\nDone. Open the `payer-claims-agent` project + experiments in LangSmith.")


if __name__ == "__main__":
    main()
