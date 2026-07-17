"""Human-judgment feedback + the FP/FN framing — the loop LangSmith closes.

The "pend for review" gate in :func:`payer_claims.nodes.finalize_settlement`
is a **binary classifier**: *does this claim need a human?* Like any
classifier it is wrong in two directions, and the two errors cost very
different things:

    ================  ==================================  ====================
    Prediction        Ground truth (reviewer judgment)    Confusion cell
    ================  ==================================  ====================
    pended            genuinely needed review             true positive  (TP)
    pended            was actually fine                   FALSE POSITIVE (FP)
    auto-settled      was actually wrong                  FALSE NEGATIVE (FN)
    auto-settled      was correct                         true negative  (TN)
    ================  ==================================  ====================

* A **false positive** wastes a reviewer's time — an operational cost.
* A **false negative** ships a wrong claim — the expensive, hard-to-detect
  miss. Tuning the confidence threshold trades one against the other.

How LangSmith helps with each:

* **Trace** every run with its intake confidence and pend decision.
* **Feedback** — the reviewer's judgment is written back onto the run via
  :func:`record_review_feedback` (``flag_warranted``, ``intake_correct``),
  turning a human decision into a queryable label.
* **Annotation queues** route pended claims to reviewers and, for a sampled
  slice of *auto-settled* claims, let a human catch the false negatives the
  gate would otherwise hide.
* **Datasets + evals** — confirmed FPs/FNs become new examples; the
  ``routing_decision_correct`` evaluator scores the gate's confusion matrix.
* **Engine** mines the FN cluster from production traces and proposes a
  threshold / prompt fix, then adds regression examples so it stays fixed.

Nothing here hardcodes a secret; the client reads credentials from the
environment via :func:`payer_claims.observability.make_client`.
"""

from __future__ import annotations

from typing import Any


# ── confusion-matrix classification ─────────────────────────────────────────


def classify_review(
    *,
    pended: bool,
    flag_warranted: bool | None,
    intake_correct: bool | None,
) -> str | None:
    """Classify one claim into a confusion-matrix cell.

    Args:
        pended: did the gate pend this claim (the classifier's *prediction*)?
        flag_warranted: for a pended claim, did it genuinely need review?
        intake_correct: for an auto-settled claim spot-checked by a human,
            was the intake/settlement actually correct?

    Returns ``"TP" | "FP" | "FN" | "TN"``, or ``None`` when there is not enough
    judgment to classify (e.g. an auto-settled claim nobody spot-checked).
    """
    if pended:
        if flag_warranted is None:
            return None
        return "TP" if flag_warranted else "FP"
    # auto-settled → only classifiable if a human spot-checked correctness
    if intake_correct is None:
        return None
    return "FN" if intake_correct is False else "TN"


def confusion_counts(cells: list[str | None]) -> dict[str, int]:
    """Tally classified cells and derive precision/recall for the pend gate."""
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for c in cells:
        if c in counts:
            counts[c] += 1
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    counts["precision"] = round(tp / (tp + fp), 4) if (tp + fp) else None
    counts["recall"] = round(tp / (tp + fn), 4) if (tp + fn) else None
    return counts


# ── capture a run id so feedback can be attached to it ──────────────────────


def invoke_with_run_id(graph, payload: dict, config: dict) -> tuple[dict, str | None]:
    """Invoke `graph` and return ``(result, run_id)``.

    Uses LangChain's ``collect_runs`` to grab the root run id, which is what
    :func:`record_review_feedback` attaches reviewer judgments to. Works whether
    or not tracing export is on; the id only resolves in the LangSmith UI when
    tracing was enabled for the run.
    """
    from langchain_core.tracers.context import collect_runs

    with collect_runs() as cb:
        result = graph.invoke(payload, config)
    run_id = str(cb.traced_runs[0].id) if cb.traced_runs else None
    return result, run_id


# ── write reviewer judgment back to LangSmith as feedback ───────────────────


def record_review_feedback(
    run_id: str,
    *,
    review_decision: str | None = None,
    flag_warranted: bool | None = None,
    intake_correct: bool | None = None,
    pended: bool = True,
    notes: str = "",
    client: Any | None = None,
) -> dict[str, Any]:
    """Attach a reviewer's judgment to a run as LangSmith feedback.

    Writes up to three feedback keys, each independently useful for filtering
    and for building eval datasets:

    * ``flag_warranted`` — 1 if pending was warranted (TP), 0 if not (FP).
    * ``intake_correct`` — 1/0 correctness of the LLM intake transcription.
    * ``review_decision`` — categorical approve/deny/return (score omitted).

    Also derives the confusion cell and records it as ``routing_outcome`` so
    FP/FN rates can be charted directly in LangSmith.

    Returns a summary dict (including the confusion ``cell``). No-op-safe: if
    ``run_id`` is falsy it returns ``{"written": []}`` without calling the API.
    """
    written: list[str] = []
    if not run_id:
        return {"written": written, "cell": None, "reason": "no run_id"}

    if client is None:
        from .observability import make_client

        client = make_client()

    if flag_warranted is not None:
        client.create_feedback(
            run_id, key="flag_warranted", score=int(bool(flag_warranted)), comment=notes or None
        )
        written.append("flag_warranted")

    if intake_correct is not None:
        client.create_feedback(
            run_id, key="intake_correct", score=int(bool(intake_correct)), comment=notes or None
        )
        written.append("intake_correct")

    if review_decision is not None:
        client.create_feedback(run_id, key="review_decision", value=review_decision)
        written.append("review_decision")

    cell = classify_review(
        pended=pended, flag_warranted=flag_warranted, intake_correct=intake_correct
    )
    if cell is not None:
        client.create_feedback(run_id, key="routing_outcome", value=cell)
        written.append("routing_outcome")

    return {"written": written, "cell": cell}
