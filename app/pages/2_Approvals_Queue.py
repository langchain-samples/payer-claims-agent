"""Streamlit demo — Approvals Queue (reviewer console).

A dedicated screen listing claims the agent pended for human review. Each is a
**classifier decision** the reviewer labels: approving/denying while judging
whether the flag was warranted (TP vs FP) and whether intake was correct. Those
judgments resume the real graph and are written back to LangSmith as feedback,
and the page tallies the live TP/FP/FN/TN confusion matrix.

The queue is shared in-process with the intake page (see ``demo_common``) and
seeded with a few pended claims so the list is populated for a live demo.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

import demo_common as dc

st.set_page_config(page_title="Approvals Queue", page_icon="✅", layout="wide")

_CELL_HELP = {
    "TP": "True positive — correctly pended",
    "FP": "False positive — pended but was fine (a wasted review)",
    "FN": "False negative — would have shipped a wrong claim",
    "TN": "True negative — correctly auto-settled",
}
_REASON_LABELS = {
    "high_value": "💰 high-value claim",
    "low_confidence": "🤖 low intake confidence",
}


def render_confusion_header(queue: dict) -> None:
    counts = dc.queue_confusion(queue)
    st.caption("Pend-gate confusion matrix (seeded sample history + this session's reviews)")
    c = st.columns(6)
    c[0].metric("TP", counts["TP"])
    c[1].metric("FP", counts["FP"], help="false positives — wasted reviews")
    c[2].metric("FN", counts["FN"], help="false negatives — missed bad claims")
    c[3].metric("TN", counts["TN"])
    c[4].metric("Precision", "—" if counts["precision"] is None else f"{counts['precision']:.0%}")
    c[5].metric("Recall", "—" if counts["recall"] is None else f"{counts['recall']:.0%}")


def render_open_item(item: dict) -> None:
    reasons = item.get("pend_reasons") or []
    why = ", ".join(_REASON_LABELS.get(r, r) for r in reasons) or "flagged"
    conf = item.get("confidence")
    header = (
        f"🟡 {item['id']} · {item['title']} — {why}"
        + (f" · conf {conf:.0%}" if conf is not None else "")
    )
    with st.expander(header, expanded=False):
        result = item["result"]
        dc.render_intake_diagnostics(result)
        dc.render_waterfall(result)
        st.caption(f"Patient responsibility: ${item.get('patient', 0):,.2f}")

        st.markdown(
            "**Reviewer judgment** — this labels the gate's decision as a true "
            "positive or a **false positive**, and flags a data-quality miss. "
            "It is written back to LangSmith as feedback."
        )
        cols = st.columns(2)
        flag_warranted = cols[0].checkbox(
            "This claim genuinely needed review",
            value=True,
            key=f"flag_{item['id']}",
            help="Unchecked = false positive: the gate pended a claim that was fine.",
        )
        intake_correct = cols[1].checkbox(
            "Intake transcription was correct",
            value=True,
            key=f"intake_{item['id']}",
            help="Unchecked marks a data-quality miss the confidence gate should catch.",
        )
        notes = st.text_input("Reviewer notes (optional)", key=f"notes_{item['id']}")

        b1, b2, b3 = st.columns(3)
        if b1.button("✅ Approve", key=f"approve_{item['id']}", use_container_width=True):
            _resolve(item, "approve", flag_warranted, intake_correct, notes)
        if b2.button("⛔ Deny", key=f"deny_{item['id']}", use_container_width=True):
            _resolve(item, "deny", flag_warranted, intake_correct, notes)
        if b3.button("↩️ Return", key=f"return_{item['id']}", use_container_width=True):
            _resolve(item, "return", flag_warranted, intake_correct, notes)


def _resolve(item, action, flag_warranted, intake_correct, notes) -> None:
    try:
        dc.resolve_item(
            item, action,
            flag_warranted=flag_warranted, intake_correct=intake_correct, notes=notes,
        )
    except Exception as exc:
        st.error(f"{type(exc).__name__}: {exc}")
        return
    st.rerun()


def render_resolved_item(item: dict) -> None:
    cell = item.get("outcome_cell")
    badge = f" · **{cell}**" if cell else ""
    st.markdown(
        f"✔️ {item['id']} · {item['title']} — decision: **{item.get('decision')}**{badge}"
    )
    if cell:
        st.caption(_CELL_HELP.get(cell, cell))
    if item.get("sample"):
        st.caption("Seeded sample review (not a live trace).")
        return
    written = item.get("feedback_written") or []
    if written:
        st.caption("LangSmith feedback: " + ", ".join(f"`{k}`" for k in written))
    elif item.get("feedback_error"):
        st.caption(f"Feedback not written ({item['feedback_error']})")
    else:
        st.caption("Tracing off — classified locally, not written to LangSmith.")


def main() -> None:
    dc.render_sidebar()
    st.header("✅ Approvals Queue")
    st.caption(
        "Claims the agent pended for a human. Each decision resumes the graph "
        "and writes the reviewer's judgment to LangSmith as feedback."
    )

    queue = dc.get_queue()
    render_confusion_header(queue)

    if st.button("↺ Reset demo queue"):
        dc.reset_queue()
        st.rerun()

    st.divider()

    open_items = [it for it in queue["items"] if it["status"] == "open"]
    resolved = [it for it in queue["items"] if it["status"] != "open"]

    st.subheader(f"Awaiting review ({len(open_items)})")
    if not open_items:
        st.success("Queue clear — no claims awaiting review.")
    for item in open_items:
        render_open_item(item)

    if resolved:
        st.divider()
        st.subheader(f"Resolved ({len(resolved)})")
        for item in resolved:
            render_resolved_item(item)


main()
