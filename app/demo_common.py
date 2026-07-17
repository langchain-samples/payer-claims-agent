"""Shared state + helpers for the multipage Streamlit demo.

Both screens — **Intake / Adjudication** (the home page) and the **Approvals
Queue** — import this module. It centralizes:

* one compiled graph + one in-memory checkpointer, shared process-wide via
  ``st.cache_resource`` so a claim pended on the intake page can be resumed on
  the approvals page;
* an in-process **review queue** (also a cached resource) seeded with a few
  pended claims so the approvals list looks realistic in a live demo;
* the render helpers and sample documents used by both pages.

Security notes (local demo — no auth, do not expose publicly):
  * Pasted claim text is UNTRUSTED. It is only ever placed in state as
    ``raw_claim_text`` and handed to the injection-hardened, structured-output
    intake node — never eval'd, exec'd, or deserialized.
  * ``thread_id``s are server-generated uuid4s; they are never read from user
    input or query params (avoids cross-claim IDOR in the queue).
  * No secrets are hardcoded; credentials come from the environment / .env.
"""

from __future__ import annotations

import os
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.types import Command

from payer_claims import (
    build_claims_graph,
    classify_review,
    configure_tracing,
    confusion_counts,
    invoke_with_run_id,
    record_review_feedback,
    run_config,
)
from payer_claims.scenarios import (
    dual_eligible,
    single_coverage,
    two_commercials,
    working_aged,
)

load_dotenv()

PROJECT = "payer-claims-agent"

# ── sample intake documents ─────────────────────────────────────────────────

# Clean, well-formed fax — the extract model handles it confidently (no escalation).
SAMPLE_RAW_TEXT_CLEAN = """\
COORDINATION OF BENEFITS INTAKE — FAX COVER
Claim: CLM-RAW-777   Provider: PR-2044   Date of service: 2025-02-03
Patient MB-RAW-777, DOB 1957-09-30, retired (not employed).

Coverage on file:
  1) Evergreen Medicare Advantage (EVG001) — pays 80%
  2) Kentucky Medicaid (MCD001) — pays 80%, payer of last resort

Services rendered:
  L1  99213  office visit ............ $450.00
  L2  85025  CBC panel ............... $125.00
"""

# Ambiguous, OCR-style fax: the fields are all *present and extractable* (so the
# claim stays valid), but several are genuinely uncertain — an unclear coverage
# type, a "verify if active" note, hand-written employment. The verify model
# returns LOW confidence, so intake ESCALATES to the stronger model; the claim
# is also high-value, so it reliably pends and lands in the Approvals Queue.
SAMPLE_RAW_TEXT_MESSY = """\
====  FAXED COB WORKSHEET  (scan quality: fair — intake clerk margin notes)  ====
Claim: CLM-RAW-8811    Provider: PR-3050    Date of service: 2025-03-12
Patient MB-RAW-8811, DOB 1952-07-19.  Employment: hand-written, hard to read ("retired?").

Coverage on file:
  1) "EVERGREEN" plan — id EVG055 — pays 80%
       clerk note: "can't tell if this is Medicare Advantage or straight Medicare — please verify"
  2) Kentucky Medicaid — id MCD022 — pays 80% (payer of last resort)
       clerk note: "verify this is still active as of the DOS"

Services rendered:
  L1  27130  total hip arthroplasty ......... $9,800.00
  L2  99213  post-op office visit ...........   $450.00
"""


# ── cached shared resources (process-wide) ──────────────────────────────────


@st.cache_resource
def ensure_tracing() -> bool:
    """Turn on LangSmith tracing once; return whether it is active."""
    return configure_tracing(project=PROJECT)


@st.cache_resource
def get_graph():
    """One compiled graph + in-memory checkpointer, shared across all pages."""
    ensure_tracing()
    return build_claims_graph(with_intake=True)


@st.cache_resource
def get_queue() -> dict:
    """Process-wide review queue, seeded once with a few pended claims."""
    queue: dict = {"items": [], "next_id": 1}
    _seed(queue)
    return queue


# ── thread ids + graph drivers ──────────────────────────────────────────────


def new_thread() -> str:
    """Server-generated, never user-supplied thread id."""
    return f"ui-{uuid.uuid4()}"


def adjuster_id() -> str:
    """Adjuster identity for per-adjuster trace filtering (env-overridable)."""
    return os.environ.get("CLAIMS_ADJUSTER_ID", "demo-adjuster")


def submit_claim(initial_state: dict) -> dict:
    """Invoke the graph on a fresh thread, capturing the trace's run id.

    Returns ``{thread_id, config, result, run_id}``. Raises on failure so the
    caller can render the error.
    """
    tid = new_thread()
    config = run_config(thread_id=tid, user_id=adjuster_id())
    result, run_id = invoke_with_run_id(get_graph(), initial_state, config)
    return {"thread_id": tid, "config": config, "result": result, "run_id": run_id}


def is_pended(result: dict) -> bool:
    return bool(result.get("__interrupt__"))


# ── queue management ────────────────────────────────────────────────────────


def _title_for(result: dict, fallback: str = "") -> str:
    claim = result.get("claim")
    if not claim:
        return fallback or "claim"
    rule = getattr(result.get("cob_rule_applied"), "value", "") or "?"
    return f"{claim.claim_id} · {rule}"


def _append_item(queue: dict, run: dict, title: str | None) -> dict:
    """Build a queue item and append it to `queue`.

    Takes the queue explicitly so it can be called *while `get_queue()` is still
    computing its cached value* (during seeding) — re-entering `get_queue()`
    there would deadlock on Streamlit's per-resource cache lock.
    """
    result = run["result"]
    item = {
        "id": f"Q{queue['next_id']}",
        "thread_id": run["thread_id"],
        "config": run["config"],
        "run_id": run.get("run_id"),
        "title": title or _title_for(result),
        "result": result,
        "pend_reasons": result.get("pend_reasons", []),
        "confidence": result.get("intake_confidence"),
        "low_confidence_fields": result.get("low_confidence_fields", []),
        "patient": result.get("patient_responsibility", 0.0),
        "status": "open",
        "decision": None,
        "outcome_cell": None,
        "feedback_written": [],
    }
    queue["items"].append(item)
    queue["next_id"] += 1
    return item


def enqueue_if_pended(run: dict, *, title: str | None = None) -> dict | None:
    """Add a pended run to the queue (idempotent by thread_id)."""
    result = run["result"]
    if not is_pended(result):
        return None
    queue = get_queue()
    for it in queue["items"]:
        if it["thread_id"] == run["thread_id"]:
            return it  # already queued
    return _append_item(queue, run, title or _title_for(result))


def resolve_item(
    item: dict,
    action: str,
    *,
    flag_warranted: bool,
    intake_correct: bool,
    notes: str = "",
) -> dict:
    """Resume the real graph for a queued claim and write reviewer feedback."""
    resumed = get_graph().invoke(
        Command(
            resume={
                "decision": action,
                "flag_warranted": flag_warranted,
                "intake_correct": intake_correct,
                "notes": notes,
            }
        ),
        item["config"],
    )
    item["result"] = resumed
    item["status"] = "resolved"
    item["decision"] = action

    if item.get("run_id") and ensure_tracing():
        try:
            fb = record_review_feedback(
                item["run_id"],
                review_decision=action,
                flag_warranted=flag_warranted,
                intake_correct=intake_correct,
                pended=True,
                notes=notes,
            )
        except Exception as exc:  # never break the demo on a feedback hiccup
            fb = {"written": [], "cell": None, "error": f"{type(exc).__name__}: {exc}"}
    else:
        fb = {
            "written": [],
            "cell": classify_review(
                pended=True, flag_warranted=flag_warranted, intake_correct=intake_correct
            ),
        }
    item["outcome_cell"] = fb.get("cell")
    item["feedback_written"] = fb.get("written", [])
    item["feedback_error"] = fb.get("error")
    return item


def queue_confusion(queue: dict) -> dict:
    """Confusion-matrix tally over resolved queue items."""
    return confusion_counts([it.get("outcome_cell") for it in queue["items"]])


def reset_queue() -> None:
    """Clear and re-seed the queue (demo reset button)."""
    queue = get_queue()
    queue["items"].clear()
    queue["next_id"] = 1
    _seed(queue)


# ── seeding (deterministic — no LLM calls) ──────────────────────────────────

# Each spec pends for a specific reason. Low-confidence seeds inject the score
# the intake node would have produced, so seeding needs no API key or tokens.
_SEED_SPECS = [
    (
        "High-value hip replacement",
        dual_eligible("CLM-Q-001", [("L1", "27130", 9500.00)]),
        {},
    ),
    (
        "Low intake confidence (garbled coverage type)",
        dual_eligible("CLM-Q-002", [("L1", "99213", 420.00)]),
        {
            "intake_confidence": 0.58,
            "low_confidence_fields": ["member.coverages[1].coverage_type"],
            "intake_model_path": [
                "extract:claude-haiku-4-5", "verify:claude-sonnet-5",
                "escalate:claude-opus-4-8", "verify:claude-sonnet-5",
            ],
            "intake_escalated": True,
        },
    ),
    (
        "High-value knee surgery (working aged)",
        working_aged("CLM-Q-003", [("L1", "27447", 14200.00)]),
        {},
    ),
    (
        "Low intake confidence (smudged amount + unclear payer)",
        two_commercials("CLM-Q-004", [("L1", "99215", 380.00)]),
        {
            "intake_confidence": 0.66,
            "low_confidence_fields": [
                "member.coverages[0].payer_name", "lines[0].billed_amount",
            ],
            "intake_model_path": [
                "extract:claude-haiku-4-5", "verify:claude-sonnet-5",
                "escalate:claude-opus-4-8", "verify:claude-sonnet-5",
            ],
            "intake_escalated": True,
        },
    ),
    (
        "High-value spinal fusion (single coverage)",
        single_coverage("CLM-Q-005", [("L1", "22633", 21500.00)]),
        {},
    ),
    (
        "Low intake confidence (ambiguous employment)",
        working_aged("CLM-Q-006", [("L1", "99214", 410.00)]),
        {
            "intake_confidence": 0.71,
            "low_confidence_fields": ["member.is_employed"],
            "intake_model_path": [
                "extract:claude-haiku-4-5", "verify:claude-sonnet-5",
                "escalate:claude-opus-4-8", "verify:claude-sonnet-5",
            ],
            "intake_escalated": True,
        },
    ),
]

# Static, display-only history of already-reviewed claims. These are NOT run
# through the graph and produce no traces/feedback — they exist so the Approvals
# page's confusion matrix (TP/FP/FN/TN + precision/recall) shows realistic
# numbers on first load. Marked `sample=True` so the UI labels them as seeded
# sample data rather than live traces. (TP=5, FP=2, FN=1, TN=4.)
_HISTORY_SPECS: list[tuple[str, str, str]] = [
    ("CLM-8841 · dual-eligible high-value hip", "approve", "TP"),
    ("CLM-8842 · low-confidence intake (garbled payer)", "return", "TP"),
    ("CLM-8843 · working-aged high-value knee", "approve", "TP"),
    ("CLM-8844 · low-confidence intake (smudged amount)", "deny", "TP"),
    ("CLM-8845 · birthday-rule high-value", "approve", "TP"),
    ("CLM-8846 · low-confidence pend — was actually fine", "approve", "FP"),
    ("CLM-8847 · high-value routine — no issue found", "approve", "FP"),
    ("CLM-8848 · auto-settled, mis-parsed coverage caught on spot-check", "spot-check", "FN"),
    ("CLM-8849 · auto-settled, spot-checked correct", "spot-check", "TN"),
    ("CLM-8850 · auto-settled, spot-checked correct", "spot-check", "TN"),
    ("CLM-8851 · auto-settled, spot-checked correct", "spot-check", "TN"),
    ("CLM-8852 · auto-settled, spot-checked correct", "spot-check", "TN"),
]


def _seed(queue: dict) -> None:
    # NB: append to `queue` directly — do NOT call enqueue_if_pended()/get_queue()
    # here. _seed runs *inside* get_queue()'s cached computation, and re-entering
    # get_queue() would deadlock on Streamlit's per-resource cache lock.
    graph = get_graph()
    for title, claim, extra in _SEED_SPECS:
        tid = new_thread()
        config = run_config(thread_id=tid, user_id=adjuster_id())
        try:
            result, run_id = invoke_with_run_id(graph, {"claim": claim, **extra}, config)
        except Exception:
            continue  # seeding is best-effort; never block the UI
        run = {"thread_id": tid, "config": config, "result": result, "run_id": run_id}
        if is_pended(result):
            _append_item(queue, run, title)

    # Static resolved history so the confusion matrix is populated on first load.
    for title, decision, cell in _HISTORY_SPECS:
        queue["items"].append(
            {
                "id": f"Q{queue['next_id']}",
                "thread_id": None,
                "config": None,
                "run_id": None,
                "title": title,
                "result": {},
                "pend_reasons": [],
                "confidence": None,
                "low_confidence_fields": [],
                "patient": 0.0,
                "status": "resolved",
                "decision": decision,
                "outcome_cell": cell,
                "feedback_written": [],
                "sample": True,
            }
        )
        queue["next_id"] += 1


# ── rendering helpers (shared by both pages) ────────────────────────────────


def fmt_enum(v) -> str:
    return getattr(v, "value", v) or "—"


def render_sidebar() -> None:
    st.sidebar.title("🏥 COB Claims Agent")
    st.sidebar.caption("Payer-to-payer coordination of benefits")

    traced = ensure_tracing()
    st.session_state.tracing = traced
    if traced:
        st.sidebar.success(f"LangSmith tracing ON → `{os.environ.get('LANGSMITH_PROJECT', 'default')}`")
        if os.environ.get("LANGSMITH_HIDE_INPUTS") == "true":
            st.sidebar.caption("PHI masking: inputs/outputs hidden in traces")
    else:
        st.sidebar.info("Tracing off (set LANGSMITH_API_KEY in .env to enable)")

    if os.environ.get("ANTHROPIC_API_KEY"):
        st.sidebar.caption("Intake models: haiku-4-5 → sonnet-5 → opus-4-8 ✓")
    else:
        st.sidebar.caption("Intake LLM: no ANTHROPIC_API_KEY — raw-text mode disabled")

    open_n = sum(1 for it in get_queue()["items"] if it["status"] == "open")
    st.sidebar.metric("Claims awaiting review", open_n)


def render_claim_summary(result: dict) -> None:
    claim = result.get("claim")
    if not claim:
        return
    st.subheader(f"Claim {claim.claim_id}")
    cols = st.columns(3)
    cols[0].markdown(f"**Provider**\n\n{claim.provider_id or '—'}")
    cols[1].markdown(f"**Date of service**\n\n{claim.date_of_service or '—'}")
    member = claim.member
    cols[2].markdown(f"**Member**\n\n{member.member_id if member else '—'}")
    if member and member.coverages:
        st.caption("Coverages on file")
        st.table(
            [
                {
                    "Payer": c.payer_name,
                    "Type": c.coverage_type.value,
                    "Coinsurance": f"{c.coverage_percent:.0%}",
                }
                for c in member.coverages
            ]
        )


def render_intake_diagnostics(result: dict) -> None:
    """Show the multi-step / multi-model intake path + confidence score."""
    model_path = result.get("intake_model_path") or []
    confidence = result.get("intake_confidence")
    if not model_path and confidence is None:
        return

    st.markdown("**Intake — multi-step, multi-model**")
    c1, c2 = st.columns([1, 2])
    if confidence is not None:
        c1.metric("Intake confidence", f"{confidence:.0%}")
        c1.progress(min(max(confidence, 0.0), 1.0))
    if model_path:
        steps = " → ".join(step.replace(":", " · ") for step in model_path)
        c2.caption("Task-based model routing")
        c2.markdown(f"`{steps}`")
        if result.get("intake_escalated"):
            c2.warning("Low confidence → **escalated** to the stronger model and re-verified.")

    low = result.get("low_confidence_fields") or []
    if low:
        st.caption("Fields the verifier was least sure about:")
        st.code(", ".join(low))


def render_waterfall(result: dict) -> None:
    claim = result.get("claim")
    billed = sum(amt for _lid, _code, amt in claim.lines) if claim else 0.0
    primary = result.get("primary_total_paid", 0.0) or 0.0
    secondary = result.get("secondary_total_paid", 0.0) or 0.0
    patient = result.get("patient_responsibility", 0.0) or 0.0

    st.markdown(f"**COB rule:** `{fmt_enum(result.get('cob_rule_applied'))}`")
    p = result.get("primary_payer")
    s = result.get("secondary_payer")
    if p:
        st.caption(
            f"Primary: {p.payer_name}"
            + (f"  •  Secondary: {s.payer_name}" if s else "  •  no secondary")
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Billed", f"${billed:,.2f}")
    c2.metric("Primary paid", f"${primary:,.2f}")
    c3.metric("Secondary paid", f"${secondary:,.2f}")
    c4.metric("Patient owes", f"${patient:,.2f}")


def render_status(result: dict) -> None:
    status = fmt_enum(result.get("workflow_status"))
    if status == "settled":
        st.success(f"Status: {status}")
    elif status == "pended_review":
        st.warning(f"Status: {status} — awaiting human review")
    elif status == "denied":
        st.error(f"Status: {status}")
    else:
        st.info(f"Status: {status}")


def render_audit(result: dict) -> None:
    with st.expander("Audit log (pipeline steps)", expanded=False):
        for entry in result.get("audit_log", []):
            st.text(entry)
