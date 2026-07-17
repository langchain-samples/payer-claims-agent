"""Streamlit demo — home page: Intake → Adjudication.

Run from the repo root::

    uv run streamlit run app/streamlit_app.py

This is a multipage app. This page handles claim intake (preset scenario or
multi-model LLM intake of raw text) and shows the adjudication result. When a
claim **pends** — because it is high-value or low intake-confidence — it is
added to the **Approvals Queue** page, where a reviewer approves / denies /
returns it and that judgment is written back to LangSmith as feedback.

Every run traces to LangSmith (PHI-masked by default), so the domain demo and
the observe → test → improve loop reinforce each other.

Security notes (local demo — do not expose publicly, no auth): pasted claim
text is UNTRUSTED and only ever handed to the injection-hardened structured
intake node; thread_ids are server-generated uuid4s. See ``demo_common``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure sibling `demo_common` imports whether launched from repo root or app/.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

import demo_common as dc
from payer_claims.scenarios import SCENARIOS as PRESETS

st.set_page_config(page_title="Payer COB Claims Agent", page_icon="🏥", layout="wide")


# ── intake screen ────────────────────────────────────────────────────────────


def render_intake() -> None:
    st.header("1 · Intake")
    mode = st.radio(
        "Claim source",
        ["Preset scenario", "Raw claim text (multi-model LLM intake)"],
        horizontal=True,
    )

    if mode == "Preset scenario":
        label = st.selectbox("Scenario", list(PRESETS.keys()))
        claim = PRESETS[label]
        with st.expander("Structured claim", expanded=False):
            st.json(
                {
                    "claim_id": claim.claim_id,
                    "member_id": claim.member.member_id if claim.member else None,
                    "coverages": [c.payer_name for c in claim.member.coverages]
                    if claim.member
                    else [],
                    "lines": [list(l) for l in claim.lines],
                }
            )
        if st.button("Adjudicate claim", type="primary"):
            _run({"claim": claim})
    else:
        has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
        if not has_key:
            st.warning(
                "Raw-text intake calls the multi-model intake pipeline "
                "(haiku → sonnet → opus) and needs `ANTHROPIC_API_KEY` in `.env`."
            )
        sample = st.radio(
            "Sample document",
            ["Clean fax", "Messy scan (triggers escalation)"],
            horizontal=True,
            help="The messy scan is garbled enough that the verify step returns "
            "low confidence, so intake escalates to the stronger model.",
        )
        default = dc.SAMPLE_RAW_TEXT_CLEAN if sample == "Clean fax" else dc.SAMPLE_RAW_TEXT_MESSY
        text = st.text_area(
            "Paste raw claim document (treated as untrusted data)",
            value=default,
            height=300,
            key=f"raw_{sample}",
        )
        if st.button("Normalize & adjudicate", type="primary", disabled=not has_key):
            with st.spinner("Multi-model intake: extract → verify → (escalate)…"):
                _run({"raw_claim_text": text})


def _run(initial_state: dict) -> None:
    try:
        run = dc.submit_claim(initial_state)
    except Exception as exc:  # surface as UI error, not a traceback
        st.session_state.active = None
        st.session_state.error = f"{type(exc).__name__}: {exc}"
        return
    st.session_state.error = None
    st.session_state.active = run
    if dc.is_pended(run["result"]):
        dc.enqueue_if_pended(run)


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    dc.render_sidebar()
    render_intake()

    if st.session_state.get("error"):
        st.error(st.session_state.error)

    run = st.session_state.get("active")
    if not run:
        return
    result = run["result"]

    st.divider()
    st.header("2 · Adjudication")
    dc.render_claim_summary(result)
    dc.render_intake_diagnostics(result)
    dc.render_waterfall(result)
    dc.render_status(result)
    dc.render_audit(result)

    if dc.is_pended(result):
        reasons = result.get("pend_reasons") or []
        labels = {"high_value": "💰 high-value", "low_confidence": "🤖 low intake confidence"}
        why = ", ".join(labels.get(r, r) for r in reasons) or "flagged"
        st.warning(f"Pended for review ({why}) → added to the **Approvals Queue**.")
        st.page_link(
            "pages/2_Approvals_Queue.py",
            label="Go to Approvals Queue →",
            icon="✅",
        )


main()
