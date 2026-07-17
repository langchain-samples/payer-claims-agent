"""Generate full end-to-end LangGraph traces with VISIBLE synthetic data.

The default demo traces are PHI-masked, so their spans show "No inputs / No
outputs". This script runs the *same* compiled graph with masking OFF, so every
span's inputs and outputs are populated — reproducing the full nested trace
structure with legible (synthetic) data:

    LangGraph
    └─ intake_claim
       └─ intake_claim                 (the multi-step pipeline)
          ├─ intake_extract  · claude-haiku-4-5      (fast first pass)
          ├─ intake_verify   · claude-sonnet-5       (confidence score)
          ├─ intake_extract  · claude-opus-4-8       (escalation re-extract)
          └─ intake_verify   · claude-sonnet-5       (re-verify)
    ├─ validate_claim → route_after_validation
    ├─ determine_cob
    ├─ adjudicate_primary → (adjudicate_secondary)
    └─ finalize_settlement → human_review / END

Usage::

    # needs LANGSMITH_API_KEY + ANTHROPIC_API_KEY
    uv run python -m evals.generate_full_trace

Every claim below is SYNTHETIC (fabricated member ids/DOB). Masking is disabled
on purpose so the trace is legible for the demo — never run the unmasked path on
real PHI. In the app, masking is ON by default (configure_tracing).
"""

from __future__ import annotations

import os
import uuid

try:
    from dotenv import load_dotenv

    load_dotenv(override=bool(os.environ.get("PAYER_CLAIMS_ENV_OVERRIDE")))
except ModuleNotFoundError:
    pass

# Make sure no env-level masking is in effect for this process.
os.environ.pop("LANGSMITH_HIDE_INPUTS", None)
os.environ.pop("LANGSMITH_HIDE_OUTPUTS", None)

from payer_claims import build_claims_graph, configure_tracing

# ── synthetic claim documents (fabricated — NOT real PHI) ────────────────────

# Ambiguous, OCR-style fax: verify confidence lands low, so intake ESCALATES to
# the stronger model (haiku → opus). Also high-value, so it pends for review —
# the full four-step intake structure plus the human_review interrupt.
MESSY_SYNTHETIC = """\
====  FAXED COB WORKSHEET  (scan quality: fair — intake clerk margin notes)  ====
Claim: CLM-TRACE-8811    Provider: PR-3050    Date of service: 2025-03-12
Patient MB-TRACE-8811, DOB 1952-07-19.  Employment: hand-written, hard to read ("retired?").

Coverage on file:
  1) "EVERGREEN" plan — id EVG055 — pays 80%
       clerk note: "can't tell if this is Medicare Advantage or straight Medicare — please verify"
  2) Kentucky Medicaid — id MCD022 — pays 80% (payer of last resort)
       clerk note: "verify this is still active as of the DOS"

Services rendered:
  L1  27130  total hip arthroplasty ......... $9,800.00
  L2  99213  post-op office visit ...........   $450.00
"""

# Clean, well-formed fax: high confidence (no escalation), low value → settles.
# Shows the full happy path validate → cob → adjudicate primary+secondary →
# finalize (settled).
CLEAN_SYNTHETIC = """\
COORDINATION OF BENEFITS INTAKE — FAX COVER
Claim: CLM-TRACE-0424   Provider: PR-2044   Date of service: 2025-02-03
Patient MB-TRACE-0424, DOB 1957-09-30, retired (not employed).

Coverage on file:
  1) Evergreen Medicare Advantage (EVG001) — pays 80%
  2) Kentucky Medicaid (MCD001) — pays 80%, payer of last resort

Services rendered:
  L1  99213  office visit ............ $450.00
  L2  85025  CBC panel ............... $125.00
"""

_RUNS = [
    ("full-trace · messy escalation (pends)", MESSY_SYNTHETIC),
    ("full-trace · clean dual-eligible (settles)", CLEAN_SYNTHETIC),
]


def main() -> None:
    # mask_phi=False → do NOT set LANGSMITH_HIDE_INPUTS/OUTPUTS, so span
    # inputs/outputs are recorded and visible in LangSmith.
    traced = configure_tracing(project="payer-claims-agent", mask_phi=False)
    print(f"tracing enabled: {traced} (masking OFF — inputs/outputs visible)")
    if not traced:
        print("  ERROR: LANGSMITH_API_KEY required. Aborting.")
        return
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("  ERROR: ANTHROPIC_API_KEY required (intake calls the LLM). Aborting.")
        return

    graph = build_claims_graph(with_intake=True)
    for name, raw_text in _RUNS:
        config = {
            "configurable": {"thread_id": f"full-{uuid.uuid4()}"},
            "run_name": name,
            "tags": ["full-trace"],
            "metadata": {"data": "synthetic sample — not real PHI", "masking": "off"},
        }
        result = graph.invoke({"raw_claim_text": raw_text}, config)
        status = getattr(result.get("workflow_status"), "value", "?")
        conf = result.get("intake_confidence")
        print(
            f"  ✓ {name}\n"
            f"      confidence={conf} escalated={result.get('intake_escalated')} "
            f"status={status} pended={'__interrupt__' in result}"
        )

    print(
        "\nDone. In LangSmith, open project 'payer-claims-agent' and filter by tag "
        "`full-trace` — the spans now show the synthetic inputs/outputs."
    )


if __name__ == "__main__":
    main()
