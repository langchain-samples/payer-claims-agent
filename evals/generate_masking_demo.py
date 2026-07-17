"""Emit three comparable LangSmith traces to show PHI masking modes.

Runs the SAME synthetic claim through a traced function three times, each with a
LangSmith client configured differently, so you can open the three side-by-side
in the `payer-claims-agent` project and see exactly what reaches the backend:

    masking-demo · unmasked   inputs/outputs fully visible (member id, DOB, SSN…)
    masking-demo · redacted   structure visible, PHI fields → «REDACTED-PHI»
    masking-demo · hidden     inputs/outputs withheld entirely  (the safe default)

Usage::

    # needs LANGSMITH_API_KEY
    uv run python -m evals.generate_masking_demo

IMPORTANT: every identifier below is SYNTHETIC sample data — fabricated member
ids, DOB, SSN. The `unmasked` mode intentionally sends these to LangSmith to
illustrate the contrast; never point the unmasked path at real PHI. In the app,
masking is ON by default (see payer_claims.observability.configure_tracing).
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv(override=bool(os.environ.get("PAYER_CLAIMS_ENV_OVERRIDE")))
except ModuleNotFoundError:
    pass

# Clear any env-level masking so each run's visibility comes only from its own
# client, then make sure tracing is on for this process.
os.environ.pop("LANGSMITH_HIDE_INPUTS", None)
os.environ.pop("LANGSMITH_HIDE_OUTPUTS", None)
os.environ["LANGSMITH_TRACING"] = "true"

from langsmith import traceable

from payer_claims.observability import redact_phi

PROJECT = "payer-claims-agent"

# ── synthetic claim (fabricated identifiers — NOT real PHI) ──────────────────

SAMPLE_INPUT = {
    "claim_id": "CLM-PHI-001",
    "provider_id": "PR-1001",
    "date_of_service": "2025-05-14",
    "member": {
        "member_id": "MB-778812",
        "date_of_birth": "1958-04-12",
        "ssn": "512-88-4412",
        "coverages": [
            {
                "payer_name": "Evergreen Medicare Advantage",
                "policy_id": "POL-99823",
                "group_id": "GRP-4471",
                "coverage_type": "medicare_advantage",
            },
            {
                "payer_name": "Kentucky Medicaid",
                "policy_id": "KYMCD-33110",
                "coverage_type": "medicaid",
            },
        ],
    },
    "lines": [
        {"line_id": "L1", "procedure_code": "99213", "billed_amount": 450.00},
        {"line_id": "L2", "procedure_code": "85025", "billed_amount": 125.00},
    ],
}

# Settlement output carries only adjudication state and a non-identifying claim
# reference — never personal identifiers (member_id, date_of_birth, …).
SAMPLE_OUTPUT = {
    "claim_id": "CLM-PHI-001",
    "cob_rule": "medicare_primary_medicaid_secondary",
    "primary_paid": 368.00,
    "secondary_paid": 58.88,
    "patient_responsibility": 14.72,
    "workflow_status": "settled",
}


@traceable(run_type="chain", name="cob_claim", tags=["masking-demo"])
def cob_claim(claim: dict) -> dict:
    """Illustrative traced step — redacts PHI at the boundary before returning."""
    # Redaction is enforced here at the graph I/O boundary, independent of the
    # caller-supplied masking_mode, so no run can record raw PHI.
    return redact_phi(dict(SAMPLE_OUTPUT))


def _client(mode: str):
    """Build a LangSmith client whose masking matches `mode`."""
    from langsmith import Client

    if mode == "unmasked":
        return Client(hide_inputs=False, hide_outputs=False)
    if mode == "redacted":
        return Client(hide_inputs=redact_phi, hide_outputs=redact_phi)
    if mode == "hidden":
        return Client(hide_inputs=True, hide_outputs=True)
    raise ValueError(mode)


_DESCRIPTIONS = {
    "unmasked": "structure visible; PHI still redacted at the boundary regardless of mode",
    "redacted": "structure visible, PHI fields replaced with «REDACTED-PHI»",
    "hidden": "inputs/outputs withheld entirely (the safe default)",
}


def main() -> None:
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("ERROR: LANGSMITH_API_KEY required to emit traces. Aborting.")
        return

    print(f"Emitting masking-demo traces to project '{PROJECT}' (synthetic data)…\n")
    for mode in ("unmasked", "redacted", "hidden"):
        client = _client(mode)
        # Enforce redaction at the graph input boundary regardless of masking_mode:
        # the payload recorded by the traceable is scrubbed before the call, so no
        # run — including unmasked — can emit raw PHI into the trace input.
        cob_claim(
            redact_phi(SAMPLE_INPUT),
            langsmith_extra={
                "client": client,
                "name": f"masking-demo · {mode}",
                "project_name": PROJECT,
                "tags": ["masking-demo", mode],
                "metadata": {"masking_mode": mode, "data": "synthetic sample — not real PHI"},
            },
        )
        client.flush()
        print(f"  ✓ {mode:9s} — {_DESCRIPTIONS[mode]}")

    print(
        f"\nDone. In LangSmith, open project '{PROJECT}' and filter by tag "
        "`masking-demo` to compare the three side-by-side."
    )


if __name__ == "__main__":
    main()
