"""COB claims graph — dual-coverage workflow.

Topology::

    START
      │
      ▼
   validate_claim ── invalid ──▶ notify_denial ──▶ END
      │
      ▼ (valid)
   determine_cob
      │
      ▼
   adjudicate_primary
      │
      ├── single coverage ──▶ finalize_settlement
      │                            │
      │               ┌────────────┼────────────┐
      │               ▼            ▼            ▼
      │           SETTLED     PENDED_REVIEW   DENIED
      │               │            │            │
      │              END      human_review  notify_denial
      │                            │            │
      │               ┌────────────┼────────────┘
      │               ▼            ▼
      │           approve        deny → notify_denial
      │               │
      │              END
      │
      └── dual coverage ──▶ adjudicate_secondary ──▶ finalize_settlement
                                (same branches as above)
"""

from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph

from .intake import intake_claim
from .nodes import (
    adjudicate_primary,
    adjudicate_secondary,
    determine_cob,
    finalize_settlement,
    human_review,
    notify_denial,
    redact_phi_boundary,
    route_after_primary,
    route_after_settlement,
    route_after_validation,
    validate_claim,
)
from .state import (
    Claim,
    ClaimsState,
    LineAdjudication,
    PayerCoverage,
    PayerEOB,
)

# Register domain types for safe checkpoint serialization. Both the JSON and
# the msgpack encode paths need the allowlist: enums/dataclasses round-trip
# through msgpack on checkpoint reads (e.g. HITL interrupt → resume), and the
# msgpack path is allowlisted separately from the JSON path.
_ALLOWED_DOMAIN_TYPES = [
    ("payer_claims.state", "Claim"),
    ("payer_claims.state", "COBRule"),
    ("payer_claims.state", "CoverageType"),
    ("payer_claims.state", "LineAdjudication"),
    ("payer_claims.state", "LineStatus"),
    ("payer_claims.state", "MemberInfo"),
    ("payer_claims.state", "PayerCoverage"),
    ("payer_claims.state", "PayerEOB"),
    ("payer_claims.state", "WorkflowStatus"),
]
_SERIALIZER = JsonPlusSerializer(
    allowed_json_modules=_ALLOWED_DOMAIN_TYPES,
    allowed_msgpack_modules=_ALLOWED_DOMAIN_TYPES,
)


def build_claims_graph(
    checkpointer: InMemorySaver | None = None,
    *,
    with_intake: bool = False,
):
    """Build and compile the COB claims graph.

    Args:
        checkpointer: checkpoint saver (defaults to an in-memory saver with the
            domain-type-aware serializer).
        with_intake: when True, prepend an LLM intake node that normalizes
            ``raw_claim_text`` into a structured ``claim`` before validation.
            When False (default), the graph starts at ``validate_claim`` and
            expects a pre-structured ``claim`` — preserving existing behavior.
    """
    builder = StateGraph(ClaimsState)

    # ── nodes ───────────────────────────────────────────────────────────────
    if with_intake:
        builder.add_node("intake_claim", intake_claim)
    builder.add_node("redact_phi_boundary", redact_phi_boundary)
    builder.add_node("validate_claim", validate_claim)
    builder.add_node("determine_cob", determine_cob)
    builder.add_node("adjudicate_primary", adjudicate_primary)
    builder.add_node("adjudicate_secondary", adjudicate_secondary)
    builder.add_node("finalize_settlement", finalize_settlement)
    builder.add_node("human_review", human_review)
    builder.add_node("notify_denial", notify_denial)

    # ── edges ───────────────────────────────────────────────────────────────
    if with_intake:
        builder.add_edge(START, "intake_claim")
        builder.add_edge("intake_claim", "redact_phi_boundary")
    else:
        builder.add_edge(START, "redact_phi_boundary")
    builder.add_edge("redact_phi_boundary", "validate_claim")

    builder.add_conditional_edges(
        "validate_claim",
        route_after_validation,
        ["determine_cob", "notify_denial"],
    )

    builder.add_edge("determine_cob", "adjudicate_primary")

    builder.add_conditional_edges(
        "adjudicate_primary",
        route_after_primary,
        ["adjudicate_secondary", "finalize_settlement"],
    )

    builder.add_edge("adjudicate_secondary", "finalize_settlement")

    builder.add_conditional_edges(
        "finalize_settlement",
        route_after_settlement,
        ["human_review", "__end__"],
    )

    # human_review uses Command(goto=...) for routing
    builder.add_edge("notify_denial", END)

    # ── compile ─────────────────────────────────────────────────────────────
    checkpointer = checkpointer or InMemorySaver(serde=_SERIALIZER)
    return builder.compile(checkpointer=checkpointer)
