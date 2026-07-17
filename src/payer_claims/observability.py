"""LangSmith observability wiring — the "observe" leg of the loop.

For LangChain/LangGraph apps tracing is automatic once the environment is
configured, so this module's job is:

  1. Turn tracing on from code in a predictable way (``configure_tracing``).
  2. Fail *safe* for PHI. Payer claims carry member identifiers, dates of
     birth, and dollar amounts — protected health information that must not
     leak into a SaaS tracing backend by accident. Masking is therefore ON by
     default and must be explicitly opted out of.
  3. Hand out a ``langsmith.Client`` pre-wired with a field-level anonymizer
     for teams that want redacted-but-still-useful traces instead of fully
     hidden inputs/outputs.

Nothing here hardcodes a secret; every credential is read from the
environment (``LANGSMITH_API_KEY``, ``ANTHROPIC_API_KEY``).
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable

# ── PHI redaction ────────────────────────────────────────────────────────────
# Coarse, defensible patterns for the identifiers that show up in claim
# payloads. This is a safety net, not a HIPAA de-identification guarantee — a
# production deployment should layer a vetted de-id pipeline on top.

_REDACTED = "«REDACTED-PHI»"

# member_id / policy_id / group_id style tokens, DOB, and free-standing member
# ids like "MB-1234". Tunable per deployment.
_PHI_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("dob", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),                 # YYYY-MM-DD
    ("member_id", re.compile(r"\bMB[-_]?\w+\b", re.IGNORECASE)),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
)

# State/JSON keys whose *values* are always PHI regardless of their content.
_PHI_KEYS = frozenset(
    {"member_id", "date_of_birth", "dob", "ssn", "policy_id", "group_id"}
)


def redact_phi(value: Any) -> Any:
    """Recursively redact PHI from an arbitrary trace payload.

    Safe to hand to ``langsmith.Client(anonymizer=...)`` or to call directly
    before logging. Returns a structurally identical object with PHI scrubbed.
    """
    if isinstance(value, dict):
        return {
            k: (_REDACTED if k in _PHI_KEYS else redact_phi(v))
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        redacted = [redact_phi(v) for v in value]
        return type(value)(redacted) if isinstance(value, tuple) else redacted
    if isinstance(value, str):
        out = value
        for _label, pattern in _PHI_PATTERNS:
            out = pattern.sub(_REDACTED, out)
        return out
    return value


def _opaque_ref(value: str) -> str:
    """Stable, non-reversible correlation key for a redacted identifier."""
    import hashlib

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"REF-{digest}"


def redact_claim(claim: Any) -> Any:
    """Return a copy of a ``Claim`` with PHI scrubbed at the pipeline boundary.

    Unlike :func:`redact_phi` (which serves the tracing anonymizer and is gated
    on the masking mode), this is applied unconditionally to the domain object so
    raw PHI never enters or leaves graph state. ``member_id`` is replaced with an
    opaque, stable reference so downstream correlation still works; the remaining
    identifiers become the ``«REDACTED-PHI»`` placeholder.
    """
    import dataclasses

    if claim is None or not dataclasses.is_dataclass(claim):
        return claim

    member = getattr(claim, "member", None)
    redacted_member = member
    if member is not None:
        coverages = [
            dataclasses.replace(
                cov,
                group_id=_REDACTED if getattr(cov, "group_id", "") else cov.group_id,
                policy_id=_REDACTED if getattr(cov, "policy_id", "") else cov.policy_id,
            )
            for cov in getattr(member, "coverages", [])
        ]
        redacted_member = dataclasses.replace(
            member,
            member_id=_opaque_ref(member.member_id) if member.member_id else member.member_id,
            date_of_birth=_REDACTED if member.date_of_birth else member.date_of_birth,
            coverages=coverages,
        )

    return dataclasses.replace(
        claim,
        date_of_service=_REDACTED if claim.date_of_service else claim.date_of_service,
        member=redacted_member,
    )


def configure_tracing(
    *,
    project: str | None = None,
    mask_phi: bool | None = None,
    enable: bool = True,
) -> bool:
    """Configure LangSmith tracing for the claims graph.

    Args:
        project: LangSmith project name. Falls back to ``LANGSMITH_PROJECT``
            then ``"payer-claims-agent"``.
        mask_phi: Deprecated and ignored. Masking is now unconditional — raw
            PHI is stripped at the pipeline boundary (see
            :func:`redact_claim`) and trace IO hiding is always on. The
            ``"unmasked"`` opt-out (including ``LANGSMITH_UNMASK_PHI``) is
            hard-disabled so it cannot be selected in production.
        enable: Set to ``False`` to explicitly leave tracing off.

    Returns:
        ``True`` if tracing was enabled (an API key was present), else ``False``.
    """
    if not enable:
        os.environ["LANGSMITH_TRACING"] = "false"
        return False

    if not os.environ.get("LANGSMITH_API_KEY"):
        # No credential → do not silently pretend to trace.
        os.environ.setdefault("LANGSMITH_TRACING", "false")
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault(
        "LANGSMITH_PROJECT", project or "payer-claims-agent"
    )
    if project:
        os.environ["LANGSMITH_PROJECT"] = project

    # Masking is unconditional: the "unmasked" opt-out is hard-disabled so raw
    # IO can never be exported regardless of `mask_phi` or LANGSMITH_UNMASK_PHI.
    os.environ["LANGSMITH_HIDE_INPUTS"] = "true"
    os.environ["LANGSMITH_HIDE_OUTPUTS"] = "true"

    return True


def make_client(
    *,
    anonymizer: Callable[[Any], Any] | None = redact_phi,
    **kwargs: Any,
):
    """Return a ``langsmith.Client`` with PHI anonymization applied by default.

    Use this when running evaluations or building a custom tracer so that trace
    inputs/outputs are *redacted* (member ids, DOB, etc.) rather than fully
    hidden. Pass ``anonymizer=None`` to disable.
    """
    from langsmith import Client

    if anonymizer is not None:
        kwargs.setdefault("hide_inputs", anonymizer)
        kwargs.setdefault("hide_outputs", anonymizer)
    return Client(**kwargs)
