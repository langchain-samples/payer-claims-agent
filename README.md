# Payer-to-Payer Insurance Claims Agent

A [LangGraph](https://langchain-ai.github.io/langgraph/) agent that automates
**Coordination of Benefits (COB)** adjudication for payer-to-payer insurance
claims — the dual-coverage workflow where a member is covered by two insurers
and the system must determine payment ordering, adjudicate against each payer,
and compute the final patient responsibility.

## Overview

Given a claim (structured, or unstructured text normalized by an LLM), the agent:

1. **Validates** the claim and member coverages.
2. **Determines COB ordering** — which payer is primary vs. secondary — using
   CMS / NAIC rules.
3. **Adjudicates** against the primary payer and, for dual coverage, submits the
   remainder to the secondary payer.
4. **Finalizes settlement** and computes patient responsibility.
5. **Routes to a human reviewer** when a claim is high-value or the LLM intake
   confidence is low, resuming on the reviewer's decision.

It runs on a `StateGraph` with an in-memory checkpointer and human-in-the-loop
interrupts, and traces to LangSmith with PHI masking on by default.

## Architecture

### Design boundary: deterministic core, LLM only at the edge

COB ordering and payment math are **100% deterministic** — they encode CMS/NAIC
statute and must be auditable and reproducible. The only LLM work is the intake
step, which normalizes messy claim documents into the strict `Claim` schema. The
LLM *parses*; it never decides payment or ordering.

### Graph topology

```
START
  │
  ▼
validate_claim ── invalid ──▶ notify_denial ──▶ END
  │
  ▼ (valid)
determine_cob                     # CMS / NAIC ordering rules
  │
  ▼
adjudicate_primary                # adjudicate against primary payer
  │
  ├── single coverage ──▶ finalize_settlement
  │                            │
  │               ┌────────────┼────────────┐
  │               ▼            ▼            ▼
  │           SETTLED     PENDED_REVIEW   (DENIED)
  │               │            │
  │              END      human_review     # interrupt() + Command(resume=...)
  │                            │
  │               ┌────────────┼────────────┐
  │               ▼            ▼            ▼
  │           approve        deny         return
  │               │            │            │
  │              END      notify_denial    END
  │
  └── dual coverage ──▶ adjudicate_secondary ──▶ finalize_settlement
```

When intake is enabled (`build_claims_graph(with_intake=True)`), an
`intake_claim` node is prepended: `START → intake_claim → validate_claim → …`.

### Multi-step, multi-model intake

Intake is a pipeline that routes each step to the model that fits the task:

| Step | Task | Model |
|---|---|---|
| `extract` | transcribe raw document → `Claim` schema | `claude-haiku-4-5` |
| `verify` | score transcription confidence (0–1) | `claude-sonnet-5` |
| `escalate` | re-extract when confidence is low | `claude-opus-4-8` |

The `verify` step returns an overall **confidence score**; when it falls below
`ESCALATION_THRESHOLD`, intake re-extracts with the stronger model and
re-verifies. Untrusted claim text is treated strictly as data (prompt-injection
hardening), and every model answers through a Pydantic schema that is validated
before being mapped to domain types.

### Confidence-gated human-in-the-loop

`finalize_settlement` pends a claim for review when it is **high-value** or
**low-confidence** (recorded in `pend_reasons`). `human_review` uses
`interrupt()` to pause and `Command(resume=..., goto=...)` to route the
reviewer's decision (approve / deny / return). The resume payload can carry a
reviewer judgment (`flag_warranted`, `intake_correct`) used for feedback.

### Observability & PHI masking

Tracing is automatic once the environment is configured. `configure_tracing()`
**masks inputs/outputs by default** so member PHI never reaches LangSmith;
`make_client(anonymizer=redact_phi)` yields redacted-but-visible traces, and
`redact_phi` scrubs member identifiers, DOB, SSN, and policy/group ids.

### Evaluation

Deterministic **code** evaluators grade the auditable core (COB rule, payment
split, settlement status, and the pend-gate routing decision); additional
evaluators grade LLM intake extraction (coverage-type + amount accuracy). An
LLM-as-judge is used only for intake-transcription fidelity. Datasets and
`client.evaluate()` runners live under `evals/`.

## COB Rules Modeled

| Scenario | Primary | Secondary | Rule |
|---|---|---|---|
| Dual-eligible (Medicare + Medicaid) | Medicare | Medicaid (last resort) | CMS dual-eligible |
| Working aged (Commercial + Medicare) | Commercial | Medicare | MSP working aged |
| Retired (Commercial + Medicare) | Medicare | Commercial | MSP retired |
| Two commercial plans | Plan A | Plan B | Birthday rule |
| Medicaid + anything | Other payer | Medicaid (last resort) | Federal Medicaid statute |
| Single coverage | The only payer | None | No COB needed |

### Payment flow

```
Billed → Primary adjudication → Patient portion → Secondary adjudication → Final patient responsibility

Example (dual-eligible, $575 claim):
  Primary (Evergreen MA, 80%): pays $368.00, patient portion = $92.00
  Secondary (Medicaid, 80%): remainder $92.00 → pays $58.88
  Patient owes: $14.72
```

## Project Layout

```
payer-claims-agent/
├── pyproject.toml
├── .env.example                    # LangSmith + Anthropic env vars
├── src/payer_claims/
│   ├── __init__.py                 # Public API
│   ├── state.py                    # State schema, dataclasses, enums
│   ├── nodes.py                    # Graph nodes + routing functions
│   ├── intake.py                   # Multi-step, multi-model LLM intake
│   ├── feedback.py                 # Reviewer-judgment feedback + run capture
│   ├── observability.py            # LangSmith tracing + PHI masking
│   ├── scenarios.py                # Shared claim fixtures
│   ├── graph.py                    # StateGraph wiring + compilation
│   └── main.py                     # Runnable COB scenarios
├── app/                            # Streamlit UI (multipage)
│   ├── streamlit_app.py            # Intake → adjudication
│   ├── demo_common.py              # Shared graph + render helpers
│   └── pages/
│       └── 2_Approvals_Queue.py    # Human-review console
├── evals/
│   ├── dataset.py                  # COB core dataset + upload
│   ├── evaluators.py               # Code evaluators + LLM-as-judge
│   ├── run_experiment.py           # COB core experiment runner
│   ├── intake_dataset.py           # Intake-extraction dataset
│   └── run_intake_experiment.py    # Intake experiment runner
└── tests/                          # pytest suite
```

## Getting Started

### Requirements

- Python 3.10+
- `langgraph >= 1.0`, `langchain >= 1.0`, `langchain-core >= 1.0`
- `langchain-anthropic >= 1.0` (LLM intake)
- `langsmith >= 0.3` (tracing + evals)
- [`uv`](https://docs.astral.sh/uv/) for dependency management

### Install

```bash
cd payer-claims-agent
uv sync --extra dev --extra evals --extra ui
```

### Configure

Credentials are read from the environment / `.env`:

```bash
cp .env.example .env    # then fill in LANGSMITH_API_KEY / ANTHROPIC_API_KEY
```

| Variable | Used for |
|---|---|
| `ANTHROPIC_API_KEY` | LLM claim intake |
| `LANGSMITH_API_KEY` | tracing + evaluation experiments |

> `.env` is loaded automatically. Standard precedence applies — a variable
> already exported in your shell **wins over `.env`**. If you hit a `401 Invalid
> token`, `unset LANGSMITH_API_KEY`, or set `PAYER_CLAIMS_ENV_OVERRIDE=1` to
> force `.env` to win.

### Run

```bash
# Run the COB scenarios end-to-end
uv run python -m payer_claims.main

# Run the test suite
uv run pytest -v

# Launch the UI (opens at http://localhost:8501)
uv run streamlit run app/streamlit_app.py

# Run the evaluation experiments
uv run python -m evals.run_experiment
uv run python -m evals.run_intake_experiment
```

### Programmatic use

```python
from payer_claims import build_claims_graph, configure_tracing

configure_tracing(project="payer-claims-agent")   # PHI-masked by default
graph = build_claims_graph(with_intake=True)
result = graph.invoke(
    {"raw_claim_text": "<faxed claim text>"},
    {"configurable": {"thread_id": "..."}},
)
```

`build_claims_graph()` with no arguments starts at `validate_claim` and expects
a pre-structured `claim` (the deterministic path).

## Key LangGraph Patterns

- `StateGraph` with a `TypedDict` state schema
- `add_conditional_edges` for branching after validation, adjudication, settlement
- `interrupt()` + `Command(resume=...)` for human-in-the-loop review
- `Command(goto=...)` for node-driven routing from review decisions
- `InMemorySaver` with `JsonPlusSerializer` (explicit domain-type allowlist)
- `Annotated[list, operator.add]` for accumulating the audit log across nodes
- Task-based model routing with confidence-driven escalation in the intake step
```
