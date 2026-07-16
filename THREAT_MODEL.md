# Threat Model

> What Stigmergic defends against, *where* in the code each control lives, and
> — kept deliberately honest — what it does **not** cover. Framed against the
> [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/).
> Claims marked *measured* are reproduced by
> [benchmarks/injection_capture/](benchmarks/injection_capture/); the rest are
> design controls asserted by the test suite.

---

## 1. Assets worth protecting

| Asset | Why it matters |
| :--- | :--- |
| **The knowledge store (the "soil")** | A self-improving RAG base; a poisoned entry compounds into every future answer. |
| **Systems of record (ERP/ITSM/IAM)** | Irreversible external effects: resolving tickets, writing rows, granting access. |
| **PII in the event stream** | Emails, SSNs, cards, phone numbers arriving inside raw tickets. |
| **The audit trail** | The `ConsensusVerdict` + `SwarmInspector` record that makes every decision explainable. |

## 2. Adversaries & entry points

- **The event itself** — a ticket/alert whose text carries a prompt injection or
  hostile instruction (the primary, untrusted ingress at `RAW`).
- **A poisoned document** — a seed doc or a prior "resolved" answer engineered to
  mislead future retrieval.
- **A compromised or broken juror** — a consensus node that has been bribed
  (approves everything) or has failed (rejects everything).
- **The environment** — crashed workers, duplicated deliveries, concurrent
  mutation, runaway reprocessing (the distributed-systems adversary).

---

## 3. OWASP LLM Top-10 mapping

| OWASP risk | Vector here | Control (where) | Status |
| :--- | :--- | :--- | :--- |
| **LLM01 — Prompt Injection** | Hostile instructions in a ticket ("ignore previous… drop table…") reach a solver's proposal | Heterogeneous **`SemanticRaft` quorum** slashes a proposal that does not follow from the premise, **before** any writeback ([consensus.py](src/stigmergic/core/consensus.py), [verifier_ant.py](src/stigmergic/agents/verifier_ant.py)) | ✅ *measured* — 45%→82% capture at equal FPR |
| **LLM02 — Sensitive Info Disclosure** | Email/SSN/card/phone inside raw event text leaking into store, logs, traces | **PII redaction before the first durable write** via the `redactor` hook on `inject_chaos` ([environment.py](src/stigmergic/core/environment.py), `redact_pii` in [concrete.py](src/stigmergic/agents/concrete.py)) | ✅ design control + tests |
| **LLM04 — Data & Model Poisoning** | A malicious ticket auto-resolved would teach the KB; a wrong answer served forever | **Consensus-gated writeback** (nothing reaches the KB unslashed) + **quarantine-not-delete** with provenance, audit log, and rollback ([knowledge_ground.py](demos/servicenow_hr/knowledge_ground.py)) | ✅ design control + tests |
| **LLM05 — Improper Output Handling** | A generative string treated as a trusted command/effect | **Propose / judge / execute are separate castes**; a proposal is inert data until a policy passes | ✅ invariant |
| **LLM06 — Excessive Agency** | An autonomous agent taking a consequential action alone | **`PENDING_HUMAN` gate** for high-risk work + capability separation (a proposer cannot execute) | ✅ design control |
| **LLM08 — Vector/Embedding Weakness** | Stale, contradictory, or quarantined knowledge still being retrieved | **Active-only retrieval** (quarantined/expired rows excluded) + per-entry provenance/confidence/expiry ([knowledge_ground.py](demos/servicenow_hr/knowledge_ground.py)) | ✅ design control + tests |
| **LLM09 — Misinformation (overreliance)** | Trusting one model's confident-but-wrong answer | **Uncorrelated quorum** (rule + NLI + LLM jurors) + human sign-off; benchmark shows a lone "real" model over-blocks and needs consensus | ✅ *measured* (`--nli`) |
| **LLM10 — Unbounded Consumption** | A poison-pill looping forever; runaway reprocessing | **Bounded retries → `DEAD_LETTER`**, **work-leases**, and idempotent injection cap the blast radius ([environment.py](src/stigmergic/core/environment.py), [base_ant.py](src/stigmergic/agents/base_ant.py)) | ✅ design control + tests |

---

## 4. Distributed-systems threats (the reliability core)

Beyond LLM-specific risks, an "agents-as-a-distributed-system" design must
survive the classic failure modes. Each is a tested control:

| Threat | Failure without a control | Control |
| :--- | :--- | :--- |
| **Worker crash mid-task** | A `CLAIMED` unit is stranded forever | Work-lease + `reclaim_expired_leases()` (a `JanitorAnt` sweeps) returns it to the pool |
| **Duplicate delivery** | The same event creates two units of work | `idempotency_key` dedup on `inject_chaos` |
| **Concurrent / stale write** | A reclaimed worker's late write clobbers newer state | Optimistic concurrency: `version` compare-and-swap rejects the stale write |
| **Poison pill** | One malformed unit wedges a caste in a retry loop | `retry_count > max_retries` → `DEAD_LETTER` with a reason |
| **Illegal lifecycle hop** | State corruption (e.g. `RESOLVED → RAW`) | Opt-in `enforce_transitions` rejects any transition outside `STATE_TRANSITIONS` |
| **One sick worker** | An exception takes down the swarm | Resilient heartbeat: a failing `tick()` is logged and swallowed |

See [STATE_MACHINE.md](STATE_MACHINE.md#5-reliability-annotations) for how these
overlay the lifecycle.

---

## 5. Byzantine fault: a compromised juror

The reason consensus is a *quorum* and not a single judge. Measured in
[BYZANTINE.md](benchmarks/injection_capture/BYZANTINE.md):

- A **bribed** juror (approves everything) collapses a *single* judge to **0%
  capture** — but the same traitor inside a 3-node quorum still **holds at
  97.6%** (two honest votes reach the 2-of-3 threshold).
- A **broken** juror (rejects everything) drives a lone judge to a **100%
  false-positive rate**; the quorum outvotes it back down to **33.3%**.

A lone model is a single point of failure in *both* directions; an uncorrelated
quorum is not. Diversity across mechanism (rule / NLI / LLM), data, and prompt is
what keeps juror errors uncorrelated.

---

## 6. Residual risks & limitations

Stated plainly so an operator knows where the edges are:

- **Not a formal BFT proof.** The quorum is *Byzantine-inspired*: empirically
  robust to one faulty juror, but with no formal safety/liveness proof, no
  identity/authentication layer for jurors, and no analysis of a coordinated
  multi-juror compromise.
- **PII scrubber ≠ DLP.** `redact_pii` is a deterministic regex over
  email/SSN/card/phone. It does **not** catch names, physical addresses, free-form
  identifiers, or non-US formats. Treat it as a first-line control, not compliance.
- **Correlated juror failure.** If jurors share a base model, prompt, or the same
  poisoned context, they can fail together — defeating the quorum. Diversity must
  be maintained deliberately.
- **Consensus ≠ correctness.** The quorum checks that a proposal *follows from* the
  premise; it does not verify ground truth. Wrong-but-plausible answers can pass,
  which is why high-risk paths keep a human gate.
- **Retrieval quality bounds everything.** A RAG answer is only as good as the
  soil; provenance/expiry/quarantine reduce but do not eliminate stale knowledge.
- **The demo runs offline jurors.** The committed headline uses `MockNLIJudge`/
  `RuleBasedJudge`; real-model behavior depends on the provider (see
  [COMPARISON.md](benchmarks/injection_capture/COMPARISON.md)).

---

## 7. Operator responsibilities

Controls the framework enables but the deployment must own:

- **Least privilege** — give each caste only the credentials/tools it needs; a
  proposer should not hold write access to the system of record.
- **Juror diversity** — pick jurors that fail for uncorrelated reasons; monitor
  inter-juror disagreement, not just aggregate accuracy.
- **Human policy** — decide per workflow which risk tier requires `PENDING_HUMAN`.
- **Secrets & retention** — keep API keys out of code; set field-level retention
  and access on the store; never commit real data.
- **Continuous red-teaming** — direct + indirect injections, poisoned documents,
  and source conflicts, run through [benchmarks/injection_capture/](benchmarks/injection_capture/).
