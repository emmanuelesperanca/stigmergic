# Architecture

> The engineering contract behind the biological metaphor. The README sells the
> *why*; this document specifies the *what* and the *how* — the abstractions,
> the invariants, and the trust boundaries — in operational language.

---

## 1. Thesis

StigmergicAI is a runtime for **agentic workflows coordinated through durable
shared state** rather than through direct calls between agents. There is no
supervisor holding the graph in memory; the state store *is* the contract.

Concretely, every worker (an "ant"):

1. **senses** eligible work on a shared ground (by status + entropy),
2. **atomically claims** exactly one unit,
3. **metabolizes** it (the cognitive step),
4. **commits** a verifiable mutation back to the ground, and
5. **releases** ownership.

Coordination is therefore **mediated by persistent artifacts**, not by an
absence of communication: agents *do* communicate — indirectly, by reading and
writing the ground — but they never invoke one another and never share memory.
This is what buys crash-recovery, horizontal scale, and a complete audit trail.

The single load-bearing rule:

> **No generative component produces an irreversible external effect directly.**
> A model may *propose*; only an authorized executor, after a policy is
> satisfied (consensus and/or a human), may write to a knowledge base, resolve a
> ticket, or touch a system of record.

---

## 2. Component map

| Component | File | Role |
| :--- | :--- | :--- |
| `AbstractGround` | [src/stigmergic_ai/core/environment.py](src/stigmergic_ai/core/environment.py) | The storage-agnostic contract every ground implements |
| `PheromoneGround` | [environment.py](src/stigmergic_ai/core/environment.py) | SQLite ground (default, in-process, WAL) |
| `PostgresGround` | [src/stigmergic_ai/core/backends/postgres.py](src/stigmergic_ai/core/backends/postgres.py) | Production ground (`SKIP LOCKED` claims, `LISTEN/NOTIFY`) |
| `create_ground(dsn)` | [src/stigmergic_ai/core/backends/__init__.py](src/stigmergic_ai/core/backends/__init__.py) | Factory that routes a DSN to a backend |
| `Pheromone` | [environment.py](src/stigmergic_ai/core/environment.py) | One unit of work (one row): payload, status, entropy, reliability fields |
| `EventSignal` / `GroundEvent` | [environment.py](src/stigmergic_ai/core/environment.py) | The event bus every mutation emits on |
| `BaseAnt` / `ConsumerAnt` / `ProducerAnt` | [src/stigmergic_ai/agents/base_ant.py](src/stigmergic_ai/agents/base_ant.py) | The worker heartbeat + claim→metabolize→commit loop |
| `SemanticRaft` + judges | [src/stigmergic_ai/core/consensus.py](src/stigmergic_ai/core/consensus.py) | The heterogeneous consensus quorum |
| `VerifierAnt` | [src/stigmergic_ai/agents/verifier_ant.py](src/stigmergic_ai/agents/verifier_ant.py) | The caste that runs the quorum before a writeback |
| `SwarmInspector` | [src/stigmergic_ai/core/observability.py](src/stigmergic_ai/core/observability.py) | The flight recorder (lifecycle, entropy, replay) |
| `latent_transfer` | [src/stigmergic_ai/core/latent_transfer.py](src/stigmergic_ai/core/latent_transfer.py) | Tensor (de)serialization for Horizon 3 |

The ServiceNow HR demo ([demos/servicenow_hr/](demos/servicenow_hr/)) is the
end-to-end reference application; the injection benchmark
([benchmarks/injection_capture/](benchmarks/injection_capture/)) is the
measurement harness.

---

## 3. Ports & adapters

The core depends on *interfaces*, not on any specific database, model, or SaaS.
Swapping a substrate never touches an ant.

| Port | Contract | Adapters today |
| :--- | :--- | :--- |
| **State store** | `AbstractGround` | `PheromoneGround` (SQLite), `PostgresGround`; Redis/DynamoDB on the roadmap |
| **Event bus** | `EventSignal` (in-proc) + backend push | SQLite in-process; Postgres `LISTEN/NOTIFY` cross-process |
| **Consensus juror** | `NLIJudge` protocol | `MockNLIJudge`, `RuleBasedJudge`, `TransformersNLIJudge`, `LLMJudge` |
| **Knowledge store** | `KnowledgeGround` API | SQLite + pure-Python cosine (demo) |
| **Embedder** | `Embedder` protocol | `StubEmbedder` (offline), `OpenAIEmbedder` |
| **System of record** | `ServiceNowClient` protocol | `MockServiceNowClient` (demo); a real REST client drops in later |
| **Human decision** | `ExpertOracle` callable | `ScriptedExpertOracle`, `approve_all_oracle` |

The core stays **torch-free and driver-free at import**: `import stigmergic_ai`
never pulls torch, transformers, or psycopg. Heavy paths are lazy and gated
behind extras (`[cognition]`, `[postgres]`).

---

## 4. The unit of work

A `Pheromone` is one row on the ground. Beyond the payload it carries the fields
that make the runtime reliable:

| Field | Purpose |
| :--- | :--- |
| `status` | The lifecycle trail a caste reacts to (see [STATE_MACHINE.md](STATE_MACHINE.md)) |
| `entropy` | Urgency in `[0, 1]`; the swarm drives the global sum to zero |
| `owner` | The ant currently holding the claim (or `None`) |
| `version` | Monotonic counter for optimistic concurrency (compare-and-swap) |
| `lease_expires_at` | When the current claim's work-lease lapses |
| `retry_count` | Claims since the last successful progress commit; trips the DLQ |
| `idempotency_key` | Dedup key so a re-delivered event never creates duplicate work |
| `dlq_reason` | Why a poison-pill was dead-lettered |
| `metadata` | JSON side-channel (original text, proposal, votes, provenance) |
| `latent_blob` | Optional hidden-state tensor (Horizon 3) |

---

## 5. Execution semantics (the reliability core)

The ground is a work queue with production guarantees, all exercised by the test
suite:

- **Atomic claim.** One ant wins a unit; SQLite uses `BEGIN IMMEDIATE`, Postgres
  uses `SELECT … FOR UPDATE SKIP LOCKED`. Terminal units are never re-claimed.
- **Work-lease + reclaim.** A claim stamps `lease_expires_at`. If the owner
  crashes, `reclaim_expired_leases()` returns the unit to its pre-claim trail so
  a healthy peer retries it (a `JanitorAnt` sweeps on a cadence).
- **Optimistic concurrency (CAS).** Every mutation bumps `version`; a commit may
  pass `expected_version`. A stale write — e.g. from a worker whose lease was
  reclaimed — matches no row and is rejected, so it can never clobber newer state.
- **At-least-once + idempotency.** Delivery is at-least-once; `idempotency_key`
  makes re-injection a no-op, so effects converge exactly-once at the ground.
- **Dead-letter queue.** A repeatedly failing unit is parked in `DEAD_LETTER`
  after a bounded number of retries (`max_retries`) with a `dlq_reason`, instead
  of wedging the swarm.
- **Formal state machine (opt-in).** With `enforce_transitions=True` the ground
  rejects any status change not in `STATE_TRANSITIONS`.
- **Resilient heartbeat.** A failing `tick()` is logged and swallowed — one sick
  ant never kills the colony.

---

## 6. Trust boundaries

Data crosses three boundaries on its way from an untrusted event to a durable,
irreversible effect. Each boundary has a dedicated control.

```
  ┌────────────┐   redact PII    ┌──────────────┐   consensus gate   ┌──────────────┐   human gate    ┌────────────┐
  │ UNTRUSTED  │  (at intake,    │  SANITIZED   │  (heterogeneous     │  VERIFIED    │  (PENDING_HUMAN │  COMMITTED  │
  │  event in  │──before write)─▶│  in the      │──quorum, slash on──▶│  proposal    │──for high-risk)─▶│  effect     │
  │ (RAW)      │                 │  ground      │   injection)        │              │                 │ (writeback) │
  └────────────┘                 └──────────────┘                     └──────────────┘                 └────────────┘
        B1                              B2                                    B3                              B4
```

- **B1 — Ingress / PII.** PII is redacted *before* the durable write (the
  `redactor` hook on `inject_chaos`), so raw emails/SSNs/cards/phones never reach
  the store, logs, or the event trail.
- **B2 — Hygiene.** The `GovernanceAnt` sanitizes the payload and preserves the
  scrubbed original as the premise the jury will judge.
- **B3 — Consensus.** No proposal advances without surviving the `SemanticRaft`
  quorum; a prompt injection or hallucination is `SLASHED` before any writeback.
- **B4 — Effect.** Only an authorized executor commits — and high-risk work is
  parked at `PENDING_HUMAN` for a human first. The knowledge store is
  **quarantine-on-reject** (never a silent delete) with provenance and rollback.

See [THREAT_MODEL.md](THREAT_MODEL.md) for the adversaries these boundaries stop.

---

## 7. Invariants

The properties the runtime is designed to hold (each is asserted by a test):

1. **No direct calls.** An ant never imports or invokes another ant; the only
   coupling is the shared ground.
2. **No effect without a policy.** A generative proposal cannot become an
   external effect until consensus (and, when configured, a human) approves.
3. **Terminal is terminal.** `RESOLVED`, `SLASHED`, and `DEAD_LETTER` are never
   re-claimed; no transition leaves them.
4. **Exactly one owner.** Two ants can never both hold the same unit.
5. **No lost work.** A crash leaves the unit durable; a lapsed lease returns it
   to the pool. Nothing is silently dropped.
6. **No duplicate work.** A re-delivered event with a known `idempotency_key`
   is deduplicated.
7. **No stale overwrite.** A reclaimed worker's late write loses the CAS.
8. **PII never persists raw.** Redaction happens before the first durable write.
9. **The KB is never poisoned.** Every writeback is consensus-gated; a wrong
   entry is quarantined (auditable, reversible), not silently deleted.
10. **Import hygiene.** The core imports no deep-learning or DB driver.

---

## 8. What this is *not*

Kept honest on purpose (see also [THREAT_MODEL.md](THREAT_MODEL.md#limitations)):

- **Not a formally-proven BFT protocol.** The quorum is *Byzantine-inspired*: it
  empirically tolerates a faulty juror (measured), but there is no formal
  safety/liveness proof, threat-model-complete quorum analysis, or identity layer.
- **Not a DLP suite.** The PII scrubber is a deterministic regex pass over common
  categories (email/SSN/card/phone) — a surface control, not names/addresses/DLP.
- **Not a replacement for an orchestrator.** It complements LangGraph/Temporal/
  ServiceNow as the *governance, consensus, and recovery layer*, not the reasoner
  or the workflow engine.
