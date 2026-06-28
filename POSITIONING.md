# StigmergicAI — Positioning One-Pager

> **Category:** Operational Entropy Management *(StigOps)*
> **Tagline:** The decomposer layer for the endless rain of events your automation can't keep up with.

---

## The one-line pitch

**Every enterprise has a forest floor: an infinite rain of events — tickets, invoices, alerts, claims — that traditional automation can't keep up with. StigmergicAI is the ecosystem of decomposers that digests that chaos into resolved, audited order, continuously — and never loses a leaf, even when the systems above it crash.**

---

## The problem

Enterprises are drowning in **operational entropy**: a never-ending, bursty stream of events that each require *judgment against a set of rules*, where being wrong is expensive and being slow is worse.

Today this work is split between two bad options:

- **Humans** — accurate but expensive, slow, and impossible to scale to the burst.
- **Traditional automation & LLM orchestrators** — fast but brittle. They model a *corporation*: a central supervisor hands strings from Agent A to Agent B. When the supervisor crashes mid-flight, state is lost; when one model hallucinates or is prompt-injected, the bad action ships straight to your ERP. They were built for linear workflows, not for an infinite, unpredictable inbox.

Neither option is a *system for continuously digesting consequential events at volume.* That is the gap.

---

## The insight

Nature solved high-volume, no-orchestrator, fault-tolerant decomposition a billion years ago. StigmergicAI ports three of its mechanisms into software:

| Horizon | In nature | What it gives you |
| :--- | :--- | :--- |
| **H1 — Stigmergic Swarms** | An ant colony coordinating through pheromone trails | A **database-as-orchestrator**: agents never call each other, they read and mutate a shared "ground." Kill any process and the work waits durably for the swarm to resume. No central point of failure. |
| **H2 — Byzantine Cognitive Consensus** | An immune system: many independent cells voting | A **heterogeneous jury** verifies every consequential action *before* it commits. Prompt injections and hallucinations get **slashed**; honest work passes; every verdict is a durable, replayable audit record. |
| **H3 — Latent State Transfer** | A neural synapse passing signal, not words | Agents can pass **tensors instead of strings**, eliminating the lossy, injectable "string tax" between cognitive steps. |

The result is not a *corporation* of agents. It is an **organism** — older, and it doesn't go down when the manager quits.

---

## Why it's defensible: entropy is measured, not metaphor

Most "agent" pitches are poetry. Ours compiles:

- The framework literally computes `global_entropy()` over the field of unresolved work.
- The `SwarmInspector` charts that entropy decaying toward zero in real time, and can replay the full life of any event.
- "Driving operational entropy to zero" is therefore a **measurable, demonstrable** claim — which is exactly why **Operational Entropy Management** is a category we can own rather than borrow.

---

## Who it's for — the forest-floor qualifier

A problem belongs to StigmergicAI when it has these **seven traits**:

1. A stable, queryable **"ground"** of truth (KB, policy, ERP state, ledger).
2. A **continuous, unpredictable** arrival of events, in bursts, with no natural end.
3. Each event needs **judgment** against the ground (reasoning + retrieval), not a CRUD write.
4. Actions are **consequential** — so *verify-before-acting* has real value.
5. Work is **decomposable and parallelizable**.
6. **Eventual consistency** is acceptable.
7. **Volume/bursts break synchronous orchestration.**

> **Essence:** *work that never stops arriving, must be judged against the rules, and can be neither lost nor executed blindly.*

---

## Target verticals

The framework is **general** (it markets a *class* of problems). The go-to-market is **focused** (prove one vertical, then expand).

| Vertical | The endless rain | Load-bearing horizon |
| :--- | :--- | :--- |
| IT service desk / ITSM triage | Tickets vs. KBs & past resolutions | H1 durability + H2 verify-before-human |
| Continuous compliance & fraud audit | Invoices/expenses vs. policy | **H2** — the auditable jury *is* the control |
| Supply-chain self-healing | Thousands of SAP delay alerts at once | **H1** — SAP dies, state freezes, swarm resumes |
| SOC / alert triage | SIEM alerts vs. threat intel & runbooks | H2 — don't close a breach or escalate noise |
| Claims adjudication | Insurance/health claims vs. policy | H2 + durable audit trail |
| Trust & Safety / moderation | Queue vs. policy | H2 heterogeneous jury as appeals bench |
| Pharmacovigilance (MDR) | Adverse-event reports vs. safety base | H1 + H2, deadline-driven |

> The two load-bearing horizons each map to a real, high-value vertical: **Fraud/Compliance showcases H2**, **Supply-chain showcases H1**. Pick the beachhead where the buyer already feels the entropy.

---

## Differentiation

| | Commercial orchestrators (LangChain / CrewAI) | Guardrail libraries (Guardrails / NeMo) | **StigmergicAI** |
| :--- | :--- | :--- | :--- |
| **Mental model** | A corporation (supervisor + nodes) | A filter on one call | An **organism** (decentralized swarm) |
| **Failure mode** | Supervisor crash = lost state | Single-model, single-point check | DB-durable state; **Byzantine quorum** |
| **Injection defense** | App-level try/catch | Pattern/policy on one model | Multi-juror **consensus that slashes** + audit |
| **State** | App memory (volatile) | N/A | **Database physics** (durable) |
| **What it measures** | Traces | Pass/fail per call | **Entropy → 0**, replayable |

---

## Proof & standards

- **Demonstrated today:** self-healing coordination (Chaos-Monkey drain, zero lost tasks), live slashing of prompt injections, and latent (tensor) hand-off — all in runnable examples, on an 83-test, torch-free suite.
- **Threat model mapped to** the **OWASP Top 10 for LLM Applications** (anchored on `LLM01: Prompt Injection`, `LLM09: Overreliance`), with **NIST AI RMF** and **MITRE ATLAS** as the governance vocabulary.
- **Benchmark roadmap (planned, not yet claimed):** injection-capture rate on **AgentDojo** / **InjecAgent** and answer faithfulness on **RAGAS**, published against a Guardrails/NeMo baseline.

> Honesty is part of the positioning: we publish the *methodology* before the *numbers*.

---

## Business model: open-core

- **The framework is open source (Apache-2.0).** Recognition is the strategy — the open-source colony is the top of the funnel and the credibility engine. *The fame of the open source* **is** *the customer-acquisition channel for the product.*
- **The product is commercial.** A managed, vertical-specific deployment — the curated ground, the retrieval layer, the human-in-the-loop console, the SLAs — is where revenue lives.
- **One monorepo, not architecture astronautics.** The framework and the productized vertical share a codebase; dispersion across many repos is a cost, not a virtue, at this stage.

---

## Status & next steps

- ✅ Horizons 1–3 implemented; pluggable grounds (SQLite + PostgreSQL); `SwarmInspector` observability; heterogeneous jurors; 83 passing tests.
- ⏭️ Pick the beachhead vertical and ship an internal pilot (one event category, measured on acceptance rate, deflection, and injection-capture).
- ⏭️ Run and publish the OWASP-mapped benchmark suite (AgentDojo / InjecAgent / RAGAS) vs. a Guardrails/NeMo baseline.
- ⏭️ Add the commercial layer: vector/retrieval ground, `PENDING_HUMAN` review console, managed deployment.
