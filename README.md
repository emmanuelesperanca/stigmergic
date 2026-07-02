# 🐜 StigmergicAI

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)

> **A zero-dependency, mathematically secure multi-agent framework based on Stigmergy, Latent State Transfer, and Byzantine Fault Tolerance.**
>
> *The decomposer layer for **operational entropy** — the endless rain of tickets, invoices, and alerts that traditional automation can't keep up with.*

Forget API chaining, central orchestrators, and fragile `Agent-A → Agent-B` string-passing. **StigmergicAI** reimagines Multi-Agent Systems (MAS) not as corporate org charts, but as **living ecosystems**.

Nature solved distributed computing 500 million years before we invented the CPU. Ant colonies coordinate thousands of workers with no manager. Your immune system runs a planet-scale security audit with no central firewall. Your neurons exchange thought without ever flattening it into words. **StigmergicAI is a shameless plagiarism of that biology** — three battle-tested survival strategies, ported into a framework for LLM agents.

If LangGraph is a rigid assembly line, StigmergicAI is an ant colony: decentralized, self-healing, and mathematically protected against hallucination.

---

## 🩻 The Disease We're Curing

Today's orchestration frameworks suffer from two chronic conditions:

- **The String Tax 💸** — agents think in rich, high-dimensional latent space, then are forced to flatten that thought into lossy text just to talk to the next agent. It is brain damage by serialization.
- **Centralized Fragility 🥀** — a supervisor node holds the whole graph in memory. The supervisor dies, the organism dies with it. No ant colony ever collapsed because the "manager ant" had a heart attack.

Biology never made these mistakes. Neither should your agents.

---

## 🌳 Forest-Floor Problems

> *Where StigmergicAI earns its keep.*

Walk into any enterprise and you'll find a **forest floor**: an endless rain of events — support tickets, invoices, fraud alerts, SAP exceptions, security alarms, insurance claims — falling faster than humans (or traditional automation) can clear them. Leaves never stop dropping. There is no "end of project," only a steady state of decay that must be continuously digested.

Nature has a name for the layer that handles exactly this: **decomposers**. StigmergicAI is the decomposer layer for your **operational entropy** — a swarm that digests the chaos into resolved, *audited* order, continuously, and never loses a leaf even when the systems above it crash.

We call this category **Operational Entropy Management**, and it is not a metaphor: the framework literally measures `global_entropy()` and drives it toward zero, while the `SwarmInspector` charts the decay in real time.

**A problem belongs on the forest floor when it has these seven traits:**

1. **A stable, queryable "ground"** — a base of truth (knowledge base, reimbursement policy, ERP state, ledger).
2. **A continuous, unpredictable arrival of events** — in bursts, with no natural end.
3. **Each event needs *judgment* against the ground** — reasoning + retrieval, not a CRUD write.
4. **Actions are consequential** — getting it wrong is expensive, so *verifying before acting* has real value.
5. **The work is decomposable and parallelizable** — many independent units.
6. **Eventual consistency is acceptable** — you need durable and resilient, not synchronous-transactional-now.
7. **Volume/bursts break synchronous orchestration** — which is exactly where the swarm beats the pipeline.

> In one line: **"work that never stops arriving, must be judged against the rules, and can be neither lost nor executed blindly."**

### Where it fits

| Vertical | The endless rain | Horizon that shines |
| :--- | :--- | :--- |
| **IT service desk** | Tickets vs. KBs & past resolutions | 🐜 H1 durability + 🛡️ H2 verify-before-human |
| **Continuous compliance & fraud** | Invoices/expenses vs. reimbursement policy | 🛡️ H2 — the auditable jury *is* the control |
| **Supply-chain self-healing** | 2,000 SAP delay alerts at once | 🐜 H1 — SAP dies, state freezes, swarm resumes |
| **SOC / alert triage** | SIEM alerts vs. threat intel & runbooks | 🛡️ H2 — don't close a real breach or escalate noise |
| **Claims adjudication** | Insurance/health claims vs. policy | 🛡️ H2 + durable audit trail |
| **Trust & Safety** | Moderation queue vs. policy | 🛡️ H2 heterogeneous jury as appeals bench |
| **Pharmacovigilance (MDR)** | Adverse-event reports vs. safety base | 🐜 H1 + 🛡️ H2, deadline-driven |

> **The framework is general; the go-to-market is one beachhead.** The architecture is domain-agnostic — but a serious deployment proves *one* vertical first. The forest is vast; you start by clearing one clearing.

---

## 🧠 The Three Architectural Horizons

Each horizon is a capability lifted straight from a living system.

### 🐜 Horizon 1 — Stigmergic Swarms · *The Ant Colony*

**The biology.** A single ant cannot comprehend the anthill's blueprint, and there is no "CEO ant" barking orders. Coordination emerges through **stigmergy**: an ant finds food, carries a piece home, and lays a chemical trail (a pheromone) on the ground. Another ant — with zero knowledge of the first — smells the trail, follows it, and reinforces it. When the food runs out, the pheromone evaporates and the traffic stops. The environment *is* the algorithm.

**The framework.** Agents here **never call each other**. They only read and mutate a shared semantic field — a **pluggable ground** that plays the role of the forest floor. It ships with SQLite (zero-config, in-process) and **PostgreSQL** — lock-free `SELECT … FOR UPDATE SKIP LOCKED` claims plus `LISTEN/NOTIFY` push — with Redis and DynamoDB on the roadmap. Swap the substrate in one line, `create_ground("postgresql://…")`, and the ants never notice.

| Biology | StigmergicAI |
| :--- | :--- |
| The forest floor | `PheromoneGround` (the shared DB) |
| Pheromone concentration | `Entropy` (`CHAOS=1.0 … ZERO=0.0`) |
| The trail's scent | `Status` (`RAW → HYGIENIZED → RESOLVED`) |
| The pheromone evaporating | An ant setting `Entropy.ZERO` |
| A colony surviving a dead ant | A killed thread; the task stays durable in the DB |

> **Zero coupling, absolute resilience.** An ant wakes, senses high entropy (unresolved work), acts, lowers the entropy, and sleeps. Kill the process and the work simply waits in the database for the swarm to resume — eventual consistency with no orchestrator. (This is the Chaos Monkey test below.)
>
> The swarm is **event-driven**: ants idle at *zero CPU* and wake the instant the ground changes — no busy-polling — and every mutation is observable through the **`SwarmInspector`**, a flight recorder that replays any pheromone's life and charts field entropy over time.

### 🛡️ Horizon 2 — Byzantine Cognitive Consensus · *The Immune System*

**The biology.** Your body does not ask the brain for permission to fight a virus. When an antigen enters the bloodstream, white blood cells detect it by molecular pattern-matching and destroy it on the spot. And crucially it is a *quorum*, not a dictator — when the immune system hallucinates and a lone cell misfires, healthy tissue gets attacked (autoimmune disease). Robustness comes from many independent sentinels having to agree.

**The framework.** You would not trust a single microservice to silently mutate production state — so why trust a single LLM? Before any mutation is committed, it must survive a **Byzantine Quorum**. The `VerifierAnt` is your white blood cell; a prompt injection (`drop table`, `ignore previous instructions`) or an LLM hallucination is the antigen.

| Biology | StigmergicAI |
| :--- | :--- |
| Antigen / pathogen | Prompt injection or hallucination |
| White blood cell | `VerifierAnt` |
| The immune response (a quorum of cells) | `SemanticRaft` (a jury of NLI judges) |
| Quarantine before the verdict | `Status.PENDING_CONSENSUS` |
| Pathogen neutralized | `Status.SLASHED` |
| All-clear, tissue healthy | `Status.RESOLVED` |

> A proposal is quarantined at `PENDING_CONSENSUS`. A jury of independent nodes each asks, in its own words, *"does this action actually follow from the original request?"* The jury can be genuinely **heterogeneous** — a fast `RuleBasedJudge`, a local NLI cross-encoder, and a provider-agnostic `LLMJudge` — so the voters fail for *uncorrelated* reasons, exactly like independent immune cells. If fewer than the threshold agree, the transaction is **slashed** and the poisoned trail evaporates so no other ant ever follows it. The vital organ (your ERP/SAP) is protected without ever consulting the user.

### 🧠 Horizon 3 — Latent State Transfer · *The Neural Synapse*

**The biology.** Neurons do not email each other sentences in English. The visual cortex does not send the amygdala the *text* "this is a lion" — it fires an electrochemical pulse, pure mathematical information, and the brain reacts in milliseconds. Forcing thought through language would mean saying every idea out loud before you could think the next one.

**The framework.** Instead of compressing reasoning into a string, agents pass the **hidden activation tensor itself**. A Reader distills a heavy document into a last-layer hidden state `[1, seq_len, hidden_dim]`, parks that tensor in the pheromone's `latent_blob`, and *erases the text*. A downstream agent injects the tensor straight into its own residual stream — pure context, zero String Tax.

| Biology | StigmergicAI |
| :--- | :--- |
| Electrochemical pulse | The serialized hidden-state tensor |
| Synaptic transmission | `latent_blob` on the `LATENT_READY` trail |
| Speaking out loud (lossy) | Classic string-based agent RPC |
| A sensory neuron encoding a stimulus | `HybridSolverAnt` (`engine="local"`) |
| A downstream neuron firing on the signal | A `Decider` agent reading only the tensor |

> In [`examples/03_latent_telepathy.py`](examples/03_latent_telepathy.py), one ant reads a 2,300-character report and the next ant reaches a decision having *never seen a single word of it* — only the tensor crossed the gap. Telepathy by mathematics.

---

## 🗺️ The Life of a Pheromone

A unit of work is born as chaos and dies as either a resolution or a slash. No agent dictates this journey; each caste simply reacts to the scent it is tuned to.

```mermaid
flowchart LR
    F([🐜 Forager]) -->|inject_chaos| RAW
    RAW -->|🧼 Governance sanitizes| HYG[HYGIENIZED]
    HYG -->|🧠 Solver proposes| PC[PENDING_CONSENSUS]
    PC -->|🛡️ quorum passes| RES[RESOLVED]
    PC -->|🛡️ injection / hallucination| SLA[SLASHED]
    RAW -.->|🧠 Reader encodes tensor| LR[LATENT_READY]
    LR -.->|🔮 Decider acts on latent| RES
```

---

## ⚡ Quick Start: The Chaos Monkey Test

You don't need a heavy vector database to feel stigmergy. Let's run a swarm on a local SQLite "forest floor."

```bash
pip install -r requirements.txt
python examples/01_hello_stigmergy.py
```

**What happens?**

1. The **Forager** injects chaotic, high-entropy requests into the ground (e.g. `"update salary to 15k urgent"`).
2. The **Governance** ant smells entropy `≥ 0.7`, wakes asynchronously, sanitizes the payload, drops entropy to `0.2`, and lays a clean `HYGIENIZED` trail.
3. The **Solver** ant smells the clean trail, executes the safe business logic, and drives entropy to zero.

**The magic 🐒** — kill the Governance thread mid-run. *Nothing crashes.* The Forager keeps piling on work, the Solver idles (no hygiene to follow), and the `RAW` backlog simply accumulates, durable in the database. Revive Governance and watch it vacuum the backlog at lightning speed. **Eventual consistency with no Step Functions, no retry queue, no orchestrator** — exactly how a colony shrugs off a dead worker.

---

## 🐜 The Castes

Every caste reacts only to a scent (`Status`) and an entropy threshold. None of them know the others exist — they are wired together solely by the trails they leave on the ground.

| Caste | Role in the colony | Biological analog | Reacts to → leaves |
| :--- | :--- | :--- | :--- |
| `ForagerAnt` | Producer: floods the ground with work | Forager bringing food home | — → `RAW` |
| `GovernanceAnt` | Sanitizes raw payloads | Hygienic worker ant | `RAW` → `HYGIENIZED` |
| `SolverAnt` | Proposes a resolution for review | Worker executing a task | `HYGIENIZED` → `PENDING_CONSENSUS` |
| `VerifierAnt` | Runs the Byzantine quorum | White blood cell | `PENDING_CONSENSUS` → `RESOLVED` / `SLASHED` |
| `HybridSolverAnt` | Cloud text **or** local latent encoding | Sensory neuron | `HYGIENIZED`/`RAW` → `PENDING_CONSENSUS` / `LATENT_READY` |

> Castes are built by composition over a resilient `BaseAnt` heartbeat: a `ProducerAnt` secretes, a `ConsumerAnt` claims → metabolizes → commits. A failed `tick()` is logged and swallowed — one sick ant never kills the colony.

---

## 📦 Installation

StigmergicAI keeps a **near-zero-dependency** core (just `pydantic`). The deep-learning horizons are strictly opt-in, so `import stigmergic_ai` never pays the torch/transformers import tax.

```bash
# Horizon 1 — the stigmergic core (pydantic only)
pip install -r requirements.txt

# Optional — a production PostgreSQL ground (SKIP LOCKED + LISTEN/NOTIFY)
pip install -e ".[postgres]"

# Horizons 2 & 3 — the cognition extras (torch + transformers)
pip install -e ".[cognition]"
```

> Requires **Python 3.10+**. Horizons 2 and 3 run fully offline in the examples and tests via deterministic mocks, so you can explore the entire architecture before downloading a single model weight.

---

## 🔬 Examples

| Demo | Horizon | What it proves |
| :--- | :--- | :--- |
| [`01_hello_stigmergy.py`](examples/01_hello_stigmergy.py) | 🐜 Swarms | Self-healing coordination — kill & revive a caste, lose zero tasks |
| [`02_byzantine_fault.py`](examples/02_byzantine_fault.py) | 🛡️ Consensus | A live jury slashing prompt injections while approving honest work |
| [`03_latent_telepathy.py`](examples/03_latent_telepathy.py) | 🧠 Latent | A decision made from a raw tensor — the source text never crossed |

```bash
python examples/02_byzantine_fault.py   # watch the immune system slash a hack
python examples/03_latent_telepathy.py  # watch one ant read another's mind
```

---

## 🧪 Testing

The whole suite is fully deterministic and **torch-free** (every juror is a `MockNLIJudge` or `RuleBasedJudge`), so it runs in a fraction of a second and proves the import hygiene holds.

```bash
pip install pytest
python -m pytest -q
```

> **83 tests**, all torch-free and sub-second, span three suites: `test_consensus.py` (quorum arithmetic — 2-of-3 passes, 1-of-3 slashes — red-flag slashing, crashing-juror resilience, heterogeneous panels), `test_ground.py` (event emission, event-driven wake, the pluggable-backend factory, SQL-injection-safe identifiers), and `test_observability.py` (lifecycle, entropy sparkline, replay, JSONL sink). Import hygiene is asserted: no `torch`, `transformers`, or `psycopg` ever leaks into a core run.

---

## 🗂️ Project Structure

```text
src/stigmergic_ai/
├── core/
│   ├── environment.py     # 🐜 PheromoneGround, AbstractGround, EventSignal — the forest floor
│   ├── backends/          # 🔌 pluggable grounds: create_ground() · PostgresGround (LISTEN/NOTIFY)
│   ├── consensus.py       # 🛡️ SemanticRaft, NLI/rule/LLM judges — the immune system
│   ├── observability.py   # 🔭 SwarmInspector — the flight recorder (lifecycle, entropy, replay)
│   └── latent_transfer.py # 🧠 tensor (de)serialization — the synapse
└── agents/
    ├── base_ant.py        # BaseAnt heartbeat · ConsumerAnt (event-driven) · ProducerAnt
    ├── concrete.py        # ForagerAnt · GovernanceAnt · SolverAnt
    ├── verifier_ant.py    # VerifierAnt (the white blood cell)
    └── hybrid_ant.py      # HybridSolverAnt (cloud text / local latent)
examples/   # runnable horizon demos
tests/      # torch-free suites: consensus · ground · observability
```

---

## 📊 Standards & Benchmarks

StigmergicAI targets a security-sensitive category, so it is designed to be *measured against recognized standards* rather than to rest on self-asserted claims.

**Threat model — mapped to the [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/).** The Byzantine quorum directly targets `LLM01: Prompt Injection` and `LLM09: Overreliance` (hallucination): no single model commits state, and every verdict carries a durable, replayable audit record (`ConsensusVerdict`).

**Benchmark methodology.** Two layers: a reproducible **internal harness** that measures the quorum directly, plus the public suites the field already trusts for external validation.

| Dimension | Yardstick | Status |
| :--- | :--- | :--- |
| Prompt-injection capture (internal) | OWASP `LLM01` · [`benchmarks/injection_capture`](benchmarks/injection_capture/RESULTS.md) | ✅ measured (torch-free) |
| Byzantine robustness (internal) | consensus under a faulty juror · [`BYZANTINE.md`](benchmarks/injection_capture/BYZANTINE.md) | ✅ measured (torch-free) |
| Prompt-injection capture (external) | [AgentDojo](https://github.com/ethz-spylab/agentdojo), [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) | 🚧 planned |
| Answer faithfulness (RAG) | [RAGAS](https://github.com/explodinggradients/ragas) | 🚧 planned |
| Throughput & self-healing | event-driven vs. polling · Chaos-Monkey drain | ✅ demonstrated in examples |

**Internal result (reproducible, torch-free).** On a hardened, adversarial 116-case corpus, the relational quorum lifts prompt-injection **capture from 45% (a keyword denylist) to 82% — at the *same* false-positive rate** — catching the keywordless paraphrased and indirect attacks the denylist waves through. Reproduce it with `python benchmarks/injection_capture/run.py`. The residual false-positive gap is adversarial-benign text — legitimate runbooks that merely *mention* `drop table` — which by design make up half the benign set; full breakdown in [`RESULTS.md`](benchmarks/injection_capture/RESULTS.md).

**A real model is not enough — consensus is (measured with `--nli`).** Swapping in a real cross-encoder NLI model does **not** simply close that gap: on its own the raw model reaches 100% capture but **over-blocks at a 90% false-positive rate** — a vivid, honest reminder that a single "real" model is not production-trustworthy. The heterogeneous panel outvotes its over-eagerness back to **82% capture at a 50% false-positive rate**. That is the whole thesis in one experiment: an uncorrelated quorum tames a single over-confident model. The cross-mode breakdown (torch-free vs. real-NLI vs. Byzantine) is in [`COMPARISON.md`](benchmarks/injection_capture/COMPARISON.md); to plug in a real LLM juror, follow [`STEP3_LLM.md`](benchmarks/injection_capture/STEP3_LLM.md).

**Why a *quorum* and not one judge (reproducible, torch-free).** The aggregate above hides what consensus is *for*, because there every juror is honest. The [Byzantine experiment](benchmarks/injection_capture/BYZANTINE.md) plants one faulty juror: a single compromised judge that approves everything **collapses to 0% capture**, yet the *same* traitor inside a 3-node quorum still **holds at 97.6%** (two honest votes reach the 2-of-3 threshold). Symmetrically, a broken juror that rejects everything drives a lone judge to a **100% false-positive rate**, while the quorum is outvoted back down to **33.3%**. A lone model is a single point of failure in both directions; an uncorrelated quorum is not. Reproduce with `python benchmarks/injection_capture/run.py --byzantine`; every mode is archived side by side in [`COMPARISON.md`](benchmarks/injection_capture/COMPARISON.md).

> Honesty first: the committed headline is the **torch-free, fully reproducible** configuration. The external suites (AgentDojo, InjecAgent, RAGAS) and the Guardrails/NeMo baselines are a published roadmap — they plug into the same harness as additional scored `Defense` configs. Throughput and self-healing are demonstrated by the runnable examples today.

---

## 🏗️ Why not LangChain / CrewAI?

| Trait | Commercial Orchestrators | StigmergicAI | In nature |
| :--- | :--- | :--- | :--- |
| **Topology** | Centralized supervisor/nodes | Decentralized stigmergy | An ant colony |
| **Communication** | String-based RPC (lossy) | Tensor / latent injection | A neural synapse |
| **Fault tolerance** | Try/catch retries | Semantic Byzantine quorum | An immune system |
| **State persistence** | App memory (volatile) | Database physics (durable) | Pheromones in soil |

Commercial frameworks model a **corporation**. StigmergicAI models an **organism**. Organisms are older, and they don't go down when the manager quits.

## 🤝 Contributing
This is an experimental research project pushing the boundaries of Distributed Cognitive Systems. We welcome PRs for new Byzantine voting mechanisms, new `AbstractGround` backends (Redis, DynamoDB), Swarm Inspector visualizations, and Latent Transfer utilities. New organs for the organism are always welcome.

## 📄 License
Apache License 2.0. See [`LICENSE`](LICENSE) for more information.
