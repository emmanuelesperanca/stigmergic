# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [0.1.0] — Initial release

The first public cut of Stigmergic: a near-zero-dependency, fault-tolerant
multi-agent framework where agents coordinate through durable shared state.

### Added

- **Horizon 1 — Stigmergic Swarms.** `PheromoneGround` (SQLite, WAL) behind an
  `AbstractGround` contract, atomic `claim`, entropy-driven scheduling, a
  resilient `BaseAnt` heartbeat, and `ConsumerAnt`/`ProducerAnt` castes.
- **Horizon 2 — Byzantine-inspired consensus.** `SemanticRaft` with pluggable,
  heterogeneous jurors (`MockNLIJudge`, `RuleBasedJudge`, `TransformersNLIJudge`,
  `LLMJudge`) and the `VerifierAnt` that gates every writeback.
- **Horizon 3 — Latent State Transfer.** Tensor (de)serialization and
  `HybridSolverAnt` for passing hidden states instead of strings.
- **Pluggable backends.** `create_ground(dsn)` factory; a production
  `PostgresGround` (`SELECT … FOR UPDATE SKIP LOCKED` + `LISTEN/NOTIFY`).
- **Observability.** `SwarmInspector` flight recorder (lifecycle, entropy
  sparkline, JSONL sink, replay).
- **Reliability core.** Work-leases + `reclaim_expired_leases`, optimistic
  concurrency (`version` compare-and-swap), idempotency keys, a `DEAD_LETTER`
  queue with bounded retries, and an opt-in formal state machine
  (`enforce_transitions` + `STATE_TRANSITIONS`).
- **Security.** PII redaction *before* the durable write (a `redactor` hook on
  `inject_chaos`); the `GovernanceAnt` hygiene caste.
- **Injection-capture benchmark.** A 116-case adversarial corpus, a uniform
  `Defense` harness, Wilson confidence intervals, per-category metrics, a
  precision-recall sweep, the Byzantine (faulty-juror) experiment, and runnable
  Guardrails AI / NeMo Guardrails baseline adapters (`--baselines`).
- **ServiceNow HR demo.** An end-to-end reference app: learning, self-healing
  (quarantine + rollback), PII scrubbing, and injection defense, with a
  crash-recovery `JanitorAnt`.
- **Documentation.** `ARCHITECTURE.md`, `STATE_MACHINE.md`, `THREAT_MODEL.md`
  (OWASP LLM Top-10 mapping), and `POSITIONING.md`.

### Notes

- The core stays torch-free and driver-free at import; deep-learning and
  Postgres paths are opt-in extras (`[cognition]`, `[postgres]`, `[benchmark]`,
  `[baselines]`).
- The committed benchmark headline is the **torch-free, fully reproducible**
  configuration; real-model/real-tool numbers depend on the provider and live in
  `benchmarks/injection_capture/COMPARISON.md`.

[Unreleased]: https://github.com/emmanuelesperanca/stigmergic/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/emmanuelesperanca/stigmergic/releases/tag/v0.1.0
