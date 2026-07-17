# Stigmergic on a real RDS: PostgreSQL + pgvector HR benefits

This demo "installs" the framework onto a real database. Same swarm as
[`../servicenow_hr`](../servicenow_hr), but every substrate is production
infrastructure:

| Piece | Demo (SQLite) | Here (your RDS) |
| --- | --- | --- |
| **Pheromone ground** (swarm working memory) | in-memory SQLite | `PostgresGround` on its own `stig_pheromones` table (`SELECT … FOR UPDATE SKIP LOCKED` + `LISTEN/NOTIFY`) |
| **Knowledge soil** (vector KB) | tiny SQLite store | your `knowledge_corporate` (pgvector), read with an **ABAC filter** and grown by approved resolutions |
| **Intake** | ServiceNow mock | your `hr_tickets` table |
| **Embedder** | stub / OpenAI | `text-embedding-3-small` (1536-d) — must match how `knowledge_corporate` was populated |

> **The two grounds are different tables.** `stig_pheromones` is the swarm's
> transient working memory (tickets *in flight*); `knowledge_corporate` is the
> durable knowledge. The framework creates and owns `stig_pheromones` — never
> point it at your knowledge table.

## The loop

```mermaid
flowchart LR
  T[("hr_tickets<br/>status='new'")] -->|HrTicketIntakeAnt<br/>(PII scrubbed, ABAC attached)| P[("stig_pheromones")]
  P -->|GovernanceAnt| P
  P -->|AbacKnowledgeSolverAnt<br/>(ABAC-filtered search)| K[("knowledge_corporate")]
  P -->|ReviewingVerifierAnt<br/>Byzantine quorum| P
  P -->|AbacGardenerAnt<br/>+ human approval| K
  P -->|resolve / reject| T
```

Nothing is ever written to `knowledge_corporate` until a **Byzantine quorum
passes** *and* a **human approves** — so a self-improving KB never becomes a
prompt-injection amplifier.

## What the POC proves (four lessons)

1. **Legitimate (+ PII scrubbed)** → the answer is learned as a new
   `chunk_type='ticket_resolution'` row (the soil grows).
2. **Wrong answer → self-healing** → the wrong row is quarantined
   (`is_active=false, soft_deleted_at=now()`) and the expert's answer planted.
3. **Prompt injection → SLASHED** → the quorum rejects it; the KB stays clean and
   the ticket is `rejected`.
4. **ABAC** → the same question returns the restricted executive policy to an
   executive but **nothing restricted** to a level-1 employee.

## Run it

### Option A — a throwaway local pgvector (recommended for the first run)

```powershell
docker compose -f demos/rds_hr/docker-compose.yml up -d   # applies both SQL files
$env:OPENAI_API_KEY = "sk-..."                            # real 1536-d embeddings
$env:STIG_PG_DSN = "postgresql://stig:stig@localhost:5432/stigdemo"
python demos/rds_hr/seed_knowledge.py                     # seed HR benefit chunks
python demos/rds_hr/run_poc.py --reset                    # drive the swarm
```

### Option B — your real RDS

```powershell
# 1. apply the intake table ONCE (your knowledge_corporate already exists):
psql "<rds-dsn>" -f demos/rds_hr/sql/01_hr_tickets.sql
# 2. point the demo at the RDS and run:
$env:OPENAI_API_KEY = "sk-..."
$env:STIG_PG_DSN = "<rds-dsn>"
python demos/rds_hr/run_poc.py --reset
```

`--reset` gives a repeatable demo: it removes learned rows
(`chunk_type='ticket_resolution'`), re-activates quarantined seed rows, truncates
`hr_tickets`, and drops the `stig_pheromones` field. Omit it to accumulate.

## Files

| File | Role |
| --- | --- |
| [sql/01_hr_tickets.sql](sql/01_hr_tickets.sql) | the intake table DDL (built from scratch, mirrors `knowledge_corporate` conventions) |
| [sql/00_knowledge_corporate.sql](sql/00_knowledge_corporate.sql) | your table, reproduced for local testing only |
| [pgvector_knowledge_ground.py](pgvector_knowledge_ground.py) | `PgVectorKnowledgeGround` — the 4-method KB adapter (ABAC search / learn / quarantine / count) |
| [hr_tickets_client.py](hr_tickets_client.py) | `HrTicketsClient` + `HrTicketIntakeAnt` |
| [ants_rds.py](ants_rds.py) | `AbacKnowledgeSolverAnt`, `AbacGardenerAnt` |
| [seed_knowledge.py](seed_knowledge.py) | seed a few benefit chunks (one wrong, one exec-restricted) |
| [run_poc.py](run_poc.py) | the end-to-end POC |

## How a learned ticket maps onto `knowledge_corporate`

| Column | Value on writeback |
| --- | --- |
| `conteudo_original` | the approved answer |
| `section_title` | the ticket question |
| `vetor` | embedding of the **question** (so look-alike tickets retrieve it) |
| `chunk_type` | `ticket_resolution` · `source_type` `form` |
| `knowledge_domain` | `rh_beneficios` · `fonte_documento`/`source_uri` = ticket number |
| `responsavel` / `aprovador` | who resolved / approved |
| `areas_liberadas` / `geografias_liberadas` | derived from the opener's clearance |
| `nivel_hierarquico_minimo` | `1` (share within the area) |
| `dado_sensivel` | `true` if the ticket carried PII |

## Notes & next steps

- **Security**: credentials come only from `STIG_PG_DSN` / `OPENAI_API_KEY`
  (never hardcoded); identifiers are validated and every value is a bound
  parameter.
- **Embeddings must match**: the KB and the swarm must use the same
  `text-embedding-3-small` or vectors are not comparable.
- **ivfflat recall**: after a bulk load, `ANALYZE knowledge_corporate;` and tune
  `SET ivfflat.probes` for recall.
- **pt-BR PII**: `redact_pii` catches email/phone/card/SSN patterns; adding CPF/RG
  patterns is a small, worthwhile enhancement for production.
- **Promotion to core**: once validated on your RDS, `PgVectorKnowledgeGround`
  is a natural fit to graduate into `stigmergic.knowledge` alongside the
  pluggable grounds.
