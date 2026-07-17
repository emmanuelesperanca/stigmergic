-- ============================================================================
-- public.hr_tickets  --  RH ticket intake for the Stigmergic swarm
-- ----------------------------------------------------------------------------
-- The intake table the swarm polls. It deliberately mirrors the conventions of
-- public.knowledge_corporate (UUID PKs, ABAC columns, timestamptz, pt-BR/EN
-- naming) so the same access-control attributes that gate the knowledge base
-- also travel with each ticket.
--
-- Lifecycle (driven by the swarm):
--   new -> in_progress -> pending_approval -> resolved | rejected | cancelled
--
-- On a human-approved resolution the GardenerAnt writes the (question, answer)
-- pair back into knowledge_corporate and stamps kb_entry_id here, closing the
-- self-improving loop.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- DROP TABLE public.hr_tickets;

CREATE TABLE IF NOT EXISTS public.hr_tickets (
    id                              uuid DEFAULT uuid_generate_v4() NOT NULL,
    ticket_number                   text,                       -- HR-000001 (auto)

    -- ---- content ------------------------------------------------------------
    assunto                         text NOT NULL,              -- short description / retrieval key
    descricao                       text NOT NULL DEFAULT '',   -- full body
    knowledge_domain                varchar(50) NOT NULL DEFAULT 'rh_beneficios',
    idioma                          varchar(10) DEFAULT 'pt-BR',
    canal                           varchar(20) DEFAULT 'portal',   -- portal | email | teams | ...

    -- ---- requester + ABAC attributes ---------------------------------------
    -- Matched at retrieval time against knowledge_corporate's ACL columns
    -- (areas_liberadas / nivel_hierarquico_minimo / geografias_liberadas), so a
    -- ticket only ever "sees" knowledge its opener is cleared for.
    solicitante_id                  text NOT NULL,
    solicitante_email               text,
    solicitante_area                text NOT NULL DEFAULT 'all',
    solicitante_nivel_hierarquico   int4 NOT NULL DEFAULT 1,
    solicitante_geografia           text NOT NULL DEFAULT 'all',
    solicitante_projetos            _text DEFAULT ARRAY['all'::text] NULL,

    -- ---- swarm lifecycle ----------------------------------------------------
    status                          varchar(20) NOT NULL DEFAULT 'new',
    pheromone_id                    int8,                       -- id in stig_pheromones
    assigned_to                     text,                       -- caste currently holding it

    -- ---- resolution + learning ----------------------------------------------
    proposta_resolucao              text,                       -- machine proposal (pre-approval)
    resolucao                       text,                       -- final, human-approved answer
    resolvido_por                   text,                       -- gardener / system
    aprovado_por                    text,                       -- human reviewer
    kb_entry_id                     uuid,                       -- knowledge_corporate.id of learned row
    kb_action                       varchar(20),                -- add | quarantine+add | quarantine | none

    -- ---- governance / audit -------------------------------------------------
    contem_pii                      bool DEFAULT false,
    pii_redigido                    jsonb DEFAULT '[]'::jsonb,  -- kinds of PII scrubbed at intake
    consensus_resultado             jsonb,                      -- the Byzantine verdict (votes)
    veredito                        varchar(20),                -- passed | slashed

    -- ---- timestamps ---------------------------------------------------------
    created_at                      timestamptz DEFAULT now() NOT NULL,
    updated_at                      timestamptz DEFAULT now() NOT NULL,
    data_resolucao                  timestamptz,

    CONSTRAINT hr_tickets_pkey PRIMARY KEY (id),
    CONSTRAINT hr_tickets_status_check CHECK (
        (status)::text = ANY (ARRAY[
            'new','in_progress','pending_approval','resolved','rejected','cancelled'
        ]::text[])
    ),
    CONSTRAINT hr_tickets_consensus_check CHECK (
        consensus_resultado IS NULL OR jsonb_typeof(consensus_resultado) = 'object'
    ),
    CONSTRAINT hr_tickets_pii_check CHECK (
        jsonb_typeof(pii_redigido) = 'array'
    )
);

CREATE UNIQUE INDEX IF NOT EXISTS hr_tickets_number_key   ON public.hr_tickets USING btree (ticket_number);
CREATE INDEX IF NOT EXISTS idx_hr_tickets_status          ON public.hr_tickets USING btree (status, created_at);
CREATE INDEX IF NOT EXISTS idx_hr_tickets_domain          ON public.hr_tickets USING btree (knowledge_domain);
CREATE INDEX IF NOT EXISTS idx_hr_tickets_solicitante     ON public.hr_tickets USING btree (solicitante_id);
CREATE INDEX IF NOT EXISTS idx_hr_tickets_area            ON public.hr_tickets USING btree (solicitante_area);
CREATE INDEX IF NOT EXISTS idx_hr_tickets_pheromone       ON public.hr_tickets USING btree (pheromone_id);

-- ---- triggers --------------------------------------------------------------

-- Keep updated_at fresh on every UPDATE.
CREATE OR REPLACE FUNCTION public.hr_tickets_touch_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS hr_tickets_set_updated_at ON public.hr_tickets;
CREATE TRIGGER hr_tickets_set_updated_at
    BEFORE UPDATE ON public.hr_tickets
    FOR EACH ROW EXECUTE FUNCTION public.hr_tickets_touch_updated_at();

-- Human-friendly, monotonic ticket_number (HR-000001) on INSERT.
CREATE SEQUENCE IF NOT EXISTS public.hr_tickets_number_seq START 1;

CREATE OR REPLACE FUNCTION public.hr_tickets_assign_number()
RETURNS trigger AS $$
BEGIN
    IF NEW.ticket_number IS NULL THEN
        NEW.ticket_number := 'HR-' || lpad(nextval('public.hr_tickets_number_seq')::text, 6, '0');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS hr_tickets_set_number ON public.hr_tickets;
CREATE TRIGGER hr_tickets_set_number
    BEFORE INSERT ON public.hr_tickets
    FOR EACH ROW EXECUTE FUNCTION public.hr_tickets_assign_number();
