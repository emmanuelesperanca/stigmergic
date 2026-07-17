-- ============================================================================
-- public.knowledge_corporate  --  reference DDL (YOURS, verbatim)
-- ----------------------------------------------------------------------------
-- This is your production table, reproduced here ONLY so a local Postgres can be
-- bootstrapped for end-to-end testing (docker compose up). Against your real RDS
-- the table already exists -- do NOT run this there. The two extra bits your
-- production DB already has (the `vector` extension and the tsvector trigger
-- function) are added at the top so a fresh local DB matches.
-- ============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- The trigger function referenced by tsvectorupdate below. On your RDS this
-- already exists; defined here so a local bootstrap does not fail.
CREATE OR REPLACE FUNCTION knowledge_tsvector_trigger()
RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        to_tsvector('portuguese',
            coalesce(NEW.section_title, '') || ' ' || coalesce(NEW.conteudo_original, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- DROP TABLE public.knowledge_corporate;

CREATE TABLE public.knowledge_corporate (
	id uuid DEFAULT uuid_generate_v4() NOT NULL,
	document_id uuid DEFAULT uuid_generate_v4() NOT NULL,
	chunk_index int4 DEFAULT 0 NULL,
	conteudo_original text NOT NULL,
	section_title text NULL,
	fonte_documento text NULL,
	source_type varchar(20) DEFAULT 'file'::character varying NULL,
	source_uri text NULL,
	content_hash text NULL,
	ingestion_batch_id uuid NULL,
	knowledge_domain varchar(50) NOT NULL,
	tags jsonb DEFAULT '{}'::jsonb NULL,
	dado_sensivel bool DEFAULT false NULL,
	apenas_para_si bool DEFAULT false NULL,
	areas_liberadas _text DEFAULT ARRAY['all'::text] NULL,
	nivel_hierarquico_minimo int4 DEFAULT 1 NULL,
	geografias_liberadas _text DEFAULT ARRAY['all'::text] NULL,
	projetos_liberados _text DEFAULT ARRAY['all'::text] NULL,
	is_active bool DEFAULT true NULL,
	soft_deleted_at timestamptz NULL,
	idioma varchar(10) DEFAULT 'pt-BR'::character varying NULL,
	data_validade date NULL,
	responsavel text NULL,
	aprovador text NULL,
	data_ingestao timestamptz DEFAULT now() NULL,
	embedding_model varchar(50) DEFAULT 'text-embedding-3-small'::character varying NULL,
	embedding_dimensions int4 DEFAULT 1536 NULL,
	vetor public.vector NOT NULL,
	search_vector tsvector NULL,
	chunk_id uuid DEFAULT uuid_generate_v4() NULL,
	parent_id uuid NULL,
	chunk_type varchar(40) DEFAULT 'text_native'::character varying NULL,
	page_number int4 NULL,
	chunk_order int4 NULL,
	extraction_method varchar(50) NULL,
	extraction_confidence float8 NULL,
	vision_model varchar(100) NULL,
	vision_prompt_version varchar(40) NULL,
	image_ref text NULL,
	image_mime_type varchar(50) NULL,
	image_width int4 NULL,
	image_height int4 NULL,
	parent_text_size int4 NULL,
	parent_page_count int4 NULL,
	CONSTRAINT knowledge_corporate_embedding_dimensions_check CHECK ((embedding_dimensions > 0)),
	CONSTRAINT knowledge_corporate_pkey PRIMARY KEY (id),
	CONSTRAINT knowledge_corporate_source_type_check CHECK (((source_type)::text = ANY ((ARRAY['file'::character varying, 'url'::character varying, 'db'::character varying, 'form'::character varying])::text[]))),
	CONSTRAINT knowledge_corporate_tags_check CHECK ((jsonb_typeof(tags) = 'object'::text))
);
CREATE INDEX idx_knowledge_areas ON public.knowledge_corporate USING gin (areas_liberadas);
CREATE INDEX idx_knowledge_batch_id ON public.knowledge_corporate USING btree (ingestion_batch_id);
CREATE INDEX idx_knowledge_chunk_type ON public.knowledge_corporate USING btree (chunk_type);
CREATE INDEX idx_knowledge_content_hash ON public.knowledge_corporate USING btree (content_hash);
CREATE INDEX idx_knowledge_document_id ON public.knowledge_corporate USING btree (document_id);
CREATE INDEX idx_knowledge_domain_active ON public.knowledge_corporate USING btree (knowledge_domain, is_active, data_validade) WHERE ((is_active = true) AND (soft_deleted_at IS NULL));
CREATE INDEX idx_knowledge_domain_parent ON public.knowledge_corporate USING btree (knowledge_domain, parent_id);
CREATE INDEX idx_knowledge_geografias ON public.knowledge_corporate USING gin (geografias_liberadas);
CREATE INDEX idx_knowledge_parent_chunk_order ON public.knowledge_corporate USING btree (parent_id, chunk_order);
CREATE INDEX idx_knowledge_parent_id ON public.knowledge_corporate USING btree (parent_id);
CREATE INDEX idx_knowledge_parent_page ON public.knowledge_corporate USING btree (parent_id, page_number);
CREATE INDEX idx_knowledge_search_vector ON public.knowledge_corporate USING gin (search_vector);
CREATE INDEX idx_knowledge_tags ON public.knowledge_corporate USING gin (tags);
CREATE INDEX idx_knowledge_vetor_ivfflat ON public.knowledge_corporate USING ivfflat (vetor vector_cosine_ops) WITH (lists='100');

CREATE TRIGGER tsvectorupdate BEFORE INSERT OR UPDATE
    ON public.knowledge_corporate
    FOR EACH ROW EXECUTE FUNCTION knowledge_tsvector_trigger();
