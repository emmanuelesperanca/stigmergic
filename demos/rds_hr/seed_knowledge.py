"""Seed a handful of HR-benefit chunks into ``knowledge_corporate`` (pt-BR).

Run this once against a fresh/local database so the swarm has soil to retrieve
from. One entry is deliberately WRONG (vacation days) so the correction/
self-healing path has something to fix, and one is ABAC-restricted to executives
so the access-control filter has something to hide from a normal employee.

    python demos/rds_hr/seed_knowledge.py --dsn postgresql://user:pw@host/db

Needs a real embedder (OPENAI_API_KEY) so the 1536-d vectors match the table.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _p in (_ROOT / "src", _ROOT / "demos" / "servicenow_hr"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from embeddings import OpenAIEmbedder  # noqa: E402

DOMAIN = "rh_beneficios"

# Each seed doc: (section_title, conteudo, areas, nivel_min, geografias, sensivel)
SEED = [
    (
        "Como aderir ao plano de previdência privada (PGBL)?",
        "Novos colaboradores podem aderir ao plano de previdência privada (PGBL) "
        "da empresa a qualquer momento pelo portal de RH, em Benefícios > "
        "Previdência. A empresa faz contrapartida de até 6% do salário base.",
        ["all"], 1, ["all"], False,
    ),
    (
        "Reembolso de cadeira ergonômica para home office",
        "Colaboradores em regime home office podem solicitar reembolso de até "
        "R$ 1.200 para uma cadeira ergonômica, uma vez a cada 3 anos, mediante "
        "nota fiscal, em Benefícios > Home Office.",
        ["all"], 1, ["all"], False,
    ),
    (
        # DELIBERATELY WRONG: the correct figure under the CLT is 30 days.
        "Quantos dias de férias por ano?",
        "Colaboradores em tempo integral têm direito a 20 dias de férias "
        "remuneradas por ano.",
        ["all"], 1, ["all"], False,
    ),
    (
        # ABAC-restricted: only executives (level 5+) may retrieve this.
        "Política de bônus e participação de executivos (PLR-EXEC)",
        "O bônus anual de executivos (níveis 5+) segue a política confidencial "
        "PLR-EXEC, com múltiplos de 0,5 a 2,0 do salário conforme metas do comitê.",
        ["executivo"], 5, ["all"], True,
    ),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=os.environ.get("STIG_PG_DSN", ""))
    args = parser.parse_args()
    if not args.dsn:
        raise SystemExit("Set --dsn or the STIG_PG_DSN environment variable.")

    import psycopg

    embedder = OpenAIEmbedder()
    conn = psycopg.connect(args.dsn, autocommit=True)
    inserted = 0
    for title, conteudo, areas, nivel, geos, sensivel in SEED:
        vec = "[" + ",".join(f"{float(x):.8f}" for x in embedder.encode(title)) + "]"
        content_hash = hashlib.sha256((title + conteudo).encode("utf-8")).hexdigest()
        conn.execute(
            "INSERT INTO knowledge_corporate "
            "(conteudo_original, section_title, knowledge_domain, source_type, "
            " fonte_documento, content_hash, chunk_type, dado_sensivel, "
            " areas_liberadas, nivel_hierarquico_minimo, geografias_liberadas, "
            " embedding_model, embedding_dimensions, vetor, idioma) "
            "VALUES (%s, %s, %s, 'file', 'seed-manual-de-beneficios', %s, "
            "        'text_native', %s, %s, %s, %s, %s, %s, %s::vector, 'pt-BR')",
            (
                conteudo,
                title,
                DOMAIN,
                content_hash,
                sensivel,
                areas,
                nivel,
                geos,
                embedder.name[:50],
                int(getattr(embedder, "dim", 1536)),
                vec,
            ),
        )
        inserted += 1
    conn.close()
    print(f"Seeded {inserted} chunks into knowledge_corporate (domain={DOMAIN}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
