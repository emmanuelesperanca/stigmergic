"""Ingest HR knowledge into the vector ground (the "seed the soil" step).

The committed, reproducible seed is :file:`hr_faq.jsonl` -- a small synthetic HR
FAQ. Point ``--docs`` at your own material to load real HR documents locally
(never commit those): ``.jsonl`` question/answer pairs load verbatim, while
``.txt`` / ``.md`` (and ``.pdf`` if :mod:`pypdf` is installed) are chunked into
entries so a plain policy document can seed the base too.

    # default: seed from the bundled synthetic FAQ, offline stub embedder
    python ingest.py

    # seed the real vector space with OpenAI embeddings into a persistent DB
    python ingest.py --embed openai --db hr_kb.sqlite

    # load your own local policy files (kept out of the repo)
    python ingest.py --docs "C:\\hr\\policies" --db hr_kb.sqlite
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent
_SRC = _HERE.parents[1] / "src"
for _p in (_HERE, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from embeddings import build_embedder  # noqa: E402
from knowledge_ground import KnowledgeGround, KnowledgeSource  # noqa: E402

DEFAULT_SEED = _HERE / "hr_faq.jsonl"


def _read_jsonl(path: pathlib.Path) -> list[dict]:
    records: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if "question" not in row or "answer" not in row:
            raise ValueError(f"{path.name}: each line needs 'question' and 'answer'.")
        records.append(row)
    return records


def _chunk_text(text: str, *, source_name: str) -> list[dict]:
    """Split a free-form document into coarse Q/A entries by blank-line blocks.

    A crude but dependency-free document loader: each paragraph becomes an
    ``answer``; its first line (or first dozen words) becomes the ``question`` so
    the block is retrievable. Good enough to seed the soil from a real policy
    file without a parsing pipeline.
    """
    blocks = [b.strip() for b in text.replace("\r\n", "\n").split("\n\n") if b.strip()]
    records: list[dict] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        heading = lines[0].lstrip("#").strip()
        question = heading if len(heading.split()) >= 3 else " ".join(block.split()[:12])
        records.append(
            {
                "question": question,
                "answer": block,
                "source": KnowledgeSource.SEED_DOC,
                "metadata": {"origin_file": source_name},
            }
        )
    return records


def _read_pdf(path: pathlib.Path) -> list[dict]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Reading PDFs needs pypdf. Install it with: pip install pypdf "
            "(or convert the file to .txt/.md/.jsonl first)."
        ) from exc
    reader = PdfReader(str(path))
    text = "\n\n".join((page.extract_text() or "") for page in reader.pages)
    return _chunk_text(text, source_name=path.name)


def read_records(path: pathlib.Path) -> list[dict]:
    """Read knowledge records from a file or a directory of files."""
    if path.is_dir():
        records: list[dict] = []
        for child in sorted(path.iterdir()):
            if child.suffix.lower() in {".jsonl", ".json", ".txt", ".md", ".pdf"}:
                records.extend(read_records(child))
        return records

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return _read_jsonl(path)
    if suffix in {".txt", ".md"}:
        return _chunk_text(path.read_text(encoding="utf-8"), source_name=path.name)
    if suffix == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"Unsupported document type: {path.suffix!r} ({path}).")


def ingest(kb: KnowledgeGround, records: list[dict]) -> list[int]:
    """Embed and persist each record; returns the new row ids."""
    ids: list[int] = []
    for row in records:
        ids.append(
            kb.add(
                row["question"],
                row["answer"],
                source=row.get("source", KnowledgeSource.SEED_DOC),
                metadata=row.get("metadata"),
            )
        )
    return ids


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the HR knowledge ground.")
    parser.add_argument(
        "--docs",
        default=str(DEFAULT_SEED),
        help="Path to a .jsonl/.txt/.md/.pdf file or a directory (default: hr_faq.jsonl).",
    )
    parser.add_argument(
        "--db",
        default=":memory:",
        help="SQLite path for the knowledge ground (default: in-memory).",
    )
    parser.add_argument(
        "--embed",
        choices=["stub", "openai"],
        default="stub",
        help="Embedder: 'stub' (offline, deterministic) or 'openai' (real).",
    )
    args = parser.parse_args(argv)

    path = pathlib.Path(args.docs)
    if not path.exists():
        parser.error(f"--docs path does not exist: {path}")

    embedder = build_embedder(args.embed)
    records = read_records(path)
    with KnowledgeGround(embedder, db_path=args.db) as kb:
        ids = ingest(kb, records)
        print(
            f"Ingested {len(ids)} knowledge entries from {path} "
            f"using {embedder.name} into {args.db}."
        )
        if args.db != ":memory:":
            print(f"Knowledge ground persisted at: {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
