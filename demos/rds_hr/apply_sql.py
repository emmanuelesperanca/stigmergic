"""Apply a .sql script to a DSN using psycopg (no psql binary needed).

    python demos/rds_hr/apply_sql.py demos/rds_hr/sql/01_hr_tickets.sql

Splits the script into individual statements (respecting ``$$``-quoted function
bodies and ``--`` line comments) and runs each on an autocommit connection, so a
full DDL script with functions and triggers applies in one shot. Reads the DSN
from ``--dsn`` or the ``STIG_PG_DSN`` environment variable.
"""

from __future__ import annotations

import argparse
import os
import pathlib


def split_statements(sql: str) -> list[str]:
    """Split a SQL script into statements, honouring ``$$`` quoting and ``--``."""
    out: list[str] = []
    buf: list[str] = []
    in_dollar = False
    i, n = 0, len(sql)
    while i < n:
        two = sql[i:i + 2]
        if not in_dollar and two == "--":
            eol = sql.find("\n", i)
            eol = n if eol == -1 else eol
            buf.append(sql[i:eol])
            i = eol
            continue
        if two == "$$":
            in_dollar = not in_dollar
            buf.append("$$")
            i += 2
            continue
        ch = sql[i]
        if ch == ";" and not in_dollar:
            stmt = "".join(buf).strip()
            if stmt:
                out.append(stmt)
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", help="path to the .sql file")
    parser.add_argument("--dsn", default=os.environ.get("STIG_PG_DSN", ""))
    args = parser.parse_args()
    if not args.dsn:
        raise SystemExit("Set --dsn or the STIG_PG_DSN environment variable.")

    import psycopg

    path = pathlib.Path(args.script)
    sql = path.read_text(encoding="utf-8")
    statements = split_statements(sql)

    conn = psycopg.connect(args.dsn, autocommit=True)
    try:
        for idx, stmt in enumerate(statements, 1):
            try:
                conn.execute(stmt)
            except Exception as exc:  # noqa: BLE001
                head = stmt.strip().splitlines()[0][:80]
                print(f"[{idx}/{len(statements)}] FAILED: {head}\n  -> {exc}")
                raise
    finally:
        conn.close()
    print(f"Applied {len(statements)} statement(s) from {path.name}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
