"""Pluggable Pheromone Ground backends and a DSN-based factory.

Horizon 1 ships a single-file SQLite ground -- perfect for a laptop, a demo, or
a single-process swarm. Real deployments outgrow it: SQLite is single-writer and
its only "notification" is a poll loop. This package lets you swap the substrate
without touching a single ant, because every backend implements the same
:class:`~stigmergic.core.environment.AbstractGround` contract.

* :func:`create_ground` -- a tiny factory that turns a DSN string into the right
  backend (``sqlite://``, ``postgresql://``, ...).
* :class:`~stigmergic.core.backends.postgres.PostgresGround` -- a production
  ground using ``SELECT ... FOR UPDATE SKIP LOCKED`` for lock-free concurrent
  claims and ``LISTEN/NOTIFY`` for true push (no polling) across processes.

Heavy drivers (``psycopg``, ``redis``, ``boto3``) are imported lazily inside the
backend that needs them, so importing this package -- or ``stigmergic`` --
never pulls a database driver into memory.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:  # avoid importing the concrete grounds at module import time
    from stigmergic.core.environment import AbstractGround

__all__ = ["create_ground"]

#: Backends that are designed but not yet shipped. They have a clear home in the
#: factory so the roadmap is visible and a helpful error points the way.
_ROADMAP = {
    "redis": "pip install redis  # RedisGround (sorted-set ground + pub/sub) is on the roadmap",
    "rediss": "pip install redis  # RedisGround (sorted-set ground + pub/sub) is on the roadmap",
    "dynamodb": "pip install boto3  # DynamoDBGround (streams-driven) is on the roadmap",
}


def create_ground(dsn: str = "sqlite://:memory:", **kwargs: object) -> "AbstractGround":
    """Build a Pheromone Ground from a DSN string.

    The scheme selects the backend; everything else is backend-specific and
    forwarded as ``kwargs``:

    * ``sqlite://:memory:`` or ``sqlite:////abs/path/swarm.db`` or a bare
      filesystem path (``swarm.db``) -> the SQLite
      :class:`~stigmergic.core.environment.PheromoneGround`.
    * ``postgresql://user:pass@host:5432/dbname`` -> the
      :class:`~stigmergic.core.backends.postgres.PostgresGround`.

    Args:
        dsn: A connection string whose scheme picks the backend.
        **kwargs: Extra backend constructor arguments (e.g. ``busy_timeout_ms``
            for SQLite, or ``table``/``channel`` for Postgres).

    Returns:
        A ready-to-use ground implementing
        :class:`~stigmergic.core.environment.AbstractGround`.

    Raises:
        NotImplementedError: For a scheme that is on the roadmap but unshipped.
        ValueError: For an unrecognized scheme.
    """
    # A bare filesystem path (no "://") is always a SQLite file. Handle it before
    # urlsplit, which would misread a Windows drive letter ("C:") as a scheme.
    if "://" not in dsn:
        from stigmergic.core.environment import PheromoneGround

        return PheromoneGround(dsn or ":memory:", **kwargs)  # type: ignore[arg-type]

    parts = urlsplit(dsn)
    scheme = parts.scheme.lower()

    if scheme in ("sqlite", "file"):
        # Local SQLite. Resolve ":memory:" vs a filesystem path from the DSN.
        from stigmergic.core.environment import PheromoneGround

        if ":memory:" in dsn:
            path = ":memory:"
        else:
            # sqlite:///rel/path or sqlite:////abs/path -> use the path part.
            path = (parts.netloc + parts.path) or ":memory:"
            # Repair a Windows drive URI such as sqlite:///C:/db -> C:/db.
            if re.match(r"^/[A-Za-z]:", path):
                path = path[1:]
        return PheromoneGround(path, **kwargs)  # type: ignore[arg-type]

    if scheme in ("postgresql", "postgres"):
        from stigmergic.core.backends.postgres import PostgresGround

        return PostgresGround(dsn, **kwargs)  # type: ignore[arg-type]

    if scheme in _ROADMAP:
        raise NotImplementedError(
            f"The {scheme!r} ground is on the roadmap, not yet shipped. "
            f"Hint: {_ROADMAP[scheme]}. Contributions welcome -- implement "
            "AbstractGround and add a branch here."
        )

    raise ValueError(
        f"Unsupported ground DSN scheme: {scheme!r}. "
        "Use 'sqlite://', 'postgresql://', or a bare file path."
    )
