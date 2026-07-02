"""A local, in-memory mock of the ServiceNow Incident (ITSM) API.

Enough of ServiceNow to make the demo feel real without a PDI, credentials, or
the network: incidents with the fields and lifecycle that matter, a numbering
scheme (``INC0000001``), work notes, and the state transitions
(New -> In Progress -> Resolved / Canceled) the ant swarm drives.

Everything sits behind the :class:`ServiceNowClient` ``Protocol`` so that a real
REST-backed client can be dropped in later with no change to the ants -- the same
"code against the contract, mock the substrate" split used for the embedder.
Field and state names follow ServiceNow's own vocabulary so the mapping to a real
instance is one-to-one.
"""

from __future__ import annotations

import secrets
import threading
import time
from enum import IntEnum
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field

__all__ = [
    "IncidentState",
    "Incident",
    "ServiceNowClient",
    "MockServiceNowClient",
]


class IncidentState(IntEnum):
    """ServiceNow incident ``state`` values (the canonical numeric codes)."""

    NEW = 1
    IN_PROGRESS = 2
    ON_HOLD = 3
    RESOLVED = 6
    CLOSED = 7
    CANCELED = 8

    @property
    def label(self) -> str:
        return {
            1: "New",
            2: "In Progress",
            3: "On Hold",
            6: "Resolved",
            7: "Closed",
            8: "Canceled",
        }[int(self)]


class Incident(BaseModel):
    """A single ServiceNow incident record (the fields this demo exercises)."""

    sys_id: str
    number: str
    short_description: str
    description: str = ""
    state: IncidentState = IncidentState.NEW
    caller_id: str = "employee"
    assignment_group: str = "HR Services"
    assigned_to: str = ""
    category: str = "HR"
    work_notes: list[str] = Field(default_factory=list)
    close_notes: str = ""
    opened_at: float = 0.0
    resolved_at: float | None = None
    closed_at: float | None = None


@runtime_checkable
class ServiceNowClient(Protocol):
    """The ITSM operations the swarm relies on (mock or real behind it)."""

    def create_incident(
        self, short_description: str, *, description: str = ..., caller_id: str = ...
    ) -> Incident: ...

    def get(self, sys_id: str) -> Incident | None: ...

    def list_incidents(self, *, state: IncidentState | None = ...) -> list[Incident]: ...

    def update(self, sys_id: str, **fields: Any) -> Incident: ...

    def add_work_note(self, sys_id: str, note: str) -> Incident: ...

    def resolve(self, sys_id: str, resolution_notes: str) -> Incident: ...

    def cancel(self, sys_id: str, reason: str) -> Incident: ...

    def reopen(self, sys_id: str, reason: str) -> Incident: ...


class MockServiceNowClient:
    """An in-memory :class:`ServiceNowClient` backed by a plain dict.

    Deterministic and dependency-free: incident numbers increment from
    ``INC0000001``; ``sys_id`` values are random 32-char hex just like the real
    platform. State transitions stamp the matching timestamps so a resolved or
    canceled ticket carries an audit trail.
    """

    def __init__(self) -> None:
        self._incidents: dict[str, Incident] = {}
        self._seq = 0
        self._lock = threading.RLock()

    def _next_number(self) -> str:
        self._seq += 1
        return f"INC{self._seq:07d}"

    def create_incident(
        self,
        short_description: str,
        *,
        description: str = "",
        caller_id: str = "employee",
        assignment_group: str = "HR Services",
    ) -> Incident:
        with self._lock:
            incident = Incident(
                sys_id=secrets.token_hex(16),
                number=self._next_number(),
                short_description=short_description,
                description=description,
                caller_id=caller_id,
                assignment_group=assignment_group,
                state=IncidentState.NEW,
                opened_at=time.time(),
            )
            self._incidents[incident.sys_id] = incident
            return incident.model_copy(deep=True)

    def get(self, sys_id: str) -> Incident | None:
        with self._lock:
            incident = self._incidents.get(sys_id)
            return incident.model_copy(deep=True) if incident is not None else None

    def list_incidents(self, *, state: IncidentState | None = None) -> list[Incident]:
        with self._lock:
            items = [
                inc.model_copy(deep=True)
                for inc in self._incidents.values()
                if state is None or inc.state == state
            ]
        items.sort(key=lambda inc: inc.number)
        return items

    def _require(self, sys_id: str) -> Incident:
        incident = self._incidents.get(sys_id)
        if incident is None:
            raise KeyError(f"No incident with sys_id={sys_id!r}.")
        return incident

    def update(self, sys_id: str, **fields: Any) -> Incident:
        with self._lock:
            incident = self._require(sys_id)
            data = incident.model_dump()
            for key, value in fields.items():
                if key not in data:
                    raise AttributeError(f"Incident has no field {key!r}.")
                data[key] = value
            updated = Incident(**data)
            self._incidents[sys_id] = updated
            return updated.model_copy(deep=True)

    def add_work_note(self, sys_id: str, note: str) -> Incident:
        with self._lock:
            incident = self._require(sys_id)
            incident.work_notes.append(note)
            return incident.model_copy(deep=True)

    def resolve(self, sys_id: str, resolution_notes: str) -> Incident:
        with self._lock:
            incident = self._require(sys_id)
            incident.state = IncidentState.RESOLVED
            incident.close_notes = resolution_notes
            incident.resolved_at = time.time()
            incident.work_notes.append(f"Resolved: {resolution_notes}")
            return incident.model_copy(deep=True)

    def cancel(self, sys_id: str, reason: str) -> Incident:
        with self._lock:
            incident = self._require(sys_id)
            incident.state = IncidentState.CANCELED
            incident.close_notes = reason
            incident.closed_at = time.time()
            incident.work_notes.append(f"Canceled: {reason}")
            return incident.model_copy(deep=True)

    def reopen(self, sys_id: str, reason: str) -> Incident:
        with self._lock:
            incident = self._require(sys_id)
            incident.state = IncidentState.IN_PROGRESS
            incident.resolved_at = None
            incident.closed_at = None
            incident.work_notes.append(f"Reopened: {reason}")
            return incident.model_copy(deep=True)
