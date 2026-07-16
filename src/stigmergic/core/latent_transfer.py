"""Latent State Transfer (LSC): move *understanding* between ants as raw tensors.

Conventional multi-agent systems pay a **String Tax**: every handoff forces a
model to compress its rich internal state into lossy text, which the next model
must re-read and re-encode. Stigmergic lets ants skip that entirely. A Reader
can distill a heavy document into the hidden-state tensor of its final layer and
park those raw activations on the Pheromone Ground; a downstream ant injects that
tensor straight into its own residual stream. No tokens, no re-reading, no loss.

This module is the plumbing for that: it (de)serializes a PyTorch tensor to and
from the ``bytes`` that live in the ``latent_blob`` SQLite column, via
``io.BytesIO`` and ``torch.save`` / ``torch.load``.

It honours the framework's lightweight-core philosophy: **torch is imported
lazily**, only when a tensor is actually (de)serialized. Importing this module --
or :mod:`stigmergic` -- never drags in the deep-learning stack. Install it with
``pip install -e ".[cognition]"``.

Security note: :func:`deserialize_tensor` defaults to ``weights_only=True`` so a
hostile ``latent_blob`` cannot smuggle arbitrary pickled code into the process
(an untrusted-deserialization / OWASP A08 hazard). Only disable that for blobs
whose origin you fully trust.
"""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # import only for type checkers, never at runtime
    from torch import Tensor

__all__ = [
    "torch_available",
    "serialize_tensor",
    "deserialize_tensor",
    "tensor_fingerprint",
]

logger = logging.getLogger("stigmergic.latent_transfer")

_MISSING_TORCH_MSG = (
    "Latent State Transfer needs the deep-learning extras (torch). "
    'Install them with: pip install -e ".[cognition]"'
)


def _require_torch() -> Any:
    """Import and return the ``torch`` module, or raise a helpful error."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised only without extras
        raise ImportError(_MISSING_TORCH_MSG) from exc
    return torch


def torch_available() -> bool:
    """Return ``True`` if ``torch`` can be imported in this environment.

    Lets callers branch to a text/cloud fallback when the local deep-learning
    stack is absent, without triggering an :class:`ImportError`.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


def serialize_tensor(tensor: "Tensor", *, detach: bool = True) -> bytes:
    """Serialize a PyTorch tensor to ``bytes`` for the ``latent_blob`` column.

    The tensor is moved to CPU and detached from the autograd graph first, so the
    blob is portable across processes/devices and carries no gradient baggage.
    Shape and dtype are preserved by ``torch.save`` and recovered automatically on
    :func:`deserialize_tensor`.

    Args:
        tensor: The tensor to serialize (e.g. a last-layer hidden state of shape
            ``[1, seq_len, hidden_dim]``).
        detach: Detach and move to CPU before saving (recommended; default).

    Returns:
        The serialized tensor as raw bytes.
    """
    torch = _require_torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"serialize_tensor expects a torch.Tensor, got {type(tensor)!r}.")
    payload = tensor.detach().cpu() if detach else tensor
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    blob = buffer.getvalue()
    logger.debug("Serialized %s -> %d bytes", tensor_fingerprint(tensor), len(blob))
    return blob


def deserialize_tensor(
    blob: bytes | bytearray | memoryview,
    *,
    map_location: Any = "cpu",
    weights_only: bool = True,
) -> "Tensor":
    """Reconstruct a tensor previously produced by :func:`serialize_tensor`.

    Args:
        blob: The raw bytes stored in ``latent_blob``.
        map_location: Where to materialize the tensor (default ``"cpu"`` so it
            loads even on machines without the original device).
        weights_only: Refuse to unpickle anything but tensor storage (default
            ``True``). This is the safe setting against malicious blobs; only set
            it ``False`` for fully trusted sources on an older torch.

    Returns:
        The reconstructed tensor, with its original shape and dtype.
    """
    torch = _require_torch()
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError(f"deserialize_tensor expects bytes-like data, got {type(blob)!r}.")
    buffer = io.BytesIO(bytes(blob))
    try:
        tensor = torch.load(buffer, map_location=map_location, weights_only=weights_only)
    except TypeError:
        # torch < 1.13 has no weights_only parameter; fall back transparently.
        buffer.seek(0)
        tensor = torch.load(buffer, map_location=map_location)
    logger.debug("Deserialized %d bytes -> %s", len(blob), tensor_fingerprint(tensor))
    return tensor


def tensor_fingerprint(tensor: "Tensor") -> str:
    """A compact ``Tensor(shape=..., dtype=...)`` string for logs and audits.

    Describes a tensor without dumping its (potentially huge) contents, so it is
    safe to drop into logs or pheromone metadata.
    """
    shape = tuple(tensor.shape)
    dtype = str(getattr(tensor, "dtype", "?")).replace("torch.", "")
    return f"Tensor(shape={list(shape)}, dtype={dtype})"
