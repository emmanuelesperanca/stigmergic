"""The Hybrid Ant: one caste, two cognitive substrates (cloud text vs. local latent).

A :class:`HybridSolverAnt` is a Solver that can run on either side of the
"String Tax" line, chosen at construction time:

* ``engine="cloud"`` -- it simulates a hosted LLM call (OpenAI / Anthropic / ...).
  Cloud providers expose only **text**, never their hidden states, so latent
  transfer is impossible here: the ant reasons over strings and records a textual
  result in metadata. This is the lossy-but-universal path.
* ``engine="local"`` -- it runs a local ``transformers`` model, performs a forward
  pass with ``output_hidden_states=True``, extracts the final layer's activation
  tensor, serializes it via :mod:`stigmergic_ai.core.latent_transfer`, and attaches
  it to the pheromone's ``latent_blob``. This is the zero-loss Latent State
  Transfer path -- pure mathematical context for a downstream ant to inject.

Both paths are decoupled from any sibling ant; the hybrid ant only reads and
mutates the shared environment. The heavy model and tokenizer are imported lazily
and cached process-wide, so constructing a hybrid ant (or a swarm of them) is
cheap and torch-free until the first *local* metabolization. For testing or
mocking, inject a ``latent_encoder`` (``text -> bytes``) or ``cloud_client``
(``text -> str``) to bypass the real model entirely.
"""

from __future__ import annotations

import threading
from typing import Any, Callable

from stigmergic_ai.agents.base_ant import ConsumerAnt, Mutation
from stigmergic_ai.core.environment import (
    Entropy,
    Pheromone,
    PheromoneGround,
    Status,
)
from stigmergic_ai.core.latent_transfer import serialize_tensor, tensor_fingerprint

__all__ = ["DEFAULT_LOCAL_MODEL", "DEFAULT_CLOUD_MODEL", "HybridSolverAnt"]

#: A tiny causal LM, fine for proving the local latent path without a big download.
DEFAULT_LOCAL_MODEL = "sshleifer/tiny-gpt2"

#: A nominal hosted model name used only in the simulated cloud path's audit trail.
DEFAULT_CLOUD_MODEL = "gpt-4o-mini"


# Module-level model cache: load each heavy (model, tokenizer) pair exactly once
# and share it across every local hybrid ant. Lock-guarded so concurrent ants
# waking together cannot trigger two parallel (expensive) loads.
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_MODEL_LOCK = threading.Lock()


def _load_causal_lm(model_name: str) -> tuple[Any, Any]:
    """Lazily load and cache a ``(model, tokenizer)`` pair for ``model_name``."""
    cached = _MODEL_CACHE.get(model_name)
    if cached is not None:
        return cached
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(model_name)
        if cached is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover - only without extras
                raise ImportError(
                    "The local engine needs the deep-learning extras. "
                    'Install them with: pip install -e ".[cognition]"'
                ) from exc
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.eval()
            cached = (model, tokenizer)
            _MODEL_CACHE[model_name] = cached
        return cached


class HybridSolverAnt(ConsumerAnt):
    """A Solver caste that runs on a ``"cloud"`` (text) or ``"local"`` (latent) engine.

    Args:
        env: The shared Pheromone Ground.
        name: Identifier / ``owner`` stamp. Defaults to the class name.
        engine: ``"cloud"`` for simulated hosted-LLM text reasoning, or
            ``"local"`` for on-device latent extraction.
        model_name: HuggingFace model id for the local engine.
        cloud_model: Nominal hosted model name recorded in the cloud audit trail.
        latent_encoder: Optional ``text -> bytes`` hook overriding local latent
            extraction (pass a mock here to run without torch).
        cloud_client: Optional ``text -> str`` hook overriding the simulated cloud
            call (wire your real OpenAI/Anthropic client here).
        target_status: The trail this ant reacts to (default
            :attr:`Status.HYGIENIZED`).
        result_status: The trail to stamp after solving (default
            :attr:`Status.PENDING_CONSENSUS`, so results still face the quorum).
        latent_marker: If set, the local engine replaces ``raw_data`` with this
            marker -- proof that the downstream ant gets the tensor, not the text.
        max_tokens: Truncation length for the local tokenizer.
        entropy_threshold: Minimum entropy to wake on.
        result_entropy: Entropy to settle to after solving.
        poll_interval: Seconds between heartbeats.
    """

    def __init__(
        self,
        env: PheromoneGround,
        name: str | None = None,
        *,
        engine: str = "cloud",
        model_name: str = DEFAULT_LOCAL_MODEL,
        cloud_model: str = DEFAULT_CLOUD_MODEL,
        latent_encoder: Callable[[str], bytes] | None = None,
        cloud_client: Callable[[str], str] | None = None,
        target_status: Status | str = Status.HYGIENIZED,
        result_status: Status | str = Status.PENDING_CONSENSUS,
        latent_marker: str | None = None,
        max_tokens: int = 512,
        entropy_threshold: float = Entropy.MIN,
        result_entropy: float = Entropy.LOW,
        poll_interval: float = 0.5,
    ) -> None:
        engine = engine.lower()
        if engine not in ("cloud", "local"):
            raise ValueError(f"engine must be 'cloud' or 'local', got {engine!r}.")
        super().__init__(
            env,
            name,
            entropy_threshold=entropy_threshold,
            target_status=target_status,
            poll_interval=poll_interval,
        )
        self.engine = engine
        self.model_name = model_name
        self.cloud_model = cloud_model
        self._latent_encoder = latent_encoder
        self._cloud_client = cloud_client
        self.result_status = result_status
        self.latent_marker = latent_marker
        self.max_tokens = max_tokens
        self.result_entropy = result_entropy

    # -- caste entry point ----------------------------------------------------

    def metabolize(self, task: Pheromone) -> Mutation:
        """Dispatch to the configured engine and propose the resulting mutation."""
        if self.engine == "local":
            return self._solve_local(task)
        return self._solve_cloud(task)

    # -- cloud engine (lossy text) -------------------------------------------

    def _solve_cloud(self, task: Pheromone) -> Mutation:
        """Reason over text via a (simulated) hosted LLM and record the result."""
        summary = (self._cloud_client or self._default_cloud_call)(task.raw_data)
        metadata = dict(task.metadata or {})
        metadata.update(
            {
                "engine": "cloud",
                "cloud_model": self.cloud_model,
                "transfer": "string",  # the String Tax: text in, text out
                "cloud_result": summary,
                "solved_by": self.name,
                "proposal": summary,
            }
        )
        self.log.info("Cloud-solved id=%s via %s (text).", task.id, self.cloud_model)
        return Mutation(
            new_entropy=self.result_entropy,
            new_status=self.result_status,
            metadata=metadata,
            release_owner=True,
        )

    def _default_cloud_call(self, text: str) -> str:
        """Simulated hosted-LLM call. No network; deterministic placeholder text.

        Replace with a real client. Note that hosted APIs return only *text* --
        you cannot retrieve hidden states from them, which is exactly why the
        cloud engine cannot do Latent State Transfer::

            from openai import OpenAI
            client = OpenAI()
            resp = client.chat.completions.create(
                model=self.cloud_model,
                messages=[{"role": "user", "content": text}],
            )
            return resp.choices[0].message.content
        """
        words = text.split()
        preview = " ".join(words[:12])
        return f"[cloud:{self.cloud_model}] reasoned over {len(words)} tokens: {preview}..."

    # -- local engine (zero-loss latent) -------------------------------------

    def _solve_local(self, task: Pheromone) -> Mutation:
        """Extract a latent tensor, serialize it, and park it on the pheromone."""
        blob = (self._latent_encoder or self._default_local_encode)(task.raw_data)
        metadata = dict(task.metadata or {})
        metadata.update(
            {
                "engine": "local",
                "local_model": self.model_name,
                "transfer": "latent",  # zero String Tax: raw activations
                "latent_nbytes": len(blob),
                "solved_by": self.name,
            }
        )
        self.log.info(
            "Local-solved id=%s via %s -> latent blob (%d bytes).",
            task.id,
            self.model_name,
            len(blob),
        )
        return Mutation(
            new_entropy=self.result_entropy,
            new_status=self.result_status,
            new_raw_data=self.latent_marker,  # withhold the text if a marker is set
            latent_blob=blob,
            metadata=metadata,
            release_owner=True,
        )

    def _default_local_encode(self, text: str) -> bytes:
        """Real local latent extraction: forward pass -> last hidden state -> bytes.

        Runs the causal LM with ``output_hidden_states=True`` and serializes the
        final layer's activations (shape ``[1, seq_len, hidden_dim]``). The tensor
        *is* the model's understanding of the text, transferred with zero loss.
        """
        import torch  # lazy: only the local path needs the deep-learning stack

        model, tokenizer = _load_causal_lm(self.model_name)
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_tokens,
        )
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]  # [1, seq_len, hidden_dim]
        self.log.debug("Extracted %s", tensor_fingerprint(last_hidden))
        return serialize_tensor(last_hidden)
