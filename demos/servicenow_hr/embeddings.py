"""Text embedders for the ServiceNow HR demo's vector "forest floor".

The knowledge base (:mod:`knowledge_ground`) turns a question into a vector and
finds the nearest stored answer -- this is the "the leaf falls near where the
answer already lies" step of the demo. That only needs one tiny contract:

    Embedder.encode(text) -> list[float]

Two implementations satisfy it:

* :class:`StubEmbedder` -- a deterministic, offline, dependency-free hashing
  embedder. Semantically crude (it is a signed bag-of-words), but *stable*: the
  same text always yields the same vector and texts that share words have a
  positive cosine similarity. This is what the committed unit tests use, so the
  whole suite stays offline, sub-second, and free -- exactly like the benchmark's
  ``MockNLIJudge`` and the latent-transfer demo's mock encoder.
* :class:`OpenAIEmbedder` -- the real embedder the *demo run* uses:
  ``text-embedding-3-small`` by default. Configured purely through environment
  variables (never hardcode a key), with every vector cached on disk so a re-run
  costs nothing and never depends on the network twice.

The split is the same tiered philosophy as the injection benchmark
(torch-free default, opt-in real model): the reproducible artifact needs no key,
and the real semantics are one flag away.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import re
import sys
from typing import Protocol, runtime_checkable

# Make sibling modules importable when run straight from the repo (no install),
# mirroring the benchmark harness bootstrap.
_HERE = pathlib.Path(__file__).resolve().parent
_SRC = _HERE.parents[1] / "src"
for _p in (_HERE, _SRC):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

__all__ = [
    "Embedder",
    "StubEmbedder",
    "OpenAIEmbedder",
    "EmbeddingConfig",
    "build_embedder",
    "cosine_similarity",
]

_CACHE_DIR = _HERE / ".cache" / "embed"


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens of length >= 2 (the stub's vocabulary)."""
    return [tok for tok in re.findall(r"[a-z0-9]+", text.lower()) if len(tok) >= 2]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors, in ``[-1.0, 1.0]``.

    Returns ``0.0`` if either vector has zero magnitude (an empty text), which
    keeps an unseeded knowledge base from raising instead of simply not matching.
    """
    if len(a) != len(b):
        raise ValueError(f"Vectors differ in length: {len(a)} != {len(b)}.")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


@runtime_checkable
class Embedder(Protocol):
    """The one-method contract the knowledge ground depends on."""

    name: str
    dim: int

    def encode(self, text: str) -> list[float]:
        """Return a fixed-length embedding vector for ``text``."""
        ...


class StubEmbedder:
    """A deterministic, offline signed-bag-of-words embedder.

    Each token is hashed to a bucket and a sign, and its contribution is
    accumulated into a fixed-dimension vector, which is then L2-normalized. It is
    not a language model -- it has no notion of synonyms -- but it is stable and
    reproducible, and two texts that reuse the same words land close together in
    cosine space. That is enough to drive and *verify* the retrieval, learning,
    and correction flows without a network, a key, or a GPU.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive.")
        self.name = f"stub-hash-{dim}"
        self.dim = dim

    def encode(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for token in _tokens(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(component * component for component in vec))
        if norm == 0.0:
            return vec
        return [component / norm for component in vec]


class EmbeddingConfig:
    """Resolved OpenAI/Azure embedding settings, read from the environment.

    Mirrors the benchmark's ``providers.LLMConfig`` so the two share one mental
    model. Never hardcode a key -- set it in your shell:

        # OpenAI (default)
        $env:OPENAI_API_KEY = "sk-..."
        $env:STIG_EMBED_MODEL = "text-embedding-3-small"   # optional

        # Azure OpenAI
        $env:STIG_EMBED_PROVIDER = "azure"
        $env:AZURE_OPENAI_API_KEY = "..."
        $env:AZURE_OPENAI_ENDPOINT = "https://<resource>.openai.azure.com"
        $env:STIG_EMBED_MODEL = "<your deployment name>"
    """

    def __init__(self) -> None:
        self.provider = os.environ.get("STIG_EMBED_PROVIDER", "openai").strip().lower()
        self.model = os.environ.get("STIG_EMBED_MODEL", "text-embedding-3-small").strip()
        self.api_version = os.environ.get("STIG_EMBED_API_VERSION", "2024-06-01").strip()
        if self.provider == "azure":
            self.api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
            self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            self.endpoint = os.environ.get("OPENAI_BASE_URL", "").strip()

    def require(self) -> "EmbeddingConfig":
        """Fail fast with an actionable message if the config is incomplete."""
        if not self.api_key:
            key = "AZURE_OPENAI_API_KEY" if self.provider == "azure" else "OPENAI_API_KEY"
            raise RuntimeError(
                f"The real embedder needs a provider key. Set {key} (see this "
                "module's docstring for the full variable list), or run the demo "
                "with --embed stub to stay fully offline."
            )
        if self.provider == "azure" and not self.endpoint:
            raise RuntimeError("Azure provider needs AZURE_OPENAI_ENDPOINT.")
        return self

    @property
    def signature(self) -> str:
        """A short, key-free fingerprint for the on-disk cache namespace."""
        return f"{self.provider}:{self.model}:{self.api_version}"


class OpenAIEmbedder:
    """A real embedder backed by an OpenAI-compatible embeddings endpoint.

    ``text-embedding-3-small`` (1536-d) by default -- cheap, fast, and the same
    family used elsewhere. Every distinct string is cached on disk under
    ``.cache/embed/`` (git-ignored) so a second run is free and deterministic and
    a flaky network never blocks the demo twice. The ``openai`` SDK is imported
    lazily, so merely importing this module never requires it.
    """

    #: Output dimensions for the common OpenAI embedding models.
    _MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, *, config: EmbeddingConfig | None = None, use_cache: bool = True) -> None:
        self.config = (config or EmbeddingConfig()).require()
        self.name = f"openai:{self.config.model}"
        self.dim = self._MODEL_DIMS.get(self.config.model, 1536)
        self.use_cache = use_cache
        self._client = None
        if use_cache:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _build_client(self):
        try:
            if self.config.provider == "azure":
                from openai import AzureOpenAI

                return AzureOpenAI(
                    api_key=self.config.api_key,
                    azure_endpoint=self.config.endpoint,
                    api_version=self.config.api_version,
                )
            from openai import OpenAI

            kwargs = {"api_key": self.config.api_key}
            if self.config.endpoint:
                kwargs["base_url"] = self.config.endpoint
            return OpenAI(**kwargs)
        except ImportError as exc:  # pragma: no cover - only without the SDK
            raise ImportError(
                "The real embedder needs the openai SDK. Install it with: "
                'pip install "openai>=1.0"  (or run the demo with --embed stub).'
            ) from exc

    def _cache_path(self, text: str) -> pathlib.Path:
        digest = hashlib.sha256(
            f"{self.config.signature}\n{text}".encode("utf-8")
        ).hexdigest()
        return _CACHE_DIR / f"{digest}.json"

    def encode(self, text: str) -> list[float]:
        cache_file = self._cache_path(text) if self.use_cache else None
        if cache_file is not None and cache_file.exists():
            return json.loads(cache_file.read_text(encoding="utf-8"))["embedding"]

        if self._client is None:
            self._client = self._build_client()
        response = self._client.embeddings.create(model=self.config.model, input=text)
        vector = list(response.data[0].embedding)
        self.dim = len(vector)

        if cache_file is not None:
            cache_file.write_text(
                json.dumps({"model": self.config.model, "text": text, "embedding": vector}),
                encoding="utf-8",
            )
        return vector


def build_embedder(kind: str = "openai", *, use_cache: bool = True) -> Embedder:
    """Factory: ``"stub"`` (offline, deterministic) or ``"openai"`` (real).

    The demo defaults to ``"openai"`` (the user's choice); the tests pass
    ``"stub"`` so they never touch the network.
    """
    normalized = kind.strip().lower()
    if normalized in {"stub", "hash", "offline"}:
        return StubEmbedder()
    if normalized in {"openai", "azure", "real"}:
        return OpenAIEmbedder(use_cache=use_cache)
    raise ValueError(f"Unknown embedder kind {kind!r}. Use 'stub' or 'openai'.")
