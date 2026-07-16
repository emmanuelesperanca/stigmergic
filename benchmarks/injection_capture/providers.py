"""Optional real-LLM backing for the benchmark's panel juror.

The benchmark runs fully offline by default. This module is only imported when a
caller passes ``--llm`` to :mod:`run`, and it turns an OpenAI-compatible endpoint
into the ``complete(prompt) -> str`` callable that
:class:`~stigmergic_ai.core.consensus.LLMJudge` expects.

Two providers are supported, both configured purely through environment
variables (never hardcode a key):

**Generic OpenAI-compatible** (OpenAI, local llama.cpp/vLLM/Ollama gateways, ...)::

    STIG_LLM_PROVIDER = openai            # the default
    OPENAI_API_KEY     = sk-...           # required
    OPENAI_BASE_URL    = https://...      # optional (defaults to OpenAI)
    STIG_LLM_MODEL     = gpt-4o-mini      # optional

**Azure OpenAI**::

    STIG_LLM_PROVIDER       = azure
    AZURE_OPENAI_API_KEY    = ...
    AZURE_OPENAI_ENDPOINT   = https://<resource>.openai.azure.com
    STIG_LLM_MODEL          = <your deployment name>
    STIG_LLM_API_VERSION    = 2024-06-01  # optional

Every distinct prompt is cached on disk (``.cache/llm/`` next to this file, git-
ignored) so a re-run is free and deterministic, and so a flaky network never
poisons a published number. The ``openai`` SDK is imported lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from typing import Callable

__all__ = ["LLMConfig", "build_llm_complete"]

_CACHE_DIR = pathlib.Path(__file__).resolve().parent / ".cache" / "llm"


class LLMConfig:
    """Resolved provider settings, read from the environment."""

    def __init__(self) -> None:
        self.provider = os.environ.get("STIG_LLM_PROVIDER", "openai").strip().lower()
        self.model = os.environ.get("STIG_LLM_MODEL", "gpt-4o-mini").strip()
        self.api_version = os.environ.get("STIG_LLM_API_VERSION", "2024-06-01").strip()
        self.temperature = float(os.environ.get("STIG_LLM_TEMPERATURE", "0.0"))
        self.max_tokens = int(os.environ.get("STIG_LLM_MAX_TOKENS", "16"))
        if self.provider == "azure":
            self.api_key = os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
            self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY", "").strip()
            self.endpoint = os.environ.get("OPENAI_BASE_URL", "").strip()

    def require(self) -> "LLMConfig":
        """Fail fast with an actionable message if the config is incomplete."""
        if not self.api_key:
            key = "AZURE_OPENAI_API_KEY" if self.provider == "azure" else "OPENAI_API_KEY"
            raise RuntimeError(
                f"--llm needs a real provider. Set {key} (and see this module's "
                "docstring for the full variable list)."
            )
        if self.provider == "azure" and not self.endpoint:
            raise RuntimeError("Azure provider needs AZURE_OPENAI_ENDPOINT.")
        return self

    @property
    def signature(self) -> str:
        """A short, key-free fingerprint of the config for the cache namespace."""
        return f"{self.provider}:{self.model}:{self.api_version}:{self.temperature}"


def _build_client(config: LLMConfig):
    """Lazily construct the appropriate OpenAI SDK client."""
    try:
        if config.provider == "azure":
            from openai import AzureOpenAI

            return AzureOpenAI(
                api_key=config.api_key,
                azure_endpoint=config.endpoint,
                api_version=config.api_version,
            )
        from openai import OpenAI

        kwargs = {"api_key": config.api_key}
        if config.endpoint:
            kwargs["base_url"] = config.endpoint
        return OpenAI(**kwargs)
    except ImportError as exc:  # pragma: no cover - exercised only without the SDK
        raise ImportError(
            "The real-LLM juror needs the openai SDK. Install it with: "
            'pip install -e ".[benchmark]"  (or: pip install openai)'
        ) from exc


def _cache_path(config: LLMConfig, prompt: str) -> pathlib.Path:
    digest = hashlib.sha256(
        f"{config.signature}\n{prompt}".encode("utf-8")
    ).hexdigest()
    return _CACHE_DIR / f"{digest}.json"


def build_llm_complete(*, use_cache: bool = True) -> Callable[[str], str]:
    """Return a cached ``complete(prompt) -> str`` backed by a real model.

    Raises immediately (before any network call) if the environment is not
    configured, so a misconfigured ``--llm`` run fails clearly rather than
    silently degrading.
    """
    config = LLMConfig().require()
    client = _build_client(config)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Side-channel token/cost accounting the caller can read after the run via
    # ``complete.usage``. Keeping it off the return value preserves the plain
    # ``complete(prompt) -> str`` contract that LLMJudge depends on.
    usage = {
        "model": config.model,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "api_calls": 0,
        "cache_hits": 0,
    }

    def complete(prompt: str) -> str:
        cache_file = _cache_path(config, prompt) if use_cache else None
        if cache_file is not None and cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            usage["prompt_tokens"] += int(cached.get("prompt_tokens", 0) or 0)
            usage["completion_tokens"] += int(cached.get("completion_tokens", 0) or 0)
            usage["cache_hits"] += 1
            return cached["response"]

        text, prompt_tokens, completion_tokens = _call_with_retries(
            client, config, prompt
        )
        usage["prompt_tokens"] += prompt_tokens
        usage["completion_tokens"] += completion_tokens
        usage["api_calls"] += 1

        if cache_file is not None:
            cache_file.write_text(
                json.dumps(
                    {
                        "prompt": prompt,
                        "response": text,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return text

    complete.usage = usage  # type: ignore[attr-defined]
    return complete


def _call_with_retries(
    client, config: LLMConfig, prompt: str, *, attempts: int = 3
) -> tuple[str, int, int]:
    """Call the chat endpoint with a small exponential backoff.

    Returns ``(text, prompt_tokens, completion_tokens)``; token counts are ``0``
    when the provider omits a ``usage`` block.
    """
    last_exc: Exception | None = None
    for attempt in range(attempts):
        try:
            completion = client.chat.completions.create(
                model=config.model,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a strict verifier. Reply with exactly "
                        "one label (ENTAILMENT, CONTRADICTION, or NEUTRAL) and a "
                        "confidence.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            text = completion.choices[0].message.content or ""
            token_usage = getattr(completion, "usage", None)
            prompt_tokens = int(getattr(token_usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(
                getattr(token_usage, "completion_tokens", 0) or 0
            )
            return text, prompt_tokens, completion_tokens
        except Exception as exc:  # noqa: BLE001 - provider SDKs raise many types
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after {attempts} attempts: {last_exc}")
