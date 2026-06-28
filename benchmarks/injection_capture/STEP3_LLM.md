# Step 3 — Plug in a real LLM juror (turnkey)

Everything is wired and tested. To add a **real LLM** as the third juror on the
heterogeneous panel, you only need to set a few environment variables and add
the `--llm` flag — no code changes. The benchmark stays fully functional without
this; `--llm` is purely opt-in.

The LLM is read through [`providers.py`](providers.py), which fails fast with a
clear message if a required variable is missing, caches every response on disk
under `.cache/llm/` (so re-runs are free and deterministic), and retries with
exponential backoff on transient errors.

## 1. Install the client (one time)

```powershell
python -m pip install "openai>=1.0"
```

(Equivalent to the `[benchmark]` extra. The `openai` package also serves Azure
OpenAI via its `AzureOpenAI` client.)

## 2a. Azure OpenAI (recommended for enterprise)

Set these in the **same PowerShell session** you will run the benchmark from.
Paste your real key and endpoint where shown — do **not** commit them.

```powershell
$env:STIG_LLM_PROVIDER   = "azure"
$env:AZURE_OPENAI_API_KEY = "<paste-your-key>"
$env:AZURE_OPENAI_ENDPOINT = "https://<your-resource>.openai.azure.com"
$env:STIG_LLM_MODEL      = "<your-deployment-name>"   # the *deployment*, not the base model
$env:STIG_LLM_API_VERSION = "2024-06-01"               # optional; this is the default
```

## 2b. OpenAI (or any OpenAI-compatible endpoint)

```powershell
$env:STIG_LLM_PROVIDER = "openai"        # this is the default, so it is optional
$env:OPENAI_API_KEY    = "<paste-your-key>"
$env:STIG_LLM_MODEL    = "gpt-4o-mini"   # optional; this is the default
# $env:OPENAI_BASE_URL = "https://..."   # optional: only for a compatible gateway
```

## 3. Run

```powershell
$env:PYTHONPATH = "src"

# real LLM juror on the panel, torch-free everywhere else:
python benchmarks/injection_capture/run.py --llm

# real LLM juror AND the real cross-encoder NLI juror (Step 1 + Step 3):
python benchmarks/injection_capture/run.py --nli --llm
```

Each run archives `results/torch-free+real-llm.json` (or `results/real-nli+real-llm.json`)
and refreshes [`COMPARISON.md`](COMPARISON.md), so the LLM run lands right next to
the torch-free and Byzantine rows for a side-by-side read. `RESULTS.md` is also
refreshed with the new mode.

## Notes

- **Cost / determinism:** responses are cached by a SHA-256 of the prompt under
  `.cache/llm/` (git-ignored), so a second run costs nothing and reproduces
  exactly. Pass `--no-cache` to force fresh calls.
- **What the LLM is asked:** a single, structured verdict (one label +
  confidence) per proposal — see the system prompt in [`providers.py`](providers.py).
  Its vote joins the `RuleBasedJudge` and the NLI/mock juror in the 2-of-3 panel.
- **Security:** keys live only in your shell environment; nothing is written to
  disk except cached *model outputs*. Never paste a key into a tracked file.
