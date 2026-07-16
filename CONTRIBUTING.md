# Contributing to Stigmergic

Thanks for your interest. Stigmergic is an experimental research project, and
new "organs for the organism" are welcome — new backends, jurors, castes,
observability, and benchmark adapters.

## Ground rules (the load-bearing ones)

These are the conventions that keep the framework coherent; a PR that breaks them
will be asked to change:

1. **No ant calls another ant.** Castes coordinate *only* by reading and mutating
   the shared ground. If you find yourself importing one caste into another, the
   design wants a new status/trail instead.
2. **The core stays light.** `import stigmergic` must never pull `torch`,
   `transformers`, or a database driver. Heavy paths are lazy-imported and gated
   behind extras. A test asserts this import hygiene — keep it green.
3. **No generative component produces an irreversible effect directly.** A model
   may *propose*; only an authorized executor commits, after consensus (and,
   where configured, a human).
4. **English for all code, comments, and docs.**
5. **Honesty first.** Don't claim what isn't measured. Benchmarks label
   reproducible (torch-free) numbers separately from model/provider-dependent ones.

## Development setup

Requires **Python 3.10+**.

```bash
# Clone, then install the core in editable mode
pip install -e ".[dev]"

# Optional extras, as needed:
pip install -e ".[postgres]"    # PostgresGround (needs a server to run its tests)
pip install -e ".[cognition]"   # real NLI jurors (torch + transformers)
pip install -e ".[benchmark]"   # the real-LLM benchmark juror
pip install -e ".[baselines]"   # Guardrails AI / NeMo Guardrails baselines
```

## Running the checks

```bash
# Tests — fully deterministic, torch-free, sub-second
python -m pytest -q

# Lint (configured to max-line-length = 300 in .flake8)
python -m flake8 src benchmarks demos tests examples

# The examples and the end-to-end demo (each exits 0)
python examples/01_hello_stigmergy.py
python demos/servicenow_hr/demo.py

# The reproducible benchmark headline
python benchmarks/injection_capture/run.py
```

CI runs the same suite across Python 3.10–3.13 (plus a Windows leg and, in
dedicated jobs, Postgres and the cognition extras). A PR must be green.

## How to add …

- **A ground backend** — implement `AbstractGround` (see `PostgresGround`) and
  route it in `core/backends/__init__.py`. Mirror the reliability primitives
  (lease, CAS, idempotency) and emit an event after every mutation.
- **A consensus juror** — implement the `NLIJudge` protocol
  (`classify(premise, hypothesis) -> (label, score)`); drop it into a
  `SemanticRaft` panel.
- **A caste** — subclass `ConsumerAnt` (implement `metabolize`) or `ProducerAnt`
  (implement `secrete`). React to a `Status`, never to another ant.
- **A benchmark baseline** — implement the `Defense` protocol
  (`run(item) -> DefenseResult`); see `benchmarks/injection_capture/baselines.py`.

## Pull requests

- Keep changes focused; add or update tests for anything you change.
- Update `CHANGELOG.md` under `[Unreleased]`.
- Describe the *why*, not just the *what*, in the PR body.
- By contributing, you agree your work is licensed under Apache-2.0.

See [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for community expectations.
