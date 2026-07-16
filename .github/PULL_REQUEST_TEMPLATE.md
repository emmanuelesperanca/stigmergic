## What & why

<!-- What does this change, and why? Link any related issue (Closes #123). -->

## Ground-rules checklist

- [ ] No ant calls another ant — coordination stays through the shared ground
- [ ] `import stigmergic` stays torch-free / driver-free (import-hygiene test green)
- [ ] Tests added/updated; `python -m pytest -q` is green
- [ ] `python -m flake8 src benchmarks demos tests examples` is clean
- [ ] `CHANGELOG.md` updated under `[Unreleased]` (if user-facing)

## Notes for the reviewer

<!-- Anything worth calling out: trade-offs, follow-ups, benchmark impact. -->
