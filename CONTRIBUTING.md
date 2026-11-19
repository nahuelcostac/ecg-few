# Contributing

Open an issue for bugs or substantial changes. Include a minimal synthetic example, the command, Python version, and relevant dependency versions. Omit credentials and patient information.

## Workflow

1. Create a focused branch.
2. Install the required extras with `uv sync --locked`.
3. Add or update tests for changed behavior.
4. Run lint, formatting, and the relevant tests.
5. Describe the change, validation, and compatibility impact in the pull request.

Keep reusable logic in `src/ecg_few/`; scripts should delegate to it. Import helpers from their defining modules. Use clear names and short docstrings; add comments only when they explain a non-obvious constraint.

Update documentation and presets when changing commands or configuration. Update `uv.lock` with dependency changes. Keep generated data, model weights, and per-patient outputs outside Git.

Final results and their aggregate mirror are reserved. Use `outputs/work/<campaign>/` for experiments and do not promote results without explicit authorization.
