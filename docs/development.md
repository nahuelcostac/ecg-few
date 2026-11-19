# Development

## Setup

```bash
uv sync --locked --extra cnn --extra real-data
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

The `dev` dependency group is installed by default. Use `--no-dev` for a runtime-only environment. CNN tests skip when PyTorch or torchvision is unavailable. The default suite uses synthetic data and mock VLM predictions; it does not require API credentials or clinical records.

For an existing environment with a custom GPU build, avoid replacing it through an exact sync. Use an isolated environment for lockfile checks, or use `uv sync --locked --inexact --no-install-package torch --no-install-package torchvision` with the required extras and run commands with `--no-sync`. Verify that your installed PyTorch and torchvision versions are compatible.

## Checks

- Unit tests cover configuration, protocol selection, response handling, and storage protection.
- Integration tests build small datasets and exercise CNN/VLM workflows in CPU mode.
- CI checks formatting and lint, runs tests, builds the distribution, and runs the installed wheel from an unrelated directory.

```bash
uv run pytest -q -m 'not integration'
uv run pytest -q -m integration
uv build
```

Test the installed package without adding the repository or `src/` to `PYTHONPATH`. Keep fixtures synthetic and small. Use temporary directories for generated outputs.

## Dependencies

Update `pyproject.toml` and `uv.lock` together. Keep backend-specific dependencies in their extras. Do not change dependency versions while comparing experimental runs.

The package follows [PyPA's src layout](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/), [pytest's integration practices](https://docs.pytest.org/en/stable/explanation/goodpractices.html), and [uv's GitHub Actions workflow](https://docs.astral.sh/uv/guides/integration/github/).
