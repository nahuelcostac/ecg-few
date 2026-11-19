# ECG Few-Shot Learning

Patient-level ECG image experiments with convolutional neural networks and vision-language models.

The project supports three workflows:

- **Morphological CNN:** predict RBBB, ST elevation, and T-wave inversion, then apply a shared Brugada rule.
- **Clinical CNN:** learn the dataset's binary clinical reference directly.
- **VLM evaluation:** compare zero-shot and in-context predictions using the same patient-level fold plans.

This is research software, not a diagnostic tool.

## Requirements

Python 3.12 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/). The quick start runs on CPU without clinical data, model downloads, or API credentials.

## Installation

```bash
git clone https://github.com/nahuelcostac/ecg-few.git
cd ecg-few
uv sync --locked
uv run ecg-few --help
```

Install the extras needed for your workflow:

```bash
uv sync --locked --extra cnn --extra real-data
uv sync --locked --extra vlm-local
```

`cnn` adds PyTorch and torchvision; `real-data` adds WFDB; `vlm-local` adds the local Hugging Face backend. Each `uv sync` installs exactly the selected extras: include all extras you want to keep. GPU builds require a compatible PyTorch installation; see [development](docs/development.md).

The distribution is named `ecg-vlm`; its import package is `ecg_few` and its CLI is `ecg-few`.

## Quick start

Generate five synthetic records and exercise the VLM pipeline with fixed mock predictions:

```bash
uv run ecg-few build-synthetic --config configs/smoke_dataset.toml
uv run ecg-few vlm --config configs/smoke_vlm.toml
```

The dataset is written to `data/synthetic/smoke/` and the provisional outputs to `outputs/work/smoke_vlm/`. The predictions test the execution pipeline; they do not measure model performance. Repeating the VLM command resumes the same run. Rebuilding an existing dataset requires `--overwrite`.

## Data preparation

Synthetic data can be generated locally:

```bash
uv run ecg-few build-synthetic --config configs/simulator.toml
```

To prepare clinical data, obtain [Brugada-HUCA 1.0.0](https://physionet.org/content/brugada-huca/1.0.0/), retain its original directory layout under `data/raw/brugada-huca/1.0.0/`, and run:

```bash
uv run --extra real-data ecg-few build-huca
```

Clinical records, processed images, patient-level manifests, predictions, and model weights stay outside Git. See [data layout and provenance](docs/data.md).

## Running experiments

```bash
uv run --extra cnn ecg-few cnn --config configs/cnn_morphology.toml
uv run --extra cnn ecg-few cnn-clinical --config configs/cnn_clinical.toml
uv run ecg-few vlm --config configs/vlm.toml --model MODEL_ID
```

Remote VLM inference requires `VLM_API_BASE` and, when the endpoint requires authentication, `VLM_API_KEY`. Set these in your shell. `.env.example` lists the supported environment variables; `.env` files are not loaded automatically. The configured endpoint receives the selected images.

Use `--vlm-runtime local_gpu` with the `vlm-local` extra to run locally. Model access may require `HF_TOKEN`. Check model compatibility and GPU memory before starting a full campaign.

Every command supports `--help`. Experiment and dataset commands accept a TOML `[experiment]` table through `--config`; CLI options override its values. Paths are relative to the current working directory.

## Outputs and reproducibility

- New experiments belong in `outputs/work/<campaign>/` and are marked provisional.
- `run_manifest.json` records configuration, input hashes, prompt hashes, source hashes, and environment details. Resume rejects incompatible runs.
- `outputs/final_results_tfg/` is the preserved canonical final campaign. Its archives and granular outputs remain local.
- `reports/loocv/` is the versioned aggregate mirror. Normal experiment commands cannot write to it or to the canonical campaign.

Final provenance is documented in the [canonical metadata](outputs/final_results_tfg/README.md). No command promotes provisional results automatically. See [experiment protocols](docs/experiments.md) for selection rules and limitations.

## Repository structure

| Path | Purpose |
| --- | --- |
| `src/ecg_few/datasets/` | Clinical and synthetic dataset construction |
| `src/ecg_few/protocols/` | Patient-level folds and sample selection |
| `src/ecg_few/cnn/` | Models, training, thresholds, and CNN pipelines |
| `src/ecg_few/vlm/` | Context selection, messages, inference, and validation |
| `src/ecg_few/prompts/templates/` | Packaged Markdown prompts |
| `src/ecg_few/evaluation/` | Metrics and campaign audits |
| `src/ecg_few/reporting/` | Aggregation, comparisons, and plots |
| `src/ecg_few/experiment_io/` | Storage, output protection, and provenance |
| `configs/` | Reproducible experiment presets |
| `scripts/` | Compatibility entry points and shell wrappers |
| `tests/unit/`, `tests/integration/` | Synthetic and offline tests |

See [architecture](docs/architecture.md) for module boundaries.

## Development

```bash
uv sync --locked --extra cnn --extra real-data
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv build
```

CI checks the code, exercises CPU workflows, and installs the wheel outside the source tree. See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow.

## License and citation

No open-source license has been granted; see [LICENSE](LICENSE). Dataset and model terms apply separately. Software citation metadata is provided in [CITATION.cff](CITATION.cff).
