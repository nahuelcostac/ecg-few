# Architecture

The installable package lives in `src/ecg_few/`. Command-line dispatch is lazy so optional CNN and local VLM dependencies are loaded only when needed.

## Boundaries

- `datasets`: read source records, render images, and export manifests.
- `simulator`: generate and validate synthetic morphology.
- `protocols`: define records, build patient-level folds, and validate selections.
- `vlm`: select context, construct messages, call a backend, and parse responses.
- `cnn`: implement the supervised reference, with models, loaders, training, inference, thresholds, and explanations.
- `evaluation`: compute metrics and audit provisional campaigns.
- `reporting`: aggregate predictions and render comparisons.
- `experiment_io`: serialize outputs, protect reserved paths, and validate resume identity.
- `cli`: dispatch commands to package modules.

Dataset builders and experiment pipelines coordinate these modules. Models and protocol functions do not parse CLI arguments. Shared storage belongs in `experiment_io`; new code should import from its defining module rather than another pipeline.

The research narrative starts with ICL: `vlm` supplies labelled examples during inference while model weights remain fixed. The CNN workflows provide the supervised comparison. Both consume rendered ECG images and patient-level selections, while the morphological and clinical branches retain their distinct annotation targets.

The CNN package separates the two prediction targets:

| Path | Responsibility |
| --- | --- |
| `cnn/clinical/` | Binary clinical-reference training, folds, thresholds, and Grad-CAM |
| `cnn/morphology/` | Morphological models, folds, training, thresholds, Grad-CAM, and adaptation |
| `cnn/data.py` | Shared image loading and preprocessing |
| `cnn/device.py` | Shared device selection and random seeds |

The CLI commands remain `cnn` and `cnn-clinical`. Python callers should import `ecg_few.cnn.morphology.pipeline` and `ecg_few.cnn.clinical.pipeline` respectively; earlier flat CNN module paths are no longer distributed.

## Entry points and local files

`ecg-few` and `python -m ecg_few` dispatch to the same package modules. Shell launchers in `scripts/run/` pass arguments to this CLI. Dataset and evaluation logic have a single implementation in the package; the old `scripts/dataset/` and `scripts/eval/` Python forwarding files are no longer distributed. Replace calls to those paths with the corresponding CLI command listed in the README.

`ecg_few.loocv` forwards to `ecg_few.protocols.loocv`, and historical pipeline helper imports remain available through compatibility exports. New code should use the package API directly.

Shell wrappers remain available for existing automation. The CLI is the portable entry point on Windows, Linux, and macOS.

Git contains the package, presets, launchers, tests, software documentation, and selected images. `docs/assets/diagrams/` holds editable draw.io sources and PNG exports; `docs/assets/results/` holds selected aggregate figures. `docs/results.md` records canonical numerical sources and image hashes.

The thesis and defense (`thesis/`, `output/`), publication tooling (`scripts/thesis/` and thesis shell launchers), GitHub configuration (`.github/`), and full result trees are ignored local material. Removing them from Git tracking does not remove local copies. New experiments write to `outputs/work/`; preserved final archives remain local under `outputs/final_results_tfg/`.

## Resources

Default prompts live only in `src/ecg_few/prompts/templates/` and are included in the wheel. `package:system/qrs_huca.md` resolves through `importlib.resources`. Explicit prompt paths override packaged defaults. Legacy `prompts/...` paths fall back to the packaged copy when no local file exists.
