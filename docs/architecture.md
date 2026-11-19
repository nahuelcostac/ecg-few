# Architecture

The installable package lives in `src/ecg_few/`. Command-line dispatch is lazy so optional CNN and local VLM dependencies are loaded only when needed.

## Boundaries

- `datasets`: read source records, render images, and export manifests.
- `simulator`: generate and validate synthetic morphology.
- `protocols`: define records, build patient-level folds, and validate selections.
- `cnn`: define models, loaders, training, inference, thresholds, and explanations.
- `vlm`: select context, construct messages, call a backend, and parse responses.
- `evaluation`: compute metrics and audit provisional campaigns.
- `reporting`: aggregate predictions and render comparisons.
- `experiment_io`: serialize outputs, protect reserved paths, and validate resume identity.
- `cli`: dispatch commands to package modules.

Dataset builders and experiment pipelines coordinate these modules. Models and protocol functions do not parse CLI arguments. Shared storage belongs in `experiment_io`; new code should import from its defining module rather than another pipeline.

## Compatibility

Existing Python script paths delegate to package modules. `ecg_few.loocv` forwards to `ecg_few.protocols.loocv`, and historical pipeline helper imports remain available through compatibility exports. New code should use the package API directly.

Shell wrappers remain available for existing automation. The CLI is the portable entry point on Windows, Linux, and macOS.

## Resources

Default prompts live only in `src/ecg_few/prompts/templates/` and are included in the wheel. `package:system/qrs_huca.md` resolves through `importlib.resources`. Explicit prompt paths override packaged defaults. Legacy `prompts/...` paths fall back to the packaged copy when no local file exists.
