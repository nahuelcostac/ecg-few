# Experiments

## Research question and input

The study asks whether in-context learning can use a small set of labelled ECG examples to guide classification while Gemma's weights remain fixed. ResNet18 supplies the supervised reference, with weights adjusted using the selected training examples. The evaluation first examines controlled morphology from the QRS/ST simulator, then real HUCA recordings, including synthetic-to-real transfer and classification with clinical examples.

The input is a rendered tracing of a central beat of V1. The numerical ECG is the source from which the clinical beat is extracted; the model receives its image. A patient-level clinical reference can depend on information absent from that beat, so concordance with the reference must not be described as a complete clinical assessment.

## Labelled examples and patient separation

K denotes the number of labelled examples used for a fold: demonstrations for Gemma or training examples for ResNet18. Demonstrations supply the requested labels and visual references at inference time without updating Gemma's weights. In the preserved clinical comparison, the CNN additionally uses eight validation patients, four per class, for early stopping and threshold selection. These are additional to K and are separate from training and test patients. Gemma does not use this extra validation set.

All leads from a patient belong to the same held-out fold. CNN and VLM workflows consume the shared fold plan. CNN training and validation sets are disjoint and exclude the held-out patient. VLM context excludes the held-out patient; the VLM does not fit or tune on a validation set. Domain adaptation excludes held-out and validation patients from the target pool.

The simulated task defines RBBB, ST elevation, and T-wave inversion explicitly. Both morphological model branches predict these findings before applying the shared Brugada rule. Clinical branches predict the dataset's binary clinical reference. Findings can coexist in simulated and real beats; the distinction is in the available annotation and prediction target. Missing clinical finding annotations must not become negative labels.

`estandar` is the retained condition identifier for standard context selection. `balanced` samples both classes. Zero-shot and control conditions are available in the VLM CLI.

The clinical comparison therefore uses K plus eight labelled patients for the CNN and K for Gemma. For a new dataset, inspect the generated fold plan and its validation configuration rather than assuming that this historical setting holds. Repeated seeds evaluate the same patients; variation across seeds is not population uncertainty.

## Configuration

Presets in `configs/` contain an `[experiment]` table. Keys match CLI destinations, using underscores. Lists such as `k_values` and `seeds` become comma-separated CLI values. Unknown keys and invalid choices are rejected. Explicit CLI options override TOML values.

```bash
uv run --extra cnn ecg-few cnn --config configs/cnn_morphology.toml --epochs 5 --output-root outputs/work/cnn_trial
```

The report directory defaults to `<output-root>/reports`. Use a new output directory when changing the configuration, data, prompts, or code.

## Resume and provenance

`run_manifest.json` identifies a provisional campaign using the effective configuration, input content hashes, prompt hashes, and Python source hash. It also records the Python version and installed backend versions. Credentials are omitted.

Resume requires the same campaign identity. Increasing `--limit-folds` is allowed to extend an interrupted run. A nonempty directory without a matching manifest is rejected. `--no-resume` requires a fresh destination.

A mock VLM run is identified by `dry_run_predictions` in its configuration. Its outputs exercise serialization and aggregation; they are not experimental model results.

## Final results

`outputs/final_results_tfg/` remains the sole canonical campaign. It and the aggregate mirror `reports/loocv/` stay local and outside Git. Their content and source archives are preserved. Selected public images live in `docs/assets/results/`; [results provenance](results.md) traces README values to exact canonical CSVs. Experiment, comparison, and audit commands write provisional outputs under `outputs/work/`; they do not promote or synchronize results.

## Campaign recipes

The README lists the primary presets. These additional commands use the same package implementation as the shell launchers in `scripts/run/`. Run from the repository root after installing the relevant extras. Each output destination below is provisional.

For CORAL adaptation from synthetic training data to the HUCA target domain:

```bash
uv run --no-sync ecg-few cnn --config configs/cnn_morphology.toml --domain-adaptation coral --k-values 16,32 --output-root outputs/work/cnn_coral_trial
```

Use `--domain-adaptation mmd` or `--domain-adaptation dann` for the other adaptation losses. For the SSL pretraining control, keep `--domain-adaptation none` and set `--ssl-pretrain-epochs 3`. Held-out and validation patients remain excluded from the target adaptation pool.

Validate a VLM setup before inference:

```bash
uv run --no-sync ecg-few validate-vlm --dataset-root data/simulator_qrs --context-dataset-root data/simulator_qrs --task morphology --models MODEL_ID --conditions zero_shot,estandar,balanced --output outputs/work/preflight/simulator.json
```

Compare completed morphology campaigns using the same K values, seeds, and fold/context selections:

```bash
uv run --no-sync ecg-few compare --cnn-summary outputs/work/cnn_simulator_qrs_loocv/reports/cnn_summary_by_seed.csv --vlm-summary outputs/work/vlm_simulator_qrs_loocv/reports/vlm_summary_by_seed.csv --vlm-condition balanced --vlm-model MODEL_ID --output-dir outputs/work/simulator_comparison/reports
```

The comparison command checks fold/context alignment and rejects incompatible runs. The selected VLM context condition must match the CNN support selections. Clinical and morphological targets must be reported separately. Use `ecg-few audit --help` and `ecg-few audit-clinical --help` to select provisional campaign inputs for their respective audits.

## Limits

Clinical references and synthetic morphology are different targets. Transfer experiments can reflect both visual differences and differences in label meaning; a performance difference does not isolate either mechanism. Real clinical examples may help ICL because they share the query's label definition and tracing variation, but this remains an interpretation. Adjusting all ResNet18 layers with few examples creates an overfitting risk, not a demonstrated explanation for every error.

The study evaluates rendered central V1 beats. It does not establish performance on photographs or scans, a complete ECG examination, diagnostic validity, or external generalization. Verify dataset provenance, model revisions, hardware, and backend settings when comparing runs. Fixed seeds alone do not guarantee identical results across devices or software versions.

Future ICL experiments can increase the number or diversity of labelled demonstrations. Updating Gemma's weights would instead require a separate fine-tuning experiment and a new evaluation protocol; it is not part of the fixed-weight results summarized here.
