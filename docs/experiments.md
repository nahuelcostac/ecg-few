# Experiments

## Protocol

All leads from a patient belong to the same held-out fold. CNN and VLM workflows consume the shared fold plan. CNN training and validation sets are disjoint and exclude the held-out patient. VLM context excludes the held-out patient; the VLM does not fit or tune on a validation set. Domain adaptation excludes held-out and validation patients from the target pool.

Morphological CNNs learn three findings and derive the Brugada decision through the shared rule. Clinical CNNs learn the binary clinical reference. VLM tasks support morphology or clinical prediction; the current clinical workflow uses V1.

`estandar` is the retained condition identifier for standard context selection. `balanced` samples both classes. Zero-shot and control conditions are available in the VLM CLI.

The CNN validation set is additional to the K training examples. VLM demonstrations use K examples. Comparisons must disclose this difference. Repeated seeds evaluate the same patients; variation across seeds is not population uncertainty.

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

Clinical references and synthetic morphology are different targets. The code does not establish diagnostic validity or external generalization. Verify dataset provenance, model revisions, hardware, and backend settings when comparing runs. Fixed seeds alone do not guarantee identical results across devices or software versions.
