# Agent instructions

These instructions apply throughout the repository.

## Canonical final results

- The only canonical experimental results are in `outputs/final_results_tfg/`.
- The base package was imported from `outputs/final_results_tfg/source/final_results_tfg.zip`. Its SHA-256 is recorded in `outputs/final_results_tfg/SOURCE.sha256`.
- The clinical binary campaign was promoted additively from `outputs/work/cnn_clinical_brugada_loocv/`. Its archive, hash, and provenance are recorded in `source/cnn_clinical_brugada_loocv.zip`, `SOURCE_CNN_CLINICAL.sha256`, and `CNN_CLINICAL_PROMOTION.json` inside the canonical directory.
- `outputs/final_results_tfg/loocv/` contains the final aggregate reports; `runs/` contains full run outputs; `tfg_figures/` contains the final figure selection.
- `reports/loocv/` is a local, unversioned aggregate mirror. If used, it must agree with the canonical reports. It must not be published to Git.
- Figures in `docs/assets/results/`, `thesis/thesis/assets/results/`, thesis text, and README tables are derived presentation artifacts, not primary numerical sources. Public figure and table provenance is documented in `docs/results.md`.

Do not consult, combine, restore, or cite results from older runs, temporary folders, backups, or other output directories. Do not select a campaign by file modification time.

## New experiments

- Do not overwrite, regenerate, or modify `outputs/final_results_tfg/` unless the user explicitly requests replacement of the final campaign.
- Store new experiments separately under `outputs/work/<campaign>/` and mark them provisional.
- Do not promote new results or synchronize them into `reports/loocv/` without explicit authorization.
- Trace every documented numerical result to a canonical CSV or JSON and keep derived artifacts consistent.
- Full run outputs may contain patient identifiers. Do not expose patient-level information; use aggregate reports for shared documentation and analysis.

## Preservation

Never delete source ZIP files or SHA-256 records. An authorized replacement campaign must record provenance and hashes, update the canonical location atomically, synchronize the aggregate mirror, and update these instructions and related READMEs.

## Software maintenance

- Publish only the experiment package, configs, tests, launchers, and supporting Markdown, diagrams, and selected result images. Keep `thesis/`, `output/`, `.github/`, `scripts/thesis/`, legacy Python launchers, datasets, and full result trees local and ignored. Never delete local thesis files, source ZIPs, or hash records when changing Git tracking.
- Keep software documentation, public messages, and new code in English.
- Put reusable code in `src/ecg_few/`; keep script entry points small.
- Preserve patient-level separation and existing experimental semantics during refactors.
- Keep dataset contents, credentials, weights, and granular outputs outside Git. Preserve structural `.gitkeep` files.
- Use synthetic fixtures and temporary directories for tests.
- Run `uv run ruff check .`, `uv run ruff format --check .`, and `uv run pytest -q` after code changes. Use `--no-sync` when preserving a custom local environment.

## Research documentation

- Apply the Nahuel TFG editorial criteria to the README and research documentation: explain the scarcity of confirmed labelled examples, introduce fixed-weight in-context learning before the supervised ResNet18 reference, and connect each result to its condition and source.
- Preserve the official thesis title, use the term QRS/ST simulator, and describe the input as a rendered tracing of a central V1 beat. Keep numerical recordings, rendered tracings, and photographs distinct.
- Define K in the method. State that the preserved clinical CNN comparison uses eight additional validation patients, four per class, whereas Gemma uses K labelled demonstrations without that extra validation set.
- Separate observed results from interpretations about demonstration relevance or overfitting. Contrast clinical ICL with no-example and synthetic-example conditions while disclosing their different annotation targets.
- Keep implementation commands intact when revising prose. Keep editorial comment mappings and thesis revision artifacts local; publish the scientific explanation and aggregate provenance only.
