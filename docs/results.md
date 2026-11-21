# Results and figure provenance

The only numerical authority is the preserved local campaign in `outputs/final_results_tfg/`. This document traces the README to its aggregate CSVs and records the exact published images. Full reports, clinical records, granular predictions, weights, thesis files, and archives are deliberately outside Git.

## Source archives

The archive names below are relative to the canonical directory. The original ZIPs and SHA-256 records remain there unchanged:

- `31F47D6BB91687F3B0F5B9B7532B6C58474CE34AB004485BB8BC05ABD9CF10DC  source/final_results_tfg.zip`
- `73373A958C25A630425F5AE3B616CBC8D8AC1FE8829C9D44F3681C0CFF3B4561  source/cnn_clinical_brugada_loocv.zip`

A clone contains selected presentation figures and this provenance record; it does not contain the canonical archives or the complete result tables. Reproducing the historical summary requires the preserved canonical campaign. Running the supplied presets creates a new provisional campaign, especially when rebuilding HUCA from public source labels.

## README table

The selection fixes K at 32 instead of choosing the best K for each method. Gemma means `google/gemma-4-E4B-it`. The displayed standard deviations are the recorded run-to-run values; three seeds repeat the same cohort and do not provide population confidence intervals. Synthetic morphology and clinical reference classification are different tasks. CNN validation examples are additional to K; VLM context contains K demonstrations.

### ResNet18 CNN — Synthetic morphology

- Canonical source: `outputs/final_results_tfg/loocv/cnn_simulator_qrs/cnn_summary_by_k.csv`
- Row selection: `k=32`; `n_runs=3`.
- Fields: `balanced_accuracy_mean`, `balanced_accuracy_std`, `f1_mean`, `f1_std`.
- Display: multiply by 100 and round to two decimal places.
- Source SHA-256: `5054a8c7597fa47e6a570cb0f4038dbfed61102f9c8668fa1b3ac236b305c591`.

### Gemma, balanced — Synthetic morphology

- Canonical source: `outputs/final_results_tfg/loocv/vlm_simulator_qrs/vlm_summary_by_model_condition_k.csv`
- Row selection: `k=32, condition=balanced, model=google/gemma-4-E4B-it`; `n_runs=3`.
- Fields: `balanced_accuracy_mean`, `balanced_accuracy_std`, `f1_mean`, `f1_std`.
- Display: multiply by 100 and round to two decimal places.
- Source SHA-256: `d7e7718edd48a9c0fd3fa270a6971c024df764eeb1283d238f9855b8c002692e`.

### ResNet18 CNN, standard — HUCA clinical reference

- Canonical source: `outputs/final_results_tfg/loocv/cnn_clinical/estandar/cnn_clinical_summary_by_k.csv`
- Row selection: `k=32, condition=estandar`; `n_runs=3`.
- Fields: `balanced_accuracy_mean`, `balanced_accuracy_std`, `f1_mean`, `f1_std`.
- Display: multiply by 100 and round to two decimal places.
- Source SHA-256: `6a21c64da7b29827db2e0ba5f4f006f8bede466731fbded1bfb89e2ce64adadd`.

### ResNet18 CNN, balanced — HUCA clinical reference

- Canonical source: `outputs/final_results_tfg/loocv/cnn_clinical/balanced/cnn_clinical_summary_by_k.csv`
- Row selection: `k=32, condition=balanced`; `n_runs=3`.
- Fields: `balanced_accuracy_mean`, `balanced_accuracy_std`, `f1_mean`, `f1_std`.
- Display: multiply by 100 and round to two decimal places.
- Source SHA-256: `eea402cdbfbd39a65f36138444a29301bd7b10ce923b4ebd641e0ff6ab3f97e3`.

### Gemma, balanced real context — HUCA clinical reference

- Canonical source: `outputs/final_results_tfg/loocv/vlm_real_context/vlm_summary_by_model_condition_k.csv`
- Row selection: `k=32, condition=balanced, model=google/gemma-4-E4B-it`; `n_runs=3`.
- Fields: `balanced_accuracy_mean`, `balanced_accuracy_std`, `f1_mean`, `f1_std`.
- Display: multiply by 100 and round to two decimal places.
- Source SHA-256: `a1d2c35da7b86c40a511323f17f7f857ba0d368a2ef3a44da0728c98fa0b14fa`.

## Published assets

All PNGs below were copied byte-for-byte from canonical aggregate reports or the final figure selection. Editable procedure diagrams were copied from the corresponding local draw.io sources. Existing diagram labels retain their original language. Diagram panels are explanatory illustrations; use the CSV selections above for reported metrics.

### `docs/assets/diagrams/experimental_pipeline.png`

- Source: `outputs/final_results_tfg/tfg_figures/results/fig_pipeline_experimental.drawio.png`.
- SHA-256: `e292c5d73dc3cd3056426474bebcd641f00d593eed678c506212e2fdcd05c149`.

### `docs/assets/diagrams/experimental_pipeline.drawio`

- Source: `thesis/thesis/assets/results/fig_pipeline_experimental.drawio`.
- SHA-256: `39b64792b1d950f6d53b8a168e172a4f9f434a8df323906d6dc333204e2964fe`.

### `docs/assets/diagrams/gaussian_generator.png`

- Source: `outputs/final_results_tfg/tfg_figures/results/gaussian_generator.drawio.png`.
- SHA-256: `536b9d63289d76e53abe26a0271cdbfcea3b52f8ab77322521e24f33ec862854`.

### `docs/assets/diagrams/gaussian_generator.drawio`

- Source: `thesis/thesis/assets/results/gaussian_generator.drawio`.
- SHA-256: `78dd064bad8c21edb3c3ffe97e50ba78615de3c49e564a968beb6c80aec1ba70`.

### `docs/assets/diagrams/cnn_pipeline.png`

- Source: `outputs/final_results_tfg/tfg_figures/results/cnn_convolutional_pipeline.drawio.png`.
- SHA-256: `ea1fb639aed53a5427b6dc739aace03a418f1794553c67de2ba58a6237304865`.

### `docs/assets/diagrams/cnn_pipeline.drawio`

- Source: `thesis/thesis/assets/results/cnn_convolutional_pipeline.drawio`.
- SHA-256: `98324735e59202cfe7664af881c1fde7f344ca7475d7fe1a7bc89efb25d2afdb`.

### `docs/assets/diagrams/vlm_context.png`

- Source: `outputs/final_results_tfg/tfg_figures/results/vlm_icl_context_window.drawio.png`.
- SHA-256: `830abd8b99fec00e509b15be71ec76d1270b1edff843120555aa9e2ee77a3dcc`.

### `docs/assets/diagrams/vlm_context.drawio`

- Source: `thesis/thesis/assets/results/vlm_icl_context_window.drawio`.
- SHA-256: `ff5de65e5863179305bfb7b04831cc51d41566a8dffca33e9397c5e9e78968a2`.

### `docs/assets/results/clinical_comparison.png`

- Source: `outputs/final_results_tfg/tfg_figures/results/clinical_cnn_vs_gemma_performance.png`.
- SHA-256: `0551e0dcf88bb1f55e447d68088356cb8e28cca3926f039a47a1f313b41a785e`.

### `docs/assets/results/synthetic_cnn.png`

- Source: `outputs/final_results_tfg/loocv/cnn_simulator_qrs/balanced_accuracy_by_k.png`.
- SHA-256: `8f024b6e7c0528ef96a904a5ec759f7c7b77cc5f2317ce59ff29bd22d47888e0`.

### `docs/assets/results/synthetic_vlm.png`

- Source: `outputs/final_results_tfg/loocv/vlm_simulator_qrs/balanced_accuracy_by_k.png`.
- SHA-256: `11481f8f4856ef397bfd382f74ddb3dac50634e13eb0945d912eb70aaf199234`.

## Plot sources

- `synthetic_cnn.png`: `outputs/final_results_tfg/loocv/cnn_simulator_qrs/cnn_summary_by_k.csv`; CNN morphology across K.
- `synthetic_vlm.png`: `outputs/final_results_tfg/loocv/vlm_simulator_qrs/vlm_summary_by_model_condition_k.csv`; Gemma standard, balanced, and zero-shot morphology conditions across K.
- `clinical_comparison.png`: `outputs/final_results_tfg/loocv/cnn_clinical/{estandar,balanced}/cnn_clinical_summary_by_k.csv` and `outputs/final_results_tfg/loocv/vlm_real_context/vlm_summary_by_model_condition_k.csv`; clinical CNN plus Gemma with real standard and balanced context across K.

To update the public summary, obtain explicit authorization for any replacement final campaign, read its aggregate CSVs, update the selected figures and this provenance record together, and check the README values against those CSVs. Never select a campaign by file modification time or combine older/provisional outputs with these results.

The archived `tfg_figures/results/comparison_simulator_ba_by_k.png` was not selected: its CNN series differs from the canonical aggregate CSV. The separate canonical CNN and VLM plots above agree with their aggregate sources. The archived figure and the full canonical campaign remain untouched.
