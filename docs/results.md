# Results and figure provenance

The only numerical authority is the preserved local campaign in `outputs/final_results_tfg/`. This document traces the README to its aggregate CSVs and records the exact published images. Full reports, clinical records, granular predictions, weights, thesis files, and archives are deliberately outside Git.

## Source archives

The archive names below are relative to the canonical directory. The original ZIPs and SHA-256 records remain there unchanged:

- `31F47D6BB91687F3B0F5B9B7532B6C58474CE34AB004485BB8BC05ABD9CF10DC  source/final_results_tfg.zip`
- `73373A958C25A630425F5AE3B616CBC8D8AC1FE8829C9D44F3681C0CFF3B4561  source/cnn_clinical_brugada_loocv.zip`

A clone contains selected presentation figures and this provenance record; it does not contain the canonical archives or the complete result tables. Reproducing the historical summary requires the preserved canonical campaign. Running the supplied presets creates a new provisional campaign, especially when rebuilding HUCA from public source labels.

## README table

The selection fixes K at 32 instead of choosing the best K for each method. Gemma means `google/gemma-4-E4B-it`. The displayed standard deviations are the recorded run-to-run values; three seeds repeat the same cohort and do not provide population confidence intervals. Synthetic morphology and clinical reference classification are different tasks. The preserved clinical CNN protocol uses K training patients plus eight additional validation patients, four per class; Gemma uses K labelled demonstrations. This is a protocol setting, not a performance metric. The validation policy is recorded in the canonical clinical run configurations; the four-per-class selection is implemented in `build_fold_plan` in `src/ecg_few/protocols/loocv.py`.

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

## Clinical error profile and comparison conditions

The clinical discussion uses the same `vlm_real_context/vlm_summary_by_model_condition_k.csv` source and `k=32, condition=balanced, model=google/gemma-4-E4B-it` selection as the main clinical table. Additional fields are `sensitivity_mean` and `specificity_mean`: after multiplication by 100 and rounding, they give **74.14% sensitivity** and **78.11% specificity**. These are cohort results; they do not establish diagnostic validity.

For the no-example clinical contrast, select `k=0, condition=zero_shot, model=google/gemma-4-E4B-it` from that same CSV. Its `sensitivity_mean=0` and aggregate `tp=0` support the statement that Gemma did not identify positive cases in that evaluation. The no-example result must not be described as a K=32 run.

The synthetic-to-HUCA contrast uses two additional aggregate sources:

- Canonical source: `outputs/final_results_tfg/loocv/vlm/vlm_summary_by_model_condition_k.csv`; selection: `k=32, condition=balanced, model=google/gemma-4-E4B-it`; field: `balanced_accuracy_mean`; SHA-256: `e74c4792b78dce090be97624bada83a6ed0e0a14ac058976579df6320d58c403`.
- Canonical source: `outputs/final_results_tfg/loocv/cnn/cnn_summary_by_k.csv`; selection: `k=32`; field: `balanced_accuracy_mean`; SHA-256: `f20b60b014ab0d877e5232c3e548777d6d570862cd44e6248528a9f052c8c74a`.

These branches predict morphological findings from which the shared rule derives a decision, evaluated against the clinical reference on HUCA. The real-example branch directly requests the clinical label. Their comparison therefore changes the annotation target as well as the origin of examples. Sharing the query's label definition and tracing variation is a possible explanation for the clinical ICL result, not an experimentally isolated cause. Overfitting is a risk when all ResNet18 layers are adjusted using few cases; the aggregate results do not establish it as the cause of the observed errors.

## Software reproduction reference

Repository: [nahuelcostac/ecg-few](https://github.com/nahuelcostac/ecg-few). The refactored experiment package was verified at revision [`10f853d`](https://github.com/nahuelcostac/ecg-few/commit/10f853d40c60c2d3aeb4e52f1d38d02f63be52ae). This identifies the checked reproduction entry points, not the software revision that produced the historical archives.

The portable entry points are `ecg-few build-synthetic`, `ecg-few build-huca`, `ecg-few vlm`, `ecg-few cnn`, and `ecg-few cnn-clinical`. Use the [README](../README.md#run-experiments) for installation and presets and the [experiment method](experiments.md) for patient separation and labelled-data use.

## Published assets

Except for the experimental pipeline, CNN, and VLM diagram exports described below, PNGs were copied byte-for-byte from canonical aggregate reports or the final figure selection. Editable procedure diagrams originate from the corresponding local draw.io sources. Green boxes in the public experimental, CNN, and VLM diagrams use the reviewed PDF palette: fill `#D9EAD3` and border `#93C47D`. Existing diagram labels retain their original language. Diagram panels are explanatory illustrations; use the CSV selections above for reported metrics.

### `docs/assets/diagrams/experimental_pipeline.png`

- Source: `docs/assets/diagrams/experimental_pipeline.drawio`.
- Export: draw.io Desktop 29.6.1, PNG, scale 1.4, border 0; all green box fills use `#D9EAD3` with `#93C47D` borders, including embedded CNN and VLM vector panels.
- SHA-256: `0e0ed2d20ad0e9cf595c8862b976303ab1e58f568d13af5f5217a0028479d3e3`.

### `docs/assets/diagrams/experimental_pipeline.drawio`

- Original source: `thesis/thesis/assets/results/fig_pipeline_experimental.drawio`; public palette aligned with the reviewed PDF and its sources in `thesis/revision_nahuel/memoria/assets/results/`.
- Public edits: headings shortened to "A: Entrada" and "D: Evaluación"; green fills and borders normalized inside the embedded vector panels.
- SHA-256: `36221ae52ca886ea8048c8e46dcf8dada339d18d39dd5115fc44ee23166702ea`.

### `docs/tfg/defensa/assets/fig_pipeline_experimental.drawio` and `.pdf`

- Defense-specific layout: `B: VLM/ICL` is above `C: CNN`. The input and evaluation blocks, connecting arrows and embedded explanatory panels are preserved.
- The editable Draw.io source records the revised node positions and headings, and the light-green sample headers embedded in both model panels now match HUCA and evaluation (`#D9EAD3`, border `#93C47D`). The corresponding PDF was derived from the retained defense vector export by exchanging the panel and heading regions; it was rendered through SVG with outlined text and a local Chromium PDF export. No experimental results were regenerated.
- Draw.io SHA-256: `126f1e5fec346df87c039a4291cbe0a6aa473f20ee8d1725d0cee129d06826c1`.
- PDF SHA-256: `07d60a0b0b6d8d08b1e28e6674541ee5ca95982832a01004b74fcea688468ad9`. The defense figure inventory also records this hash.

### Defense presentation crops and Grad-CAM legibility

The defense aligns the simulator and HUCA counts with equal-width vector crops having symmetric artwork bounds. Numbered figure captions credit original work or the published source. The ICL schematic preserves its visible adapted graphic while omitting hidden article text outside the crop. The Grad-CAM teaching slide now uses a native-resolution crop of the first row of figure 15 from [Sundaramoorthy and Karunanidhi (2026)](https://doi.org/10.3389/fphys.2026.1848316), licensed CC BY 4.0. It is explicitly presented as a published illustration of the method, rather than a result of this thesis. The original canonical Grad-CAM example remains preserved locally and its public derivative is unchanged. [Derivative provenance](tfg/defensa/data/presentation_figure_derivatives.json) records crop coordinates, attribution, license and checksums. [Canonical Grad-CAM provenance](tfg/defensa/data/gradcam_procedencia.json) remains available. PowerPoint pages are exported at 3840 × 2160.

### Defense implementation diagram palette

The defense implementation figures `docs/tfg/defensa/assets/vlm_icl_context_window.pdf` and `cnn_convolutional_pipeline.pdf` use the same light-green test-sample fill (`#D9EAD3`) and border (`#93C47D`). Their PDF vector color operators were normalized while preserving all text, page sizes and drawing positions. Current checksums are recorded in the [defense figure inventory](tfg/defensa/inventario_figuras.json).

### `docs/assets/diagrams/gaussian_generator.png`

- Source: `outputs/final_results_tfg/tfg_figures/results/gaussian_generator.drawio.png`.
- SHA-256: `536b9d63289d76e53abe26a0271cdbfcea3b52f8ab77322521e24f33ec862854`.

### `docs/assets/diagrams/gaussian_generator.drawio`

- Source: `thesis/thesis/assets/results/gaussian_generator.drawio`.
- SHA-256: `78dd064bad8c21edb3c3ffe97e50ba78615de3c49e564a968beb6c80aec1ba70`.

### `docs/assets/diagrams/cnn_pipeline.png`

- Source: `docs/assets/diagrams/cnn_pipeline.drawio`.
- Export: draw.io Desktop 29.6.1, PNG, scale 1.4, border 0; all green box fills use `#D9EAD3` with `#93C47D` borders, including embedded CNN and VLM vector panels.
- SHA-256: `2184243c1a1c1b8669bcdc72ea7e89b10281b510b7107fb7a23f82e6795eaa14`.

### `docs/assets/diagrams/cnn_pipeline.drawio`

- Original source: `thesis/thesis/assets/results/cnn_convolutional_pipeline.drawio`; public palette aligned with the reviewed PDF and its sources in `thesis/revision_nahuel/memoria/assets/results/`.
- Public edits: the test-sample box fill and border normalized to the common light-green palette; text, geometry, and signal content preserved.
- SHA-256: `ed3f62731009f68985eeaab6818d3390a7e03aa4fe13230c280ccd14cb159157`.

### `docs/assets/diagrams/vlm_context.png`

- Source: `docs/assets/diagrams/vlm_context.drawio`.
- Export: draw.io Desktop 29.6.1, PNG, scale 1.4, border 0; all green box fills use `#D9EAD3` with `#93C47D` borders, including embedded CNN and VLM vector panels.
- SHA-256: `20a43ee46fef61bfa80881b417d454dfb0b31f9f1dd329b52cd2642a667c4624`.

### `docs/assets/diagrams/vlm_context.drawio`

- Original source: `thesis/thesis/assets/results/vlm_icl_context_window.drawio`; public palette aligned with the reviewed PDF and its sources in `thesis/revision_nahuel/memoria/assets/results/`.
- Public edits: the test-sample box fill and border normalized to the common light-green palette; text, geometry, and signal content preserved.
- SHA-256: `962986524a2753bdad026c93eb223c335f7768896200e21c66ba60b2af191dfe`.

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
