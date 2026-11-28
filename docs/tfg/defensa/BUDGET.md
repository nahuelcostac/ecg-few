# Approximate research budget

The support slide after the thank-you page values the work at **EUR 9,713 (approximately EUR 9,700)**. It is an illustrative replacement-cost scenario dated 28 November 2026, not a record of expenditure. All personnel hours, rates, equipment allocations and compute reservations are assumptions. The estimate covers the existing retrospective study; a new clinical study would need a separate budget.

## Structure and assumptions

The structure follows section 6.2, printed page 72, of Nahuel A. Costa Cortez's 2019 undergraduate thesis, *Preprocesado y tratamiento inteligente de electrocardiogramas intracardíacos*: personnel, materials and a combined total. The local reference is named `TFG.pdf`; its SHA-256 is recorded in [the calculation record](data/budget_estimate.json). The reference itself is not redistributed, and its historical prices are not reused.

| Item | Assumption | EUR |
| --- | --- | ---: |
| Research and development | 300 h × EUR 20/h | 6,000 |
| Academic supervision by teaching and research staff | 40 h × EUR 50/h | 2,000 |
| Clinical advice and data review | 10 h × EUR 60/h | 600 |
| Gemma inference service | 20 reserved service hours × EUR 2/h | 40 |
| CNN compute, including domain adaptation | 10 reserved GPU hours × EUR 1/h | 10 |
| Workstation depreciation | EUR 1,200 / 48 months × 6 months | 150 |
| Storage and backups | 6 months × EUR 5/month | 30 |
| Subtotal | | 8,830 |
| Contingency | 10% of subtotal | 883 |
| **Estimated total** | | **9,713** |

Research and development includes literature review, the QRS/ST simulator, software, experiment operation, analysis, thesis and defense. Personnel rates represent assumed institutional costs including employer charges; they are not official pay scales or verified timesheets. The clinical allowance is hypothetical retrospective advice and review, not the measured cost of collecting or labeling the source cohort.

Software licenses are valued at zero because the study uses freely available software. GPU electricity is included in the assumed compute rates. Workstation depreciation excludes a separately purchased GPU, avoiding double counting cloud compute. Values are before applicable indirect taxes; VAT is not added to personnel. Institutional overhead, new ECG acquisition or labeling, prospective or multicentre validation, clinical deployment and the original pretraining of Gemma and ImageNet are outside this estimate.

## Recorded times and their limits

Only `outputs/final_results_tfg/` was inspected for experimental timings. No experiment was rerun and no canonical result or aggregate mirror was changed. The public [calculation record](data/budget_estimate.json) contains aggregate durations, source checksums, formulas and assumptions, without patient identifiers or predictions.

| Canonical campaign | Recorded time | Accounting |
| --- | ---: | --- |
| Gemma with simulated demonstrations, clinical queries (`loocv/vlm/`) | 4.1805 h | 10,461 queries |
| Gemma with clinical demonstrations (`loocv/vlm_real_context/`) | 2.8313 h | 10,461 queries |
| Gemma on the QRS/ST simulator (`loocv/vlm_simulator_qrs/`) | 1.3280 h | 3,300 queries |
| **Gemma total** | **8.3397 h** | **24,222 queries** |
| Clinical CNN, both context conditions | 1.5952 h | 9,510 fold-specific fits |
| CNN on the QRS/ST simulator | 1.1576 h | 1,500 fold-specific fits |
| CNN trained on simulated examples, clinical queries | 0.0130 h | 15 shared fits |
| CNN domain adaptation, including SimCLR | 0.0593 h | 24 shared fits |
| **CNN total** | **2.8251 h** | **11,049 supervised fits plus recorded SimCLR time** |

For Gemma, sum `n_leads * latency_seconds_mean` over all rows of `vlm_summary_by_seed.csv` in the three listed canonical aggregate directories, then divide by 3,600. The three files contain 33 runs each, including no-example, standard-context and balanced-context conditions. This is cumulative client request latency, including network and service overhead, not elapsed campaign time or measured GPU occupation. Server hardware, concurrency and token billing are not available, so these values cannot reconstruct actual API expenditure.

For CNNs, read only timing fields from the canonical `runs/*/fold_predictions.jsonl` files. Sum `train_seconds` across unique fold keys for the clinical and simulator campaigns. For `cnn_loocv` and `cnn_domain_adaptation`, the training payload is repeated across predictions from the same shared model. Verify that timings agree within each run and that `run_config.json` declares shared training, then count `train_seconds` and `ssl_pretrain_seconds` once per run. Counting them for every held-out patient would greatly overstate training time. No patient-level contents are published.

The CNN counters exclude parts of preprocessing, model construction, inference, Grad-CAM and job overhead. Complete scheduler wall-clock or GPU accounting is unavailable. The recorded times therefore inform scale, while the **20-hour Gemma and 10-hour CNN allowances remain explicit planning assumptions**, allowing for setup, preprocessing, analysis and modest reruns. They are not inferred GPU-hour measurements or benchmarks on a replacement device. Unrecorded exploratory work cannot be reconstructed from the final campaign.

As a market context check, [Runpod's Cloud GPUs page](https://www.runpod.io/product/cloud-gpus), consulted on 28 November 2026, displayed USD 2.09/h for A100 PCIe 80 GB and USD 0.89/h for RTX 4090 24 GB. The rounded EUR 2/h and EUR 1/h budget assumptions are independent illustrative rates, not exchange-rate conversions, historical invoices or vendor quotes. Neither the hardware nor its throughput is assumed to match the recorded runs. Using these published prices is a cost comparison, not a proposal to upload clinical data to that service.
