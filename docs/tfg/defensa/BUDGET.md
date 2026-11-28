# Approximate research budget

The support slide values the work and resources at **EUR 9,715 (approximately EUR 9,700)**. This is an approximate economic allocation, not an invoice. Personnel hours, depreciation, reserved equipment time, electricity and rounding are explicit assumptions. The author reports that Gemma ran on a borrowed, self-managed cluster with two A100 GPUs and the CNN ran on the author's own RTX 3060 PC. The former generic cloud-rental rates have been replaced.

## Personnel, equipment and total

The structure follows section 6.2, printed page 72, of Nahuel A. Costa Cortez's 2019 thesis, *Preprocesado y tratamiento inteligente de electrocardiogramas intracardíacos*. Its local source hash remains in [the calculation record](data/budget_estimate.json). Historical prices from that thesis are not reused.

| Item | Calculation or allowance | EUR |
| --- | --- | ---: |
| Research and development | 300 h × EUR 20/h | 6,000 |
| Academic supervision | 40 h × EUR 50/h | 2,000 |
| Clinical advice and data review | 10 h × EUR 60/h | 600 |
| Borrowed two-A100 cluster: depreciation allocation | 20 node-hours × EUR 2.25/h | 45 |
| Borrowed cluster: electricity and cooling | 20 h × 1 kW × 1.2 × EUR 0.25/kWh | 6 |
| RTX 3060 PC: electricity | 10 h × 0.30 kW × EUR 0.25/kWh = EUR 0.75, rounded | 1 |
| Existing workstation depreciation | EUR 1,200 / 48 months × 6 months | 150 |
| Storage and backups | 6 months × EUR 5/month | 30 |
| **Subtotal** | | **8,832** |
| Contingency | 10%, rounded to whole euros | 883 |
| **Estimated total** | | **9,715** |

Personnel rates are hypothetical loaded institutional costs, not official pay scales or recorded timesheets. Research includes literature review, the QRS/ST simulator, software, experiment operation, analysis, thesis and defense. Clinical advice is an allowance, not a measured cost of collecting or labeling the cohort. The existing workstation allocation includes the RTX 3060, so no separate RTX depreciation is charged in compute. Software licenses add zero euros.

## Self-managed compute assumptions

A100 memory capacity, form factor, acquisition price, useful life and scheduler accounting have not been confirmed. The calculation therefore uses a documented reference scenario for the complete two-GPU node, rather than claiming to know the configuration or expenditure of the borrowed cluster.

For replacement-cost context, the [HydraGPU refurbished A100 80 GB listing](https://www.hydragpu.com/nvidia-a100-80gb-refurbished.html) was indexed at EUR 7,479 on 28 November 2026. The direct product page returned HTTP 403 during retrieval, so this is limited to its published search listing. It is not the actual acquisition cost of the cards used. The scenario rounds this to EUR 7,500 per GPU and assumes EUR 3,000 for the host and other hardware. The resulting EUR 18,000 node is allocated over four years and 2,000 productive node-hours per year: EUR 2.25 per node-hour, or EUR 45 for 20 hours. Both GPUs are included in this rate. These are 20 node-hours, corresponding to 40 available GPU-hours, not 20 hours billed twice.

This is an **imputed resource cost**, not a purchase or rental payment by the student. The number of productive hours per year strongly affects depreciation per hour. For example, using 4,000 rather than 2,000 hours per year halves that component. Actual acquisition records and utilization would be needed for institutional cost accounting.

The [NVIDIA A100 data sheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf) lists maximum TDPs of 250 W for 40 GB PCIe, 300 W for 80 GB PCIe, and 400 W for SXM variants. As a conservative allowance, the calculation uses two 400 W GPUs plus 200 W for the host, giving 1 kW for the node, then an assumed 1.2 cooling/overhead factor. At an assumed EUR 0.25/kWh for 20 hours, electricity is EUR 6.

[NVIDIA's RTX 3060 specifications](https://www.nvidia.com/en-au/geforce/graphics-cards/30-series/rtx-3060-3060ti/) list 170 W graphics-card power. Adding an assumed 130 W host load gives 0.30 kW for the complete PC. Ten hours at EUR 0.25/kWh gives EUR 0.75, rounded to EUR 1 on the slide. Thus the combined estimated **electricity cost is approximately EUR 7**, while depreciation allocated to the borrowed cluster adds EUR 45.

TDP is a device specification, not measured power consumption. Cooling, host load, useful life, utilization, reserved hours and the electricity tariff are assumptions. The EUR 0.25/kWh rate is not a verified university or household contract. [Eurostat's electricity price statistics](https://ec.europa.eu/eurostat/statistics-explained/index.php?title=Electricity_price_statistics) provide market context, including an EU household mean of EUR 0.2832/kWh in H1 2026, rather than a quote for these machines. No extra electricity charge is added elsewhere.

## Canonical timing evidence

Only `outputs/final_results_tfg/` supplies experimental timings. No experiment was rerun and no canonical result or aggregate mirror was changed. [The calculation record](data/budget_estimate.json) retains the source checksums, aggregate durations and formulas without patient identifiers or predictions.

| Campaign | Recorded time | Accounting |
| --- | ---: | --- |
| Gemma with simulated demonstrations, clinical queries (`loocv/vlm/`) | 4.1805 h | 10,461 queries |
| Gemma with clinical demonstrations (`loocv/vlm_real_context/`) | 2.8313 h | 10,461 queries |
| Gemma on the QRS/ST simulator (`loocv/vlm_simulator_qrs/`) | 1.3280 h | 3,300 queries |
| **Gemma total** | **8.3397 h** | **24,222 queries** |
| Clinical CNN, both conditions | 1.5952 h | 9,510 fold-specific fits |
| CNN on the simulator | 1.1576 h | 1,500 fold-specific fits |
| CNN trained on simulated examples, clinical queries | 0.0130 h | 15 shared fits |
| CNN domain adaptation, including SimCLR | 0.0593 h | 24 shared fits |
| **CNN total** | **2.8251 h** | **11,049 supervised fits plus recorded SimCLR time** |

For Gemma, sum `n_leads * latency_seconds_mean` over the three canonical `vlm_summary_by_seed.csv` reports and divide by 3,600. This is cumulative client request latency, including service overhead, rather than reserved-node wall time, GPU occupation or energy. A self-managed server can still be accessed through an API.

For CNNs, sum `train_seconds` over unique fold keys in the clinical and simulator campaigns. In the shared-training campaigns, verify that `run_config.json` declares shared training and count `train_seconds` and `ssl_pretrain_seconds` once per run, not once per held-out patient. Training counters omit parts of preprocessing, model construction, inference, Grad-CAM and job overhead. The 20 node-hours and 10 PC-hours remain allowances, not measured wall-clock totals.

Existing supplied clinical data are assumed. New recruitment or labeling, prospective/multicentre validation, production deployment, original foundation-model pretraining, institutional overhead allocations and additional indirect taxes are outside this estimate. Hardware access was provided, so no cash purchase or rental invoice is claimed.
