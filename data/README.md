# Local datasets

Data contents are ignored by Git. Tracked `.gitkeep` files preserve the expected layout.

- `raw/brugada-huca/1.0.0/`: original PhysioNet records and metadata.
- `brugada_huca/`: processed clinical images and manifests.
- `simulator_qrs/`: generated morphology dataset.
- `synthetic/`: small demonstrations and custom synthetic datasets.

See [data preparation](../docs/data.md) for commands, schemas, provenance, and rebuild behavior. Do not commit clinical records or patient-level artifacts.
