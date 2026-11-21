# Data

## Layout

```text
data/
    raw/brugada-huca/1.0.0/
    brugada_huca/
        images/
        labels/
        vlm/
        examples/
    simulator_qrs/
        images/
        labels/
        vlm/
        examples/
        metadata/
    synthetic/
```

`.gitkeep` files preserve empty structural directories. Patient and class subdirectories are generated on demand. Git tracks the layout and documentation, not dataset contents. `tests/fixtures/` is reserved for small synthetic fixtures.

## Clinical source

Obtain [Brugada-HUCA 1.0.0](https://physionet.org/content/brugada-huca/1.0.0/) from PhysioNet and follow its terms and citation requirements. Place the original `metadata.csv` and WFDB records under `data/raw/brugada-huca/1.0.0/`, preserving their paths. The builder does not download or change the source data.

```bash
uv run --extra real-data ecg-few build-huca
```

The builder reads numerical ECG recordings, extracts a central beat of V1 around an R peak, and renders that tracing as an image. The model receives this image; the held-out evaluation unit remains the patient. The input does not contain the full recording, other leads, or the patient's complete clinical assessment. Its default filtering excludes the atypical label and borderline positive records. Review the flags before changing the cohort definition.

The released source metadata and the labels used by the preserved final campaign differ. Rebuilding from the public source is a new provisional dataset; it does not reproduce or replace the final clinical labels. Keep provenance for any reviewed label changes.

## Generated files

| File | Contents |
| --- | --- |
| `labels/all_labels.csv` | Image paths, patient grouping, lead, and available labels |
| `labels/loocv_folds.jsonl` | Held-out patients and context/validation selections |
| `labels/label_schema.json` | Label definitions |
| `labels/dataset_summary.json` | Aggregate dataset description |
| `labels/excluded_patients.csv` | Exclusions and reasons; local only |
| `vlm/all_records.jsonl` | VLM input records; local only |

The QRS/ST simulator generates rendered beats with explicit morphological labels. Clinical records contain the patient-level clinical reference and may lack morphological annotations. RBBB, ST elevation, and T-wave inversion can coexist in either domain; the difference is what was annotated. Missing finding annotations must not be converted to negative labels or treated as proof that the finding is absent.

The experiments use rendered tracings. Photographs or scans of ECG documents are a separate input domain whose capture conditions are not validated by these results.

## Rebuilding

Builders accept an empty directory containing only layout files. Existing generated files require `--overwrite`. Rebuilds preserve `.gitkeep`, `.gitignore`, and `README.md`, and reject raw-data paths, final-result paths, symbolic links, or unrelated files at the dataset root.

Do not publish clinical records, patient identifiers, fold manifests, per-patient predictions, or clinical previews. Share reviewed aggregates only. Large public assets may use a separate release or data repository when their terms permit redistribution.
