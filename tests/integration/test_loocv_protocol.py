from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from ecg_few.datasets.huca import (
    BrugadaRecord,
    ExcludedRecord,
    build_dataset,
    detect_central_r_peak,
    extract_beat_window,
    read_metadata,
    read_metadata_with_exclusions,
    render_single_beat,
)
from ecg_few.datasets.synthetic import build_dataset as build_simulator_dataset
from ecg_few.protocols.loocv import (
    BrugadaImageRow,
    build_fold_plan,
    patients_from_rows,
    read_jsonl,
    read_manifest,
    selection_for,
    validate_fold_plan,
    write_jsonl,
    write_manifest,
)
from ecg_few.reporting.comparison import verify_alignment
from ecg_few.vlm.validation import validate_setup


class FakeBrugadaRecord:
    def __init__(self) -> None:
        self.fs = 100
        self.sig_name = ["I", "II", "V1", "V2", "V3"]
        signals = np.zeros((1200, len(self.sig_name)), dtype=np.float32)
        for peak in (250, 600, 950):
            for lead_index in range(len(self.sig_name)):
                signals[peak, lead_index] = 1.0 + lead_index * 0.1
                signals[peak + 4, lead_index] = 0.25
        self.p_signal = signals


def test_metadata_excludes_atypical_and_borderline_by_default(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    (raw_root / "metadata.csv").write_text(
        "patient_id,basal_pattern,sudden_death,brugada\n"
        "100,0,0,0\n"
        "101,1,0,1\n"
        "102,0,0,2\n"
        "103,0,0,1\n",
        encoding="utf-8",
    )

    records, excluded = read_metadata_with_exclusions(raw_root)

    assert [record.patient_id for record in records] == ["100", "103"]
    assert [record.patient_id for record in read_metadata(raw_root)] == ["100", "103"]
    assert {record.reason for record in excluded} == {
        "borderline_basal_pattern_1_brugada_1",
        "brugada_2_atypical",
    }


def test_detect_central_r_peak_and_extracts_fixed_window(monkeypatch) -> None:
    record = FakeBrugadaRecord()
    leads = {
        lead_name: record.p_signal[:, index] for index, lead_name in enumerate(record.sig_name)
    }
    monkeypatch.setattr(
        "ecg_few.datasets.huca._detect_r_peaks",
        lambda _signal, *, fs, detector: np.asarray([250, 600, 950], dtype=np.int64),
    )

    extraction = detect_central_r_peak(record, leads)
    beat = extract_beat_window(
        leads["V2"],
        fs=record.fs,
        r_peak_sample=extraction.r_peak_sample,
        pre_r_ms=300,
        post_r_ms=600,
    )

    assert extraction.r_peak_sample == 600
    assert extraction.r_peak_lead == "V1"
    assert beat.shape[0] == 90
    assert int(np.argmax(beat)) == 30


def test_render_single_beat_writes_image(tmp_path: Path) -> None:
    output = tmp_path / "v2.png"
    beat = np.zeros(90, dtype=np.float32)
    beat[30] = 1.0

    render_single_beat(beat, output, fs=100, dpi=80)

    assert output.exists()
    assert output.stat().st_size > 0


def test_build_dataset_writes_single_manifest_and_loocv_folds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        BrugadaRecord("100", 0, 0, 0),
        BrugadaRecord("101", 0, 0, 1),
        BrugadaRecord("102", 0, 0, 0),
        BrugadaRecord("103", 0, 0, 1),
    ]
    excluded = [ExcludedRecord("104", 1, 0, 1, "borderline_basal_pattern_1_brugada_1")]
    monkeypatch.setattr(
        "ecg_few.datasets.huca.read_wfdb_record",
        lambda _raw_root, _patient_id: FakeBrugadaRecord(),
    )
    rows = build_dataset(
        records=records,
        excluded=excluded,
        raw_root=tmp_path / "raw",
        outdir=tmp_path / "out",
        dpi=80,
        pre_r_ms=300,
        post_r_ms=600,
        k_values=[2],
        seeds=[42],
        val_per_class=1,
        include_borderline_positive=False,
    )

    loaded = read_manifest(tmp_path / "out" / "labels" / "all_labels.csv")
    folds = read_jsonl(tmp_path / "out" / "labels" / "loocv_folds.jsonl")

    assert len(rows) == 4
    assert len(loaded) == 4
    assert len(folds) == 4
    assert not (tmp_path / "out" / "labels" / "train_labels.csv").exists()
    assert (tmp_path / "out" / "labels" / "excluded_patients.csv").exists()
    validate_fold_plan(folds, loaded, k_values=[2], seeds=[42])


def test_build_simulator_dataset_writes_loocv_manifest(tmp_path: Path) -> None:
    rows = build_simulator_dataset(
        outdir=tmp_path / "simulator",
        patients_per_source_family=1,
        base_seed=2026,
        fs=100,
        duration_ms=900,
        dpi=80,
        k_values=[1],
        seeds=[42],
        val_per_class=0,
        max_patient_attempts=200,
        overwrite=True,
    )
    loaded = read_manifest(tmp_path / "simulator" / "labels" / "all_labels.csv")
    folds = read_jsonl(tmp_path / "simulator" / "labels" / "loocv_folds.jsonl")

    assert len(rows) == 5
    assert len(loaded) == 5
    assert len(folds) == 5
    assert {row.lead for row in loaded} == {"V1"}
    assert (tmp_path / "simulator" / "labels" / "dataset_summary.json").exists()
    assert any(row.derived_brugada == 1 for row in loaded)
    validate_fold_plan(folds, loaded, k_values=[1], seeds=[42])


def test_fold_plan_has_no_patient_leakage_and_shared_context() -> None:
    rows = _mini_rows(Path("data"))
    folds = build_fold_plan(rows, k_values=[2], seeds=[42], val_per_class=1)

    validate_fold_plan(folds, rows, k_values=[2], seeds=[42])
    first = folds[0]
    selection = selection_for(first, k=2, seed=42)

    assert first["test_patient_id"] not in selection["context_patient_ids"]
    assert first["test_patient_id"] not in selection["validation_patient_ids"]
    assert not set(selection["context_patient_ids"]).intersection(
        selection["validation_patient_ids"]
    )
    assert selection == selection_for(first, k=2, seed=42)


def test_validate_vlm_loocv_preflight_requires_remote_api_base(tmp_path: Path) -> None:
    dataset_root = _write_mini_dataset(tmp_path)

    missing = validate_setup(_vlm_args(dataset_root, api_base=""))
    ready = validate_setup(_vlm_args(dataset_root, api_base="http://nahuel:8000/v1"))

    assert missing["ok"] is False
    assert any("VLM_API_BASE" in error for error in missing["errors"])
    assert ready["ok"] is True
    assert ready["planned_patient_predictions"] == 4
    assert ready["planned_image_requests"] == 12
    assert ready["network_calls"] == 0
    assert ready["model_loads"] == 0


def test_compare_alignment_rejects_mismatched_context(tmp_path: Path) -> None:
    cnn_pred = tmp_path / "cnn.csv"
    vlm_pred = tmp_path / "vlm.csv"
    _write_prediction_csv(cnn_pred, context="101|102")
    _write_prediction_csv(vlm_pred, context="101|103")

    cnn_summary = [
        {"k": "2", "seed": "42", "balanced_accuracy": "0.5", "predictions": str(cnn_pred)}
    ]
    vlm_summary = [
        {"k": "2", "seed": "42", "balanced_accuracy": "0.5", "predictions": str(vlm_pred)}
    ]

    with pytest.raises(ValueError, match="mismatch"):
        verify_alignment(cnn_summary, vlm_summary)


def test_cnn_and_vlm_smoke_share_loocv_outputs(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from ecg_few.cnn.pipeline import run_with_args as run_cnn
    from ecg_few.vlm.pipeline import run_with_args as run_vlm

    dataset_root = _write_mini_dataset(tmp_path)
    cnn_report = tmp_path / "reports" / "cnn"
    vlm_report = tmp_path / "reports" / "vlm"

    run_cnn(
        _cnn_args(
            dataset_root=dataset_root,
            output_root=tmp_path / "outputs" / "cnn",
            report_dir=cnn_report,
            limit_folds=2,
        )
    )
    run_vlm(
        Namespace(
            dataset_root=dataset_root,
            folds=dataset_root / "labels" / "loocv_folds.jsonl",
            output_root=tmp_path / "outputs" / "vlm",
            report_dir=vlm_report,
            k_values="2",
            seeds="42",
            vlm_runtime="remote_api",
            model="fake/medgemma",
            api_base="http://nahuel:8000/v1",
            api_key="",
            system_prompt_file="prompts/system/qrs_huca.md",
            prompt_file="prompts/qrs/right_precordial_morphology.md",
            response_format="json_schema",
            temperature=0.0,
            max_output_tokens=32,
            timeout=1.0,
            local_batch_size=1,
            local_device="cpu",
            local_dtype="float32",
            local_attn_implementation="sdpa",
            local_offload_dir=tmp_path / "offload",
            limit_folds=2,
            resume=False,
            dry_run_predictions="expected",
            context_dataset_root=Path(""),
        )
    )

    assert (cnn_report / "cnn_summary_by_seed.csv").exists()
    assert (vlm_report / "vlm_summary_by_seed.csv").exists()
    verify_alignment(
        _read_csv(cnn_report / "cnn_summary_by_seed.csv"),
        _read_csv(vlm_report / "vlm_summary_by_seed.csv"),
    )


def test_vlm_campaign_writes_multicondition_artifacts(tmp_path: Path) -> None:
    from ecg_few.vlm.pipeline import run_with_args as run_vlm

    dataset_root = _write_mini_dataset(tmp_path)
    report_dir = tmp_path / "reports" / "vlm_campaign"

    run_vlm(
        Namespace(
            dataset_root=dataset_root,
            folds=dataset_root / "labels" / "loocv_folds.jsonl",
            output_root=tmp_path / "outputs" / "vlm_campaign",
            report_dir=report_dir,
            k_values="0,2",
            control_k_values="2",
            seeds="42",
            models="fake/gemma4,fake/medgemma",
            conditions="zero_shot,estandar,balanced,permuted,no_support_images",
            vlm_runtime="remote_api",
            model="",
            api_base="http://nahuel:8000/v1",
            api_key="",
            system_prompt_file="prompts/system/qrs_huca.md",
            prompt_file="prompts/qrs/right_precordial_morphology.md",
            response_format="json_schema",
            temperature=0.0,
            max_output_tokens=32,
            timeout=1.0,
            local_batch_size=1,
            local_device="cpu",
            local_dtype="float32",
            local_attn_implementation="sdpa",
            local_offload_dir=tmp_path / "offload",
            limit_folds=2,
            resume=False,
            dry_run_predictions="negative",
            context_dataset_root=Path(""),
        )
    )

    rows = _read_csv(report_dir / "vlm_summary_by_seed.csv")
    assert {row["condition"] for row in rows} == {
        "zero_shot",
        "estandar",
        "balanced",
        "permuted",
        "no_support_images",
    }
    assert {row["model"] for row in rows} == {"fake/gemma4", "fake/medgemma"}
    assert (report_dir / "vlm_campaign_manifest.json").exists()
    assert (report_dir / "vlm_summary_by_model_condition_k.csv").exists()
    assert (report_dir / "balanced_accuracy_by_k.png").exists()
    assert (report_dir / "f1_by_k.png").exists()
    assert any((report_dir / "confusion_by_model_condition_k").glob("*.png"))


def test_vlm_clinical_real_context_uses_balanced_huca_examples(tmp_path: Path) -> None:
    from ecg_few.vlm.pipeline import run_with_args as run_vlm

    dataset_root = _write_mini_clinical_dataset(tmp_path)
    report_dir = tmp_path / "reports" / "vlm_clinical"

    setup = validate_setup(
        Namespace(
            dataset_root=dataset_root,
            context_dataset_root=Path(""),
            folds=dataset_root / "labels" / "loocv_folds.jsonl",
            k_values="0,2",
            control_k_values="2",
            seeds="42",
            models="fake/clinical",
            conditions="zero_shot,estandar,permuted,no_support_images",
            task="clinical",
            clinical_lead="V1",
            vlm_runtime="local_gpu",
            model="",
            api_base="",
            system_prompt_file="",
            prompt_file="",
            image_check="all",
            limit_folds=2,
            output=Path(""),
        )
    )
    assert setup["ok"] is True
    assert setup["planned_model_requests"] == 8
    assert setup["planned_image_requests"] == 8
    assert setup["planned_context_images"] == 8

    run_vlm(
        Namespace(
            dataset_root=dataset_root,
            folds=dataset_root / "labels" / "loocv_folds.jsonl",
            output_root=tmp_path / "outputs" / "vlm_clinical",
            report_dir=report_dir,
            k_values="0,2",
            control_k_values="2",
            seeds="42",
            models="fake/clinical",
            conditions="zero_shot,estandar,permuted,no_support_images",
            task="clinical",
            clinical_lead="V1",
            vlm_runtime="remote_api",
            model="",
            api_base="http://nahuel:8000/v1",
            api_key="",
            system_prompt_file="",
            prompt_file="",
            response_format="json_schema",
            temperature=0.0,
            max_output_tokens=32,
            timeout=1.0,
            local_batch_size=1,
            local_device="cpu",
            local_dtype="float32",
            local_attn_implementation="sdpa",
            local_offload_dir=tmp_path / "offload",
            limit_folds=2,
            resume=False,
            dry_run_predictions="expected",
            context_dataset_root=Path(""),
        )
    )

    rows = _read_csv(report_dir / "vlm_summary_by_seed.csv")
    assert {row["task"] for row in rows} == {"clinical"}
    assert {row["condition"] for row in rows} == {
        "zero_shot",
        "estandar",
        "permuted",
        "no_support_images",
    }
    predictions = _read_csv(
        tmp_path
        / "outputs"
        / "vlm_clinical"
        / "fake_clinical"
        / "estandar"
        / "k2_seed42"
        / "fold_predictions.csv"
    )
    label_by_patient = {
        patient.patient_id: patient.reference_brugada
        for patient in patients_from_rows(read_manifest(dataset_root / "labels" / "all_labels.csv"))
    }
    for row in predictions:
        labels = {
            label_by_patient[patient_id] for patient_id in row["context_patient_ids"].split("|")
        }
        assert labels == {0, 1}
        assert row["clinical_lead"] == "V1"


def test_cnn_ssl_pretraining_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from ecg_few.cnn.pipeline import run_with_args as run_cnn

    dataset_root = _write_mini_dataset(tmp_path)
    report_dir = tmp_path / "reports" / "cnn_ssl"

    run_cnn(
        _cnn_args(
            dataset_root=dataset_root,
            output_root=tmp_path / "outputs" / "cnn_ssl",
            report_dir=report_dir,
            limit_folds=1,
            target_dataset_root=dataset_root,
            ssl_pretrain_epochs=1,
        )
    )

    assert (report_dir / "cnn_summary_by_seed.csv").exists()
    assert (
        tmp_path
        / "outputs"
        / "cnn_ssl"
        / "resnet18"
        / "k2_seed42"
        / "fold_0000"
        / "ssl_pretrain"
        / "simclr_history.json"
    ).exists()


def test_cnn_domain_adaptation_epoch_variants_train() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    from ecg_few.cnn.domain_adaptation import DomainClassifier
    from ecg_few.cnn.pipeline import build_model, run_domain_train_epoch

    source_images = torch.rand(4, 3, 32, 32)
    source_labels = torch.tensor(
        [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    target_images = torch.rand(4, 3, 32, 32)

    for method in ("coral", "mmd", "dann"):
        model = build_model(resnet_weights="none")
        domain_classifier = DomainClassifier() if method == "dann" else None
        optimizer = torch.optim.AdamW(
            list(model.parameters())
            + ([] if domain_classifier is None else list(domain_classifier.parameters())),
            lr=1e-3,
        )
        payload = run_domain_train_epoch(
            model=model,
            source_loader=DataLoader(
                TensorDataset(source_images, source_labels),
                batch_size=2,
                shuffle=False,
            ),
            criterion=nn.BCEWithLogitsLoss(),
            optimizer=optimizer,
            device=torch.device("cpu"),
            args=Namespace(
                domain_adaptation=method,
                domain_adaptation_weight=0.1,
                epochs=1,
                mmd_kernel_scales="0.5,1.0",
            ),
            epoch=1,
            target_loader=DataLoader(target_images, batch_size=2, shuffle=False),
            domain_classifier=domain_classifier,
        )

        assert np.isfinite(payload["train_loss"])
        assert payload["domain_weight"] > 0
        assert payload["adaptation_loss"] >= 0


def _mini_rows(dataset_root: Path) -> list[BrugadaImageRow]:
    rows: list[BrugadaImageRow] = []
    for patient_index, label in enumerate((0, 1, 0, 1), start=100):
        family = "COMBINED_QRS_ST" if label else "NORMAL"
        finding_value = int(label)
        for lead in ("V1", "V2", "V3"):
            rows.append(
                BrugadaImageRow(
                    image_path=f"images/{family}/{patient_index}_{lead}.png",
                    patient_id=str(patient_index),
                    lead=lead,
                    source_family=family,
                    label_rbbb=finding_value,
                    label_st_elevation=finding_value,
                    label_t_wave_inversion=finding_value,
                    clinical_brugada=label,
                    basal_pattern=0,
                    sudden_death=0,
                    sample_index=patient_index,
                    aggregation_group_id=str(patient_index),
                )
            )
    return rows


def _write_mini_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "brugada_huca"
    rows = _mini_rows(dataset_root)
    for row in rows:
        _write_image(dataset_root / row.image_path, invert=bool(row.reference_brugada))
    write_manifest(dataset_root / "labels" / "all_labels.csv", rows)
    folds = build_fold_plan(rows, k_values=[2], seeds=[42], val_per_class=1)
    write_jsonl(dataset_root / "labels" / "loocv_folds.jsonl", folds)
    (dataset_root / "labels" / "label_schema.json").write_text(
        json.dumps(
            {
                "labels": ["RBBB", "ST_ELEVATION", "T_WAVE_INVERSION"],
                "clinical_reference_column": "clinical_brugada",
            }
        ),
        encoding="utf-8",
    )
    return dataset_root


def _write_mini_clinical_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "brugada_huca_clinical"
    rows: list[BrugadaImageRow] = []
    for patient_index, label in enumerate((0, 1, 0, 1), start=200):
        family = "CLINICAL_BRUGADA" if label else "CLINICAL_NORMAL"
        for lead in ("V1", "V2", "V3"):
            rows.append(
                BrugadaImageRow(
                    image_path=f"images/{family}/{patient_index}_{lead}.png",
                    patient_id=str(patient_index),
                    lead=lead,
                    source_family=family,
                    label_rbbb=None,
                    label_st_elevation=None,
                    label_t_wave_inversion=None,
                    clinical_brugada=label,
                    basal_pattern=0,
                    sudden_death=0,
                    sample_index=patient_index,
                    aggregation_group_id=str(patient_index),
                )
            )
    for row in rows:
        _write_image(dataset_root / row.image_path, invert=bool(row.reference_brugada))
    write_manifest(dataset_root / "labels" / "all_labels.csv", rows)
    folds = build_fold_plan(rows, k_values=[2], seeds=[42], val_per_class=1)
    write_jsonl(dataset_root / "labels" / "loocv_folds.jsonl", folds)
    return dataset_root


def _cnn_args(
    *,
    dataset_root: Path,
    output_root: Path,
    report_dir: Path,
    limit_folds: int,
    target_dataset_root: Path = Path(""),
    ssl_pretrain_epochs: int = 0,
) -> Namespace:
    return Namespace(
        dataset_root=dataset_root,
        folds=dataset_root / "labels" / "loocv_folds.jsonl",
        output_root=output_root,
        report_dir=report_dir,
        k_values="2",
        seeds="42",
        epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=1e-4,
        patience=1,
        num_workers=0,
        image_size=32,
        threshold=0.5,
        threshold_strategy="fixed",
        loss_pos_weight="none",
        device="cpu",
        resnet_weights="none",
        gradcam_count=0,
        limit_folds=limit_folds,
        save_fold_models=False,
        train_dataset_root=Path(""),
        external_train_mode="shared",
        target_dataset_root=target_dataset_root,
        domain_adaptation="none",
        domain_adaptation_weight=0.1,
        mmd_kernel_scales="0.5,1.0,2.0,4.0",
        ssl_pretrain_epochs=ssl_pretrain_epochs,
        ssl_pretrain_lr=1e-4,
        ssl_projection_dim=64,
        ssl_temperature=0.2,
        resume=False,
    )


def _write_image(path: Path, *, invert: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("L", (64, 64), color=255)
    draw = ImageDraw.Draw(image)
    y = 18 if invert else 36
    draw.line([(4, 36), (18, 36), (28, y), (40, 36), (60, 36)], fill=0, width=3)
    image.save(path)


def _vlm_args(dataset_root: Path, *, api_base: str) -> Namespace:
    return Namespace(
        dataset_root=dataset_root,
        folds=dataset_root / "labels" / "loocv_folds.jsonl",
        k_values="2",
        seeds="42",
        vlm_runtime="remote_api",
        model="fake/medgemma",
        api_base=api_base,
        task="morphology",
        clinical_lead="V1",
        system_prompt_file="prompts/system/qrs_huca.md",
        prompt_file="prompts/qrs/right_precordial_morphology.md",
        image_check="all",
        limit_folds=0,
        output=Path(""),
        context_dataset_root=Path(""),
    )


def _write_prediction_csv(path: Path, *, context: str) -> None:
    path.write_text(
        f"fold_id,test_patient_id,context_patient_ids\n0,100,{context}\n",
        encoding="utf-8",
    )


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
