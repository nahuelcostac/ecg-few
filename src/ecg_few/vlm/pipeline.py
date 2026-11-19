"""Pipeline for ECG experiments."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ecg_few.configuration import parse_config_args
from ecg_few.experiment_io import (
    append_jsonl,
    existing_keys,
    load_existing_predictions,
    prepare_campaign,
    validate_output_path,
    write_csv,
)
from ecg_few.findings import LABEL_NAMES, findings_to_brugada
from ecg_few.prompts import (
    DEFAULT_SYSTEM_INSTRUCTIONS,
    load_markdown_prompt,
)
from ecg_few.protocols.loocv import (
    DEFAULT_K_VALUES,
    DEFAULT_SEEDS,
    BrugadaImageRow,
    parse_int_list,
    read_jsonl,
    read_manifest,
    resolve_folds_path,
    rows_for_patient_ids,
    validate_fold_plan,
)
from ecg_few.reporting.vlm import summarize_predictions, write_summary_reports
from ecg_few.vlm.inference import (
    aggregate_clinical_lead_predictions,
    aggregate_lead_findings,
    call_model,
    dry_run_clinical_prediction,
    dry_run_prediction,
    parse_clinical_prediction,
    parse_morphology_prediction,
)
from ecg_few.vlm.messages import build_clinical_messages, build_messages, row_for_lead
from ecg_few.vlm.runtime import (
    LOCAL_RUNTIME,
    REMOTE_RUNTIME,
    LocalGPUGenerator,
    resolve_api_base,
    resolve_api_key,
)
from ecg_few.vlm.selection import (
    answer_rows_for_condition,
    k_values_for_condition,
    model_slug,
    only_v1_rows,
    parse_string_list,
    resolve_clinical_leads,
    resolve_context_dataset,
    resolve_models,
    resolve_prompt_path,
    select_balanced_context_patient_ids,
    select_qrs_context_patient_ids,
    selection_for_condition,
)
from ecg_few.vlm.settings import (
    ALL_CONDITIONS,
    BALANCED_CONDITION,
    CLINICAL_AGGREGATIONS,
    CLINICAL_LABEL_NAME,
    CLINICAL_TASK,
    DEFAULT_CLINICAL_LEADS,
    ESTANDAR_CONDITION,
    MORPHOLOGY_TASK,
    NO_SUPPORT_IMAGES_CONDITION,
    PERMUTED_CONDITION,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VLM few-shot LOOCV for QRS findings and derived Brugada."
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/brugada_huca"))
    parser.add_argument(
        "--context-dataset-root",
        type=Path,
        default=Path(""),
        help="Optional synthetic QRS dataset used for ICL examples.",
    )
    parser.add_argument("--folds", type=Path, default=Path(""))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/work/vlm_loocv"))
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--k-values", default=",".join(str(k) for k in DEFAULT_K_VALUES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model ids. Overrides --model when provided.",
    )
    parser.add_argument(
        "--conditions",
        default=ESTANDAR_CONDITION,
        help=(
            "Comma-separated VLM conditions: "
            "zero_shot,estandar,balanced,permuted,no_support_images."
        ),
    )
    parser.add_argument(
        "--control-k-values",
        default="8,16,32",
        help="K values used by control conditions such as permuted and no_support_images.",
    )
    parser.add_argument(
        "--vlm-runtime",
        choices=(REMOTE_RUNTIME, LOCAL_RUNTIME),
        default=REMOTE_RUNTIME,
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--api-base", default="")
    parser.add_argument("--api-key", default="")
    parser.add_argument(
        "--task",
        choices=(MORPHOLOGY_TASK, CLINICAL_TASK),
        default=MORPHOLOGY_TASK,
        help=(
            "morphology predicts RBBB/ST_ELEVATION/T_WAVE_INVERSION per lead; "
            "clinical predicts clinical_brugada per configured lead and aggregates by patient."
        ),
    )
    parser.add_argument(
        "--clinical-lead",
        default="V1",
        help="Legacy single lead used by TASK=clinical when --clinical-leads is empty.",
    )
    parser.add_argument(
        "--clinical-leads",
        default=",".join(DEFAULT_CLINICAL_LEADS),
        help="Comma-separated leads used by TASK=clinical before patient-level aggregation.",
    )
    parser.add_argument(
        "--clinical-aggregation",
        choices=CLINICAL_AGGREGATIONS,
        default="majority",
        help="Patient-level aggregation rule for TASK=clinical multi-lead predictions.",
    )
    parser.add_argument("--system-prompt-file", default="")
    parser.add_argument(
        "--prompt-file",
        default="",
    )
    parser.add_argument(
        "--response-format",
        choices=("json_schema", "json_object", "none"),
        default="json_schema",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--local-batch-size", type=int, default=4)
    parser.add_argument("--local-device", default="cuda")
    parser.add_argument(
        "--local-dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--local-attn-implementation", default="sdpa")
    parser.add_argument("--local-offload-dir", type=Path)
    parser.add_argument("--limit-folds", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument(
        "--dry-run-predictions",
        choices=("none", "expected", "negative"),
        default="none",
        help="Write deterministic predictions without loading a model or calling an API.",
    )
    return parse_config_args(parser, argv)


def main(argv: list[str] | None = None) -> None:
    run_with_args(parse_args(argv))


def run_with_args(args: argparse.Namespace) -> None:
    validate_output_path(Path(args.output_root))
    validate_output_path(
        Path(args.report_dir) if args.report_dir else Path(args.output_root) / "reports"
    )
    dataset_root = Path(args.dataset_root).resolve()
    folds_path = resolve_folds_path(dataset_root, args.folds)
    rows = only_v1_rows(read_manifest(dataset_root / "labels" / "all_labels.csv"))
    context_dataset_root, has_context_dataset = resolve_context_dataset(
        dataset_root,
        args.context_dataset_root,
    )
    context_pool_rows = (
        only_v1_rows(read_manifest(context_dataset_root / "labels" / "all_labels.csv"))
        if has_context_dataset
        else rows
    )
    folds = read_jsonl(folds_path.resolve())
    k_values = parse_int_list(str(args.k_values))
    seeds = parse_int_list(str(args.seeds))
    fold_k_values = [k for k in k_values if k > 0]
    if fold_k_values:
        validate_fold_plan(folds, rows, k_values=fold_k_values, seeds=seeds)
    if int(args.limit_folds) > 0:
        folds = folds[: int(args.limit_folds)]

    runtime = str(args.vlm_runtime)
    task = str(getattr(args, "task", MORPHOLOGY_TASK))
    models = resolve_models(args, runtime)
    conditions = parse_string_list(getattr(args, "conditions", ESTANDAR_CONDITION))
    invalid_conditions = sorted(set(conditions) - set(ALL_CONDITIONS))
    if invalid_conditions:
        raise ValueError(f"Unsupported VLM conditions: {invalid_conditions}")
    control_k_values = parse_int_list(str(getattr(args, "control_k_values", "8,16,32")))
    api_base = resolve_api_base(str(args.api_base), runtime)
    api_key = resolve_api_key(str(args.api_key), runtime)
    system_prompt_file = resolve_prompt_path(
        str(args.system_prompt_file),
        task=task,
        kind="system",
    )
    prompt_file = resolve_prompt_path(str(args.prompt_file), task=task, kind="user")
    system_prompt = (
        load_markdown_prompt(system_prompt_file)
        if system_prompt_file
        else DEFAULT_SYSTEM_INSTRUCTIONS
    )
    prompt_template = load_markdown_prompt(prompt_file)
    output_root = Path(args.output_root).resolve()
    report_dir = (Path(args.report_dir) if args.report_dir else output_root / "reports").resolve()

    offload_root = (
        Path(args.local_offload_dir) if args.local_offload_dir else output_root / "offload"
    )
    validate_output_path(offload_root)
    prepare_campaign(
        output_root,
        {**vars(args), "resolved_models": models, "resolved_api_base": api_base},
        inputs={
            "dataset_manifest": dataset_root / "labels" / "all_labels.csv",
            "context_manifest": context_dataset_root / "labels" / "all_labels.csv",
            "folds": folds_path.resolve(),
        },
        resources={"system_prompt": system_prompt, "user_prompt": prompt_template},
        resume=bool(args.resume),
    )
    summary_rows: list[dict[str, object]] = []
    for model in models:
        generator: LocalGPUGenerator | None = None
        if runtime == LOCAL_RUNTIME and args.dry_run_predictions == "none":
            generator = LocalGPUGenerator(
                model_id=model,
                device=str(args.local_device),
                dtype=str(args.local_dtype),
                attn_implementation=str(args.local_attn_implementation),
                offload_dir=offload_root / model_slug(model),
                token=api_key,
            )
        try:
            for condition in conditions:
                condition_k_values = k_values_for_condition(
                    condition,
                    k_values=k_values,
                    control_k_values=control_k_values,
                )
                if not condition_k_values:
                    continue
                for k in condition_k_values:
                    for seed in seeds:
                        run_dir = output_root / model_slug(model) / condition / f"k{k}_seed{seed}"
                        predictions = run_k_seed(
                            rows=rows,
                            context_pool_rows=context_pool_rows,
                            folds=folds,
                            dataset_root=dataset_root,
                            context_dataset_root=context_dataset_root,
                            run_dir=run_dir,
                            k=k,
                            seed=seed,
                            task=task,
                            condition=condition,
                            runtime=runtime,
                            model=model,
                            api_base=api_base,
                            api_key=api_key,
                            system_prompt=system_prompt,
                            prompt_template=prompt_template,
                            args=args,
                            generator=generator,
                        )
                        summary_rows.append(
                            summarize_predictions(
                                predictions,
                                k=k,
                                seed=seed,
                                task=task,
                                condition=condition,
                                model=model,
                                run_dir=run_dir,
                            )
                        )
        finally:
            if generator is not None:
                generator.close()
    write_summary_reports(summary_rows, report_dir)


def run_k_seed(
    *,
    rows: list[BrugadaImageRow],
    context_pool_rows: list[BrugadaImageRow],
    folds: list[dict[str, Any]],
    dataset_root: Path,
    context_dataset_root: Path,
    run_dir: Path,
    k: int,
    seed: int,
    task: str,
    condition: str,
    runtime: str,
    model: str,
    api_base: str | None,
    api_key: str,
    system_prompt: str,
    prompt_template: str,
    args: argparse.Namespace,
    generator: LocalGPUGenerator | None,
) -> list[dict[str, object]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "fold_predictions.jsonl"
    existing = existing_keys(output_path) if args.resume else set()
    if output_path.exists() and not args.resume:
        output_path.unlink()
    predictions: list[dict[str, object]] = load_existing_predictions(output_path)
    for fold in folds:
        fold_id = int(fold["fold_id"])
        test_patient_id = str(fold["test_patient_id"])
        key = f"{condition}:{fold_id}:{test_patient_id}:{k}:{seed}"
        if key in existing:
            continue
        selection = selection_for_condition(
            fold=fold,
            rows=rows,
            context_pool_rows=context_pool_rows,
            test_patient_id=test_patient_id,
            k=k,
            seed=seed,
            fold_id=fold_id,
            task=task,
            condition=condition,
        )
        context_ids = selection["context_patient_ids"]
        if (
            task != CLINICAL_TASK
            and context_pool_rows is not rows
            and condition in {ESTANDAR_CONDITION, PERMUTED_CONDITION, NO_SUPPORT_IMAGES_CONDITION}
            and k > 0
        ):
            context_ids = select_qrs_context_patient_ids(
                context_pool_rows,
                k=k,
                seed=seed + fold_id * 100_003,
            )
        elif (
            task != CLINICAL_TASK
            and context_pool_rows is not rows
            and condition == BALANCED_CONDITION
            and k > 0
        ):
            context_ids = select_balanced_context_patient_ids(
                context_pool_rows,
                test_patient_id="__real_eval_patient__",
                k=k,
                seed=seed + fold_id * 100_003,
            )
        if (
            dataset_root.resolve() == context_dataset_root.resolve()
            and test_patient_id in context_ids
        ):
            raise RuntimeError(
                f"Fold {fold_id} leaks test patient {test_patient_id} into VLM context."
            )
        context_rows = rows_for_patient_ids(context_pool_rows, context_ids)
        answer_rows = answer_rows_for_condition(
            context_rows,
            condition=condition,
            seed=seed + fold_id * 100_003,
        )
        test_rows = rows_for_patient_ids(rows, [test_patient_id])
        print(
            "[INFO] VLM "
            f"model={model} task={task} condition={condition} k={k} seed={seed} "
            f"fold={fold_id} test_patient={test_patient_id}"
        )
        started = time.perf_counter()
        if task == CLINICAL_TASK:
            record = run_clinical_fold(
                rows=rows,
                context_rows=context_rows,
                dataset_root=dataset_root,
                context_dataset_root=context_dataset_root,
                test_patient_id=test_patient_id,
                fold_id=fold_id,
                k=k,
                seed=seed,
                key=key,
                condition=condition,
                runtime=runtime,
                model=model,
                api_base=api_base,
                api_key=api_key,
                system_prompt=system_prompt,
                prompt_template=prompt_template,
                include_support_images=condition != NO_SUPPORT_IMAGES_CONDITION,
                clinical_leads=resolve_clinical_leads(args),
                clinical_aggregation=str(getattr(args, "clinical_aggregation", "majority")),
                started=started,
                selection=selection,
                context_ids=context_ids,
                args=args,
                generator=generator,
            )
            append_jsonl(output_path, record)
            predictions.append(record)
            continue
        lead_predictions: dict[str, dict[str, bool]] = {}
        lead_errors: dict[str, str] = {}
        raw_outputs: dict[str, str] = {}
        for test_row in test_rows:
            try:
                if args.dry_run_predictions != "none":
                    prediction = dry_run_prediction(test_row, mode=str(args.dry_run_predictions))
                    raw_text = json.dumps(prediction, sort_keys=True)
                else:
                    messages = build_messages(
                        dataset_root=dataset_root,
                        context_dataset_root=context_dataset_root,
                        system_prompt=system_prompt,
                        prompt_template=prompt_template,
                        context_rows=context_rows,
                        answer_rows=answer_rows,
                        test_row=test_row,
                        include_support_images=condition != NO_SUPPORT_IMAGES_CONDITION,
                        local=runtime == LOCAL_RUNTIME,
                    )
                    raw_text = call_model(
                        messages,
                        runtime=runtime,
                        model=model,
                        api_base=api_base,
                        api_key=api_key,
                        task=task,
                        args=args,
                        generator=generator,
                    )
                    prediction = parse_morphology_prediction(raw_text)
                lead_predictions[test_row.lead] = {
                    label_name: bool(prediction[label_name]) for label_name in LABEL_NAMES
                }
                raw_outputs[test_row.lead] = raw_text
            except Exception as exc:  # noqa: BLE001
                lead_errors[test_row.lead] = str(exc)
                raw_outputs[test_row.lead] = ""
        pred_findings = aggregate_lead_findings(lead_predictions)
        pred_label = findings_to_brugada(pred_findings)
        record = {
            "key": key,
            "fold_id": fold_id,
            "k": k,
            "seed": seed,
            "condition": condition,
            "task": task,
            "test_patient_id": test_patient_id,
            "patient_id": test_patient_id,
            "true_label": int(test_rows[0].reference_brugada),
            "pred_label": pred_label,
            **{f"pred_{label.lower()}": int(pred_findings[label]) for label in LABEL_NAMES},
            "context_patient_ids": "|".join(context_ids),
            "validation_patient_ids": "|".join(selection["validation_patient_ids"]),
            "valid_leads": len(lead_predictions),
            "invalid_leads": len(lead_errors),
            "json_invalid": int(bool(lead_errors)),
            "lead_predictions": json.dumps(lead_predictions, sort_keys=True),
            "raw_outputs": json.dumps(raw_outputs, sort_keys=True),
            "errors": json.dumps(lead_errors, sort_keys=True),
            "latency_seconds": time.perf_counter() - started,
            "model": model,
            "model_slug": model_slug(model),
            "vlm_runtime": runtime,
        }
        append_jsonl(output_path, record)
        predictions.append(record)
    write_csv(run_dir / "fold_predictions.csv", predictions)
    return predictions


def run_clinical_fold(
    *,
    rows: list[BrugadaImageRow],
    context_rows: list[BrugadaImageRow],
    dataset_root: Path,
    context_dataset_root: Path,
    test_patient_id: str,
    fold_id: int,
    k: int,
    seed: int,
    key: str,
    condition: str,
    runtime: str,
    model: str,
    api_base: str | None,
    api_key: str,
    system_prompt: str,
    prompt_template: str,
    include_support_images: bool,
    clinical_leads: list[str],
    clinical_aggregation: str,
    started: float,
    selection: dict[str, list[str]],
    context_ids: list[str],
    args: argparse.Namespace,
    generator: LocalGPUGenerator | None,
) -> dict[str, object]:
    test_rows = rows_for_patient_ids(rows, [test_patient_id])
    lead_predictions: dict[str, dict[str, bool]] = {}
    lead_errors: dict[str, str] = {}
    raw_outputs: dict[str, str] = {}
    for clinical_lead in clinical_leads:
        try:
            test_row = row_for_lead(test_rows, clinical_lead)
            if args.dry_run_predictions != "none":
                prediction = dry_run_clinical_prediction(
                    test_row,
                    mode=str(args.dry_run_predictions),
                )
                raw_text = json.dumps(prediction, sort_keys=True)
            else:
                messages = build_clinical_messages(
                    dataset_root=dataset_root,
                    context_dataset_root=context_dataset_root,
                    system_prompt=system_prompt,
                    prompt_template=prompt_template,
                    context_rows=context_rows,
                    condition=condition,
                    seed=seed,
                    test_row=test_row,
                    clinical_leads=clinical_leads,
                    include_support_images=include_support_images,
                    local=runtime == LOCAL_RUNTIME,
                )
                raw_text = call_model(
                    messages,
                    runtime=runtime,
                    model=model,
                    api_base=api_base,
                    api_key=api_key,
                    task=CLINICAL_TASK,
                    args=args,
                    generator=generator,
                )
                prediction = parse_clinical_prediction(raw_text)
            lead_predictions[test_row.lead] = {
                CLINICAL_LABEL_NAME: bool(prediction[CLINICAL_LABEL_NAME])
            }
            raw_outputs[test_row.lead] = raw_text
        except Exception as exc:  # noqa: BLE001
            lead_errors[clinical_lead.upper()] = str(exc)
            raw_outputs[clinical_lead.upper()] = ""
    pred_label = int(
        aggregate_clinical_lead_predictions(
            {
                lead: bool(prediction[CLINICAL_LABEL_NAME])
                for lead, prediction in lead_predictions.items()
            },
            method=clinical_aggregation,
        )
    )
    return {
        "key": key,
        "fold_id": fold_id,
        "k": k,
        "seed": seed,
        "condition": condition,
        "task": CLINICAL_TASK,
        "test_patient_id": test_patient_id,
        "patient_id": test_patient_id,
        "true_label": int(test_rows[0].reference_brugada),
        "pred_label": pred_label,
        "pred_clinical_brugada": pred_label,
        "clinical_lead": ",".join(clinical_leads),
        "clinical_leads": "|".join(clinical_leads),
        "clinical_aggregation": clinical_aggregation,
        "context_patient_ids": "|".join(context_ids),
        "validation_patient_ids": "|".join(selection["validation_patient_ids"]),
        "valid_leads": len(lead_predictions),
        "invalid_leads": len(lead_errors),
        "json_invalid": int(bool(lead_errors)),
        "lead_predictions": json.dumps(lead_predictions, sort_keys=True),
        "raw_outputs": json.dumps(raw_outputs, sort_keys=True),
        "errors": json.dumps(lead_errors, sort_keys=True),
        "latency_seconds": time.perf_counter() - started,
        "model": model,
        "model_slug": model_slug(model),
        "vlm_runtime": runtime,
    }


_LEGACY_EXPORTS = {
    "resolve_models": "ecg_few.vlm.selection",
    "parse_string_list": "ecg_few.vlm.selection",
    "only_v1_rows": "ecg_few.vlm.selection",
    "resolve_context_dataset": "ecg_few.vlm.selection",
    "resolve_clinical_leads": "ecg_few.vlm.selection",
    "resolve_prompt_path": "ecg_few.vlm.selection",
    "model_slug": "ecg_few.vlm.selection",
    "k_values_for_condition": "ecg_few.vlm.selection",
    "selection_for_condition": "ecg_few.vlm.selection",
    "answer_rows_for_condition": "ecg_few.vlm.selection",
    "select_qrs_context_patient_ids": "ecg_few.vlm.selection",
    "select_balanced_context_patient_ids": "ecg_few.vlm.selection",
    "select_balanced_clinical_context_patient_ids": "ecg_few.vlm.selection",
    "build_clinical_messages": "ecg_few.vlm.messages",
    "build_messages": "ecg_few.vlm.messages",
    "prompt_for_row": "ecg_few.vlm.messages",
    "patient_sort_key": "ecg_few.vlm.messages",
    "sorted_rows_by_lead": "ecg_few.vlm.messages",
    "row_for_lead": "ecg_few.vlm.messages",
    "rows_for_leads": "ecg_few.vlm.messages",
    "rows_by_patient": "ecg_few.vlm.messages",
    "clinical_answers_by_patient": "ecg_few.vlm.messages",
    "clinical_answer_labels_for_condition": "ecg_few.vlm.messages",
    "clinical_answer_text": "ecg_few.vlm.messages",
    "support_user_message": "ecg_few.vlm.messages",
    "support_patient_message": "ecg_few.vlm.messages",
    "user_message": "ecg_few.vlm.messages",
    "patient_message": "ecg_few.vlm.messages",
    "data_url_for_image": "ecg_few.vlm.messages",
    "call_model": "ecg_few.vlm.inference",
    "dry_run_prediction": "ecg_few.vlm.inference",
    "dry_run_clinical_prediction": "ecg_few.vlm.inference",
    "response_schema_for_task": "ecg_few.vlm.inference",
    "parse_json_object": "ecg_few.vlm.inference",
    "parse_morphology_prediction": "ecg_few.vlm.inference",
    "parse_clinical_prediction": "ecg_few.vlm.inference",
    "aggregate_lead_findings": "ecg_few.vlm.inference",
    "aggregate_clinical_lead_predictions": "ecg_few.vlm.inference",
    "summarize_predictions": "ecg_few.reporting.vlm",
    "first_nonempty": "ecg_few.reporting.vlm",
    "write_summary_reports": "ecg_few.reporting.vlm",
    "aggregate_by_group": "ecg_few.reporting.vlm",
    "write_confusion_matrices_by_group": "ecg_few.reporting.vlm",
    "plot_confusion_matrix": "ecg_few.reporting.vlm",
    "plot_metric_by_group": "ecg_few.reporting.vlm",
    "write_campaign_manifest": "ecg_few.reporting.vlm",
}


def __getattr__(name: str):
    from importlib import import_module

    if name in _LEGACY_EXPORTS:
        return getattr(import_module(_LEGACY_EXPORTS[name]), name)
    settings = import_module("ecg_few.vlm.settings")
    if hasattr(settings, name):
        return getattr(settings, name)
    raise AttributeError(name)


if __name__ == "__main__":
    main()
