#!/usr/bin/env python3
"""Compare simulated-QRS and real-HUCA CNN summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

from ecg_few.configuration import parse_config_args
from ecg_few.experiment_io import validate_output_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CNN sim-vs-real reports by k.")
    parser.add_argument(
        "--sim-summary",
        type=Path,
        default=Path("outputs/work/cnn_simulator_qrs_loocv/reports/cnn_summary_by_k.csv"),
    )
    parser.add_argument(
        "--real-summary",
        type=Path,
        default=Path("outputs/work/cnn_loocv/reports/cnn_summary_by_k.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/work/cnn_comparison/reports")
    )
    return parse_config_args(parser, argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sim_rows = keyed_by_k(read_csv(args.sim_summary))
    real_rows = keyed_by_k(read_csv(args.real_summary))
    rows = build_comparison(sim_rows, real_rows)
    validate_output_path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "cnn_simulated_vs_real_by_k.csv", rows)
    plot_metric(
        args.output_dir / "balanced_accuracy_simulated_vs_real_by_k.png",
        rows,
        metric="balanced_accuracy",
        ylabel="Balanced accuracy",
    )
    plot_metric(
        args.output_dir / "f1_simulated_vs_real_by_k.png",
        rows,
        metric="f1",
        ylabel="F1",
    )
    print(f"[OK] Wrote CNN sim-vs-real comparison under: {args.output_dir}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def keyed_by_k(rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {int(row["k"]): row for row in rows}


def build_comparison(
    sim_rows: dict[int, dict[str, str]],
    real_rows: dict[int, dict[str, str]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for k in sorted(set(sim_rows) & set(real_rows)):
        row: dict[str, object] = {"k": k}
        for metric in (
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "f1",
            "roc_auc",
            "average_precision",
        ):
            row[f"sim_{metric}_mean"] = sim_rows[k].get(f"{metric}_mean", "")
            row[f"sim_{metric}_std"] = sim_rows[k].get(f"{metric}_std", "")
            row[f"real_{metric}_mean"] = real_rows[k].get(f"{metric}_mean", "")
            row[f"real_{metric}_std"] = real_rows[k].get(f"{metric}_std", "")
        output.append(row)
    return output


def plot_metric(path: Path, rows: list[dict[str, object]], *, metric: str, ylabel: str) -> None:
    if not rows:
        return
    k_values = [int(row["k"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for prefix, label in (("sim", "Simulated QRS"), ("real", "Real HUCA")):
        means = [float(row[f"{prefix}_{metric}_mean"]) for row in rows]
        stds = [float(row[f"{prefix}_{metric}_std"]) for row in rows]
        ax.errorbar(k_values, means, yerr=stds, marker="o", capsize=4, label=label)
    ax.set_xlabel("k simulated training patients")
    ax.set_ylabel(ylabel)
    ax.set_title(f"CNN {ylabel.lower()} by k")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
