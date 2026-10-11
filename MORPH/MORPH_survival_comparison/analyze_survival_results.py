from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


MORPH = "MORPH-Cox"
DATASETS = ("LAML", "SKCM")
SEEDS = (1028, 2024, 3407)
FOLDS = tuple(range(1, 6))
MODELS = (
    "MORPH-Cox",
    "LASSO-Cox",
    "VAE-Cox",
    "Adapted OmiEmbed-Survival",
    "SurvPath-derived MaskedOmics",
    "MCAT-derived SNN",
)
METRICS = ("c_index", "auc_1y", "auc_3y")
COMPLEXITY_COLUMNS = (
    "total_parameter_count",
    "trainable_parameter_count",
    "nonzero_parameter_count",
    "model_size_bytes",
    "training_seconds",
    "inference_ms_per_sample",
    "peak_memory_mb",
)


def atomic_json(payload: Dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(str(temporary), str(path))


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        frame.groupby(["dataset", "model"], sort=False)
        .agg(
            successful_runs=("status", "size"),
            c_index_mean=("c_index", "mean"),
            c_index_std=("c_index", "std"),
            auc_1y_mean=("auc_1y", "mean"),
            auc_1y_std=("auc_1y", "std"),
            auc_3y_mean=("auc_3y", "mean"),
            auc_3y_std=("auc_3y", "std"),
        )
        .reset_index()
    )
    for metric in METRICS:
        grouped[f"{metric}_rank"] = grouped.groupby("dataset")[
            f"{metric}_mean"
        ].rank(ascending=False, method="min")
    return grouped


def holm_adjust(p_values: List[float]) -> List[float]:
    adjusted = [np.nan] * len(p_values)
    valid = [index for index, value in enumerate(p_values) if np.isfinite(value)]
    order = sorted(valid, key=lambda index: p_values[index])
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def paired_tests(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for dataset in DATASETS:
        dataset_rows = frame[frame["dataset"].eq(dataset)]
        for metric in METRICS:
            baseline_means = (
                dataset_rows[~dataset_rows["model"].eq(MORPH)]
                .groupby("model")[metric]
                .mean()
            )
            best_baseline = str(baseline_means.idxmax())
            morph_rows = dataset_rows[dataset_rows["model"].eq(MORPH)][
                ["seed", "fold", metric]
            ].rename(columns={metric: "morph"})
            baseline_rows = dataset_rows[dataset_rows["model"].eq(best_baseline)][
                ["seed", "fold", metric]
            ].rename(columns={metric: "baseline"})
            pairs = morph_rows.merge(baseline_rows, on=["seed", "fold"]).dropna()
            difference = pairs["morph"] - pairs["baseline"]
            if difference.empty or np.allclose(difference.to_numpy(), 0.0):
                p_value = 1.0
            else:
                p_value = float(
                    wilcoxon(
                        difference.to_numpy(),
                        alternative="two-sided",
                        zero_method="wilcox",
                        mode=(
                            "approx"
                            if np.any(difference.to_numpy() == 0.0)
                            else "exact"
                        ),
                    ).pvalue
                )
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "best_baseline": best_baseline,
                    "paired_runs": int(len(pairs)),
                    "morph_mean": float(pairs["morph"].mean()),
                    "baseline_mean": float(pairs["baseline"].mean()),
                    "mean_difference": float(difference.mean()),
                    "morph_wins": int((difference > 0).sum()),
                    "ties": int((difference == 0).sum()),
                    "morph_losses": int((difference < 0).sum()),
                    "wilcoxon_p_raw": p_value,
                }
            )
    adjusted = holm_adjust([float(row["wilcoxon_p_raw"]) for row in rows])
    for row, p_value in zip(rows, adjusted):
        row["holm_p_across_six_tests"] = p_value
        row["significant_at_0_05"] = bool(np.isfinite(p_value) and p_value < 0.05)
    return pd.DataFrame(rows)


def complexity_summary(frame: pd.DataFrame) -> pd.DataFrame:
    aggregations = {}
    for column in COMPLEXITY_COLUMNS:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_std"] = (column, "std")
    return (
        frame.groupby(["dataset", "model"], sort=False)
        .agg(**aggregations)
        .reset_index()
    )


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for values in frame.itertuples(index=False, name=None):
        cells = []
        for value in values:
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def validate_results(frame: pd.DataFrame, oof: pd.DataFrame) -> Dict[str, object]:
    expected_keys = {
        (dataset, model, seed, fold)
        for dataset in DATASETS
        for model in MODELS
        for seed in SEEDS
        for fold in FOLDS
    }
    actual_keys = set(
        frame[["dataset", "model", "seed", "fold"]].itertuples(index=False, name=None)
    )
    duplicate_fold_keys = int(
        frame.duplicated(["dataset", "model", "seed", "fold"]).sum()
    )
    failed_rows = int((~frame["status"].eq("ok")).sum())
    nonfinite_metrics = int(
        (~np.isfinite(frame[list(METRICS)].apply(pd.to_numeric, errors="coerce"))).sum().sum()
    )
    duplicate_oof_keys = int(
        oof.duplicated(["dataset", "model", "seed", "fold", "sample_id"]).sum()
    )
    coverage_errors = 0
    for dataset in DATASETS:
        expected_samples = int(frame[frame["dataset"].eq(dataset)]["n_samples"].iloc[0])
        for model in MODELS:
            for seed in SEEDS:
                subset = oof[
                    oof["dataset"].eq(dataset)
                    & oof["model"].eq(model)
                    & oof["seed"].eq(seed)
                ]
                if len(subset) != expected_samples or subset["sample_id"].nunique() != expected_samples:
                    coverage_errors += 1
    valid = (
        actual_keys == expected_keys
        and duplicate_fold_keys == 0
        and failed_rows == 0
        and nonfinite_metrics == 0
        and duplicate_oof_keys == 0
        and coverage_errors == 0
    )
    if not valid:
        raise RuntimeError("Result integrity validation failed; see integrity.json fields")
    return {
        "protocol": "fixed-final",
        "expected_fold_rows": len(expected_keys),
        "fold_rows": int(len(frame)),
        "successful_rows": int(frame["status"].eq("ok").sum()),
        "failed_rows": failed_rows,
        "missing_or_extra_fold_keys": len(expected_keys.symmetric_difference(actual_keys)),
        "duplicate_fold_keys": duplicate_fold_keys,
        "nonfinite_metric_values": nonfinite_metrics,
        "oof_rows": int(len(oof)),
        "duplicate_oof_keys": duplicate_oof_keys,
        "oof_coverage_errors": coverage_errors,
        "validation_passed": valid,
    }


def write_report(
    results_dir: Path,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    complexity: pd.DataFrame,
) -> None:
    lines = [
        "# Fixed final survival comparison",
        "",
        "These are numerical cross-validation estimates for the fixed selected configuration.",
        "They are not described as an untouched external validation.",
        "",
        "## Performance",
        "",
        markdown_table(summary),
        "",
        "## Paired comparison with the strongest baseline for each metric",
        "",
        markdown_table(tests),
        "",
        "No result should be called statistically significant unless the Holm-adjusted p-value is below 0.05.",
        "",
        "## Complexity",
        "",
        markdown_table(complexity),
        "",
    ]
    (results_dir / "survival_comparison_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    fold_file = args.results_dir / "fold_results.csv"
    oof_file = args.results_dir / "oof_predictions.csv"
    if not fold_file.exists() or not oof_file.exists():
        raise SystemExit(f"Missing fixed result inputs in {args.results_dir}")
    frame = pd.read_csv(fold_file)
    oof = pd.read_csv(oof_file)
    integrity = validate_results(frame, oof)
    ok = frame[frame["status"].eq("ok")].copy()
    summary = summarize(ok)
    tests = paired_tests(ok)
    complexity = complexity_summary(ok)
    summary.to_csv(args.results_dir / "comparison_summary.csv", index=False)
    tests.to_csv(args.results_dir / "paired_wilcoxon_holm.csv", index=False)
    complexity.to_csv(args.results_dir / "complexity_summary.csv", index=False)
    write_report(args.results_dir, summary, tests, complexity)
    integrity["analysis_outputs"] = [
        "comparison_summary.csv",
        "paired_wilcoxon_holm.csv",
        "complexity_summary.csv",
        "survival_comparison_report.md",
    ]
    atomic_json(integrity, args.results_dir / "integrity.json")
    print(summary.to_string(index=False))
    print("\nPaired tests:\n" + tests.to_string(index=False))
    print("\nIntegrity: passed")


if __name__ == "__main__":
    main()
