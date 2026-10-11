from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE
sys.path.insert(0, str(PROJECT_ROOT))

from train_morph_survival import (
    HORIZONS,
    build_fold_priors,
    load_dataset,
    load_prior_resources,
    safe_cindex,
    safe_td_auc,
    set_seeds,
)


DEFAULT_DATASETS = ("LAML", "SKCM")
DEFAULT_SEEDS = (1028, 2024, 3407)
DEFAULT_RESULTS = (
    PROJECT_ROOT / "results" / "survpath_mcat"
)
def select_fixed_count(x_train: np.ndarray, count: int) -> np.ndarray:

    variance = np.nanvar(x_train, axis=0)
    valid = np.flatnonzero(np.isfinite(variance) & (variance > 0))
    if valid.size == 0:
        raise ValueError("No nonconstant features in training data")
    order = np.argsort(variance[valid], kind="stable")[::-1]
    return valid[order[: min(count, valid.size)]]


def prepare_modalities(
    x_mrna: np.ndarray,
    x_mirna: np.ndarray,
    inner_train_idx: np.ndarray,
    all_indices: Sequence[np.ndarray],
    mrna_features: int,
    mirna_features: int,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:

    mrna_keep = select_fixed_count(x_mrna[inner_train_idx], mrna_features)
    mirna_keep = select_fixed_count(x_mirna[inner_train_idx], mirna_features)
    mrna_scaler = StandardScaler().fit(x_mrna[inner_train_idx][:, mrna_keep])
    mirna_scaler = StandardScaler().fit(x_mirna[inner_train_idx][:, mirna_keep])
    transformed: List[np.ndarray] = []
    for indices in all_indices:
        mrna = mrna_scaler.transform(
            x_mrna[indices][:, mrna_keep]
        ).astype(np.float32)
        mirna = mirna_scaler.transform(
            x_mirna[indices][:, mirna_keep]
        ).astype(np.float32)
        transformed.append(np.concatenate([mrna, mirna], axis=1))
    return transformed, mrna_keep, mirna_keep


def make_discrete_bins(
    train_times: np.ndarray,
    train_events: np.ndarray,
    n_classes: int,
) -> np.ndarray:

    event_times = np.asarray(train_times, dtype=float)[
        np.asarray(train_events, dtype=int) == 1
    ]
    if event_times.size < n_classes:
        event_times = np.asarray(train_times, dtype=float)
    quantiles = np.linspace(0.0, 1.0, n_classes + 1)
    edges = np.asarray(np.quantile(event_times, quantiles), dtype=float)
    scale = max(float(np.nanmax(np.abs(event_times))), 1.0)
    eps = np.finfo(float).eps * scale * 16.0
    for index in range(1, len(edges)):
        if not edges[index] > edges[index - 1]:
            edges[index] = edges[index - 1] + eps
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_discrete_bins(times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    labels = np.digitize(
        np.asarray(times, dtype=float), edges[1:-1], right=False
    ).astype(np.int64)
    return np.clip(labels, 0, len(edges) - 2)


def nll_surv_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    events: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:

    y = labels.to(torch.int64).unsqueeze(1)
    censorship = (1.0 - events).to(torch.int64).unsqueeze(1)
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    survival_padded = torch.cat(
        [torch.ones_like(censorship, dtype=survival.dtype), survival], dim=1
    )
    s_prev = torch.gather(survival_padded, 1, y).clamp_min(eps)
    h_this = torch.gather(hazards, 1, y).clamp_min(eps)
    s_this = torch.gather(survival_padded, 1, y + 1).clamp_min(eps)
    uncensored = -(1 - censorship) * (torch.log(s_prev) + torch.log(h_this))
    censored = -censorship * torch.log(s_this)
    return (uncensored + censored).mean()


def init_max_weights(module: nn.Module) -> None:

    for layer in module.modules():
        if type(layer) is nn.Linear:
            stdv = 1.0 / math.sqrt(layer.weight.size(1))
            layer.weight.data.normal_(0, stdv)
            layer.bias.data.zero_()


def snn_block(dim1: int, dim2: int, dropout: float = 0.25) -> nn.Sequential:

    return nn.Sequential(
        nn.Linear(dim1, dim2),
        nn.ELU(),
        nn.AlphaDropout(p=dropout, inplace=False),
    )


class MCATSNN(nn.Module):


    def __init__(self, input_dim: int, n_classes: int = 4):
        super().__init__()
        hidden = [256, 256, 256, 256]
        layers: List[nn.Module] = [snn_block(input_dim, hidden[0])]
        for index in range(len(hidden) - 1):
            layers.append(snn_block(hidden[index], hidden[index + 1], 0.25))
        self.fc_omic = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden[-1], n_classes)
        init_max_weights(self)

    def forward(self, data_omics: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.fc_omic(data_omics))


class SurvPathMaskedOmics(nn.Module):


    def __init__(
        self,
        composition: np.ndarray,
        dim_per_path_1: int = 8,
        dim_per_path_2: int = 16,
        dropout: float = 0.25,
        n_classes: int = 4,
    ):
        super().__init__()
        input_dim, n_pathways = composition.shape
        mask_1 = torch.repeat_interleave(
            torch.as_tensor(composition, dtype=torch.float32),
            dim_per_path_1,
            dim=1,
        )
        mask_2 = torch.zeros(
            dim_per_path_1 * n_pathways,
            dim_per_path_2 * n_pathways,
            dtype=torch.float32,
        )
        for index in range(n_pathways):
            row = index * dim_per_path_1
            col = index * dim_per_path_2
            mask_2[
                row : row + dim_per_path_1,
                col : col + dim_per_path_2,
            ] = 1.0
        self.register_buffer("mask_1", mask_1)
        self.register_buffer("mask_2", mask_2)
        self.fc_1_weight = nn.Parameter(
            torch.empty(input_dim, dim_per_path_1 * n_pathways)
        )
        self.fc_1_bias = nn.Parameter(torch.rand(dim_per_path_1 * n_pathways))
        self.fc_2_weight = nn.Parameter(
            torch.empty(
                dim_per_path_1 * n_pathways,
                dim_per_path_2 * n_pathways,
            )
        )
        self.fc_2_bias = nn.Parameter(torch.rand(dim_per_path_2 * n_pathways))
        nn.init.xavier_normal_(self.fc_1_weight)
        nn.init.xavier_normal_(self.fc_2_weight)
        encoded_dim = dim_per_path_2 * n_pathways
        self.to_logits = nn.Sequential(
            nn.Linear(encoded_dim, encoded_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoded_dim // 4, n_classes),
        )

    def forward(self, data_omics: torch.Tensor) -> torch.Tensor:
        hidden = (
            torch.matmul(data_omics, self.fc_1_weight * self.mask_1)
            + self.fc_1_bias
        )
        hidden = (
            torch.matmul(hidden, self.fc_2_weight * self.mask_2)
            + self.fc_2_bias
        )
        return self.to_logits(hidden)


def risk_from_logits(logits: torch.Tensor) -> torch.Tensor:
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1.0 - hazards, dim=1)
    return -torch.sum(survival, dim=1)


def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def predict_risk(model: nn.Module, features: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        risk = risk_from_logits(model(features))
    return risk.detach().cpu().numpy().astype(float)


def fit_model(
    model_name: str,
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    train_labels: pd.DataFrame,
    val_labels: pd.DataFrame,
    composition: Optional[np.ndarray],
    *,
    device: torch.device,
    n_classes: int,
    max_epochs: int,
    patience: int,
    survpath_lr: float,
    mcat_lr: float,
    weight_decay: float,
) -> Tuple[np.ndarray, Dict[str, object]]:
    if model_name == "SurvPath-MaskedOmics":
        if composition is None:
            raise ValueError("SurvPath-MaskedOmics requires a pathway composition")
        model: nn.Module = SurvPathMaskedOmics(
            composition=composition,
            dropout=0.25,
            n_classes=n_classes,
        )
        learning_rate = survpath_lr
    elif model_name == "MCAT-SNN":
        model = MCATSNN(input_dim=train_x.shape[1], n_classes=n_classes)
        learning_rate = mcat_lr
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_t = torch.from_numpy(train_x).to(device)
    val_t = torch.from_numpy(val_x).to(device)
    test_t = torch.from_numpy(test_x).to(device)
    train_times = train_labels["time"].to_numpy(dtype=float)
    train_events_np = train_labels["event"].to_numpy(dtype=int)
    bin_edges = make_discrete_bins(train_times, train_events_np, n_classes)
    train_bins = torch.as_tensor(
        assign_discrete_bins(train_times, bin_edges),
        dtype=torch.long,
        device=device,
    )
    train_events = torch.as_tensor(
        train_events_np, dtype=torch.float32, device=device
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = -np.inf
    best_epoch = 0
    stale = 0
    final_loss = np.nan
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        epochs_run = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_t)
        loss = nll_surv_loss(logits, train_bins, train_events)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite {model_name} loss at epoch {epoch}"
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())

        val_risk = predict_risk(model, val_t)
        val_c = safe_cindex(val_labels, val_risk)
        if np.isfinite(val_c) and val_c > best_val + 1e-8:
            best_val = val_c
            best_epoch = epoch
            best_state = clone_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError(f"{model_name} produced no valid validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_risk = predict_risk(model, test_t)
    info: Dict[str, object] = {
        "parameter_count": int(
            sum(parameter.numel() for parameter in model.parameters())
        ),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "best_val_c_index": best_val,
        "final_train_loss": final_loss,
        "n_time_bins": n_classes,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "bin_edges": json.dumps(bin_edges.tolist()),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_risk, info


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    ok = results[results["status"] == "ok"].copy()
    for (dataset, model), group in ok.groupby(["dataset", "model"], sort=True):
        row: Dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "runs": int(len(group)),
        }
        for metric in ("c_index", "auc_1y", "auc_3y", "auc_5y"):
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std(ddof=1))
            row[f"{metric}_valid"] = int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run no-WSI SurvPath/MCAT repository counterparts"
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["SurvPath-MaskedOmics", "MCAT-SNN"],
        default=["SurvPath-MaskedOmics", "MCAT-SNN"],
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--mrna-features", type=int, default=1000)
    parser.add_argument("--mirna-features", type=int, default=300)
    parser.add_argument("--n-classes", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--survpath-lr", type=float, default=1e-4)
    parser.add_argument("--mcat-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.datasets = [item.strip().upper() for item in args.datasets]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    fold_file = args.results_dir / "survpath_mcat_fold_results.csv"
    oof_file = args.results_dir / "survpath_mcat_oof_predictions.csv"
    rows: List[Dict[str, object]] = []
    oof_rows: List[Dict[str, object]] = []
    if args.resume and fold_file.exists():
        rows = pd.read_csv(fold_file).to_dict("records")
    if args.resume and oof_file.exists():
        oof_rows = pd.read_csv(oof_file).to_dict("records")
    completed = {
        (str(row["dataset"]), str(row["model"]), int(row["seed"]), int(row["fold"]))
        for row in rows
        if str(row.get("status")) == "ok"
    }

    protocol = {
        "purpose": (
            "No-WSI implementations of the omics-only counterparts distributed "
            "with the reviewer-named SurvPath and MCAT repositories"
        ),
        "claim_boundary": {
            "SurvPath-MaskedOmics": (
                "Official SurvPath pathway-masked omics-only ablation; not full "
                "WSI+omics SurvPath"
            ),
            "MCAT-SNN": (
                "Official MCAT genomic-only SNN baseline; not full WSI+omics MCAT"
            ),
        },
        "source_adjustments": [
            (
                "MCAT models/model_genomic.py calls super(MaxNet, self) although "
                "the class is named SNN; adapter corrects it to standard super()."
            ),
            (
                "SurvPath MaskedOmics is transcribed without the unused "
                "x_transformers import, which is absent from the local environment."
            ),
            (
                "mRNA and miRNA are concatenated after separate fold-local scaling; "
                "the SurvPath mask concatenates the corresponding mRNA/miRNA pathway "
                "membership matrices."
            ),
        ],
        "datasets": args.datasets,
        "seeds": args.seeds,
        "outer_folds": args.folds,
        "inner_validation_fraction": 0.2,
        "preprocessing": (
            "inner-training-only variance selection and separate StandardScaler; "
            f"{args.mrna_features} mRNA + {args.mirna_features} miRNA maximum"
        ),
        "time_discretization": (
            f"{args.n_classes} event-time quantile bins fit on inner training only; "
            "validation/test values clipped to the outer intervals"
        ),
        "loss": "official discrete-time NLL; censoring indicator=1-event",
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "learning_rates": {
            "SurvPath-MaskedOmics": args.survpath_lr,
            "MCAT-SNN": args.mcat_lr,
        },
        "weight_decay": args.weight_decay,
        "metrics": {"Harrell_C_index": "all folds", **HORIZONS},
        "device": str(device),
        "python": sys.executable,
    }
    (args.results_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("=== No-WSI SurvPath/MCAT counterparts ===", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"Models  : {args.models}", flush=True)
    print(f"Device  : {device}", flush=True)
    resources = (
        load_prior_resources()
        if "SurvPath-MaskedOmics" in args.models
        else None
    )

    for dataset in args.datasets:
        x_mrna, x_mirna, gene_names, mirna_names, labels = load_dataset(dataset)
        for seed in args.seeds:
            splitter = StratifiedKFold(
                n_splits=args.folds, shuffle=True, random_state=seed
            )
            for fold, (outer_train_idx, test_idx) in enumerate(
                splitter.split(np.arange(len(labels)), labels["event"].to_numpy()),
                start=1,
            ):
                if args.max_folds is not None and fold > args.max_folds:
                    continue
                local_train, local_val = train_test_split(
                    np.arange(len(outer_train_idx)),
                    test_size=0.2,
                    stratify=labels.iloc[outer_train_idx]["event"].to_numpy(),
                    random_state=seed,
                )
                train_idx = outer_train_idx[local_train]
                val_idx = outer_train_idx[local_val]
                transformed, mrna_keep, mirna_keep = prepare_modalities(
                    x_mrna,
                    x_mirna,
                    train_idx,
                    [train_idx, val_idx, test_idx],
                    args.mrna_features,
                    args.mirna_features,
                )
                train_x, val_x, test_x = transformed
                train_labels = labels.iloc[train_idx].reset_index(drop=True)
                val_labels = labels.iloc[val_idx].reset_index(drop=True)
                test_labels = labels.iloc[test_idx].reset_index(drop=True)
                outer_train_labels = labels.iloc[
                    outer_train_idx
                ].reset_index(drop=True)

                composition: Optional[np.ndarray] = None
                active_pathways = np.nan
                if "SurvPath-MaskedOmics" in args.models:
                    if resources is None:
                        raise RuntimeError("Pathway resources were not loaded")
                    (
                        gene_mask,
                        mirna_mask,
                        _,
                        _,
                        _,
                        prior_stats,
                        _,
                    ) = build_fold_priors(
                        [gene_names[index] for index in mrna_keep],
                        [mirna_names[index] for index in mirna_keep],
                        resources,
                    )
                    composition = np.concatenate(
                        [gene_mask, mirna_mask], axis=0
                    ).astype(np.float32)
                    active_pathways = prior_stats["active_pathways"]

                for model_name in args.models:
                    key = (dataset, model_name, seed, fold)
                    if key in completed:
                        print(f"[SKIP] {key}", flush=True)
                        continue
                    set_seeds(seed)
                    started = time.perf_counter()
                    row: Dict[str, object] = {
                        "dataset": dataset,
                        "model": model_name,
                        "seed": seed,
                        "fold": fold,
                        "status": "failed",
                        "reason": "",
                        "n_samples": len(labels),
                        "n_outer_train": len(outer_train_idx),
                        "n_train": len(train_idx),
                        "n_val": len(val_idx),
                        "n_test": len(test_idx),
                        "train_events": int(train_labels["event"].sum()),
                        "val_events": int(val_labels["event"].sum()),
                        "test_events": int(test_labels["event"].sum()),
                        "selected_mRNA_features": len(mrna_keep),
                        "selected_miRNA_features": len(mirna_keep),
                        "active_pathways": (
                            active_pathways
                            if model_name == "SurvPath-MaskedOmics"
                            else np.nan
                        ),
                    }
                    print(
                        f"[RUN] {dataset} {model_name} seed={seed} fold={fold}",
                        flush=True,
                    )
                    try:
                        risk, info = fit_model(
                            model_name,
                            train_x,
                            val_x,
                            test_x,
                            train_labels,
                            val_labels,
                            composition,
                            device=device,
                            n_classes=args.n_classes,
                            max_epochs=args.max_epochs,
                            patience=args.patience,
                            survpath_lr=args.survpath_lr,
                            mcat_lr=args.mcat_lr,
                            weight_decay=args.weight_decay,
                        )
                        row.update(info)
                        row["c_index"] = safe_cindex(test_labels, risk)
                        for metric, horizon in HORIZONS.items():
                            row[metric] = safe_td_auc(
                                outer_train_labels,
                                test_labels,
                                risk,
                                horizon,
                            )
                        row["status"] = "ok"
                        for sample_index, sample_risk in zip(test_idx, risk):
                            label = labels.iloc[int(sample_index)]
                            oof_rows.append(
                                {
                                    "dataset": dataset,
                                    "model": model_name,
                                    "seed": seed,
                                    "fold": fold,
                                    "sample_index": int(sample_index),
                                    "sample_id": str(label["sample_id"]),
                                    "time": float(label["time"]),
                                    "event": int(label["event"]),
                                    "risk": float(sample_risk),
                                }
                            )
                    except Exception as exc:
                        row["reason"] = f"{type(exc).__name__}: {exc}"
                        print(f"[FAILED] {row['reason']}", flush=True)
                    row["runtime_seconds"] = time.perf_counter() - started
                    row["device"] = str(device)
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(fold_file, index=False)
                    pd.DataFrame(oof_rows).to_csv(oof_file, index=False)
                    if row["status"] == "ok":
                        print(
                            f"[DONE] C={row['c_index']:.4f}, "
                            f"epoch={row['best_epoch']}, "
                            f"{row['runtime_seconds']:.1f}s",
                            flush=True,
                        )

    results = pd.DataFrame(rows)
    summary = summarize(results)
    summary.to_csv(
        args.results_dir / "survpath_mcat_summary.csv", index=False
    )
    failures = results[results["status"] != "ok"].copy()
    failures.to_csv(args.results_dir / "failed_runs.csv", index=False)
    print(f"Results   : {args.results_dir}", flush=True)
    print(f"Successful: {(results['status'] == 'ok').sum()}/{len(results)}", flush=True)
    print(f"Failed    : {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
