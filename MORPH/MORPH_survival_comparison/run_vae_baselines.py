from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
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
    cox_breslow_loss,
    load_dataset,
    safe_cindex,
    safe_td_auc,
    set_seeds,
)


DEFAULT_DATASETS = ("LAML", "SKCM")
DEFAULT_SEEDS = (1028, 2024, 3407)
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "vae"
OFFICIAL_ROOT = (
    PROJECT_ROOT
    / "external_sources"
    / "omiembed"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OFFICIAL_NETWORKS = load_module(
    "omiembed_official_networks", OFFICIAL_ROOT / "models" / "networks.py"
)
OFFICIAL_LOSSES = load_module(
    "omiembed_official_losses", OFFICIAL_ROOT / "models" / "losses.py"
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
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], np.ndarray, np.ndarray]:

    mrna_keep = select_fixed_count(x_mrna[inner_train_idx], mrna_features)
    mirna_keep = select_fixed_count(x_mirna[inner_train_idx], mirna_features)
    mrna_scaler = StandardScaler().fit(x_mrna[inner_train_idx][:, mrna_keep])
    mirna_scaler = StandardScaler().fit(x_mirna[inner_train_idx][:, mirna_keep])
    transformed: List[Tuple[np.ndarray, np.ndarray]] = []
    for indices in all_indices:
        transformed.append(
            (
                mrna_scaler.transform(x_mrna[indices][:, mrna_keep]).astype(np.float32),
                mirna_scaler.transform(x_mirna[indices][:, mirna_keep]).astype(np.float32),
            )
        )
    return transformed, mrna_keep, mirna_keep


def kl_loss(mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1.0 + log_var - mean.pow(2) - log_var.exp())


def make_mtlr_targets(
    times: np.ndarray,
    events: np.ndarray,
    t_max: float,
    time_num: int,
) -> torch.Tensor:

    time_points = np.linspace(0.0, t_max * 1.1, time_num + 1)
    targets = np.zeros((len(times), time_num + 1), dtype=np.float32)
    for i, (time_value, event) in enumerate(zip(times, events)):
        time_index = int(np.argmin(np.abs(time_points[:-1] - float(time_value))))
        if int(event) == 1:
            targets[i, time_index] = 1.0
        else:
            targets[i, time_index:] = 1.0
    return torch.from_numpy(targets)


class OfficialOmiEmbedSurvival(nn.Module):


    def __init__(self, n_mrna: int, n_mirna: int, latent_dim: int, time_num: int):
        super().__init__()



        self.embed = OFFICIAL_NETWORKS.FcVaeAB(
            [n_mrna, n_mirna],
            norm_layer=nn.BatchNorm1d,
            leaky_slope=0.2,
            dropout_p=0.2,
            latent_dim=latent_dim,
        )
        self.down = OFFICIAL_NETWORKS.MultiFcSurvival(
            time_num=time_num,
            latent_dim=latent_dim,
            norm_layer=nn.BatchNorm1d,
            leaky_slope=0.2,
            dropout_p=0.2,
        )
        self.time_num = time_num
        self.register_buffer("tri_1", torch.tril(torch.ones(time_num, time_num + 1)))
        self.register_buffer("tri_2", torch.tril(torch.ones(time_num + 1, time_num + 1)))

    def forward(
        self, mrna: torch.Tensor, mirna: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor]:
        _, recon, mean, log_var = self.embed([mrna, mirna])
        logits = self.down(mean)
        return logits, recon, mean, log_var

    def risk_from_logits(self, logits: torch.Tensor) -> torch.Tensor:



        log_phi = torch.mm(logits, self.tri_1)
        density = torch.softmax(log_phi, dim=1)
        survival = torch.mm(density, self.tri_2).clamp_min(1e-8)
        hazard = density[:, :-1] / survival[:, 1:]
        return torch.cumsum(hazard, dim=1).sum(dim=1)


class VAECox(nn.Module):


    def __init__(self, n_mrna: int, n_mirna: int, latent_dim: int = 32):
        super().__init__()
        self.enc_mrna = nn.Sequential(
            nn.Linear(n_mrna, 128), nn.ReLU(), nn.Dropout(0.2)
        )
        self.enc_mirna = nn.Sequential(
            nn.Linear(n_mirna, 64), nn.ReLU(), nn.Dropout(0.2)
        )
        self.mean = nn.Linear(192, latent_dim)
        self.log_var = nn.Linear(192, latent_dim)
        self.dec_mrna = nn.Linear(latent_dim, n_mrna)
        self.dec_mirna = nn.Linear(latent_dim, n_mirna)
        self.cox = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(
        self, mrna: torch.Tensor, mirna: torch.Tensor, sample: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = torch.cat([self.enc_mrna(mrna), self.enc_mirna(mirna)], dim=1)
        mean = self.mean(hidden)
        log_var = self.log_var(hidden).clamp(-8.0, 8.0)
        if sample:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * log_var)
        else:
            latent = mean
        return (
            self.cox(latent).squeeze(1),
            self.dec_mrna(latent),
            self.dec_mirna(latent),
            mean,
            log_var,
        )


def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def predict_omiembed(
    model: OfficialOmiEmbedSurvival, mrna: torch.Tensor, mirna: torch.Tensor
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(mrna, mirna)[0]
        risk = model.risk_from_logits(logits)
    return risk.detach().cpu().numpy().astype(float)


def fit_official_omiembed(
    train_mrna: np.ndarray,
    train_mirna: np.ndarray,
    val_mrna: np.ndarray,
    val_mirna: np.ndarray,
    test_mrna: np.ndarray,
    test_mirna: np.ndarray,
    train_labels: pd.DataFrame,
    val_labels: pd.DataFrame,
    *,
    device: torch.device,
    latent_dim: int,
    time_num: int,
    phase_epochs: Tuple[int, int, int],
) -> Tuple[np.ndarray, Dict[str, object]]:
    model = OfficialOmiEmbedSurvival(
        train_mrna.shape[1], train_mirna.shape[1], latent_dim, time_num
    ).to(device)
    embed_optimizer = torch.optim.Adam(
        model.embed.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-4
    )
    down_optimizer = torch.optim.Adam(
        model.down.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-4
    )
    tensors = [
        torch.from_numpy(array).to(device)
        for array in (
            train_mrna,
            train_mirna,
            val_mrna,
            val_mirna,
            test_mrna,
            test_mirna,
        )
    ]
    tr_m, tr_mi, va_m, va_mi, te_m, te_mi = tensors
    t_max = float(train_labels["time"].max())
    y_true = make_mtlr_targets(
        train_labels["time"].to_numpy(),
        train_labels["event"].to_numpy(),
        t_max,
        time_num,
    ).to(device)
    events = torch.as_tensor(
        train_labels["event"].to_numpy(), dtype=torch.float32, device=device
    )

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = -np.inf
    best_epoch = 0
    total_epoch = 0
    final_loss = np.nan

    for phase, epochs in enumerate(phase_epochs, start=1):
        for _ in range(epochs):
            total_epoch += 1
            model.train()
            embed_optimizer.zero_grad(set_to_none=True)
            down_optimizer.zero_grad(set_to_none=True)
            logits, recon, mean, log_var = model(tr_m, tr_mi)
            reconstruction = 0.5 * (
                torch.mean((recon[0] - tr_m) ** 2)
                + torch.mean((recon[1] - tr_mi) ** 2)
            )
            embedding = reconstruction + 0.01 * kl_loss(mean, log_var)
            survival = OFFICIAL_LOSSES.MTLR_survival_loss(
                logits, y_true, events, model.tri_1, reduction="mean"
            )
            if phase == 1:
                loss = embedding
            elif phase == 2:


                loss = OFFICIAL_LOSSES.MTLR_survival_loss(
                    model.down(mean.detach()),
                    y_true,
                    events,
                    model.tri_1,
                    reduction="mean",
                )
            else:
                loss = survival + 0.001 * embedding
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite OmiEmbed loss in phase {phase}, epoch {total_epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            if phase in (1, 3):
                embed_optimizer.step()
            if phase in (2, 3):
                down_optimizer.step()
            final_loss = float(loss.detach().cpu())

            if phase >= 2:
                val_risk = predict_omiembed(model, va_m, va_mi)
                val_c = safe_cindex(val_labels, val_risk)
                if np.isfinite(val_c) and val_c > best_val + 1e-8:
                    best_val = val_c
                    best_epoch = total_epoch
                    best_state = clone_state(model)

    if best_state is None:
        raise RuntimeError("OmiEmbed produced no valid validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    test_risk = predict_omiembed(model, te_m, te_mi)
    info = {
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best_epoch,
        "best_val_c_index": best_val,
        "final_train_loss": final_loss,
        "time_num": time_num,
        "latent_dim": latent_dim,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_risk, info


def fit_vae_cox(
    train_mrna: np.ndarray,
    train_mirna: np.ndarray,
    val_mrna: np.ndarray,
    val_mirna: np.ndarray,
    test_mrna: np.ndarray,
    test_mirna: np.ndarray,
    train_labels: pd.DataFrame,
    val_labels: pd.DataFrame,
    *,
    device: torch.device,
    epochs: int,
    patience: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    model = VAECox(train_mrna.shape[1], train_mirna.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    tensors = [
        torch.from_numpy(array).to(device)
        for array in (
            train_mrna,
            train_mirna,
            val_mrna,
            val_mirna,
            test_mrna,
            test_mirna,
        )
    ]
    tr_m, tr_mi, va_m, va_mi, te_m, te_mi = tensors
    times = torch.as_tensor(
        train_labels["time"].to_numpy(), dtype=torch.float32, device=device
    )
    events = torch.as_tensor(
        train_labels["event"].to_numpy(), dtype=torch.float32, device=device
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = -np.inf
    best_epoch = 0
    stale = 0
    final_loss = np.nan
    epochs_run = 0
    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        risk, rec_m, rec_mi, mean, log_var = model(tr_m, tr_mi, sample=True)
        cox = cox_breslow_loss(risk, times, events)
        reconstruction = 0.5 * (
            torch.mean((rec_m - tr_m) ** 2)
            + torch.mean((rec_mi - tr_mi) ** 2)
        )
        loss = cox + 0.05 * reconstruction + 0.001 * kl_loss(mean, log_var)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite VAE-Cox loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())

        model.eval()
        with torch.no_grad():
            val_risk = model(va_m, va_mi, sample=False)[0].detach().cpu().numpy()
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
        raise RuntimeError("VAE-Cox produced no valid validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    with torch.no_grad():
        test_risk = model(te_m, te_mi, sample=False)[0].detach().cpu().numpy()
    info = {
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "best_val_c_index": best_val,
        "final_train_loss": final_loss,
        "time_num": np.nan,
        "latent_dim": 32,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return test_risk.astype(float), info


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
        description="Run official-architecture OmiEmbed and VAE-Cox survival baselines"
    )
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--models", nargs="+", choices=["OmiEmbed-Survival", "VAE-Cox"],
        default=["OmiEmbed-Survival", "VAE-Cox"],
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--mrna-features", type=int, default=1000)
    parser.add_argument("--mirna-features", type=int, default=300)
    parser.add_argument("--omi-latent-dim", type=int, default=128)
    parser.add_argument("--omi-time-num", type=int, default=256)
    parser.add_argument("--omi-phase-epochs", nargs=3, type=int, default=[50, 50, 100])
    parser.add_argument("--vae-cox-epochs", type=int, default=100)
    parser.add_argument("--vae-cox-patience", type=int, default=15)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.datasets = [item.strip().upper() for item in args.datasets]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    fold_file = args.results_dir / "vae_baseline_fold_results.csv"
    oof_file = args.results_dir / "vae_baseline_oof_predictions.csv"
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
        "purpose": "VAE multi-omics survival baselines",
        "models": {
            "OmiEmbed-Survival": {
                "source": str(OFFICIAL_ROOT),
                "source_version": "v0.2.1 (Zenodo record 4854112)",
                "official_components": [
                    "models.networks.FcVaeAB",
                    "models.networks.MultiFcSurvival",
                    "models.losses.MTLR_survival_loss",
                ],
                "adapter": "mRNA->omics A, miRNA->omics B; local nested CV interface",
                "phases": {
                    "VAE_pretrain": args.omi_phase_epochs[0],
                    "survival_head": args.omi_phase_epochs[1],
                    "joint_finetune": args.omi_phase_epochs[2],
                },
                "time_intervals": args.omi_time_num,
                "latent_dim": args.omi_latent_dim,
            },
            "VAE-Cox": {
                "role": "representative related VAE-based multi-omics integration baseline",
                "objective": "Cox partial likelihood + reconstruction + KL",
                "epochs": args.vae_cox_epochs,
                "patience": args.vae_cox_patience,
            },
        },
        "datasets": args.datasets,
        "seeds": args.seeds,
        "outer_folds": args.folds,
        "inner_validation_fraction": 0.2,
        "preprocessing": (
            "inner-training-only variance selection and StandardScaler; "
            f"{args.mrna_features} mRNA + {args.mirna_features} miRNA maximum"
        ),
        "metrics": {"Harrell_C_index": "all folds", **HORIZONS},
        "device": str(device),
        "python": sys.executable,
    }
    (args.results_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("VAE survival baselines", flush=True)
    print(f"Datasets: {args.datasets}", flush=True)
    print(f"Models  : {args.models}", flush=True)
    print(f"Device  : {device}", flush=True)
    for dataset in args.datasets:
        x_mrna, x_mirna, _, _, labels = load_dataset(dataset)
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
                (train_mrna, train_mirna), (val_mrna, val_mirna), (
                    test_mrna,
                    test_mirna,
                ) = transformed
                train_labels = labels.iloc[train_idx].reset_index(drop=True)
                val_labels = labels.iloc[val_idx].reset_index(drop=True)
                test_labels = labels.iloc[test_idx].reset_index(drop=True)
                outer_train_labels = labels.iloc[outer_train_idx].reset_index(drop=True)

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
                        "n_train": len(train_idx),
                        "n_val": len(val_idx),
                        "n_test": len(test_idx),
                        "train_events": int(train_labels["event"].sum()),
                        "val_events": int(val_labels["event"].sum()),
                        "test_events": int(test_labels["event"].sum()),
                        "selected_mRNA_features": len(mrna_keep),
                        "selected_miRNA_features": len(mirna_keep),
                    }
                    print(
                        f"[RUN] {dataset} {model_name} seed={seed} fold={fold}",
                        flush=True,
                    )
                    try:
                        if model_name == "OmiEmbed-Survival":
                            risk, info = fit_official_omiembed(
                                train_mrna,
                                train_mirna,
                                val_mrna,
                                val_mirna,
                                test_mrna,
                                test_mirna,
                                train_labels,
                                val_labels,
                                device=device,
                                latent_dim=args.omi_latent_dim,
                                time_num=args.omi_time_num,
                                phase_epochs=tuple(args.omi_phase_epochs),
                            )
                        else:
                            risk, info = fit_vae_cox(
                                train_mrna,
                                train_mirna,
                                val_mrna,
                                val_mirna,
                                test_mrna,
                                test_mirna,
                                train_labels,
                                val_labels,
                                device=device,
                                epochs=args.vae_cox_epochs,
                                patience=args.vae_cox_patience,
                            )
                        row.update(info)
                        row["c_index"] = safe_cindex(test_labels, risk)
                        for metric, horizon in HORIZONS.items():
                            row[metric] = safe_td_auc(
                                outer_train_labels, test_labels, risk, horizon
                            )
                        row["status"] = "ok"
                        for sample_index, sample_risk in zip(test_idx, risk):
                            oof_rows.append(
                                {
                                    "dataset": dataset,
                                    "model": model_name,
                                    "seed": seed,
                                    "fold": fold,
                                    "sample_index": int(sample_index),
                                    "sample_id": labels.iloc[sample_index]["sample_id"],
                                    "time": float(labels.iloc[sample_index]["time"]),
                                    "event": int(labels.iloc[sample_index]["event"]),
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
                            f"{row['runtime_seconds']:.1f}s",
                            flush=True,
                        )

    results = pd.DataFrame(rows)
    summary = summarize(results)
    summary.to_csv(args.results_dir / "vae_baseline_summary.csv", index=False)
    failures = results[results["status"] != "ok"].copy()
    failures.to_csv(args.results_dir / "failed_runs.csv", index=False)
    print(f"Results: {args.results_dir}", flush=True)
    print(f"Successful runs: {(results['status'] == 'ok').sum()}/{len(results)}", flush=True)


if __name__ == "__main__":
    main()
