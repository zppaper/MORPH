from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import pickle
import platform
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import sklearn
import sksurv
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxnetSurvivalAnalysis


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE
DATA_ROOT = Path(os.environ.get("MORPH_SURVIVAL_DATA_ROOT", str(PACKAGE_ROOT / "data" / "Survival_Analysis_Two_Omics")))
sys.path.insert(0, str(HERE))

import train_morph_survival as morph


def load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


vae_source = HERE / "run_vae_baselines.py"
surv_source = HERE / "run_survpath_mcat_omics.py"
morph_impl = load_source_module("survival_model", HERE / "survival_model.py")
vae_impl = load_source_module("final_protocol_vae_impl", vae_source)
surv_impl = load_source_module("final_protocol_surv_impl", surv_source)


DATASETS = ("LAML", "SKCM")
SEEDS = (1028, 2024, 3407)
MODELS = (
    "MORPH-Cox",
    "LASSO-Cox",
    "VAE-Cox",
    "Adapted OmiEmbed-Survival",
    "SurvPath-derived MaskedOmics",
    "MCAT-derived SNN",
)
NEURAL_MODELS = set(MODELS) - {"LASSO-Cox"}
SURVIVAL_HORIZONS = {"auc_1y": 365.0, "auc_3y": 1095.0}
FINAL_FEATURE_COUNTS = {
    "LAML": {"mRNA": 798, "miRNA": 200},
    "SKCM": {"mRNA": 997, "miRNA": 300},
}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    for attempt in range(8):
        try:
            os.replace(str(temporary), str(path))
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def atomic_json(payload: Dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for attempt in range(8):
        try:
            os.replace(str(temporary), str(path))
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.25 * (attempt + 1))


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def torch_model_size_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(buffer.tell())


def pickle_size_bytes(model: object) -> int:
    return int(len(pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_path_key(path: Path) -> str:
    for prefix, root in (("data", DATA_ROOT), ("priors", morph.PATHWAY_DIR), ("code", HERE)):
        try:
            return prefix + "/" + path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    raise ValueError(f"Input outside configured directories: {path}")


def experiment_input_hashes(datasets: Sequence[str]) -> Dict[str, str]:
    paths = [
        HERE / "run_survival_protocol.py",
        HERE / "train_morph_survival.py",
        HERE / "survival_model.py",
        vae_source,
        surv_source,
        HERE / "external_sources" / "omiembed" / "models" / "networks.py",
        HERE / "external_sources" / "omiembed" / "models" / "losses.py",
        morph.GMT_GENE_FILE,
        morph.GMT_MIRNA_FILE,
        morph.INTERACTION_FILE,
        morph.DRIVER_FILE,
    ]
    for dataset in datasets:
        paths.extend(data_file_paths(dataset))
    return {
        input_path_key(path): file_sha256(path)
        for path in paths
    }


def parameter_info(model: nn.Module) -> Dict[str, object]:
    return {
        "total_parameter_count": int(sum(item.numel() for item in model.parameters())),
        "trainable_parameter_count": int(
            sum(item.numel() for item in model.parameters() if item.requires_grad)
        ),
        "nonzero_parameter_count": np.nan,
        "model_size_bytes": torch_model_size_bytes(model),
    }


def timed_prediction(
    predict: Callable[[], np.ndarray],
    device: torch.device,
    repeats: int,
) -> Tuple[np.ndarray, float]:
    for _ in range(2):
        predict()
    synchronize(device)
    durations: List[float] = []
    risk: Optional[np.ndarray] = None
    for _ in range(max(1, repeats)):
        synchronize(device)
        started = time.perf_counter()
        risk = np.asarray(predict(), dtype=float)
        synchronize(device)
        durations.append(time.perf_counter() - started)
    if risk is None:
        raise RuntimeError("Prediction timing produced no output")
    return risk, float(np.median(durations))


def select_features(x_train: np.ndarray, count: int) -> Tuple[np.ndarray, np.ndarray]:
    variance = np.nanvar(x_train, axis=0)
    valid = np.flatnonzero(np.isfinite(variance) & (variance > 0))
    if valid.size == 0:
        raise ValueError("No finite nonconstant features in outer training fold")
    order = np.argsort(variance[valid], kind="stable")[::-1]
    keep = valid[order[: min(count, valid.size)]]
    return keep, variance[keep]


def prepare_common_fold(
    x_mrna: np.ndarray,
    x_mirna: np.ndarray,
    outer_train_idx: np.ndarray,
    test_idx: np.ndarray,
    mrna_count: int,
    mirna_count: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    mrna_keep, mrna_variance = select_features(x_mrna[outer_train_idx], mrna_count)
    mirna_keep, mirna_variance = select_features(x_mirna[outer_train_idx], mirna_count)
    mrna_scaler = StandardScaler().fit(x_mrna[outer_train_idx][:, mrna_keep])
    mirna_scaler = StandardScaler().fit(x_mirna[outer_train_idx][:, mirna_keep])
    train_mrna = mrna_scaler.transform(
        x_mrna[outer_train_idx][:, mrna_keep]
    ).astype(np.float32)
    train_mirna = mirna_scaler.transform(
        x_mirna[outer_train_idx][:, mirna_keep]
    ).astype(np.float32)
    test_mrna = mrna_scaler.transform(x_mrna[test_idx][:, mrna_keep]).astype(np.float32)
    test_mirna = mirna_scaler.transform(x_mirna[test_idx][:, mirna_keep]).astype(np.float32)
    return (
        train_mrna,
        train_mirna,
        test_mrna,
        test_mirna,
        mrna_keep,
        mirna_keep,
        mrna_variance,
        mirna_variance,
        time.perf_counter() - started,
    )


def fit_lasso(
    train_x: np.ndarray,
    test_x: np.ndarray,
    labels: pd.DataFrame,
    seed: int,
    inner_folds: int,
    n_alphas: int,
    alpha_min_ratio: float,
    max_iter: int,
    repeats: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    started = time.perf_counter()
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    score_rows: List[np.ndarray] = []
    for inner_train, inner_val in splitter.split(
        np.arange(len(labels)), labels["event"].to_numpy()
    ):
        path = CoxnetSurvivalAnalysis(
            l1_ratio=1.0,
            n_alphas=n_alphas,
            alpha_min_ratio=alpha_min_ratio,
            max_iter=max_iter,
        )
        path.fit(train_x[inner_train], morph.make_survival_array(labels.iloc[inner_train]))
        scores = np.full(n_alphas, np.nan, dtype=float)
        available = min(n_alphas, len(path.alphas_))
        for index in range(available):
            try:
                prediction = path.predict(train_x[inner_val], alpha=float(path.alphas_[index]))
                if float(np.std(prediction)) > 1e-12:
                    scores[index] = morph.safe_cindex(labels.iloc[inner_val], prediction)
            except Exception:
                continue
        score_rows.append(scores)
    score_matrix = np.vstack(score_rows)
    valid_counts = np.sum(np.isfinite(score_matrix), axis=0)
    sums = np.nansum(score_matrix, axis=0)
    mean_scores = np.divide(
        sums,
        valid_counts,
        out=np.full_like(sums, np.nan, dtype=float),
        where=valid_counts > 0,
    )
    if not np.isfinite(mean_scores).any():
        raise RuntimeError("No valid LASSO alpha ratio in inner cross-validation")
    best_index = int(np.nanargmax(mean_scores))
    final_model = CoxnetSurvivalAnalysis(
        l1_ratio=1.0,
        n_alphas=n_alphas,
        alpha_min_ratio=alpha_min_ratio,
        max_iter=max_iter,
    )
    final_model.fit(train_x, morph.make_survival_array(labels))
    selected_index = min(best_index, len(final_model.alphas_) - 1)
    alpha = float(final_model.alphas_[selected_index])
    alpha_ratio = float(alpha / final_model.alphas_[0])
    training_seconds = time.perf_counter() - started
    risk, inference_seconds = timed_prediction(
        lambda: final_model.predict(test_x, alpha=alpha), torch.device("cpu"), repeats
    )
    coefficients = np.asarray(final_model.coef_[:, selected_index])
    info = {
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "total_parameter_count": int(coefficients.size),
        "trainable_parameter_count": int(coefficients.size),
        "nonzero_parameter_count": int(np.count_nonzero(np.abs(coefficients) > 1e-12)),
        "model_size_bytes": pickle_size_bytes(final_model),
        "alpha": alpha,
        "alpha_ratio": alpha_ratio,
        "lasso_path_index": selected_index,
        "lasso_inner_mean_c_index": float(mean_scores[best_index]),
        "best_epoch": np.nan,
        "epochs_run": np.nan,
    }
    return risk, info


def initialize_neural_timing(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    return time.perf_counter()


def finish_neural_training(started: float, device: torch.device) -> Tuple[float, float]:
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak = (
        float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))
        if device.type == "cuda"
        else np.nan
    )
    return elapsed, peak


def fit_vae_cox(
    train_mrna: np.ndarray,
    train_mirna: np.ndarray,
    test_mrna: np.ndarray,
    test_mirna: np.ndarray,
    labels: pd.DataFrame,
    local_train: np.ndarray,
    local_val: np.ndarray,
    device: torch.device,
    epochs: int,
    patience: int,
    repeats: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    started = initialize_neural_timing(device)
    model = vae_impl.VAECox(train_mrna.shape[1], train_mirna.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    train_m = torch.from_numpy(train_mrna).to(device)
    train_mi = torch.from_numpy(train_mirna).to(device)
    test_m = torch.from_numpy(test_mrna).to(device)
    test_mi = torch.from_numpy(test_mirna).to(device)
    time_t = torch.as_tensor(labels["time"].to_numpy(), dtype=torch.float32, device=device)
    event_t = torch.as_tensor(labels["event"].to_numpy(), dtype=torch.float32, device=device)
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
        risk, rec_m, rec_mi, mean, log_var = model(
            train_m[local_train], train_mi[local_train], sample=True
        )
        cox = morph.cox_breslow_loss(risk, time_t[local_train], event_t[local_train])
        reconstruction = 0.5 * (
            torch.mean((rec_m - train_m[local_train]) ** 2)
            + torch.mean((rec_mi - train_mi[local_train]) ** 2)
        )
        loss = cox + 0.05 * reconstruction + 0.001 * vae_impl.kl_loss(mean, log_var)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite VAE-Cox loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        model.eval()
        with torch.no_grad():
            validation = model(
                train_m[local_val], train_mi[local_val], sample=False
            )[0].detach().cpu().numpy()
        score = morph.safe_cindex(labels.iloc[local_val], validation)
        if np.isfinite(score) and score > best_val + 1e-8:
            best_val = score
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
    training_seconds, peak_memory = finish_neural_training(started, device)

    def predict() -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return model(test_m, test_mi, sample=False)[0].detach().cpu().numpy()

    risk, inference_seconds = timed_prediction(predict, device, repeats)
    info = parameter_info(model)
    info.update(
        {
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "peak_memory_mb": peak_memory,
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "best_val_c_index": best_val,
            "final_train_loss": final_loss,
            "alpha": np.nan,
            "alpha_ratio": np.nan,
        }
    )
    del model
    return risk, info


def fit_omiembed(
    train_mrna: np.ndarray,
    train_mirna: np.ndarray,
    test_mrna: np.ndarray,
    test_mirna: np.ndarray,
    labels: pd.DataFrame,
    local_train: np.ndarray,
    local_val: np.ndarray,
    device: torch.device,
    latent_dim: int,
    time_num: int,
    phase_epochs: Sequence[int],
    repeats: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    started = initialize_neural_timing(device)
    model = vae_impl.OfficialOmiEmbedSurvival(
        train_mrna.shape[1], train_mirna.shape[1], latent_dim, time_num
    ).to(device)
    embed_optimizer = torch.optim.Adam(
        model.embed.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-4
    )
    down_optimizer = torch.optim.Adam(
        model.down.parameters(), lr=1e-4, betas=(0.5, 0.999), weight_decay=1e-4
    )
    train_m = torch.from_numpy(train_mrna).to(device)
    train_mi = torch.from_numpy(train_mirna).to(device)
    test_m = torch.from_numpy(test_mrna).to(device)
    test_mi = torch.from_numpy(test_mirna).to(device)
    inner_labels = labels.iloc[local_train].reset_index(drop=True)
    y_true = vae_impl.make_mtlr_targets(
        inner_labels["time"].to_numpy(),
        inner_labels["event"].to_numpy(),
        float(inner_labels["time"].max()),
        time_num,
    ).to(device)
    events = torch.as_tensor(
        inner_labels["event"].to_numpy(), dtype=torch.float32, device=device
    )
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val = -np.inf
    best_epoch = 0
    total_epoch = 0
    final_loss = np.nan
    for phase, phase_length in enumerate(phase_epochs, start=1):
        for _ in range(int(phase_length)):
            total_epoch += 1
            model.train()
            embed_optimizer.zero_grad(set_to_none=True)
            down_optimizer.zero_grad(set_to_none=True)
            logits, recon, mean, log_var = model(
                train_m[local_train], train_mi[local_train]
            )
            reconstruction = 0.5 * (
                torch.mean((recon[0] - train_m[local_train]) ** 2)
                + torch.mean((recon[1] - train_mi[local_train]) ** 2)
            )
            embedding = reconstruction + 0.01 * vae_impl.kl_loss(mean, log_var)
            survival = vae_impl.OFFICIAL_LOSSES.MTLR_survival_loss(
                logits, y_true, events, model.tri_1, reduction="mean"
            )
            if phase == 1:
                loss = embedding
            elif phase == 2:
                loss = vae_impl.OFFICIAL_LOSSES.MTLR_survival_loss(
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
                validation = vae_impl.predict_omiembed(
                    model, train_m[local_val], train_mi[local_val]
                )
                score = morph.safe_cindex(labels.iloc[local_val], validation)
                if np.isfinite(score) and score > best_val + 1e-8:
                    best_val = score
                    best_epoch = total_epoch
                    best_state = clone_state(model)
    if best_state is None:
        raise RuntimeError("OmiEmbed produced no valid validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    training_seconds, peak_memory = finish_neural_training(started, device)

    def predict() -> np.ndarray:
        return vae_impl.predict_omiembed(model, test_m, test_mi)

    risk, inference_seconds = timed_prediction(predict, device, repeats)
    info = parameter_info(model)
    info.update(
        {
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "peak_memory_mb": peak_memory,
            "best_epoch": best_epoch,
            "epochs_run": total_epoch,
            "best_val_c_index": best_val,
            "final_train_loss": final_loss,
            "alpha": np.nan,
            "alpha_ratio": np.nan,
        }
    )
    del model
    return risk, info


def fit_discrete_model(
    model_name: str,
    train_x: np.ndarray,
    test_x: np.ndarray,
    labels: pd.DataFrame,
    local_train: np.ndarray,
    local_val: np.ndarray,
    composition: Optional[np.ndarray],
    device: torch.device,
    epochs: int,
    patience: int,
    repeats: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    started = initialize_neural_timing(device)
    if model_name == "SurvPath-derived MaskedOmics":
        if composition is None:
            raise ValueError("SurvPath-derived MaskedOmics requires a pathway mask")
        model: nn.Module = surv_impl.SurvPathMaskedOmics(
            composition=composition, dropout=0.25, n_classes=4
        )
        learning_rate = 1e-4
    else:
        model = surv_impl.MCATSNN(input_dim=train_x.shape[1], n_classes=4)
        learning_rate = 2e-4
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    train_t = torch.from_numpy(train_x).to(device)
    test_t = torch.from_numpy(test_x).to(device)
    inner_labels = labels.iloc[local_train].reset_index(drop=True)
    edges = surv_impl.make_discrete_bins(
        inner_labels["time"].to_numpy(), inner_labels["event"].to_numpy(), 4
    )
    bins = torch.as_tensor(
        surv_impl.assign_discrete_bins(inner_labels["time"].to_numpy(), edges),
        dtype=torch.long,
        device=device,
    )
    events = torch.as_tensor(
        inner_labels["event"].to_numpy(), dtype=torch.float32, device=device
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
        logits = model(train_t[local_train])
        loss = surv_impl.nll_surv_loss(logits, bins, events)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {model_name} loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        model.eval()
        with torch.no_grad():
            validation = surv_impl.risk_from_logits(
                model(train_t[local_val])
            ).detach().cpu().numpy()
        score = morph.safe_cindex(labels.iloc[local_val], validation)
        if np.isfinite(score) and score > best_val + 1e-8:
            best_val = score
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
    model.to(device).eval()
    training_seconds, peak_memory = finish_neural_training(started, device)

    def predict() -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return surv_impl.risk_from_logits(model(test_t)).detach().cpu().numpy()

    risk, inference_seconds = timed_prediction(predict, device, repeats)
    info = parameter_info(model)
    info.update(
        {
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "peak_memory_mb": peak_memory,
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "best_val_c_index": best_val,
            "final_train_loss": final_loss,
            "n_time_bins": 4,
            "bin_edges": json.dumps(edges.tolist()),
            "alpha": np.nan,
            "alpha_ratio": np.nan,
        }
    )
    del model
    return risk, info


def fit_morph(
    train_mrna: np.ndarray,
    train_mirna: np.ndarray,
    test_mrna: np.ndarray,
    test_mirna: np.ndarray,
    selected_genes: Sequence[str],
    selected_mirnas: Sequence[str],
    labels: pd.DataFrame,
    local_train: np.ndarray,
    local_val: np.ndarray,
    resources: morph.PriorResources,
    device: torch.device,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    repeats: int,
) -> Tuple[np.ndarray, Dict[str, object]]:
    (
        gene_mask,
        mirna_mask,
        reg_matrix,
        driver_idx,
        path_prior,
        prior_stats,
        _,
    ) = morph.build_fold_priors(selected_genes, selected_mirnas, resources)
    started = initialize_neural_timing(device)
    gene_mask_df = pd.DataFrame(
        gene_mask,
        index=list(selected_genes),
        columns=[f"path_{index}" for index in range(gene_mask.shape[1])],
    )
    mirna_mask_df = pd.DataFrame(
        mirna_mask,
        index=list(selected_mirnas),
        columns=[f"path_{index}" for index in range(mirna_mask.shape[1])],
    )
    model = morph_impl.PathformerBio(
        dim_mrna=train_mrna.shape[1],
        dim_mirna=train_mirna.shape[1],
        gene_path_mask_df=gene_mask_df,
        mirna_path_mask_df=mirna_mask_df,
        reg_matrix=reg_matrix,
        driver_idx=driver_idx,
        path_prior=path_prior,
        hidden_dim=128,
        depth=2,
        ff_dropout=0.2,
        classifier_dropout=0.3,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_m = torch.from_numpy(train_mrna).to(device)
    train_mi = torch.from_numpy(train_mirna).to(device)
    test_m = torch.from_numpy(test_mrna).to(device)
    test_mi = torch.from_numpy(test_mirna).to(device)
    time_t = torch.as_tensor(labels["time"].to_numpy(), dtype=torch.float32, device=device)
    event_t = torch.as_tensor(labels["event"].to_numpy(), dtype=torch.float32, device=device)
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
        risk = model(train_m[local_train], train_mi[local_train])
        risk = risk.squeeze(-1)
        loss = morph.cox_breslow_loss(risk, time_t[local_train], event_t[local_train])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite MORPH-Cox loss at epoch {epoch}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        model.eval()
        with torch.no_grad():
            validation = model(
                train_m[local_val], train_mi[local_val]
            ).squeeze(-1).detach().cpu().numpy()
        score = morph.safe_cindex(labels.iloc[local_val], validation)
        if np.isfinite(score) and score > best_val + 1e-8:
            best_val = score
            best_epoch = epoch
            best_state = clone_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("MORPH-Cox produced no valid validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device).eval()
    training_seconds, peak_memory = finish_neural_training(started, device)

    def predict() -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return model(test_m, test_mi).squeeze(-1).detach().cpu().numpy()

    risk, inference_seconds = timed_prediction(predict, device, repeats)
    info = parameter_info(model)
    info.update(prior_stats)
    info.update(
        {
            "training_seconds": training_seconds,
            "inference_seconds": inference_seconds,
            "peak_memory_mb": peak_memory,
            "best_epoch": best_epoch,
            "epochs_run": epochs_run,
            "best_val_c_index": best_val,
            "final_train_loss": final_loss,
            "alpha": np.nan,
            "alpha_ratio": np.nan,
        }
    )
    del model
    return risk, info


def data_file_paths(dataset: str) -> List[Path]:
    root = DATA_ROOT / dataset
    mirna_candidates = [root / "miRNA.csv", root / "mirna.csv"]
    mirna_path = next((path for path in mirna_candidates if path.exists()), mirna_candidates[0])
    return [root / "mRNA.csv", mirna_path, root / "label.csv"]


def feature_count_plan(
    datasets: Sequence[str], feature_counts: Dict[str, Dict[str, int]]
) -> Dict[str, Dict[str, int]]:
    plan: Dict[str, Dict[str, int]] = {}
    for dataset in datasets:
        mrna_count = int(feature_counts[dataset]["mRNA"])
        mirna_count = int(feature_counts[dataset]["miRNA"])
        if mrna_count < 1 or mirna_count < 1:
            raise ValueError(f"{dataset}: fixed feature counts must be positive")
        mrna_path, mirna_path, _ = data_file_paths(dataset)
        mrna_total = int(pd.read_csv(mrna_path, usecols=[0]).shape[0])
        mirna_total = int(pd.read_csv(mirna_path, usecols=[0]).shape[0])
        if mrna_count > mrna_total or mirna_count > mirna_total:
            raise ValueError(
                f"{dataset}: requested {mrna_count} mRNA + {mirna_count} miRNA, "
                f"but only {mrna_total} + {mirna_total} are available"
            )
        plan[dataset] = {
            "mRNA_total": mrna_total,
            "miRNA_total": mirna_total,
            "mRNA_selected": int(mrna_count),
            "miRNA_selected": int(mirna_count),
        }
    return plan


def build_protocol(args: argparse.Namespace, device: torch.device) -> Dict[str, object]:
    counts_by_dataset = feature_count_plan(args.datasets, args.feature_counts)
    payload: Dict[str, object] = {
        "protocol_version": "survival-fixed-final-v1",
        "configuration_status": (
            "fixed selected configuration; no search is performed by this package"
        ),
        "selection_provenance": (
            "Feature counts and optimization settings were frozen after exploratory "
            "development; reported cross-validation results are numerical estimates "
            "for the selected configuration, not an untouched external validation."
        ),
        "models": list(args.models),
        "model_name_boundaries": {
            "MORPH-Cox": "survival-only PathformerBio backbone with a Cox risk objective",
            "Adapted OmiEmbed-Survival": "official v0.2.1 components with local mRNA+miRNA CV adapter",
            "SurvPath-derived MaskedOmics": "omics-only pathway-masked derivative; no WSI",
            "MCAT-derived SNN": "genomic-only SNN derivative; no WSI",
        },
        "datasets": list(args.datasets),
        "seeds": list(args.seeds),
        "outer_folds": args.folds,
        "inner_validation_fraction": 0.2,
        "features": {
            "mode": "fixed count per modality per dataset",
            "requested_counts": args.feature_counts,
            "per_dataset_counts": counts_by_dataset,
            "selection": "outer-training top variance with stable tie order",
            "scaling": "separate outer-training StandardScaler per modality",
            "shared_feature_identity_across_models": True,
        },
        "metrics": {"Harrell_C_index": "primary", **SURVIVAL_HORIZONS},
        "cox_ties": "Breslow for neural Cox models",
        "morph_survival_parameters": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "epochs": args.epochs,
            "patience": args.patience,
            "architecture_source": "code/survival_model.py",
            "architecture_fixed": True,
        },
        "lasso": {
            "inner_folds": args.lasso_inner_folds,
            "n_alphas": args.lasso_n_alphas,
            "alpha_min_ratio": args.lasso_alpha_min_ratio,
            "max_iter": args.lasso_max_iter,
            "selection": "mean inner-fold C-index at a relative regularization-path index",
            "multiple_testing_correction": "none",
        },
        "neural": {
            "epochs": args.epochs,
            "patience": args.patience,
            "OmiEmbed_phase_epochs": list(args.omi_phase_epochs),
            "OmiEmbed_latent_dim": args.omi_latent_dim,
            "OmiEmbed_time_num": args.omi_time_num,
        },
        "baseline_optimizers": {
            "VAE-Cox": {"learning_rate": 5e-4, "weight_decay": 1e-4},
            "Adapted OmiEmbed-Survival": {"learning_rate": 1e-4, "weight_decay": 1e-4},
            "SurvPath-derived MaskedOmics": {"learning_rate": 1e-4, "weight_decay": 1e-5},
            "MCAT-derived SNN": {"learning_rate": 2e-4, "weight_decay": 1e-5},
        },
        "device": str(device),
        "inference_repeats": args.inference_repeats,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "sksurv": sksurv.__version__,
        },
        "input_sha256": experiment_input_hashes(args.datasets),
    }
    return payload


def validate_or_write_protocol(
    protocol: Dict[str, object], results_dir: Path, resume: bool
) -> None:
    path = results_dir / "protocol.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise SystemExit(
                "Protocol mismatch. Refusing to mix results; use a new results directory."
            )
        if not resume and (results_dir / "fold_results.csv").exists():
            raise SystemExit("Results already exist. Use --resume only with the identical protocol.")
    atomic_json(protocol, path)


def run_model(
    model_name: str,
    train_mrna: np.ndarray,
    train_mirna: np.ndarray,
    test_mrna: np.ndarray,
    test_mirna: np.ndarray,
    selected_genes: Sequence[str],
    selected_mirnas: Sequence[str],
    labels: pd.DataFrame,
    local_train: np.ndarray,
    local_val: np.ndarray,
    resources: morph.PriorResources,
    composition: np.ndarray,
    device: torch.device,
    model_seed: int,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, Dict[str, object]]:
    train_x = np.concatenate([train_mrna, train_mirna], axis=1)
    test_x = np.concatenate([test_mrna, test_mirna], axis=1)
    if model_name == "LASSO-Cox":
        return fit_lasso(
            train_x,
            test_x,
            labels,
            model_seed,
            args.lasso_inner_folds,
            args.lasso_n_alphas,
            args.lasso_alpha_min_ratio,
            args.lasso_max_iter,
            args.inference_repeats,
        )
    if model_name == "VAE-Cox":
        return fit_vae_cox(
            train_mrna,
            train_mirna,
            test_mrna,
            test_mirna,
            labels,
            local_train,
            local_val,
            device,
            args.epochs,
            args.patience,
            args.inference_repeats,
        )
    if model_name == "Adapted OmiEmbed-Survival":
        return fit_omiembed(
            train_mrna,
            train_mirna,
            test_mrna,
            test_mirna,
            labels,
            local_train,
            local_val,
            device,
            args.omi_latent_dim,
            args.omi_time_num,
            args.omi_phase_epochs,
            args.inference_repeats,
        )
    if model_name in ("SurvPath-derived MaskedOmics", "MCAT-derived SNN"):
        return fit_discrete_model(
            model_name,
            train_x,
            test_x,
            labels,
            local_train,
            local_val,
            composition if model_name == "SurvPath-derived MaskedOmics" else None,
            device,
            args.epochs,
            args.patience,
            args.inference_repeats,
        )
    if model_name == "MORPH-Cox":
        return fit_morph(
            train_mrna,
            train_mirna,
            test_mrna,
            test_mirna,
            selected_genes,
            selected_mirnas,
            labels,
            local_train,
            local_val,
            resources,
            device,
            args.epochs,
            args.patience,
            args.learning_rate,
            args.weight_decay,
            args.inference_repeats,
        )
    raise ValueError(f"Unknown model: {model_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=PACKAGE_ROOT / "results" / "reproduced"
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--mrna-features", type=int, default=798)
    parser.add_argument("--mirna-features", type=int, default=200)
    parser.add_argument(
        "--feature-counts",
        nargs="+",
        default=["LAML=798,200", "SKCM=997,300"],
        help="Locked final counts: LAML=798,200 SKCM=997,300",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--omi-phase-epochs", nargs=3, type=int, default=[50, 50, 100])
    parser.add_argument("--omi-latent-dim", type=int, default=128)
    parser.add_argument("--omi-time-num", type=int, default=256)
    parser.add_argument("--lasso-inner-folds", type=int, default=3)
    parser.add_argument("--lasso-n-alphas", type=int, default=50)
    parser.add_argument("--lasso-alpha-min-ratio", type=float, default=0.05)
    parser.add_argument("--lasso-max-iter", type=int, default=100000)
    parser.add_argument("--inference-repeats", type=int, default=20)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.datasets = [item.strip().upper() for item in args.datasets]
    if args.feature_counts is None:
        args.feature_counts = {
            dataset: {"mRNA": int(args.mrna_features), "miRNA": int(args.mirna_features)}
            for dataset in args.datasets
        }
    else:
        parsed_counts: Dict[str, Dict[str, int]] = {}
        for value in args.feature_counts:
            if "=" not in value:
                raise SystemExit(
                    f"Invalid feature count: {value}; expected DATASET=MRNA_COUNT,mirna_COUNT"
                )
            dataset, counts = value.split("=", 1)
            dataset = dataset.strip().upper()
            pieces = counts.replace(":", ",").split(",")
            if len(pieces) != 2:
                raise SystemExit(
                    f"Invalid feature count: {value}; expected DATASET=MRNA_COUNT,mirna_COUNT"
                )
            parsed_counts[dataset] = {
                "mRNA": int(pieces[0]),
                "miRNA": int(pieces[1]),
            }
        missing = [dataset for dataset in args.datasets if dataset not in parsed_counts]
        if missing:
            raise SystemExit("Missing feature counts for: " + ", ".join(missing))
        args.feature_counts = {dataset: parsed_counts[dataset] for dataset in args.datasets}
    fixed_checks = {
        "datasets": (tuple(args.datasets), DATASETS),
        "seeds": (tuple(args.seeds), SEEDS),
        "models": (tuple(args.models), MODELS),
        "folds": (int(args.folds), 5),
        "feature_counts": (args.feature_counts, FINAL_FEATURE_COUNTS),
        "epochs": (int(args.epochs), 200),
        "patience": (int(args.patience), 30),
        "learning_rate": (float(args.learning_rate), 5e-4),
        "weight_decay": (float(args.weight_decay), 1e-4),
        "omi_phase_epochs": (tuple(args.omi_phase_epochs), (50, 50, 100)),
        "omi_latent_dim": (int(args.omi_latent_dim), 128),
        "omi_time_num": (int(args.omi_time_num), 256),
        "lasso_inner_folds": (int(args.lasso_inner_folds), 3),
        "lasso_n_alphas": (int(args.lasso_n_alphas), 50),
        "lasso_alpha_min_ratio": (float(args.lasso_alpha_min_ratio), 0.05),
        "lasso_max_iter": (int(args.lasso_max_iter), 100000),
        "inference_repeats": (int(args.inference_repeats), 20),
    }
    changed = [name for name, (actual, expected) in fixed_checks.items() if actual != expected]
    if changed:
        raise SystemExit(
            "This is a fixed final protocol. Changed settings are not permitted: "
            + ", ".join(changed)
        )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    protocol = build_protocol(args, device)
    validate_or_write_protocol(protocol, args.results_dir, args.resume)
    runtime = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "sksurv": sksurv.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    atomic_json(runtime, args.results_dir / "runtime_environment.json")

    fold_file = args.results_dir / "fold_results.csv"
    oof_file = args.results_dir / "oof_predictions.csv"
    split_file = args.results_dir / "split_manifest.csv"
    feature_file = args.results_dir / "feature_manifest.csv"
    rows = pd.read_csv(fold_file).to_dict("records") if args.resume and fold_file.exists() else []
    oof_rows = pd.read_csv(oof_file).to_dict("records") if args.resume and oof_file.exists() else []
    split_rows = pd.read_csv(split_file).to_dict("records") if args.resume and split_file.exists() else []
    feature_rows = pd.read_csv(feature_file).to_dict("records") if args.resume and feature_file.exists() else []
    completed = {
        (
            str(row.get("dataset")),
            str(row.get("model")),
            int(row.get("seed")),
            int(row.get("fold")),
        )
        for row in rows
        if str(row.get("status")) == "ok"
    }
    manifest_folds = {
        (str(row.get("dataset")), int(row.get("seed")), int(row.get("fold")))
        for row in feature_rows
    }
    resources = morph.load_prior_resources()
    task_count = 0
    stop = False
    print("=== Unified final survival protocol ===", flush=True)
    print(f"Results : {args.results_dir}", flush=True)
    print(f"Device  : {device}", flush=True)

    for dataset in args.datasets:
        x_mrna, x_mirna, gene_names, mirna_names, labels = morph.load_dataset(dataset)
        dataset_feature_plan = protocol["features"]["per_dataset_counts"][dataset]
        if int(dataset_feature_plan["mRNA_total"]) != x_mrna.shape[1]:
            raise RuntimeError(f"{dataset}: mRNA feature count changed after protocol creation")
        if int(dataset_feature_plan["miRNA_total"]) != x_mirna.shape[1]:
            raise RuntimeError(f"{dataset}: miRNA feature count changed after protocol creation")
        mrna_count = int(dataset_feature_plan["mRNA_selected"])
        mirna_count = int(dataset_feature_plan["miRNA_selected"])
        print(
            f"[FEATURES] {dataset}: mRNA={mrna_count}/{x_mrna.shape[1]}, "
            f"miRNA={mirna_count}/{x_mirna.shape[1]}",
            flush=True,
        )
        for seed in args.seeds:
            splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
            for fold, (outer_train_idx, test_idx) in enumerate(
                splitter.split(np.arange(len(labels)), labels["event"].to_numpy()), start=1
            ):
                fold_key = (dataset, seed, fold)
                (
                    train_mrna,
                    train_mirna,
                    test_mrna,
                    test_mirna,
                    mrna_keep,
                    mirna_keep,
                    mrna_variance,
                    mirna_variance,
                    preprocessing_seconds,
                ) = prepare_common_fold(
                    x_mrna,
                    x_mirna,
                    outer_train_idx,
                    test_idx,
                    mrna_count,
                    mirna_count,
                )
                outer_labels = labels.iloc[outer_train_idx].reset_index(drop=True)
                test_labels = labels.iloc[test_idx].reset_index(drop=True)
                local_train, local_val = train_test_split(
                    np.arange(len(outer_train_idx)),
                    test_size=0.2,
                    stratify=outer_labels["event"].to_numpy(),
                    random_state=seed,
                )
                selected_genes = [gene_names[index] for index in mrna_keep]
                selected_mirnas = [mirna_names[index] for index in mirna_keep]
                (
                    gene_mask,
                    mirna_mask,
                    _,
                    _,
                    _,
                    prior_stats,
                    _,
                ) = morph.build_fold_priors(selected_genes, selected_mirnas, resources)
                composition = np.concatenate([gene_mask, mirna_mask], axis=0).astype(np.float32)

                if fold_key not in manifest_folds:
                    for role, indices in (("outer_train", outer_train_idx), ("test", test_idx)):
                        for index in indices:
                            split_rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "fold": fold,
                                    "sample_index": int(index),
                                    "sample_id": labels.iloc[int(index)]["sample_id"],
                                    "role": role,
                                }
                            )
                    for modality, keep, variances, names in (
                        ("mRNA", mrna_keep, mrna_variance, gene_names),
                        ("miRNA", mirna_keep, mirna_variance, mirna_names),
                    ):
                        for rank, (index, variance) in enumerate(zip(keep, variances), start=1):
                            feature_rows.append(
                                {
                                    "dataset": dataset,
                                    "seed": seed,
                                    "fold": fold,
                                    "modality": modality,
                                    "rank": rank,
                                    "feature_index": int(index),
                                    "feature_name": names[int(index)],
                                    "training_variance": float(variance),
                                }
                            )
                    manifest_folds.add(fold_key)
                    atomic_csv(pd.DataFrame(split_rows), split_file)
                    atomic_csv(pd.DataFrame(feature_rows), feature_file)

                for model_name in args.models:
                    key = (dataset, model_name, seed, fold)
                    if key in completed:
                        print(f"[SKIP] {dataset} {model_name} seed={seed} fold={fold}", flush=True)
                        continue
                    if args.max_tasks is not None and task_count >= args.max_tasks:
                        stop = True
                        break
                    task_count += 1
                    model_seed = seed * 100 + fold
                    set_seeds(model_seed)
                    row: Dict[str, object] = {
                        "dataset": dataset,
                        "model": model_name,
                        "seed": seed,
                        "model_seed": model_seed,
                        "fold": fold,
                        "status": "failed",
                        "reason": "",
                        "n_samples": int(len(labels)),
                        "n_outer_train": int(len(outer_train_idx)),
                        "n_inner_train": int(len(local_train)),
                        "n_validation": int(len(local_val)),
                        "n_test": int(len(test_idx)),
                        "train_events": int(outer_labels["event"].sum()),
                        "test_events": int(test_labels["event"].sum()),
                        "available_mRNA_features": int(x_mrna.shape[1]),
                        "available_miRNA_features": int(x_mirna.shape[1]),
                        "selected_mRNA_features": int(len(mrna_keep)),
                        "selected_miRNA_features": int(len(mirna_keep)),
                        "active_pathways": int(prior_stats["active_pathways"]),
                        "driver_genes": int(prior_stats["driver_genes"]),
                        "regulatory_edges": int(prior_stats["regulatory_edges"]),
                        "preprocessing_seconds": preprocessing_seconds,
                        "device": (
                            str(device)
                            if model_name in NEURAL_MODELS
                            else "cpu"
                        ),
                    }
                    print(f"[RUN] {dataset} {model_name} seed={seed} fold={fold}", flush=True)
                    try:
                        risk, info = run_model(
                            model_name,
                            train_mrna,
                            train_mirna,
                            test_mrna,
                            test_mirna,
                            selected_genes,
                            selected_mirnas,
                            outer_labels,
                            local_train,
                            local_val,
                            resources,
                            composition,
                            device,
                            model_seed,
                            args,
                        )
                        row.update(info)
                        row["inference_ms_per_sample"] = (
                            1000.0 * float(row["inference_seconds"]) / len(test_idx)
                        )
                        row["c_index"] = morph.safe_cindex(test_labels, risk)
                        for metric_name, horizon in SURVIVAL_HORIZONS.items():
                            row[metric_name] = morph.safe_td_auc(
                                outer_labels, test_labels, risk, horizon
                            )
                        row["status"] = "ok"
                        for local_index, sample_index in enumerate(test_idx):
                            label = labels.iloc[int(sample_index)]
                            oof_rows.append(
                                {
                                    "dataset": dataset,
                                    "model": model_name,
                                    "seed": seed,
                                    "fold": fold,
                                    "sample_id": label["sample_id"],
                                    "time": float(label["time"]),
                                    "event": int(label["event"]),
                                    "risk": float(risk[local_index]),
                                }
                            )
                    except Exception as exc:
                        row["reason"] = f"{type(exc).__name__}: {exc}"
                        row["traceback"] = traceback.format_exc(limit=8)
                    finally:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    rows.append(row)
                    atomic_csv(pd.DataFrame(rows), fold_file)
                    atomic_csv(pd.DataFrame(oof_rows), oof_file)
                    print(
                        f"[DONE] status={row['status']} C={row.get('c_index', np.nan):.4f} "
                        f"train={row.get('training_seconds', np.nan):.2f}s",
                        flush=True,
                    )
                if stop:
                    break
            if stop:
                break
        if stop:
            break
    frame = pd.DataFrame(rows)
    failures = frame[frame["status"] != "ok"].copy() if not frame.empty else frame
    atomic_csv(failures, args.results_dir / "failed_runs.csv")
    print(f"Completed rows: {len(frame)}; failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
