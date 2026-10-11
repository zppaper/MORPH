from __future__ import annotations
import io
import random
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import cox_data as morph
import cox_model as morph_impl

def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def synchronize(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize(device)

def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for (name, value) in model.state_dict().items()}

def torch_model_size_bytes(model: nn.Module) -> int:
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return int(buffer.tell())

def parameter_info(model: nn.Module) -> Dict[str, object]:
    return {'total_parameter_count': int(sum((item.numel() for item in model.parameters()))), 'trainable_parameter_count': int(sum((item.numel() for item in model.parameters() if item.requires_grad))), 'nonzero_parameter_count': np.nan, 'model_size_bytes': torch_model_size_bytes(model)}

def timed_prediction(predict: Callable[[], np.ndarray], device: torch.device, repeats: int) -> Tuple[np.ndarray, float]:
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
        raise RuntimeError('Prediction timing produced no output')
    return (risk, float(np.median(durations)))

def select_features(x_train: np.ndarray, count: int) -> Tuple[np.ndarray, np.ndarray]:
    variance = np.nanvar(x_train, axis=0)
    valid = np.flatnonzero(np.isfinite(variance) & (variance > 0))
    if valid.size == 0:
        raise ValueError('No finite nonconstant features in outer training fold')
    order = np.argsort(variance[valid], kind='stable')[::-1]
    keep = valid[order[:min(count, valid.size)]]
    return (keep, variance[keep])

def prepare_common_fold(x_mrna: np.ndarray, x_mirna: np.ndarray, outer_train_idx: np.ndarray, test_idx: np.ndarray, mrna_count: int, mirna_count: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    started = time.perf_counter()
    (mrna_keep, mrna_variance) = select_features(x_mrna[outer_train_idx], mrna_count)
    (mirna_keep, mirna_variance) = select_features(x_mirna[outer_train_idx], mirna_count)
    mrna_scaler = StandardScaler().fit(x_mrna[outer_train_idx][:, mrna_keep])
    mirna_scaler = StandardScaler().fit(x_mirna[outer_train_idx][:, mirna_keep])
    train_mrna = mrna_scaler.transform(x_mrna[outer_train_idx][:, mrna_keep]).astype(np.float32)
    train_mirna = mirna_scaler.transform(x_mirna[outer_train_idx][:, mirna_keep]).astype(np.float32)
    test_mrna = mrna_scaler.transform(x_mrna[test_idx][:, mrna_keep]).astype(np.float32)
    test_mirna = mirna_scaler.transform(x_mirna[test_idx][:, mirna_keep]).astype(np.float32)
    return (train_mrna, train_mirna, test_mrna, test_mirna, mrna_keep, mirna_keep, mrna_variance, mirna_variance, time.perf_counter() - started)

def initialize_neural_timing(device: torch.device) -> float:
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    return time.perf_counter()

def finish_neural_training(started: float, device: torch.device) -> Tuple[float, float]:
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)) if device.type == 'cuda' else np.nan
    return (elapsed, peak)

def fit_morph(train_mrna: np.ndarray, train_mirna: np.ndarray, test_mrna: np.ndarray, test_mirna: np.ndarray, selected_genes: Sequence[str], selected_mirnas: Sequence[str], labels: pd.DataFrame, local_train: np.ndarray, local_val: np.ndarray, resources: morph.PriorResources, device: torch.device, epochs: int, patience: int, learning_rate: float, weight_decay: float, repeats: int) -> Tuple[np.ndarray, Dict[str, object]]:
    (gene_mask, mirna_mask, reg_matrix, driver_idx, path_prior, prior_stats, _) = morph.build_fold_priors(selected_genes, selected_mirnas, resources)
    started = initialize_neural_timing(device)
    gene_mask_df = pd.DataFrame(gene_mask, index=list(selected_genes), columns=[f'path_{index}' for index in range(gene_mask.shape[1])])
    mirna_mask_df = pd.DataFrame(mirna_mask, index=list(selected_mirnas), columns=[f'path_{index}' for index in range(mirna_mask.shape[1])])
    model = morph_impl.PathformerBio(dim_mrna=train_mrna.shape[1], dim_mirna=train_mirna.shape[1], gene_path_mask_df=gene_mask_df, mirna_path_mask_df=mirna_mask_df, reg_matrix=reg_matrix, driver_idx=driver_idx, path_prior=path_prior, hidden_dim=128, depth=2, ff_dropout=0.2, classifier_dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_m = torch.from_numpy(train_mrna).to(device)
    train_mi = torch.from_numpy(train_mirna).to(device)
    test_m = torch.from_numpy(test_mrna).to(device)
    test_mi = torch.from_numpy(test_mirna).to(device)
    time_t = torch.as_tensor(labels['time'].to_numpy(), dtype=torch.float32, device=device)
    event_t = torch.as_tensor(labels['event'].to_numpy(), dtype=torch.float32, device=device)
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
            raise FloatingPointError(f'Non-finite MORPH-Cox loss at epoch {epoch}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        final_loss = float(loss.detach().cpu())
        model.eval()
        with torch.no_grad():
            validation = model(train_m[local_val], train_mi[local_val]).squeeze(-1).detach().cpu().numpy()
        score = morph.safe_cindex(labels.iloc[local_val], validation)
        if np.isfinite(score) and score > best_val + 1e-08:
            best_val = score
            best_epoch = epoch
            best_state = clone_state(model)
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError('MORPH-Cox produced no valid validation checkpoint')
    model.load_state_dict(best_state)
    model.to(device).eval()
    (training_seconds, peak_memory) = finish_neural_training(started, device)

    def predict() -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return model(test_m, test_mi).squeeze(-1).detach().cpu().numpy()
    (risk, inference_seconds) = timed_prediction(predict, device, repeats)
    info = parameter_info(model)
    info.update(prior_stats)
    info.update({'training_seconds': training_seconds, 'inference_seconds': inference_seconds, 'peak_memory_mb': peak_memory, 'best_epoch': best_epoch, 'epochs_run': epochs_run, 'best_val_c_index': best_val, 'final_train_loss': final_loss, 'alpha': np.nan, 'alpha_ratio': np.nan})
    del model
    return (risk, info)
