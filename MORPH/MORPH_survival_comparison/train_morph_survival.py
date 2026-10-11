from __future__ import annotations

import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR
DATA_ROOT = Path(os.environ.get("MORPH_SURVIVAL_DATA_ROOT", str(PACKAGE_ROOT / "data" / "Survival_Analysis_Two_Omics")))
PATHWAY_DIR = Path(os.environ.get("MORPH_SURVIVAL_PRIOR_ROOT", str(PACKAGE_ROOT / "resources" / "KEGG_pathways")))

DEFAULT_DATASETS = ("LAML", "SKCM")
DEFAULT_SEEDS = (1028, 2024, 3407)
DEFAULT_FOLDS = 5

GMT_GENE_FILE = PATHWAY_DIR / "20230205_kegg_hsa.gmt"
GMT_MIRNA_FILE = PATHWAY_DIR / "kegg_anano.txt"
INTERACTION_FILE = PATHWAY_DIR / "miRTarBase_MTI_processed_cleaned.csv"
DRIVER_FILE = PATHWAY_DIR / "driver_genes_list.txt"
HORIZONS = {"auc_1y": 365.0, "auc_3y": 1095.0, "auc_5y": 1825.0}


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_mirna_key(name: object) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"^(hsa|mmu)-", "", text)
    text = re.sub(r"-(3|5)p$", "", text)
    return text.strip()


def standardize_mirna_id(name: object) -> str:
    text = str(name).strip()
    text = re.sub(r"miR", "mir", text)
    if "-" in text:
        base, suffix = text.rsplit("-", 1)
        if suffix and not suffix[0].isdigit():
            text = base
    return text


@dataclass
class PriorResources:
    pathways: List[str]
    gene_to_paths: Dict[str, List[int]]
    mirna_to_paths: Dict[str, List[int]]
    driver_genes: set
    driver_to_mirnas: Dict[str, set]


def parse_gene_pathways(path: Path) -> Dict[str, set]:
    pathways: Dict[str, set] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            path_id = parts[0].split("_")[0].strip()
            pathways[path_id] = {
                str(gene).strip().upper() for gene in parts[2:] if str(gene).strip()
            }
    return pathways


def parse_mirna_pathways(path: Path) -> Dict[str, set]:
    pathways: Dict[str, set] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 2:
                continue
            key_parts = parts[0].split("|")
            if len(key_parts) < 2:
                continue
            path_id = key_parts[1].strip()
            pathways[path_id] = {
                normalize_mirna_key(item) for item in parts[1:] if str(item).strip()
            }
    return pathways


def load_prior_resources() -> PriorResources:
    gene_paths = parse_gene_pathways(GMT_GENE_FILE)
    mirna_paths = parse_mirna_pathways(GMT_MIRNA_FILE)
    pathways = sorted(set(gene_paths).intersection(mirna_paths))
    gene_to_paths: Dict[str, List[int]] = {}
    mirna_to_paths: Dict[str, List[int]] = {}
    for path_index, path_id in enumerate(pathways):
        for gene in gene_paths[path_id]:
            gene_to_paths.setdefault(gene, []).append(path_index)
        for mirna in mirna_paths[path_id]:
            mirna_to_paths.setdefault(mirna, []).append(path_index)
    driver_genes = {
        line.strip().upper()
        for line in DRIVER_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    interactions = pd.read_csv(
        INTERACTION_FILE,
        usecols=["miRNA", "Target Gene"],
        dtype=str,
        low_memory=False,
    )
    interactions["gene"] = interactions["Target Gene"].str.strip().str.upper()
    interactions = interactions[interactions["gene"].isin(driver_genes)].copy()
    interactions["mirna"] = interactions["miRNA"].map(normalize_mirna_key)
    interactions = interactions[
        (interactions["gene"] != "") & (interactions["mirna"] != "")
    ][["gene", "mirna"]].drop_duplicates()
    driver_to_mirnas = {
        gene: set(group["mirna"].tolist())
        for gene, group in interactions.groupby("gene", sort=False)
    }
    return PriorResources(
        pathways=pathways,
        gene_to_paths=gene_to_paths,
        mirna_to_paths=mirna_to_paths,
        driver_genes=driver_genes,
        driver_to_mirnas=driver_to_mirnas,
    )


def load_dataset(
    dataset: str,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], pd.DataFrame]:
    dataset_dir = DATA_ROOT / dataset
    labels = pd.read_csv(dataset_dir / "label.csv")[["sample_id", "time", "event"]].copy()
    labels["time"] = pd.to_numeric(labels["time"], errors="raise").astype(float)
    labels["event"] = pd.to_numeric(labels["event"], errors="raise").astype(int)
    mrna = pd.read_csv(dataset_dir / "mRNA.csv", index_col=0, low_memory=False)
    mirna = pd.read_csv(dataset_dir / "miRNA.csv", index_col=0, low_memory=False)
    ids = labels["sample_id"].astype(str).tolist()
    if ids != list(mrna.columns) or ids != list(mirna.columns):
        raise ValueError(f"{dataset}: expression columns do not match labels")
    return (
        mrna.T.to_numpy(dtype=np.float32, copy=True),
        mirna.T.to_numpy(dtype=np.float32, copy=True),
        [str(item).strip().upper() for item in mrna.index],
        [standardize_mirna_id(item) for item in mirna.index],
        labels,
    )


def select_top_variance_features(x_train: np.ndarray, keep_ratio: float) -> np.ndarray:
    variance = np.nanvar(x_train, axis=0)
    valid = np.flatnonzero(np.isfinite(variance) & (variance > 0))
    if valid.size == 0:
        raise ValueError("No nonconstant features in training data")
    keep_count = max(1, int(math.ceil(valid.size * keep_ratio)))
    if keep_count >= valid.size:
        return valid
    order = np.argsort(variance[valid], kind="stable")[::-1]
    return valid[order[:keep_count]]


def build_fold_priors(
    gene_names: Sequence[str],
    mirna_names: Sequence[str],
    resources: PriorResources,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, int],
    List[str],
]:
    n_genes = len(gene_names)
    n_mirnas = len(mirna_names)
    n_paths_all = len(resources.pathways)
    gene_mask_all = np.zeros((n_genes, n_paths_all), dtype=np.float32)
    mirna_mask_all = np.zeros((n_mirnas, n_paths_all), dtype=np.float32)
    for row, gene in enumerate(gene_names):
        gene_mask_all[row, resources.gene_to_paths.get(str(gene).upper(), [])] = 1.0
    normalized_mirnas = [normalize_mirna_key(item) for item in mirna_names]
    for row, mirna in enumerate(normalized_mirnas):
        mirna_mask_all[row, resources.mirna_to_paths.get(mirna, [])] = 1.0
    active_paths = (gene_mask_all.sum(axis=0) > 0) & (mirna_mask_all.sum(axis=0) > 0)
    if int(active_paths.sum()) < 2:
        raise ValueError("Too few active shared pathways after fold-local feature selection")
    gene_mask = gene_mask_all[:, active_paths]
    mirna_mask = mirna_mask_all[:, active_paths]

    mirna_index = {normalize_mirna_key(name): index for index, name in enumerate(mirna_names)}
    driver_candidates: List[Tuple[int, str, List[int]]] = []
    for gene_index, gene in enumerate(gene_names):
        gene_key = str(gene).upper()
        if gene_key not in resources.driver_genes:
            continue
        matched_mirnas = [
            mirna_index[mirna]
            for mirna in resources.driver_to_mirnas.get(gene_key, set())
            if mirna in mirna_index
        ]
        if matched_mirnas:
            driver_candidates.append((gene_index, gene_key, sorted(set(matched_mirnas))))
    if not driver_candidates:
        raise ValueError("No selected driver genes have matched miRTarBase miRNAs")

    driver_idx = np.asarray([item[0] for item in driver_candidates], dtype=np.int64)
    reg_matrix = np.zeros((n_mirnas, len(driver_candidates)), dtype=np.float32)
    for driver_col, (_, _, mirna_rows) in enumerate(driver_candidates):
        reg_matrix[mirna_rows, driver_col] = 1.0
    driver_gene_mask = gene_mask[driver_idx]
    raw_prior = (mirna_mask.T @ reg_matrix @ driver_gene_mask).T
    raw_prior += np.eye(raw_prior.shape[0], dtype=np.float32)
    row_sum = raw_prior.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    path_prior = raw_prior / row_sum
    stats = {
        "active_pathways": int(active_paths.sum()),
        "driver_genes": int(len(driver_idx)),
        "regulatory_edges": int(reg_matrix.sum()),
    }
    active_path_names = [
        resources.pathways[index] for index in np.flatnonzero(active_paths)
    ]
    return gene_mask, mirna_mask, reg_matrix, driver_idx, path_prior, stats, active_path_names


def cox_breslow_loss(
    risk: torch.Tensor, time_values: torch.Tensor, events: torch.Tensor
) -> torch.Tensor:
    event_mask = events > 0.5
    if int(event_mask.sum()) == 0:
        raise ValueError("Cox loss requires at least one event")
    event_times = torch.unique(time_values[event_mask])
    log_likelihood = risk.new_tensor(0.0)
    event_count = risk.new_tensor(0.0)
    for event_time in event_times:
        tied_events = event_mask & (time_values == event_time)
        deaths = tied_events.sum()
        risk_set = time_values >= event_time
        log_likelihood = log_likelihood + risk[tied_events].sum()
        log_likelihood = log_likelihood - deaths * torch.logsumexp(
            risk[risk_set], dim=0
        )
        event_count = event_count + deaths
    return -log_likelihood / event_count.clamp_min(1.0)


def make_survival_array(labels: pd.DataFrame) -> np.ndarray:
    return Surv.from_arrays(
        event=labels["event"].astype(bool).to_numpy(),
        time=labels["time"].astype(float).to_numpy(),
    )


def safe_cindex(labels: pd.DataFrame, risk: np.ndarray) -> float:
    try:
        return float(
            concordance_index_censored(
                labels["event"].astype(bool).to_numpy(),
                labels["time"].astype(float).to_numpy(),
                np.asarray(risk, dtype=float),
            )[0]
        )
    except Exception:
        return np.nan


def administratively_censor_test(
    train_labels: pd.DataFrame, test_labels: pd.DataFrame
) -> pd.DataFrame:
    tau = np.nextafter(float(train_labels["time"].max()), 0.0)
    adjusted = test_labels.copy()
    beyond = adjusted["time"] >= tau
    adjusted.loc[beyond, "time"] = tau
    adjusted.loc[beyond, "event"] = 0
    return adjusted


def safe_td_auc(
    train_labels: pd.DataFrame,
    test_labels: pd.DataFrame,
    risk: np.ndarray,
    horizon: float,
) -> float:
    adjusted = administratively_censor_test(train_labels, test_labels)
    times = adjusted["time"].to_numpy(dtype=float)
    events = adjusted["event"].to_numpy(dtype=int)
    if int(np.sum((events == 1) & (times <= horizon))) == 0 or int(np.sum(times > horizon)) == 0:
        return np.nan
    if not (float(times.min()) < horizon < float(times.max())):
        return np.nan
    try:
        auc, _ = cumulative_dynamic_auc(
            make_survival_array(train_labels),
            make_survival_array(adjusted),
            np.asarray(risk, dtype=float),
            np.asarray([horizon], dtype=float),
        )
        return float(auc[0])
    except Exception:
        return np.nan
