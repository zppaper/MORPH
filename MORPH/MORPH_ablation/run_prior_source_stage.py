from _settings import data_dir, prior_dir, output_dir, external_root, driver_file
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import random
import warnings
import re
from pathlib import Path
warnings.filterwarnings('ignore')
DATASETS = ['KIRC', 'MESO', 'STAD', 'THCA']
SEEDS = [1028, 2024, 3407]
N_FOLDS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 100
PATIENCE = 15
BASE_DATA_DIR = data_dir('stage')
PATHWAY_DIR = prior_dir()
GMT_GENE_FILE = PATHWAY_DIR / '20230205_kegg_hsa.gmt'
GMT_MIRNA_FILE = PATHWAY_DIR / 'kegg_anano.txt'
INTERACTION_FILE = PATHWAY_DIR / 'miRTarBase_MTI_processed_cleaned.csv'
DRIVER_FILE = driver_file()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def clean_name_logic(name):
    name = str(name).lower()
    name = re.sub('-(3|5)p', '', name)
    return name.strip()

def normalize_mirna_key(name):
    name = str(name).lower()
    name = re.sub('^(hsa|mmu)-', '', name)
    name = re.sub('-(3|5)p', '', name)
    return name.strip()

def build_mirna_gene_interaction_matrix(gene_list, mirna_list):
    (n_genes, n_mirnas) = (len(gene_list), len(mirna_list))
    interaction = np.zeros((n_mirnas, n_genes), dtype=np.float32)
    if not INTERACTION_FILE.exists():
        return interaction
    try:
        df = pd.read_csv(INTERACTION_FILE, usecols=['miRNA', 'Target Gene'], low_memory=False)
    except Exception as e:
        return interaction
    if df.empty:
        return interaction
    mirna_map = {normalize_mirna_key(m): i for (i, m) in enumerate(mirna_list)}
    gene_map = {str(g).strip().upper(): i for (i, g) in enumerate(gene_list)}
    mirna_series = df['miRNA'].astype(str).str.strip().str.lower()
    gene_series = df['Target Gene'].astype(str).str.strip().str.upper()
    mi_idx = mirna_series.map(mirna_map)
    g_idx = gene_series.map(gene_map)
    valid = mi_idx.notna() & g_idx.notna()
    if valid.sum() == 0:
        return interaction
    mi_arr = mi_idx[valid].astype(np.int64).values
    g_arr = g_idx[valid].astype(np.int64).values
    interaction[mi_arr, g_arr] = 1.0
    return interaction

def load_driver_genes(gene_list):
    if not DRIVER_FILE.exists():
        return gene_list
    try:
        with open(DRIVER_FILE, 'r', encoding='utf-8') as f:
            drivers = [line.strip() for line in f if line.strip()]
    except Exception as e:
        return gene_list
    driver_set = {d.strip().upper() for d in drivers}
    filtered = [g for g in gene_list if str(g).strip().upper() in driver_set]
    if len(filtered) == 0:
        return gene_list
    return filtered

def parse_gmt(file_path):
    pathway_dict = {}
    if not file_path.exists():
        return pathway_dict
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            path_id = parts[0].split('_')[0].strip()
            members = [m.strip() for m in parts[2:] if m.strip()]
            pathway_dict[path_id] = members
    return pathway_dict

def parse_mirna_pathway_file(file_path):
    pathway_dict = {}
    if not file_path.exists():
        return pathway_dict
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            key_parts = parts[0].split('|')
            if len(key_parts) < 2:
                continue
            path_id = key_parts[1].strip()
            members = list(set([m.strip() for m in parts[1:] if m.strip()]))
            pathway_dict[path_id] = members
    return pathway_dict

def load_dual_pathway_structure(gene_list, mirna_list):
    gene_path_dict = parse_gmt(GMT_GENE_FILE)
    mirna_path_dict = parse_mirna_pathway_file(GMT_MIRNA_FILE)
    common_paths = sorted(list(set(gene_path_dict.keys()) & set(mirna_path_dict.keys())))
    if len(common_paths) == 0:
        n_dummy = 10
        g_mask = pd.DataFrame(np.random.randint(0, 2, (len(gene_list), n_dummy)), index=gene_list)
        m_mask = pd.DataFrame(np.random.randint(0, 2, (len(mirna_list), n_dummy)), index=mirna_list)
        return (g_mask, m_mask)
    g_mask = np.zeros((len(gene_list), len(common_paths)))
    for (j, pid) in enumerate(common_paths):
        members = gene_path_dict[pid]
        indices = [i for (i, g) in enumerate(gene_list) if g in members]
        g_mask[indices, j] = 1
    gene_mask_df = pd.DataFrame(g_mask, index=gene_list, columns=common_paths)
    m_mask = np.zeros((len(mirna_list), len(common_paths)))
    mirna_map = {clean_name_logic(m): i for (i, m) in enumerate(mirna_list)}
    for (j, pid) in enumerate(common_paths):
        members = mirna_path_dict[pid]
        for m in members:
            clean_m = clean_name_logic(m)
            if clean_m in mirna_map:
                m_mask[mirna_map[clean_m], j] = 1
    mirna_mask_df = pd.DataFrame(m_mask, index=mirna_list, columns=common_paths)
    return (gene_mask_df, mirna_mask_df)

def build_pathway_prior(gene_list, mirna_list, gene_mask_df, mirna_mask_df, use_driver_only=True, driver_idx=None):
    if use_driver_only and driver_idx is not None and (len(driver_idx) > 0):
        gene_list_use = [gene_list[i] for i in driver_idx]
        G = gene_mask_df.values.astype(np.float32)[driver_idx]
    else:
        gene_list_use = gene_list
        G = gene_mask_df.values.astype(np.float32)
    R = build_mirna_gene_interaction_matrix(gene_list_use, mirna_list)
    M = mirna_mask_df.values.astype(np.float32)
    A_raw = M.T @ R @ G
    if A_raw.size == 0:
        return np.zeros((G.shape[1], M.shape[1]), dtype=np.float32)
    A_raw = A_raw.T
    I = np.eye(A_raw.shape[0], dtype=np.float32)
    A_raw = A_raw + I
    row_sum = A_raw.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    A_raw = A_raw / row_sum
    return A_raw.astype(np.float32)

def load_data(dataset_name):
    data_dir = BASE_DATA_DIR / dataset_name
    try:
        mrna_df = pd.read_csv(data_dir / 'mRNA.csv', index_col=0).T
        mirna_df = pd.read_csv(data_dir / 'miRNA.csv', index_col=0).T
        label_df = pd.read_csv(data_dir / 'label.csv', index_col=0)
    except FileNotFoundError:
        return (None, None, None, None, None, None, None, None)
    common_samples = mrna_df.index.intersection(mirna_df.index).intersection(label_df.index)
    if len(common_samples) == 0:
        raise ValueError('No common samples found!')
    mrna_df = mrna_df.loc[common_samples]
    mirna_df = mirna_df.loc[common_samples]
    label_df = label_df.loc[common_samples]
    gene_list = list(mrna_df.columns)
    mirna_list = list(mirna_df.columns)
    driver_genes = load_driver_genes(gene_list)
    driver_idx = [i for (i, g) in enumerate(gene_list) if g in driver_genes]
    gene_list_driver = [gene_list[i] for i in driver_idx]
    if len(driver_idx) == 0:
        driver_idx = list(range(len(gene_list)))
        gene_list_driver = gene_list
    reg_matrix = build_mirna_gene_interaction_matrix(gene_list_driver, mirna_list)
    edge_mask = reg_matrix.sum(axis=0) > 0
    if edge_mask.any():
        driver_idx = [idx for (idx, keep) in zip(driver_idx, edge_mask) if keep]
        reg_matrix = reg_matrix[:, edge_mask]
    (gene_path_mask_df, mirna_path_mask_df) = load_dual_pathway_structure(gene_list, mirna_list)
    path_prior = build_pathway_prior(gene_list, mirna_list, gene_path_mask_df, mirna_path_mask_df, use_driver_only=True, driver_idx=driver_idx)
    return (mrna_df, mirna_df, gene_path_mask_df, mirna_path_mask_df, reg_matrix, driver_idx, path_prior, label_df['label'].values)

class ResNetBlock(nn.Module):

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, x):
        return x + self.ff(self.norm(x))

class PathwayCrossAttnBlock(nn.Module):

    def __init__(self, n_genes, n_mirnas, n_pathways, gene_mask, mirna_mask, path_embed_dim=16, out_dim=128, path_prior=None, num_heads=4, topk_ratio=0.025, topk_min=5, topk_max=30, tau_init=0.35, alpha_init=0.18, attn_dropout=0.4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = path_embed_dim // num_heads
        g_mask_norm = gene_mask / (gene_mask.sum(dim=0, keepdim=True) + 1e-08)
        m_mask_norm = mirna_mask / (mirna_mask.sum(dim=0, keepdim=True) + 1e-08)
        self.register_buffer('gene_mask', g_mask_norm)
        self.register_buffer('mirna_mask', m_mask_norm)
        self.g_proj = nn.Linear(1, path_embed_dim)
        self.m_proj = nn.Linear(1, path_embed_dim)
        self.to_q = nn.Linear(path_embed_dim, path_embed_dim, bias=False)
        self.to_k = nn.Linear(path_embed_dim, path_embed_dim, bias=False)
        self.to_v = nn.Linear(path_embed_dim, path_embed_dim, bias=False)
        self.scale = self.head_dim ** (-0.5)
        self.norm = nn.LayerNorm(path_embed_dim)
        self.tau = nn.Parameter(torch.tensor(tau_init))
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.attn_drop = nn.Dropout(attn_dropout)
        if path_prior is not None:
            prior_tensor = torch.tensor(path_prior, dtype=torch.float32)
            num_paths = prior_tensor.shape[0]
            k_val = int(num_paths * topk_ratio)
            k_val = max(topk_min, min(topk_max, k_val))
            noise = torch.rand_like(prior_tensor) * 1e-05
            (_, topk_idx) = torch.topk(prior_tensor + noise, k=k_val, dim=-1)
            sparse_mask = torch.zeros_like(prior_tensor).scatter_(-1, topk_idx, 1.0)
            sparse_mask.fill_diagonal_(1.0)
            norm_mask = sparse_mask / (sparse_mask.sum(dim=-1, keepdim=True) + 1e-08)
            self.register_buffer('prior_norm_mask', norm_mask)
        else:
            self.register_buffer('prior_norm_mask', None)
        self.gate_proj = nn.Linear(path_embed_dim, path_embed_dim)
        self.final_proj = nn.Sequential(nn.Linear(path_embed_dim * 2, out_dim), nn.LayerNorm(out_dim), nn.ReLU(), nn.Dropout(0.3))

    def forward(self, x_mrna, x_mirna):
        B = x_mrna.shape[0]
        h_g_local = F.gelu(self.g_proj(x_mrna.unsqueeze(-1)))
        h_m_local = F.gelu(self.m_proj(x_mirna.unsqueeze(-1)))
        path_g = torch.einsum('b g d, g p -> b p d', h_g_local, self.gene_mask)
        path_m = torch.einsum('b m d, m p -> b p d', h_m_local, self.mirna_mask)
        q = self.to_q(path_g).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(path_m).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(path_m).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        tau = torch.clamp(self.tau, min=0.01)
        dots = dots / tau
        attn = F.softmax(dots, dim=-1)
        if self.prior_norm_mask is not None:
            alpha = torch.sigmoid(self.alpha)
            prior = self.prior_norm_mask.unsqueeze(0).unsqueeze(0)
            attn = alpha * prior + (1.0 - alpha) * attn
        attn = self.attn_drop(attn)
        update = torch.matmul(attn, v)
        update = update.transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
        path_g_refined = self.norm(path_g + update)
        gate = torch.sigmoid(self.gate_proj(path_g_refined))
        path_g_gated = path_g_refined * gate
        pool_mean = path_g_gated.mean(dim=1)
        pool_max = path_g_gated.max(dim=1)[0]
        pooled = torch.cat([pool_mean, pool_max], dim=-1)
        out = self.final_proj(pooled)
        return out

class MicroRegCrossAttnBlock(nn.Module):

    def __init__(self, n_genes, n_mirnas, reg_matrix, embed_dim=32, out_dim=128):
        super().__init__()
        self.gene_proj = nn.Linear(1, embed_dim)
        self.mirna_proj = nn.Linear(1, embed_dim)
        self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** (-0.5)
        self.norm = nn.LayerNorm(embed_dim)
        self.out_proj = nn.Sequential(nn.Linear(embed_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU(), nn.Dropout(0.3))
        self.pool_score = nn.Linear(embed_dim, 1)
        reg_bias = torch.log1p(torch.as_tensor(reg_matrix, dtype=torch.float32).t())
        self.register_buffer('reg_bias', reg_bias)
        self.reg_scale_raw = nn.Parameter(torch.tensor(0.0))
        mix_hidden = max(8, embed_dim // 2)
        self.prior_mix_net = nn.Sequential(nn.Linear(embed_dim, mix_hidden), nn.GELU(), nn.Linear(mix_hidden, 1))

    def forward(self, x_mrna, x_mirna):
        g = self.gene_proj(x_mrna.unsqueeze(-1))
        m = self.mirna_proj(x_mirna.unsqueeze(-1))
        q = self.to_q(g)
        k = self.to_k(m)
        v = self.to_v(m)
        dots_free = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        reg_scale = F.softplus(self.reg_scale_raw)
        dots_prior = dots_free + reg_scale * self.reg_bias.unsqueeze(0)
        attn_free = F.softmax(dots_free, dim=-1)
        attn_prior = F.softmax(dots_prior, dim=-1)
        prior_mix = torch.sigmoid(self.prior_mix_net(q))
        attn = prior_mix * attn_prior + (1.0 - prior_mix) * attn_free
        update = torch.matmul(attn, v)
        g_refined = self.norm(g + update)
        pool_w = torch.softmax(self.pool_score(g_refined), dim=1)
        pooled = torch.sum(pool_w * g_refined, dim=1)
        out = self.out_proj(pooled)
        return out

class PathformerBio(nn.Module):

    def __init__(self, dim_mrna, dim_mirna, gene_path_mask_df, mirna_path_mask_df, reg_matrix, driver_idx, path_prior, hidden_dim=128, depth=2, ff_dropout=0.2, classifier_dropout=0.3):
        super().__init__()
        gene_path_tensor = torch.tensor(gene_path_mask_df.values, dtype=torch.float32)
        mirna_path_tensor = torch.tensor(mirna_path_mask_df.values, dtype=torch.float32)
        reg_tensor = torch.tensor(reg_matrix, dtype=torch.float32)
        col_sum = reg_tensor.sum(dim=0, keepdim=True)
        col_sum = torch.where(col_sum == 0, torch.ones_like(col_sum), col_sum)
        reg_tensor = reg_tensor / col_sum
        self.register_buffer('reg_matrix', reg_tensor)
        self.register_buffer('driver_idx', torch.tensor(driver_idx, dtype=torch.long))
        self.mrna_embedding = nn.Sequential(nn.Linear(dim_mrna, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim))
        self.mrna_encoder = nn.ModuleList([ResNetBlock(hidden_dim, dropout=ff_dropout) for _ in range(depth)])
        self.mirna_embedding = nn.Sequential(nn.Linear(dim_mirna, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.3), nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
        self.mirna_encoder = nn.ModuleList([ResNetBlock(hidden_dim, dropout=ff_dropout) for _ in range(depth)])
        n_pathways = gene_path_mask_df.shape[1]
        self.pathway_block = PathwayCrossAttnBlock(n_genes=dim_mrna, n_mirnas=dim_mirna, n_pathways=n_pathways, gene_mask=gene_path_tensor, mirna_mask=mirna_path_tensor, path_embed_dim=16, out_dim=hidden_dim, path_prior=path_prior)
        self.micro_reg_block = MicroRegCrossAttnBlock(n_genes=len(driver_idx), n_mirnas=dim_mirna, reg_matrix=reg_tensor, embed_dim=16, out_dim=hidden_dim)
        fusion_dim = hidden_dim * 4
        self.classifier = nn.Sequential(nn.Linear(fusion_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(classifier_dropout), nn.Linear(hidden_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(classifier_dropout), nn.Linear(64, 1))

    def forward(self, x_mrna, x_mirna):
        h_mrna_glob = self.mrna_embedding(x_mrna)
        for blk in self.mrna_encoder:
            h_mrna_glob = blk(h_mrna_glob)
        h_mirna = self.mirna_embedding(x_mirna)
        for blk in self.mirna_encoder:
            h_mirna = blk(h_mirna)
        h_path = self.pathway_block(x_mrna, x_mirna)
        x_mrna_driver = x_mrna.index_select(1, self.driver_idx)
        h_micro = self.micro_reg_block(x_mrna_driver, x_mirna)
        h_concat = torch.cat([h_mrna_glob, h_mirna, h_path, h_micro], dim=1)
        logits_fusion = self.classifier(h_concat)
        return logits_fusion
import argparse
import hashlib
import json
import platform
import sys
import time
import sklearn
from datetime import datetime, timedelta, timezone
VARIANTS = ['regulatory_matched', 'independent_jaccard']
GRAPH_TIE_SEED = 6142026
PROTOCOL_VERSION = 'stage-prior-source-common-supported-degree-v1'

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda : stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def array_hash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def parameter_hash(model):
    h = hashlib.sha256()
    for (name, p) in model.named_parameters():
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def atomic_csv(frame, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temp, index=False, encoding='utf-8-sig')
    temp.replace(path)

def atomic_json(value, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temp.replace(path)

def metrics(labels, probs):
    (labels, probs) = (np.asarray(labels), np.asarray(probs))
    assert np.isfinite(probs).all() and len(np.unique(labels)) == 2
    return (float(roc_auc_score(labels, probs)), float(average_precision_score(labels, probs)))

def independent_scores(gmt_file, pathway_ids):
    annotations = parse_gmt(gmt_file)
    gene_sets = [set(annotations[p]) for p in pathway_ids]
    assert all(gene_sets), 'All evaluated pathways must have GMT gene annotations'
    p = len(pathway_ids)
    result = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(i, p):
            value = len(gene_sets[i] & gene_sets[j]) / len(gene_sets[i] | gene_sets[j])
            result[i, j] = result[j, i] = value
    return (result, np.array([len(s) for s in gene_sets], dtype=np.int64))

def select_supported(scores, degrees, tie):
    selected = np.zeros(scores.shape, dtype=np.float32)
    for (i, degree) in enumerate(degrees):
        candidates = np.flatnonzero((scores[i] > 0) & (np.arange(len(scores)) != i))
        assert len(candidates) >= degree
        order = np.lexsort((tie[i, candidates], -scores[i, candidates]))
        selected[i, candidates[order[:int(degree)]]] = 1.0
    return selected

def graph_stats(adjacency):
    graph = adjacency.astype(bool)
    nodes = len(graph)
    incoming = graph.sum(axis=0)
    symmetric = graph | graph.T
    unseen = set(range(nodes))
    sizes = []
    while unseen:
        todo = [unseen.pop()]
        size = 0
        while todo:
            i = todo.pop()
            size += 1
            next_nodes = set(np.flatnonzero(symmetric[i])) & unseen
            unseen.difference_update(next_nodes)
            todo.extend(next_nodes)
        sizes.append(size)
    edges = int(graph.sum())
    return {'offdiag_edges': edges, 'offdiag_density': edges / (nodes * (nodes - 1)), 'incoming_degree_std': float(incoming.std()), 'incoming_degree_max': int(incoming.max()), 'zero_incoming_nodes': int((incoming == 0).sum()), 'reciprocity': float((graph & graph.T).sum() / edges) if edges else 0.0, 'weak_components': len(sizes), 'largest_weak_component': max(sizes)}

def load_checked(dataset):
    res = load_data(dataset)
    assert res[0] is not None, 'Missing cohort data: ' + dataset
    (mrna, mirna, gm, mm, reg, drivers, prior, y) = res
    assert mrna.index.is_unique and mirna.index.is_unique and mrna.index.equals(mirna.index)
    assert mrna.columns.is_unique and mirna.columns.is_unique
    assert np.isfinite(mrna.values).all() and np.isfinite(mirna.values).all()
    assert set(np.unique(y)) == {0, 1} and len(y) == len(mrna)
    assert list(gm.columns) == list(mm.columns) and gm.shape[1] == 237
    assert len(drivers) and reg.sum() > 0 and (reg.sum(axis=0) > 0).all()
    return res

def prepare_graphs(res, dataset, output):
    (mrna, mirna, gm, mm, reg, drivers, original_prior, y) = res
    ids = list(gm.columns)
    n = len(ids)
    regulatory = (gm.values[drivers].astype(np.float32).T @ reg.T @ mm.values.astype(np.float32)).astype(np.float64)
    (independent, gene_counts) = independent_scores(GMT_GENE_FILE, ids)
    tmp = regulatory + np.eye(n)
    reconstructed = tmp / tmp.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(original_prior, reconstructed, rtol=1e-06, atol=1e-08)
    graph_rng = torch.Generator(device='cpu').manual_seed(GRAPH_TIE_SEED)
    tie = torch.rand((n, n), generator=graph_rng).numpy()
    k = max(5, min(30, int(n * 0.025)))
    original_tensor = torch.tensor(original_prior, dtype=torch.float32)
    indices = torch.topk(original_tensor + torch.from_numpy(tie) * 1e-05, k=k, dim=-1).indices
    legacy = torch.zeros_like(original_tensor).scatter_(-1, indices, 1.0).numpy()
    np.fill_diagonal(legacy, 0)
    original_degree = legacy.sum(axis=1).astype(np.int64)
    np.fill_diagonal(regulatory, 0)
    np.fill_diagonal(independent, 0)
    reg_positive = (regulatory > 0).sum(axis=1)
    ind_positive = (independent > 0).sum(axis=1)
    degree = np.minimum(np.minimum(original_degree, reg_positive), ind_positive)
    adjacency_reg = select_supported(regulatory, degree, tie)
    adjacency_ind = select_supported(independent, degree, tie)
    normalized = {}
    for (variant, adjacency) in zip(VARIANTS, [adjacency_reg, adjacency_ind]):
        with_self = adjacency + np.eye(n, dtype=np.float32)
        normalized[variant] = with_self / with_self.sum(axis=1, keepdims=True)
        assert np.array_equal(adjacency.sum(axis=1), degree)
        assert np.all(np.diag(normalized[variant]) > 0)
        np.testing.assert_allclose(normalized[variant].sum(axis=1), 1, atol=1e-06)
    assert np.array_equal(np.sort(normalized[VARIANTS[0]], axis=1), np.sort(normalized[VARIANTS[1]], axis=1))
    assert not np.array_equal(adjacency_reg, adjacency_ind), 'No-op: two graph supports are identical'
    directory = output / 'graphs' / dataset
    directory.mkdir(parents=True, exist_ok=True)
    arrays = dict(regulatory_scores=regulatory, independent_jaccard_scores=independent, regulatory_adjacency=adjacency_reg, independent_adjacency=adjacency_ind, regulatory_prior=normalized[VARIANTS[0]], independent_prior=normalized[VARIANTS[1]], tie_break_matrix=tie, original_nonself_degree=original_degree, common_nonself_degree=degree)
    archive = directory / 'prior_matrices.npz'
    if archive.exists():
        with np.load(archive) as saved:
            assert set(saved.files) == set(arrays)
            for name in arrays:
                assert np.array_equal(saved[name], arrays[name]), 'Graph changed: ' + name
    else:
        np.savez_compressed(archive, **arrays)
    atomic_csv(pd.DataFrame({'pathway_id': ids, 'gmt_gene_count': gene_counts, 'original_nonself_degree': original_degree, 'regulatory_positive_neighbors': reg_positive, 'jaccard_positive_neighbors': ind_positive, 'common_nonself_degree': degree, 'budget_reduced': degree < original_degree, 'regulatory_indegree': adjacency_reg.sum(axis=0).astype(int), 'independent_indegree': adjacency_ind.sum(axis=0).astype(int)}), directory / 'pathway_degrees.csv')
    edge_rows = []
    for (variant, adjacency, scores) in zip(VARIANTS, [adjacency_reg, adjacency_ind], [regulatory, independent]):
        for (i, j) in zip(*np.where(adjacency)):
            edge_rows.append({'variant': variant, 'query_pathway': ids[i], 'key_pathway': ids[j], 'source_score': float(scores[i, j]), 'prior_weight': float(normalized[variant][i, j])})
    atomic_csv(pd.DataFrame(edge_rows), directory / 'selected_offdiagonal_edges.csv')
    intersection = int(((adjacency_reg > 0) & (adjacency_ind > 0)).sum())
    union = int(((adjacency_reg > 0) | (adjacency_ind > 0)).sum())
    info = {'dataset': dataset, 'patients': len(y), 'pathways': n, 'stage_nominal_k': k, 'driver_genes': len(drivers), 'regulatory_molecular_edges': int(reg.sum()), 'jaccard_gene_universe': 'complete gene memberships in fixed GMT; no cohort feature filtering', 'budget_reduced_rows': int((degree < original_degree).sum()), 'budget_reduced_pathways': [ids[i] for i in np.flatnonzero(degree < original_degree)], 'self_only_rows': int((degree == 0).sum()), 'common_degree_min': int(degree.min()), 'common_degree_max': int(degree.max()), 'row_weight_multisets_identical': True, 'offdiag_edge_jaccard': intersection / union, 'intersecting_offdiag_edges': intersection, 'regulatory_graph': graph_stats(adjacency_reg), 'independent_graph': graph_stats(adjacency_ind), 'prior_sha256': {v: array_hash(a) for (v, a) in normalized.items()}, 'independent_raw_score_sha256': array_hash(independent)}
    atomic_json(info, directory / 'graph_diagnostics.json')
    return (normalized, info)

def model_kwargs(res):
    (a, b, gm, mm, reg, drivers, prior, y) = res
    return dict(dim_mrna=a.shape[1], dim_mirna=b.shape[1], gene_path_mask_df=gm, mirna_path_mask_df=mm, reg_matrix=reg, driver_idx=drivers, path_prior=prior, hidden_dim=128)

def create_model(res, prior, initial_seed):
    set_seeds(initial_seed)
    model = PathformerBio(**model_kwargs(res)).to(device)
    model.pathway_block.prior_norm_mask.copy_(torch.as_tensor(prior, dtype=torch.float32, device=device))
    return model

def check_model_pair(res, priors):
    left = create_model(res, priors[VARIANTS[0]], 1028)
    right = create_model(res, priors[VARIANTS[1]], 1028)
    assert parameter_hash(left) == parameter_hash(right)
    different = [k for (k, v) in left.state_dict().items() if not torch.equal(v, right.state_dict()[k])]
    assert different == ['pathway_block.prior_norm_mask'], different
    x = torch.tensor(StandardScaler().fit_transform(res[0].values[:8]).astype(np.float32), device=device)
    mi = torch.tensor(StandardScaler().fit_transform(res[1].values[:8]).astype(np.float32), device=device)
    left.eval()
    right.eval()
    with torch.no_grad():
        delta = float((left(x, mi) - right(x, mi)).abs().max().cpu())
    assert delta > 0, 'Prior replacement does not affect predictions'
    for model in [left, right]:
        model.train()
        set_seeds(117)
        logits = model(x, mi).squeeze(-1)
        loss = nn.BCEWithLogitsLoss()(logits, torch.tensor(res[-1][:8], dtype=torch.float32, device=device))
        loss.backward()
        assert all((p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        assert model.micro_reg_block.gene_proj.weight.grad.abs().sum() > 0
        assert model.pathway_block.g_proj.weight.grad.abs().sum() > 0
    return {'all_trainable_initial_parameters_identical': True, 'state_difference_only': different, 'both_branches_receive_finite_gradients': True, 'eval_logit_max_difference': delta, 'trainable_parameters': sum((p.numel() for p in left.parameters())), 'initial_parameters_sha256': parameter_hash(left)}

def collect(model, loader):
    model.eval()
    (true, probs) = ([], [])
    with torch.no_grad():
        for (x, mi, y) in loader:
            logits = model(x.to(device), mi.to(device))
            probs.extend(torch.sigmoid(logits.squeeze(-1)).cpu().numpy().tolist())
            true.extend(y.numpy().tolist())
    return (np.asarray(true, dtype=int), np.asarray(probs, dtype=np.float64))

def is_complete(directory):
    marker_path = directory / 'complete.json'
    if not marker_path.exists():
        return False
    marker = json.loads(marker_path.read_text(encoding='utf-8'))
    for (name, checksum) in marker['sha256'].items():
        assert sha256(directory / name) == checksum, 'Completed result changed: ' + str(directory / name)
    prediction = pd.read_csv(directory / 'test_predictions.csv')
    row = json.loads((directory / 'fold_metrics.json').read_text(encoding='utf-8'))
    np.testing.assert_allclose(metrics(prediction.label, prediction.probability), [row['auc'], row['aupr']], atol=1e-12, rtol=0)
    return True

def train_fold(res, prior, variant, dataset, seed, fold, pair_directory, arrays):
    directory = pair_directory / variant
    directory.mkdir(parents=True, exist_ok=True)
    if is_complete(directory):
        print('SKIP', dataset, seed, fold, variant, flush=True)
        return
    (tx, tm, ty, vx, vm, vy, testx, testm, testy, test_ids) = arrays
    model_seed = int(seed + 100000 * (fold - 1))
    batch_seed = int(model_seed + 17000000)
    model = create_model(res, prior, model_seed)
    initial_hash = parameter_hash(model)
    loader_generator = torch.Generator().manual_seed(batch_seed)
    train_loader = DataLoader(TensorDataset(torch.tensor(tx), torch.tensor(tm), torch.tensor(ty).float()), batch_size=BATCH_SIZE, shuffle=True, drop_last=False, generator=loader_generator)
    val_loader = DataLoader(TensorDataset(torch.tensor(vx), torch.tensor(vm), torch.tensor(vy).float()), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TensorDataset(torch.tensor(testx), torch.tensor(testm), torch.tensor(testy).float()), batch_size=BATCH_SIZE, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    criterion = nn.BCEWithLogitsLoss()
    (best_score, patience, best_state, best_epoch) = (-float('inf'), 0, None, 0)
    history = []
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for epoch in range(EPOCHS):
        model.train()
        batch_state = array_hash(loader_generator.get_state().numpy())
        loss_sum = 0.0
        for (x, mi, y) in train_loader:
            (x, mi, y) = (x.to(device), mi.to(device), y.to(device))
            loss = criterion(model(x, mi).squeeze(-1), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(y)
        (val_true, val_probs) = collect(model, val_loader)
        val_auc = roc_auc_score(val_true, val_probs)
        assert np.isfinite(val_auc) and np.isfinite(loss_sum)
        history.append({'epoch': epoch + 1, 'train_loss': loss_sum / len(ty), 'validation_auc': val_auc, 'shuffle_generator_state_sha256': batch_state})
        if val_auc > best_score:
            (best_score, best_epoch, patience) = (val_auc, epoch + 1, 0)
            best_state = model.state_dict()
        else:
            patience += 1
            if patience >= PATIENCE:
                break
    model.load_state_dict(best_state)
    (val_true, val_probs) = collect(model, val_loader)
    (test_true, test_probs) = collect(model, test_loader)
    (auc, aupr) = metrics(test_true, test_probs)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    row = {'dataset': dataset, 'seed': int(seed), 'fold': int(fold), 'variant': variant, 'auc': auc, 'aupr': aupr, 'model_seed': model_seed, 'batch_seed': batch_seed, 'initial_parameters_sha256': initial_hash, 'prior_sha256': array_hash(prior), 'trainable_parameters': sum((p.numel() for p in model.parameters())), 'train_samples': len(ty), 'validation_samples': len(vy), 'test_samples': len(testy), 'epochs_trained': epoch + 1, 'best_validation_epoch': best_epoch, 'evaluated_epoch': epoch + 1, 'best_validation_auc': float(best_score), 'seconds': elapsed, 'learned_pathway_alpha': float(torch.sigmoid(model.pathway_block.alpha).detach().cpu()), 'learned_pathway_temperature': float(model.pathway_block.tau.detach().clamp(min=0.01).cpu())}
    atomic_json(row, directory / 'fold_metrics.json')
    atomic_csv(pd.DataFrame(history), directory / 'training_history.csv')
    atomic_csv(pd.DataFrame({'sample_id': test_ids, 'label': test_true, 'probability': test_probs}), directory / 'test_predictions.csv')
    marker = {'sha256': {name: sha256(directory / name) for name in ['fold_metrics.json', 'training_history.csv', 'test_predictions.csv']}}
    atomic_json(marker, directory / 'complete.json')
    assert is_complete(directory)
    print(f'{dataset} seed={seed} fold={fold}/5 {variant} AUC={auc:.6f} AUPR={aupr:.6f} epochs={epoch + 1} time={elapsed:.1f}s', flush=True)

def summarize(output):
    records = []
    for marker in sorted((output / 'runs').glob('*/*/*/*/complete.json')):
        assert is_complete(marker.parent)
        records.append(json.loads((marker.parent / 'fold_metrics.json').read_text(encoding='utf-8')))
    if not records:
        return
    frame = pd.DataFrame(records)
    atomic_csv(frame, output / 'fold_metrics.csv')
    seed_records = []
    for (key, group) in frame.groupby(['dataset', 'seed', 'variant'], sort=False):
        if len(group) == 5 and set(group.fold) == set(range(1, 6)):
            seed_records.append({'dataset': key[0], 'seed': key[1], 'variant': key[2], 'auc': group.auc.mean(), 'aupr': group.aupr.mean()})
    if not seed_records:
        return
    seeds = pd.DataFrame(seed_records)
    atomic_csv(seeds, output / 'seed_means.csv')
    summary = []
    for ((dataset, variant), group) in seeds.groupby(['dataset', 'variant'], sort=False):
        summary.append({'dataset': dataset, 'variant': variant, 'n_seeds': len(group), 'complete': len(group) == 3, 'auc_mean': group.auc.mean(), 'auc_std': group.auc.std(ddof=1), 'aupr_mean': group.aupr.mean(), 'aupr_std': group.aupr.std(ddof=1)})
    atomic_csv(pd.DataFrame(summary), output / 'summary.csv')

def main():
    global BASE_DATA_DIR, PATHWAY_DIR, GMT_GENE_FILE, GMT_MIRNA_FILE, INTERACTION_FILE, DRIVER_FILE
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=data_dir('stage'))
    parser.add_argument('--pathway-dir', type=Path, default=prior_dir())
    parser.add_argument('--output-dir', type=Path, default=output_dir('run_prior_source_stage'))
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    (BASE_DATA_DIR, PATHWAY_DIR) = (args.data_dir.resolve(), args.pathway_dir.resolve())
    GMT_GENE_FILE = PATHWAY_DIR / '20230205_kegg_hsa.gmt'
    GMT_MIRNA_FILE = PATHWAY_DIR / 'kegg_anano.txt'
    INTERACTION_FILE = PATHWAY_DIR / 'miRTarBase_MTI_processed_cleaned.csv'
    DRIVER_FILE = driver_file()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = [Path(__file__).resolve(), GMT_GENE_FILE, GMT_MIRNA_FILE, INTERACTION_FILE, DRIVER_FILE]
    files += [BASE_DATA_DIR / ds / name for ds in DATASETS for name in ['mRNA.csv', 'miRNA.csv', 'label.csv']]
    manifest = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files]
    config = {'protocol_version': PROTOCOL_VERSION, 'datasets': DATASETS, 'variants': VARIANTS, 'seeds': SEEDS, 'n_folds': N_FOLDS, 'batch_size': BATCH_SIZE, 'epochs': EPOCHS, 'patience': PATIENCE, 'learning_rate': LEARNING_RATE, 'weight_decay': 0.0001, 'graph_tie_seed': GRAPH_TIE_SEED, 'checkpoint_policy': 'source-compatible stopping-epoch weights', 'model_seed_formula': 'seed + 100000*(fold-1)', 'batch_seed_formula': 'model_seed + 17000000', 'graph_policy': 'd=min(original nonself degree, regulatory positive support, Jaccard positive support); positive-only top-d; binary+self+RowNorm', 'python_executable': sys.executable, 'python_version': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': sklearn.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', 'torch_num_threads': torch.get_num_threads(), 'platform': platform.platform(), 'inputs': manifest}
    config_path = output / 'run_config.json'
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding='utf-8')) == config, 'Changed configuration/input: use a new output directory'
    else:
        atomic_json(config, config_path)
    atomic_csv(pd.DataFrame(manifest), output / 'input_manifest.csv')
    cohorts = {}
    all_priors = {}
    graph_info = []
    for dataset in DATASETS:
        cohorts[dataset] = load_checked(dataset)
        (all_priors[dataset], info) = prepare_graphs(cohorts[dataset], dataset, output)
        graph_info.append(info)
        print('GRAPH', dataset, 'edges=', info['regulatory_graph']['offdiag_edges'], 'Jaccard=', round(info['offdiag_edge_jaccard'], 4), 'reduced_rows=', info['budget_reduced_rows'], flush=True)
    assert len({i['independent_raw_score_sha256'] for i in graph_info}) == 1
    atomic_json(graph_info, output / 'graph_diagnostics.json')
    checks = check_model_pair(cohorts['KIRC'], all_priors['KIRC'])
    checks['script_sha256'] = sha256(__file__)
    atomic_json(checks, output / 'implementation_checks.json')
    print('PAIR CHECKS', json.dumps(checks), flush=True)
    if args.prepare_only:
        print('PREPARATION COMPLETE; no training performed.', flush=True)
        return
    print('TRAINING START', datetime.now(timezone(timedelta(hours=8))).isoformat(), flush=True)
    for dataset in DATASETS:
        res = cohorts[dataset]
        x = res[0].values.astype(np.float32)
        mi = res[1].values.astype(np.float32)
        y = res[-1]
        ids = res[0].index
        for seed in SEEDS:
            splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
            for (fold, (outer_train, test)) in enumerate(splitter.split(x, y), 1):
                (tr, va) = train_test_split(np.arange(len(outer_train)), test_size=0.2, stratify=y[outer_train], random_state=seed)
                (train, valid) = (outer_train[tr], outer_train[va])
                assert len(set(train) | set(valid) | set(test)) == len(y)
                pair_directory = output / 'runs' / dataset / str(seed) / str(fold)
                pair_directory.mkdir(parents=True, exist_ok=True)
                splits = []
                for (role, indices) in [('train', train), ('validation', valid), ('test', test)]:
                    splits.extend(({'sample_id': str(ids[i]), 'label': int(y[i]), 'split': role} for i in indices))
                atomic_csv(pd.DataFrame(splits), pair_directory / 'sample_splits.csv')
                scaler = StandardScaler().fit(x[train])
                scaler_m = StandardScaler().fit(mi[train])
                assert len(train) % BATCH_SIZE != 1, 'Source BatchNorm cannot train singleton batches'
                arrays = (scaler.transform(x[train]), scaler_m.transform(mi[train]), y[train], scaler.transform(x[valid]), scaler_m.transform(mi[valid]), y[valid], scaler.transform(x[test]), scaler_m.transform(mi[test]), y[test], ids[test])
                scale_info = {'fit_sample_ids_sha256': hashlib.sha256('\n'.join((str(ids[i]) for i in train)).encode()).hexdigest(), 'mrna_mean_sha256': array_hash(scaler.mean_), 'mrna_scale_sha256': array_hash(scaler.scale_), 'mirna_mean_sha256': array_hash(scaler_m.mean_), 'mirna_scale_sha256': array_hash(scaler_m.scale_)}
                atomic_json(scale_info, pair_directory / 'normalization_fingerprint.json')
                for variant in VARIANTS:
                    train_fold(res, all_priors[dataset][variant], variant, dataset, seed, fold, pair_directory, arrays)
                pair_rows = [json.loads((pair_directory / v / 'fold_metrics.json').read_text(encoding='utf-8')) for v in VARIANTS]
                assert pair_rows[0]['initial_parameters_sha256'] == pair_rows[1]['initial_parameters_sha256']
                assert pair_rows[0]['trainable_parameters'] == pair_rows[1]['trainable_parameters']
                histories = [pd.read_csv(pair_directory / v / 'training_history.csv') for v in VARIANTS]
                n = min((len(h) for h in histories))
                assert histories[0].shuffle_generator_state_sha256.iloc[:n].equals(histories[1].shuffle_generator_state_sha256.iloc[:n])
            summarize(output)
    summarize(output)
    assert len(pd.read_csv(output / 'fold_metrics.csv')) == 120
    print('COMPLETE:', str(output / 'summary.csv'), flush=True)
if __name__ == '__main__':
    main()
