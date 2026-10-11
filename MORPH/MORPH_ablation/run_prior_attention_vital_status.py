from __future__ import annotations
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
DATASETS = ['LAML', 'SKCM', 'UVM']
SEEDS = [1028, 2024, 3407]
N_FOLDS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 100
PATIENCE = 15
BASE_DATA_DIR = data_dir('vital_status')
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

    def __init__(self, n_genes, n_mirnas, n_pathways, gene_mask, mirna_mask, path_embed_dim=16, out_dim=128, path_prior=None, num_heads=4, topk_ratio=0.028, topk_min=5, topk_max=22, tau_init=0.23, alpha_init=0.57, attn_dropout=0.21):
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
        self.micro_reg_block = MicroRegCrossAttnBlock(n_genes=len(driver_idx), n_mirnas=dim_mirna, reg_matrix=reg_tensor, embed_dim=8, out_dim=hidden_dim)
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

def collect_probs_and_labels(model, data_loader):
    (all_probs, all_true) = ([], [])
    model.eval()
    with torch.no_grad():
        for (bx_m, bx_mi, by) in data_loader:
            (bx_m, bx_mi) = (bx_m.to(device), bx_mi.to(device))
            logits = model(bx_m, bx_mi)
            probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
            if np.ndim(probs) == 0:
                probs = np.expand_dims(probs, 0)
            all_probs.extend(probs.tolist())
            all_true.extend(by.numpy().tolist())
    return (np.array(all_true).astype(int), np.array(all_probs, dtype=np.float64))

def compute_metrics(y_true, y_probs):
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs, dtype=np.float64)
    if len(np.unique(y_true)) < 2:
        return [np.nan, np.nan]
    try:
        auc = roc_auc_score(y_true, y_probs)
        aupr = average_precision_score(y_true, y_probs)
    except:
        (auc, aupr) = (np.nan, np.nan)
    return [auc, aupr]

def nanstd_ddof1(values, axis=0):
    values = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(values, axis=axis, keepdims=True)
    sq_diff = (values - mean) ** 2
    valid_count = np.sum(~np.isnan(values), axis=axis)
    denom = np.where(valid_count > 1, valid_count - 1, np.nan)
    return np.sqrt(np.nansum(sq_diff, axis=axis) / denom)

def run_experiment(dataset_name, seed):
    set_seeds(seed)
    res = load_data(dataset_name)
    if res[0] is None:
        return None
    (df_mrna, df_mirna, gene_path_mask_df, mirna_path_mask_df, reg_matrix, driver_idx, path_prior, y) = res
    X_mrna = df_mrna.values.astype(np.float32)
    X_mirna = df_mirna.values.astype(np.float32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_metrics = []
    for (fold_idx, (train_idx, test_idx)) in enumerate(skf.split(X_mrna, y)):
        (X_m_train, X_m_test) = (X_mrna[train_idx], X_mrna[test_idx])
        (X_mi_train, X_mi_test) = (X_mirna[train_idx], X_mirna[test_idx])
        (y_train, y_test) = (y[train_idx], y[test_idx])
        (sub_train_idx, sub_val_idx) = train_test_split(np.arange(len(y_train)), test_size=0.2, stratify=y_train, random_state=seed)
        (tm_x, vm_x) = (X_m_train[sub_train_idx], X_m_train[sub_val_idx])
        (tmi_x, vmi_x) = (X_mi_train[sub_train_idx], X_mi_train[sub_val_idx])
        (ty, vy) = (y_train[sub_train_idx], y_train[sub_val_idx])
        (test_m_x, test_mi_x) = (X_m_test, X_mi_test)
        scaler_m = StandardScaler().fit(tm_x)
        tm_x = scaler_m.transform(tm_x)
        vm_x = scaler_m.transform(vm_x)
        test_m_x = scaler_m.transform(test_m_x)
        scaler_mi = StandardScaler().fit(tmi_x)
        tmi_x = scaler_mi.transform(tmi_x)
        vmi_x = scaler_mi.transform(vmi_x)
        test_mi_x = scaler_mi.transform(test_mi_x)
        train_dataset = TensorDataset(torch.tensor(tm_x), torch.tensor(tmi_x), torch.tensor(ty).float())
        val_dataset = TensorDataset(torch.tensor(vm_x), torch.tensor(vmi_x), torch.tensor(vy).float())
        test_dataset = TensorDataset(torch.tensor(test_m_x), torch.tensor(test_mi_x), torch.tensor(y_test).float())
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        model = PathformerBio(dim_mrna=X_mrna.shape[1], dim_mirna=X_mirna.shape[1], gene_path_mask_df=gene_path_mask_df, mirna_path_mask_df=mirna_path_mask_df, reg_matrix=reg_matrix, driver_idx=driver_idx, path_prior=path_prior, hidden_dim=128).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
        criterion = nn.BCEWithLogitsLoss()
        best_val_score = float('-inf')
        patience_counter = 0
        best_model_state = None
        for epoch in range(EPOCHS):
            model.train()
            for (bx_m, bx_mi, by) in train_loader:
                (bx_m, bx_mi, by) = (bx_m.to(device), bx_mi.to(device), by.to(device))
                logits_fusion = model(bx_m, bx_mi)
                loss = criterion(logits_fusion.squeeze(), by)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            (val_probs, val_true) = ([], [])
            with torch.no_grad():
                for (bx_m, bx_mi, by) in val_loader:
                    (bx_m, bx_mi) = (bx_m.to(device), bx_mi.to(device))
                    logits = model(bx_m, bx_mi)
                    probs = torch.sigmoid(logits.squeeze()).cpu().numpy()
                    if np.ndim(probs) == 0:
                        probs = np.expand_dims(probs, 0)
                    val_probs.extend(probs.tolist())
                    val_true.extend(by.numpy().tolist())
            try:
                val_auc = roc_auc_score(val_true, val_probs)
            except:
                val_auc = 0.5
            if val_auc > best_val_score:
                best_val_score = val_auc
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        (val_true, val_probs) = collect_probs_and_labels(model, val_loader)
        (test_true, test_probs) = collect_probs_and_labels(model, test_loader)
        metrics_list = compute_metrics(test_true, test_probs)
        fold_metrics.append(metrics_list)
    fold_metrics = np.array(fold_metrics, dtype=np.float64)
    seed_mean = np.nanmean(fold_metrics, axis=0)
    return seed_mean
import argparse
import hashlib
import itertools
import json
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
SCRIPT_ROOT = Path(__file__).resolve().parent
TASK_NAME = 'survival_status'
TASK_LABEL = 'Vital-status classification'
TASK_DATASETS = ('LAML', 'SKCM', 'UVM')
TASK_SEEDS = (1028, 2024, 3407)
DEFAULT_DATA_ROOT = data_dir('vital_status')
DEFAULT_PATHWAY_DIR = prior_dir()
DEFAULT_OUTPUT_DIR = output_dir('run_prior_attention_vital_status')
LEGACY_TOPK_RATIO = 0.028
LEGACY_TOPK_MIN = 5
LEGACY_TOPK_MAX = 22
LEGACY_TAU_INIT = 0.23
PAPER_FULL_RESULTS = {'LAML': {'auc_mean': 0.7118, 'auc_std': 0.0373, 'aupr_mean': 0.8095, 'aupr_std': 0.0179}, 'SKCM': {'auc_mean': 0.6619, 'auc_std': 0.0181, 'aupr_mean': 0.5226, 'aupr_std': 0.0258}, 'UVM': {'auc_mean': 0.785, 'auc_std': 0.0116, 'aupr_mean': 0.6405, 'aupr_std': 0.0303}}

@dataclass(frozen=True)
class AblationSpec:
    name: str
    group: str
    query_mode: str = 'driver'
    random_replicate: int | None = None
    alpha_mode: str | float = 'adaptive'
    beta_mode: str | float = 'adaptive'
    mask_mode: str = 'legacy'
    k: int | None = None
    tau_mode: str | float = 'learnable'
    description: str = ''
SPECS = {'full_original': AblationSpec('full_original', 'baseline', description='Exact supplied paper model; must run first.'), 'gate_fixed_both': AblationSpec('gate_fixed_both', 'gating', alpha_mode=0.5, beta_mode=0.5), 'gate_data_only': AblationSpec('gate_data_only', 'gating', alpha_mode=0.0, beta_mode=0.0, description='Same-capacity attention with both prior gates set to zero.'), 'gate_prior_biased_only': AblationSpec('gate_prior_biased_only', 'gating', alpha_mode=1.0, beta_mode=1.0, description='Micro branch uses data logits plus miRTarBase bias; it is prior-biased attention, not a static prior matrix.'), 'gate_fixed_micro': AblationSpec('gate_fixed_micro', 'gating', alpha_mode='adaptive', beta_mode=0.5), 'gate_fixed_pathway': AblationSpec('gate_fixed_pathway', 'gating', alpha_mode=0.5, beta_mode='adaptive'), 'path_dense_weighted': AblationSpec('path_dense_weighted', 'pathway', mask_mode='dense'), 'path_topk_2': AblationSpec('path_topk_2', 'pathway', mask_mode='topk', k=2), 'path_topk_10': AblationSpec('path_topk_10', 'pathway', mask_mode='topk', k=10), 'path_topk_20': AblationSpec('path_topk_20', 'pathway', mask_mode='topk', k=20), 'path_topk_40': AblationSpec('path_topk_40', 'pathway', mask_mode='topk', k=40), 'path_random_matched': AblationSpec('path_random_matched', 'pathway', mask_mode='random_matched'), 'temperature_fixed_0_5': AblationSpec('temperature_fixed_0_5', 'temperature', tau_mode=0.5), 'temperature_fixed_1': AblationSpec('temperature_fixed_1', 'temperature', tau_mode=1.0), 'temperature_fixed_2': AblationSpec('temperature_fixed_2', 'temperature', tau_mode=2.0)}
VARIANT_ORDER = tuple(SPECS)

def script_sha256() -> str:
    digest = hashlib.sha256()
    with Path(__file__).open('rb') as handle:
        for block in iter(lambda : handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def stable_sampling_seed(dataset: str, replicate: int) -> int:
    token = f'MORPH|{TASK_NAME}|random-non-driver|{dataset}|{replicate}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(token).digest()[:8], 'little')

def normalize_gene(value) -> str:
    return str(value).strip().upper()

def require_inputs(data_root: Path, pathway_dir: Path, datasets) -> None:
    required_prior = (pathway_dir / '20230205_kegg_hsa.gmt', pathway_dir / 'kegg_anano.txt', pathway_dir / 'miRTarBase_MTI_processed_cleaned.csv', driver_file())
    missing = [str(path) for path in required_prior if not path.is_file()]
    for dataset in datasets:
        for filename in ('mRNA.csv', 'miRNA.csv', 'label.csv'):
            path = data_root / dataset / filename
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError('Required inputs are missing:\n' + '\n'.join(missing))

def install_runtime_paths(data_root: Path, pathway_dir: Path) -> None:
    global BASE_DATA_DIR, PATHWAY_DIR
    global GMT_GENE_FILE, GMT_MIRNA_FILE, INTERACTION_FILE, DRIVER_FILE
    BASE_DATA_DIR = data_root
    PATHWAY_DIR = pathway_dir
    GMT_GENE_FILE = pathway_dir / '20230205_kegg_hsa.gmt'
    GMT_MIRNA_FILE = pathway_dir / 'kegg_anano.txt'
    INTERACTION_FILE = pathway_dir / 'miRTarBase_MTI_processed_cleaned.csv'
    DRIVER_FILE = driver_file()

def driver_reference_set() -> set[str]:
    with DRIVER_FILE.open('r', encoding='utf-8') as handle:
        values = {normalize_gene(line) for line in handle if line.strip()}
    if not values:
        raise ValueError(f'Driver file is empty: {DRIVER_FILE}')
    return values

def prepare_cohort(dataset: str) -> dict:
    loaded = load_data(dataset)
    if loaded[0] is None:
        raise FileNotFoundError(f'Unable to load {TASK_NAME}/{dataset}')
    (mrna, mirna, gene_mask, mirna_mask, driver_regulation, driver_indices, path_prior, labels) = loaded
    genes = list(mrna.columns)
    mirnas = list(mirna.columns)
    labels = np.asarray(labels, dtype=int)
    driver_indices = [int(index) for index in driver_indices]
    if not driver_indices:
        raise ValueError(f'{dataset}: no driver queries were retained.')
    if gene_mask.shape[1] == 0 or mirna_mask.shape[1] == 0:
        raise ValueError(f'{dataset}: no shared pathway annotations.')
    all_regulation = build_mirna_gene_interaction_matrix(genes, mirnas)
    reconstructed = all_regulation[:, np.asarray(driver_indices, dtype=np.int64)]
    if not np.array_equal(reconstructed, np.asarray(driver_regulation, dtype=np.float32)):
        raise RuntimeError(f'{dataset}: driver regulation matrix is inconsistent.')
    source_drivers = driver_reference_set()
    eligible_non_driver = np.asarray([index for (index, gene) in enumerate(genes) if normalize_gene(gene) not in source_drivers and all_regulation[:, index].sum() > 0], dtype=np.int64)
    if len(eligible_non_driver) < len(driver_indices):
        raise ValueError(f'{dataset}: insufficient miRTarBase-linked non-driver genes.')
    queries = {'driver': {'indices': tuple(driver_indices), 'regulation': np.asarray(driver_regulation, dtype=np.float32), 'sampling_seed': None}, 'all_gene': {'indices': tuple(range(len(genes))), 'regulation': np.asarray(all_regulation, dtype=np.float32), 'sampling_seed': None}}
    random_rows = []
    for replicate in (1, 2, 3):
        sample_seed = stable_sampling_seed(dataset, replicate)
        rng = np.random.default_rng(sample_seed)
        selected = np.sort(rng.choice(eligible_non_driver, size=len(driver_indices), replace=False))
        key = f'random_non_driver_r{replicate}'
        queries[key] = {'indices': tuple((int(index) for index in selected)), 'regulation': np.asarray(all_regulation[:, selected], dtype=np.float32), 'sampling_seed': sample_seed}
        for index in selected:
            random_rows.append({'task': TASK_NAME, 'dataset': dataset, 'replicate': replicate, 'sampling_seed': sample_seed, 'gene_index': int(index), 'gene': genes[int(index)]})
    path_prior = np.asarray(path_prior, dtype=np.float32)
    supported = path_prior > 0
    degree = supported.sum(axis=1)
    p = int(path_prior.shape[0])
    actual_k = max(LEGACY_TOPK_MIN, min(LEGACY_TOPK_MAX, int(p * LEGACY_TOPK_RATIO)))
    raw_prior = {'task': TASK_NAME, 'dataset': dataset, 'pathway_count': p, 'raw_nonzero_edges_with_self': int(supported.sum()), 'raw_density_with_self': float(supported.mean()), 'raw_row_degree_min': int(degree.min()), 'raw_row_degree_mean': float(degree.mean()), 'raw_row_degree_max': int(degree.max()), 'raw_self_loops': int(np.diag(supported).sum()), 'legacy_topk_ratio': LEGACY_TOPK_RATIO, 'legacy_topk_min': LEGACY_TOPK_MIN, 'legacy_topk_max': LEGACY_TOPK_MAX, 'actual_legacy_k': actual_k, 'direction_semantics': 'column miRNA-associated pathway to row target-gene pathway', 'evidence_boundary': 'projected directional regulatory association; not experimentally validated pathway signaling or causality'}
    return {'dataset': dataset, 'mrna': mrna, 'mirna': mirna, 'gene_mask': gene_mask, 'mirna_mask': mirna_mask, 'path_prior': path_prior, 'labels': labels, 'sample_ids': np.asarray(mrna.index.astype(str)), 'sample_count': int(len(labels)), 'label_0_count': int(np.sum(labels == 0)), 'label_1_count': int(np.sum(labels == 1)), 'queries': queries, 'random_rows': random_rows, 'raw_prior': raw_prior}

def resolve_alpha(block) -> torch.Tensor:
    mode = getattr(block, '_revision_alpha_mode', 'adaptive')
    if mode == 'adaptive':
        return torch.sigmoid(block.alpha)
    return torch.as_tensor(float(mode), dtype=block.alpha.dtype, device=block.alpha.device)

def resolve_beta(block, q: torch.Tensor) -> torch.Tensor:
    mode = getattr(block, '_revision_beta_mode', 'adaptive')
    if mode == 'adaptive':
        return torch.sigmoid(block.prior_mix_net(q))
    return torch.full_like(q[..., :1], float(mode))

def resolve_tau(block) -> torch.Tensor:
    mode = getattr(block, '_revision_tau_mode', 'learnable')
    if mode == 'learnable':
        return torch.clamp(block.tau, min=0.01)
    return torch.as_tensor(float(mode), dtype=block.tau.dtype, device=block.tau.device)

def revision_pathway_forward(self, x_mrna, x_mirna):
    batch_size = x_mrna.shape[0]
    h_g_local = F.gelu(self.g_proj(x_mrna.unsqueeze(-1)))
    h_m_local = F.gelu(self.m_proj(x_mirna.unsqueeze(-1)))
    path_g = torch.einsum('b g d, g p -> b p d', h_g_local, self.gene_mask)
    path_m = torch.einsum('b m d, m p -> b p d', h_m_local, self.mirna_mask)
    q = self.to_q(path_g).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.to_k(path_m).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.to_v(path_m).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    dots = dots / resolve_tau(self)
    data_attention = F.softmax(dots, dim=-1)
    if self.prior_norm_mask is not None:
        alpha = resolve_alpha(self)
        prior = self.prior_norm_mask.unsqueeze(0).unsqueeze(0)
        attention = alpha * prior + (1.0 - alpha) * data_attention
    else:
        attention = data_attention
    attention = self.attn_drop(attention)
    update = torch.matmul(attention, v)
    update = update.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_dim)
    refined = self.norm(path_g + update)
    gate = torch.sigmoid(self.gate_proj(refined))
    gated = refined * gate
    pooled = torch.cat([gated.mean(dim=1), gated.max(dim=1)[0]], dim=-1)
    return self.final_proj(pooled)

def revision_micro_forward(self, x_mrna, x_mirna):
    genes = self.gene_proj(x_mrna.unsqueeze(-1))
    mirnas = self.mirna_proj(x_mirna.unsqueeze(-1))
    q = self.to_q(genes)
    k = self.to_k(mirnas)
    v = self.to_v(mirnas)
    data_logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale
    prior_scale = F.softplus(self.reg_scale_raw)
    prior_logits = data_logits + prior_scale * self.reg_bias.unsqueeze(0)
    data_attention = F.softmax(data_logits, dim=-1)
    prior_biased_attention = F.softmax(prior_logits, dim=-1)
    beta = resolve_beta(self, q)
    attention = beta * prior_biased_attention + (1.0 - beta) * data_attention
    update = torch.matmul(attention, v)
    refined = self.norm(genes + update)
    pooling = torch.softmax(self.pool_score(refined), dim=1)
    return self.out_proj(torch.sum(pooling * refined, dim=1))
PathwayCrossAttnBlock.forward = revision_pathway_forward
MicroRegCrossAttnBlock.forward = revision_micro_forward

def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    return mask / (mask.sum(dim=-1, keepdim=True) + 1e-08)

def deterministic_topk_mask(path_prior: np.ndarray, k: int) -> torch.Tensor:
    prior = np.asarray(path_prior, dtype=np.float32)
    p = prior.shape[0]
    effective_k = max(1, min(int(k), p))
    mask = np.zeros_like(prior, dtype=np.float32)
    columns = np.arange(p)
    for row in range(p):
        order = np.lexsort((columns, -prior[row]))
        mask[row, order[:effective_k]] = 1.0
        mask[row, row] = 1.0
    return normalize_mask(torch.tensor(mask, dtype=torch.float32))

def matched_random_mask(reference_mask: torch.Tensor, dataset: str, seed: int, fold: int) -> torch.Tensor:
    reference = reference_mask.detach().cpu().numpy() > 0
    p = reference.shape[0]
    dataset_code = sum(((index + 1) * ord(character) for (index, character) in enumerate(dataset)))
    rng = np.random.default_rng(91000003 + dataset_code * 101 + seed * 17 + fold)
    output = np.zeros((p, p), dtype=np.float32)
    columns = np.arange(p)
    for row in range(p):
        degree = int(reference[row].sum())
        output[row, row] = 1.0
        remaining = max(0, degree - 1)
        if remaining:
            candidates = columns[columns != row]
            chosen = rng.choice(candidates, size=remaining, replace=False)
            output[row, chosen] = 1.0
    return normalize_mask(torch.tensor(output, dtype=torch.float32))

def mask_diagnostics(mask: torch.Tensor) -> dict:
    binary = mask.detach().cpu().numpy() > 0
    degrees = binary.sum(axis=1)
    return {'retained_edges': int(binary.sum()), 'retained_density': float(binary.mean()), 'retained_row_degree_min': int(degrees.min()), 'retained_row_degree_mean': float(degrees.mean()), 'retained_row_degree_max': int(degrees.max()), 'retained_self_loops': int(np.diag(binary).sum())}

def query_key(spec: AblationSpec) -> str:
    if spec.query_mode == 'random_non_driver':
        return f'random_non_driver_r{spec.random_replicate}'
    return spec.query_mode

def make_model(cohort: dict, spec: AblationSpec, seed: int, fold: int):
    query = cohort['queries'][query_key(spec)]
    model = PathformerBio(dim_mrna=cohort['mrna'].shape[1], dim_mirna=cohort['mirna'].shape[1], gene_path_mask_df=cohort['gene_mask'], mirna_path_mask_df=cohort['mirna_mask'], reg_matrix=query['regulation'], driver_idx=list(query['indices']), path_prior=cohort['path_prior'], hidden_dim=128).to(device)
    model.pathway_block._revision_alpha_mode = spec.alpha_mode
    model.micro_reg_block._revision_beta_mode = spec.beta_mode
    model.pathway_block._revision_tau_mode = spec.tau_mode
    original_mask = model.pathway_block.prior_norm_mask.detach().clone()
    if spec.mask_mode == 'legacy':
        configured = original_mask
    elif spec.mask_mode == 'dense':
        configured = torch.tensor(cohort['path_prior'], dtype=torch.float32)
        configured = normalize_mask(configured)
    elif spec.mask_mode == 'topk':
        configured = deterministic_topk_mask(cohort['path_prior'], int(spec.k))
    elif spec.mask_mode == 'random_matched':
        configured = matched_random_mask(original_mask, cohort['dataset'], seed, fold)
    else:
        raise ValueError(f'Unknown mask mode: {spec.mask_mode}')
    model.pathway_block.prior_norm_mask.copy_(configured.to(device=model.pathway_block.prior_norm_mask.device))
    model._revision_metadata = {'experiment_group': spec.group, 'query_mode': spec.query_mode, 'random_replicate': spec.random_replicate, 'query_sampling_seed': query['sampling_seed'], 'alpha_mode': spec.alpha_mode, 'beta_mode': spec.beta_mode, 'mask_mode': spec.mask_mode, 'nominal_k': spec.k, 'tau_mode': spec.tau_mode, **mask_diagnostics(model.pathway_block.prior_norm_mask)}
    return model

def collect_probabilities(model, loader):
    probabilities = []
    labels = []
    model.eval()
    with torch.no_grad():
        for (batch_mrna, batch_mirna, batch_labels) in loader:
            batch_mrna = batch_mrna.to(device)
            batch_mirna = batch_mirna.to(device)
            logits = model(batch_mrna, batch_mirna)
            values = torch.sigmoid(logits.squeeze()).cpu().numpy()
            if np.ndim(values) == 0:
                values = np.expand_dims(values, 0)
            probabilities.extend(values.tolist())
            labels.extend(batch_labels.numpy().tolist())
    return (np.asarray(labels, dtype=int), np.asarray(probabilities, dtype=np.float64))

def describe_gate(values: np.ndarray, prefix: str) -> dict:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return {f'{prefix}_count': int(values.size), f'{prefix}_mean': float(values.mean()), f'{prefix}_std': float(values.std(ddof=0)), f'{prefix}_median': float(np.median(values)), f'{prefix}_q25': float(np.quantile(values, 0.25)), f'{prefix}_q75': float(np.quantile(values, 0.75)), f'{prefix}_lt_0_05': float(np.mean(values < 0.05)), f'{prefix}_gt_0_95': float(np.mean(values > 0.95)), f'{prefix}_saturated': float(np.mean((values < 0.05) | (values > 0.95)))}

def collect_diagnostics(model, loader) -> dict:
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    beta_chunks = []
    try:
        model.eval()
        with torch.no_grad():
            alpha = float(resolve_alpha(model.pathway_block).detach().cpu())
            alpha_raw = float(model.pathway_block.alpha.detach().cpu())
            tau = float(resolve_tau(model.pathway_block).detach().cpu())
            tau_raw = float(model.pathway_block.tau.detach().cpu())
            reg_scale = float(F.softplus(model.micro_reg_block.reg_scale_raw).detach().cpu())
            for (batch_mrna, _, _) in loader:
                batch_mrna = batch_mrna.to(device)
                driver_values = batch_mrna.index_select(1, model.driver_idx)
                embedded = model.micro_reg_block.gene_proj(driver_values.unsqueeze(-1))
                q = model.micro_reg_block.to_q(embedded)
                beta = resolve_beta(model.micro_reg_block, q)
                beta_chunks.append(beta.detach().cpu().numpy().reshape(-1))
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    result = {'alpha': alpha, 'alpha_raw_parameter': alpha_raw, 'alpha_lt_0_05': float(alpha < 0.05), 'alpha_gt_0_95': float(alpha > 0.95), 'alpha_saturated': float(alpha < 0.05 or alpha > 0.95), 'tau': tau, 'tau_raw_parameter': tau_raw, 'reg_scale': reg_scale}
    result.update(describe_gate(np.concatenate(beta_chunks), 'beta'))
    return result

def metric_pair(labels, probabilities):
    if len(np.unique(labels)) < 2:
        return (np.nan, np.nan)
    return (float(roc_auc_score(labels, probabilities)), float(average_precision_score(labels, probabilities)))

def run_seed(cohort: dict, spec: AblationSpec, seed: int):
    set_seeds(seed)
    started = time.perf_counter()
    x_mrna = cohort['mrna'].values.astype(np.float32)
    x_mirna = cohort['mirna'].values.astype(np.float32)
    labels = cohort['labels']
    splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_rows = []
    prediction_rows = []
    diagnostic_rows = []
    for (fold, (train_index, test_index)) in enumerate(splitter.split(x_mrna, labels)):
        fold_started = time.perf_counter()
        (x_m_train, x_m_test) = (x_mrna[train_index], x_mrna[test_index])
        (x_mi_train, x_mi_test) = (x_mirna[train_index], x_mirna[test_index])
        (y_train, y_test) = (labels[train_index], labels[test_index])
        (sub_train_index, validation_index) = train_test_split(np.arange(len(y_train)), test_size=0.2, stratify=y_train, random_state=seed)
        train_mrna = x_m_train[sub_train_index]
        validation_mrna = x_m_train[validation_index]
        train_mirna = x_mi_train[sub_train_index]
        validation_mirna = x_mi_train[validation_index]
        train_labels = y_train[sub_train_index]
        validation_labels = y_train[validation_index]
        mrna_scaler = StandardScaler().fit(train_mrna)
        train_mrna = mrna_scaler.transform(train_mrna)
        validation_mrna = mrna_scaler.transform(validation_mrna)
        test_mrna = mrna_scaler.transform(x_m_test)
        mirna_scaler = StandardScaler().fit(train_mirna)
        train_mirna = mirna_scaler.transform(train_mirna)
        validation_mirna = mirna_scaler.transform(validation_mirna)
        test_mirna = mirna_scaler.transform(x_mi_test)
        train_loader = DataLoader(TensorDataset(torch.tensor(train_mrna), torch.tensor(train_mirna), torch.tensor(train_labels).float()), batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
        validation_loader = DataLoader(TensorDataset(torch.tensor(validation_mrna), torch.tensor(validation_mirna), torch.tensor(validation_labels).float()), batch_size=BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TensorDataset(torch.tensor(test_mrna), torch.tensor(test_mirna), torch.tensor(y_test).float()), batch_size=BATCH_SIZE, shuffle=False)
        model = make_model(cohort, spec, seed, fold)
        metadata = dict(model._revision_metadata)
        parameter_count = sum((parameter.numel() for parameter in model.parameters()))
        trainable_count = sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
        criterion = nn.BCEWithLogitsLoss()
        best_validation_auc = float('-inf')
        best_epoch = -1
        best_model_state = None
        best_diagnostics = None
        patience_counter = 0
        epochs_trained = 0
        for epoch in range(EPOCHS):
            epochs_trained = epoch + 1
            model.train()
            for (batch_mrna, batch_mirna, batch_labels) in train_loader:
                batch_mrna = batch_mrna.to(device)
                batch_mirna = batch_mirna.to(device)
                batch_labels = batch_labels.to(device)
                logits = model(batch_mrna, batch_mirna)
                loss = criterion(logits.squeeze(), batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            (validation_true, validation_probabilities) = collect_probabilities(model, validation_loader)
            try:
                validation_auc = roc_auc_score(validation_true, validation_probabilities)
            except ValueError:
                validation_auc = 0.5
            if validation_auc > best_validation_auc:
                best_validation_auc = float(validation_auc)
                best_epoch = epoch + 1
                best_diagnostics = collect_diagnostics(model, validation_loader)
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        collect_probabilities(model, validation_loader)
        (test_true, test_probabilities) = collect_probabilities(model, test_loader)
        (test_auc, test_aupr) = metric_pair(test_true, test_probabilities)
        test_diagnostics = collect_diagnostics(model, test_loader)
        common = {'task': TASK_NAME, 'dataset': cohort['dataset'], 'variant': spec.name, 'experiment_group': spec.group, 'seed': int(seed), 'fold': int(fold + 1)}
        fold_rows.append({**common, 'test_auc': test_auc, 'test_aupr': test_aupr, 'best_validation_auc': best_validation_auc, 'best_validation_epoch': best_epoch, 'epochs_trained': epochs_trained, 'fold_seconds': time.perf_counter() - fold_started, 'sample_count': cohort['sample_count'], 'label_0_count': cohort['label_0_count'], 'label_1_count': cohort['label_1_count'], 'query_count': int(model.driver_idx.numel()), 'parameters': parameter_count, 'trainable_parameters': trainable_count, **metadata})
        diagnostic_rows.append({**common, 'best_validation_epoch': best_epoch, **{f'best_{key}': value for (key, value) in (best_diagnostics or {}).items()}, **{f'test_{key}': value for (key, value) in test_diagnostics.items()}, **metadata})
        for (sample_id, label, probability) in zip(cohort['sample_ids'][test_index], test_true, test_probabilities):
            prediction_rows.append({**common, 'sample_id': sample_id, 'label': int(label), 'probability': float(probability)})
        print(f"{cohort['dataset']} {spec.name} seed={seed} fold={fold + 1} AUC={test_auc:.4f} AUPR={test_aupr:.4f}", flush=True)
    seed_row = {'task': TASK_NAME, 'dataset': cohort['dataset'], 'variant': spec.name, 'experiment_group': spec.group, 'seed': int(seed), 'seed_mean_auc': float(np.mean([row['test_auc'] for row in fold_rows])), 'seed_mean_aupr': float(np.mean([row['test_aupr'] for row in fold_rows])), 'sample_count': cohort['sample_count'], 'label_0_count': cohort['label_0_count'], 'label_1_count': cohort['label_1_count'], 'query_count': fold_rows[0]['query_count'], 'parameters': fold_rows[0]['parameters'], 'trainable_parameters': fold_rows[0]['trainable_parameters'], 'seed_seconds': time.perf_counter() - started}
    return (fold_rows, seed_row, prediction_rows, diagnostic_rows)

def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)

def write_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding='utf-8-sig')

def append_rows(path: Path, rows, keys) -> None:
    if not rows:
        return
    previous = load_csv(path)
    combined = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
    combined = combined.drop_duplicates(list(keys), keep='last')
    combined = combined.sort_values(list(keys)).reset_index(drop=True)
    write_csv(combined, path)

def with_random_aggregate(seed_frame: pd.DataFrame) -> pd.DataFrame:
    if seed_frame.empty:
        return seed_frame
    random_names = {'driver_random_non_driver_r1', 'driver_random_non_driver_r2', 'driver_random_non_driver_r3'}
    random_frame = seed_frame[seed_frame['variant'].isin(random_names)]
    rows = []
    for ((dataset, seed), group) in random_frame.groupby(['dataset', 'seed'], sort=False):
        if set(group['variant']) != random_names:
            continue
        rows.append({'task': TASK_NAME, 'dataset': dataset, 'variant': 'driver_random_non_driver_mean', 'experiment_group': 'driver', 'seed': int(seed), 'seed_mean_auc': float(group['seed_mean_auc'].mean()), 'seed_mean_aupr': float(group['seed_mean_aupr'].mean()), 'sample_count': int(group['sample_count'].iloc[0]), 'label_0_count': int(group['label_0_count'].iloc[0]), 'label_1_count': int(group['label_1_count'].iloc[0]), 'query_count': int(group['query_count'].iloc[0]), 'parameters': int(group['parameters'].iloc[0]), 'trainable_parameters': int(group['trainable_parameters'].iloc[0]), 'seed_seconds': float(group['seed_seconds'].sum())})
    if not rows:
        return seed_frame
    return pd.concat([seed_frame, pd.DataFrame(rows)], ignore_index=True)

def make_summary(seed_frame: pd.DataFrame) -> pd.DataFrame:
    expanded = with_random_aggregate(seed_frame)
    rows = []
    for ((dataset, variant), group) in expanded.groupby(['dataset', 'variant'], sort=False):
        rows.append({'task': TASK_NAME, 'dataset': dataset, 'variant': variant, 'experiment_group': group['experiment_group'].iloc[0], 'seed_runs': int(len(group)), 'auc_mean': float(group['seed_mean_auc'].mean()), 'auc_std_across_seed_means': float(group['seed_mean_auc'].std(ddof=1)), 'aupr_mean': float(group['seed_mean_aupr'].mean()), 'aupr_std_across_seed_means': float(group['seed_mean_aupr'].std(ddof=1)), 'sample_count': int(group['sample_count'].iloc[0]), 'query_count': int(group['query_count'].iloc[0]), 'parameters': int(group['parameters'].iloc[0]), 'trainable_parameters': int(group['trainable_parameters'].iloc[0]), 'total_seconds': float(group['seed_seconds'].sum())})
    return pd.DataFrame(rows)

def make_macro_summary(seed_frame: pd.DataFrame) -> pd.DataFrame:
    expanded = with_random_aggregate(seed_frame)
    rows = []
    for ((variant, seed), group) in expanded.groupby(['variant', 'seed'], sort=False):
        rows.append({'variant': variant, 'seed': int(seed), 'dataset_count': int(group['dataset'].nunique()), 'macro_auc': float(group['seed_mean_auc'].mean()), 'macro_aupr': float(group['seed_mean_aupr'].mean())})
    seed_macro = pd.DataFrame(rows)
    output = []
    for (variant, group) in seed_macro.groupby('variant', sort=False):
        output.append({'task': TASK_NAME, 'variant': variant, 'seed_runs': int(len(group)), 'dataset_count': int(group['dataset_count'].max()), 'macro_auc_mean': float(group['macro_auc'].mean()), 'macro_auc_std': float(group['macro_auc'].std(ddof=1)), 'macro_aupr_mean': float(group['macro_aupr'].mean()), 'macro_aupr_std': float(group['macro_aupr'].std(ddof=1))})
    return pd.DataFrame(output)

def exact_signflip_p(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences)]
    if differences.size == 0:
        return np.nan
    observed = abs(float(differences.mean()))
    values = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(differences)):
        values.append(abs(float(np.mean(differences * np.asarray(signs)))))
    return float(np.mean(np.asarray(values) >= observed - 1e-15))

def make_paired_comparisons(seed_frame: pd.DataFrame) -> pd.DataFrame:
    expanded = with_random_aggregate(seed_frame)
    full = expanded[expanded['variant'] == 'full_original'][['dataset', 'seed', 'seed_mean_auc', 'seed_mean_aupr']].rename(columns={'seed_mean_auc': 'full_auc', 'seed_mean_aupr': 'full_aupr'})
    rows = []
    for variant in expanded['variant'].drop_duplicates():
        if variant == 'full_original':
            continue
        candidate = expanded[expanded['variant'] == variant][['dataset', 'seed', 'seed_mean_auc', 'seed_mean_aupr']].rename(columns={'seed_mean_auc': 'variant_auc', 'seed_mean_aupr': 'variant_aupr'})
        merged = full.merge(candidate, on=['dataset', 'seed'])
        for (dataset, group) in merged.groupby('dataset', sort=False):
            auc_delta = (group['full_auc'] - group['variant_auc']).to_numpy()
            aupr_delta = (group['full_aupr'] - group['variant_aupr']).to_numpy()
            rows.append({'task': TASK_NAME, 'dataset': dataset, 'variant': variant, 'paired_seed_count': int(len(group)), 'full_minus_variant_auc_mean': float(auc_delta.mean()), 'full_minus_variant_aupr_mean': float(aupr_delta.mean()), 'seeds_full_better_auc': int(np.sum(auc_delta > 0)), 'seeds_full_better_aupr': int(np.sum(aupr_delta > 0)), 'exact_seed_signflip_p_auc': exact_signflip_p(auc_delta), 'exact_seed_signflip_p_aupr': exact_signflip_p(aupr_delta), 'statistical_note': 'Exact paired sign-flip test on three seed-level five-fold means; p-values are necessarily coarse.'})
    return pd.DataFrame(rows)

def make_gate_summary(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    selected = diagnostics[diagnostics['experiment_group'].isin(['baseline', 'gating'])]
    rows = []
    for ((dataset, variant), group) in selected.groupby(['dataset', 'variant'], sort=False):
        rows.append({'task': TASK_NAME, 'dataset': dataset, 'variant': variant, 'fold_runs': int(len(group)), 'sample_count': int(PAPER_SAMPLE_COUNTS.get(dataset, np.nan)), 'alpha_mean': float(group['test_alpha'].mean()), 'alpha_std': float(group['test_alpha'].std(ddof=1)), 'alpha_saturated_fraction': float(group['test_alpha_saturated'].mean()), 'beta_mean': float(group['test_beta_mean'].mean()), 'beta_std_across_folds': float(group['test_beta_mean'].std(ddof=1)), 'beta_saturated_fraction': float(group['test_beta_saturated'].mean()), 'tau_mean': float(group['test_tau'].mean()), 'tau_std': float(group['test_tau'].std(ddof=1))})
    return pd.DataFrame(rows)
PAPER_SAMPLE_COUNTS = {'KIRC': 508, 'MESO': 85, 'STAD': 363, 'THCA': 501, 'LAML': 130, 'SKCM': 432, 'UVM': 80}

def make_reproduction_check(summary: pd.DataFrame) -> pd.DataFrame:
    full = summary[summary['variant'] == 'full_original']
    rows = []
    for dataset in TASK_DATASETS:
        observed = full[full['dataset'] == dataset]
        if observed.empty or int(observed.iloc[0]['seed_runs']) != len(TASK_SEEDS):
            continue
        observed = observed.iloc[0]
        expected = PAPER_FULL_RESULTS[dataset]
        checks = {'auc_mean_match': round(observed['auc_mean'], 4) == expected['auc_mean'], 'auc_std_match': round(observed['auc_std_across_seed_means'], 4) == expected['auc_std'], 'aupr_mean_match': round(observed['aupr_mean'], 4) == expected['aupr_mean'], 'aupr_std_match': round(observed['aupr_std_across_seed_means'], 4) == expected['aupr_std']}
        rows.append({'task': TASK_NAME, 'dataset': dataset, 'observed_auc_mean': observed['auc_mean'], 'paper_auc_mean': expected['auc_mean'], 'observed_auc_std': observed['auc_std_across_seed_means'], 'paper_auc_std': expected['auc_std'], 'observed_aupr_mean': observed['aupr_mean'], 'paper_aupr_mean': expected['aupr_mean'], 'observed_aupr_std': observed['aupr_std_across_seed_means'], 'paper_aupr_std': expected['aupr_std'], **checks, 'all_rounded_values_match': all(checks.values())})
    return pd.DataFrame(rows)

def refresh_outputs(output_dir: Path):
    seeds = load_csv(output_dir / 'seed_results.csv')
    diagnostics = load_csv(output_dir / 'diagnostics.csv')
    summary = make_summary(seeds) if not seeds.empty else pd.DataFrame()
    macro = make_macro_summary(seeds) if not seeds.empty else pd.DataFrame()
    paired = make_paired_comparisons(seeds) if not seeds.empty else pd.DataFrame()
    gates = make_gate_summary(diagnostics)
    checks = make_reproduction_check(summary)
    write_csv(summary, output_dir / 'summary.csv')
    write_csv(macro, output_dir / 'macro_summary.csv')
    write_csv(paired, output_dir / 'paired_seed_comparisons.csv')
    write_csv(gates, output_dir / 'gate_summary.csv')
    write_csv(checks, output_dir / 'reproduction_check.csv')
    return (summary, checks)

def baseline_passed(checks: pd.DataFrame, datasets) -> bool:
    for dataset in datasets:
        row = checks[checks['dataset'] == dataset]
        if row.empty or not bool(row.iloc[0]['all_rounded_values_match']):
            return False
    return True

def parse_csv_list(value: str, allowed, cast=str):
    parsed = [cast(item.strip()) for item in value.split(',') if item.strip()]
    invalid = [item for item in parsed if item not in allowed]
    if invalid:
        raise ValueError(f'Unsupported values: {invalid}')
    return parsed

def parse_args():
    parser = argparse.ArgumentParser(description=f'Standalone MORPH {TASK_NAME} ablations for three reviewer concerns.')
    parser.add_argument('--data-root', type=Path, default=data_dir('vital_status'))
    parser.add_argument('--pathway-dir', type=Path, default=prior_dir())
    parser.add_argument('--output-dir', type=Path, default=output_dir('run_prior_attention_vital_status'))
    parser.add_argument('--datasets', default=','.join(TASK_DATASETS))
    parser.add_argument('--seeds', default=','.join(map(str, TASK_SEEDS)))
    parser.add_argument('--variants', default=','.join(VARIANT_ORDER), help='Comma-separated variants; full_original is always run first.')
    parser.add_argument('--allow-paper-mismatch', action='store_true', help='Allow ablations after the full model fails the exact four-decimal paper reproduction gate.')
    return parser.parse_args()

def main():
    args = parse_args()
    datasets = parse_csv_list(args.datasets, TASK_DATASETS)
    seeds = parse_csv_list(args.seeds, TASK_SEEDS, int)
    requested = parse_csv_list(args.variants, VARIANT_ORDER)
    variants = ['full_original'] + [name for name in requested if name != 'full_original']
    data_root = args.data_root.resolve()
    pathway_dir = args.pathway_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    require_inputs(data_root, pathway_dir, datasets)
    install_runtime_paths(data_root, pathway_dir)
    config = {'created_at': datetime.now().isoformat(timespec='seconds'), 'script': str(Path(__file__).resolve()), 'script_sha256': script_sha256(), 'standalone': True, 'imports_local_python_modules': False, 'task': TASK_NAME, 'task_label': TASK_LABEL, 'data_root': str(data_root), 'pathway_dir': str(pathway_dir), 'output_dir': str(output_dir), 'datasets': datasets, 'seeds': seeds, 'folds': N_FOLDS, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'epochs': EPOCHS, 'patience': PATIENCE, 'variants_in_execution_order': variants, 'variant_specs': {name: asdict(SPECS[name]) for name in variants}, 'full_model_first': True, 'paper_reproduction_gate': not args.allow_paper_mismatch, 'legacy_checkpoint_behavior': 'Shallow state_dict behavior preserved from supplied code.', 'gate_definition': {'alpha': 'global sigmoid-transformed pathway scalar', 'beta': 'sample-by-driver-query sigmoid output; not a scalar', 'saturation': 'value < 0.05 or value > 0.95'}, 'path_prior_definition': 'Projected directional association from miRNA-associated pathways to target-gene pathways; not causal signaling.', 'software': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'device': str(device), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}}
    (output_dir / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(pd.DataFrame([{**asdict(spec), 'execution_order': VARIANT_ORDER.index(name) + 1} for (name, spec) in SPECS.items() if name in variants]), output_dir / 'experiment_catalog.csv')
    cohorts = {}
    random_rows = []
    raw_prior_rows = []
    for dataset in datasets:
        cohort = prepare_cohort(dataset)
        cohorts[dataset] = cohort
        random_rows.extend(cohort['random_rows'])
        raw_prior_rows.append(cohort['raw_prior'])
    write_csv(pd.DataFrame(random_rows), output_dir / 'random_non_driver_query_sets.csv')
    write_csv(pd.DataFrame(raw_prior_rows), output_dir / 'raw_pathway_prior_audit.csv')
    fold_path = output_dir / 'fold_results.csv'
    seed_path = output_dir / 'seed_results.csv'
    prediction_path = output_dir / 'oof_predictions.csv'
    diagnostic_path = output_dir / 'diagnostics.csv'

    def run_variant(variant_name: str):
        spec = SPECS[variant_name]
        for dataset in datasets:
            cohort = cohorts[dataset]
            for seed in seeds:
                existing = load_csv(seed_path)
                complete = not existing.empty and ((existing['dataset'] == dataset) & (existing['variant'] == variant_name) & (existing['seed'] == seed)).any()
                if complete:
                    print(f'SKIP {dataset} {variant_name} seed={seed}', flush=True)
                    continue
                print(f'RUN {dataset} {variant_name} seed={seed}', flush=True)
                (folds, seed_row, predictions, diagnostics) = run_seed(cohort, spec, seed)
                append_rows(fold_path, folds, ('task', 'dataset', 'variant', 'seed', 'fold'))
                append_rows(prediction_path, predictions, ('task', 'dataset', 'variant', 'seed', 'fold', 'sample_id'))
                append_rows(diagnostic_path, diagnostics, ('task', 'dataset', 'variant', 'seed', 'fold'))
                append_rows(seed_path, [seed_row], ('task', 'dataset', 'variant', 'seed'))
                refresh_outputs(output_dir)
    run_variant('full_original')
    (_, checks) = refresh_outputs(output_dir)
    if len(variants) > 1 and (not args.allow_paper_mismatch):
        if set(seeds) != set(TASK_SEEDS):
            raise RuntimeError('All paper seeds are required before ablations. Use --allow-paper-mismatch only for deliberate test runs.')
        if not baseline_passed(checks, datasets):
            raise RuntimeError('The full original model did not reproduce every selected paper result at four decimals; refusing to run ablations.')
    for variant_name in variants[1:]:
        run_variant(variant_name)
    (summary, checks) = refresh_outputs(output_dir)
    print('\nFINAL SUMMARY', flush=True)
    print(summary.to_string(index=False), flush=True)
    print('\nPAPER REPRODUCTION CHECK', flush=True)
    print(checks.to_string(index=False), flush=True)
    print(f'\nOutputs: {output_dir}', flush=True)
if __name__ == '__main__':
    main()
