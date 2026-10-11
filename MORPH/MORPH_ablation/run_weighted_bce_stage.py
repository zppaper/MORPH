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
import copy
from pathlib import Path
warnings.filterwarnings('ignore')
DATASETS = ['KIRC', 'MESO', 'STAD', 'THCA']
SEEDS = [1028, 2024, 3407]
N_FOLDS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
EPOCHS = 100
PATIENCE = 15
LOSS_CONDITIONS = (('Unweighted BCE', False), ('Class-weighted BCE', True))
SCRIPT_DIR = Path(__file__).resolve().parent
FOLD_RESULTS_FILE = output_dir('run_weighted_bce_stage') / 'fold_metrics.csv'
SUMMARY_RESULTS_FILE = output_dir('run_weighted_bce_stage') / 'summary.csv'
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

def capture_rng_state():
    state = {'python': random.getstate(), 'numpy': copy.deepcopy(np.random.get_state()), 'torch_cpu': torch.get_rng_state().clone()}
    if torch.cuda.is_available():
        state['torch_cuda'] = [s.clone() for s in torch.cuda.get_rng_state_all()]
    return state

def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(copy.deepcopy(state['numpy']))
    torch.set_rng_state(state['torch_cpu'].clone())
    if torch.cuda.is_available() and 'torch_cuda' in state:
        torch.cuda.set_rng_state_all([s.clone() for s in state['torch_cuda']])

def make_pos_weight(y_train):
    y_train = np.asarray(y_train).reshape(-1)
    if not np.all(np.isin(y_train, [0, 1])):
        raise ValueError('Class-weighted BCE requires binary labels encoded as 0 and 1.')
    class_counts = np.bincount(y_train.astype(np.int64), minlength=2)
    (n_negative, n_positive) = (int(class_counts[0]), int(class_counts[1]))
    if n_negative == 0 or n_positive == 0:
        raise ValueError(f'Cannot compute pos_weight because the inner-training subset contains n_negative={n_negative}, n_positive={n_positive}.')
    ratio = n_negative / n_positive
    pos_weight = torch.tensor([ratio], dtype=torch.float32, device=device)
    return (pos_weight, n_negative, n_positive, ratio)

def collect_probs_and_labels(model, data_loader):
    (all_probs, all_true) = ([], [])
    model.eval()
    with torch.no_grad():
        for (bx_m, bx_mi, by) in data_loader:
            (bx_m, bx_mi) = (bx_m.to(device), bx_mi.to(device))
            logits = model(bx_m, bx_mi)
            probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
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
    except Exception:
        (auc, aupr) = (np.nan, np.nan)
    return [auc, aupr]

def nanstd_ddof1(values, axis=0):
    values = np.asarray(values, dtype=np.float64)
    mean = np.nanmean(values, axis=axis, keepdims=True)
    sq_diff = (values - mean) ** 2
    valid_count = np.sum(~np.isnan(values), axis=axis)
    denom = np.where(valid_count > 1, valid_count - 1, np.nan)
    return np.sqrt(np.nansum(sq_diff, axis=axis) / denom)

def train_one_condition(condition_name, use_class_weight, dataset_name, seed, fold_idx, train_dataset, val_dataset, test_dataset, ty, dim_mrna, dim_mirna, gene_path_mask_df, mirna_path_mask_df, reg_matrix, driver_idx, path_prior):
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = PathformerBio(dim_mrna=dim_mrna, dim_mirna=dim_mirna, gene_path_mask_df=gene_path_mask_df, mirna_path_mask_df=mirna_path_mask_df, reg_matrix=reg_matrix, driver_idx=driver_idx, path_prior=path_prior, hidden_dim=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    (pos_weight, n_negative, n_positive, class_ratio) = make_pos_weight(ty)
    if use_class_weight:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        applied_weight = class_ratio
    else:
        criterion = nn.BCEWithLogitsLoss()
        applied_weight = 1.0
    print(f'  [{condition_name}] seed={seed}, fold={fold_idx + 1}/{N_FOLDS}: train class 0={n_negative}, class 1={n_positive}, N0/N1={class_ratio:.4f}, applied pos_weight={applied_weight:.4f}')
    best_val_score = float('-inf')
    patience_counter = 0
    best_val_epoch = 0
    for epoch in range(EPOCHS):
        model.train()
        for (bx_m, bx_mi, by) in train_loader:
            (bx_m, bx_mi, by) = (bx_m.to(device), bx_mi.to(device), by.to(device))
            logits_fusion = model(bx_m, bx_mi)
            loss = criterion(logits_fusion.squeeze(-1), by)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        (val_probs, val_true) = ([], [])
        with torch.no_grad():
            for (bx_m, bx_mi, by) in val_loader:
                (bx_m, bx_mi) = (bx_m.to(device), bx_mi.to(device))
                logits = model(bx_m, bx_mi)
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
                if np.ndim(probs) == 0:
                    probs = np.expand_dims(probs, 0)
                val_probs.extend(probs.tolist())
                val_true.extend(by.numpy().tolist())
        try:
            val_auc = roc_auc_score(val_true, val_probs)
        except Exception:
            val_auc = 0.5
        if val_auc > best_val_score:
            best_val_score = val_auc
            best_val_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break
    stop_epoch = epoch + 1
    print(f'  [{condition_name}] best val AUC={best_val_score:.4f} at epoch {best_val_epoch}; stopped at epoch {stop_epoch}; tested termination model (paper-baseline protocol)')
    collect_probs_and_labels(model, val_loader)
    (test_true, test_probs) = collect_probs_and_labels(model, test_loader)
    (test_auc, test_aupr) = compute_metrics(test_true, test_probs)
    return {'dataset': dataset_name, 'seed': seed, 'fold': fold_idx + 1, 'condition': condition_name, 'class_weighted': use_class_weight, 'train_class_0': n_negative, 'train_class_1': n_positive, 'class_ratio_n0_over_n1': class_ratio, 'applied_pos_weight': applied_weight, 'best_val_auc': best_val_score, 'best_epoch': best_val_epoch, 'stop_epoch': stop_epoch, 'test_auc': test_auc, 'test_aupr': test_aupr}

def run_experiment(dataset_name, seed):
    set_seeds(seed)
    res = load_data(dataset_name)
    if res[0] is None:
        return None
    (df_mrna, df_mirna, gene_path_mask_df, mirna_path_mask_df, reg_matrix, driver_idx, path_prior, y) = res
    X_mrna = df_mrna.values.astype(np.float32)
    X_mirna = df_mirna.values.astype(np.float32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    paired_records = []
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
        fold_start_rng_state = capture_rng_state()
        unweighted_end_rng_state = None
        for (condition_name, use_class_weight) in LOSS_CONDITIONS:
            restore_rng_state(fold_start_rng_state)
            record = train_one_condition(condition_name=condition_name, use_class_weight=use_class_weight, dataset_name=dataset_name, seed=seed, fold_idx=fold_idx, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, ty=ty, dim_mrna=X_mrna.shape[1], dim_mirna=X_mirna.shape[1], gene_path_mask_df=gene_path_mask_df, mirna_path_mask_df=mirna_path_mask_df, reg_matrix=reg_matrix, driver_idx=driver_idx, path_prior=path_prior)
            paired_records.append(record)
            if not use_class_weight:
                unweighted_end_rng_state = capture_rng_state()
        if unweighted_end_rng_state is None:
            raise RuntimeError('Unweighted reference condition did not run.')
        restore_rng_state(unweighted_end_rng_state)
    return paired_records

def summarize_and_save(records):
    if not records:
        print('No valid results collected.')
        return
    fold_df = pd.DataFrame(records)
    fold_df.to_csv(FOLD_RESULTS_FILE, index=False, encoding='utf-8-sig')
    seed_df = fold_df.groupby(['dataset', 'condition', 'seed'], as_index=False)[['test_auc', 'test_aupr']].mean()
    summary_rows = []
    for dataset_name in DATASETS:
        dataset_seed_df = seed_df[seed_df['dataset'] == dataset_name]
        if dataset_seed_df.empty:
            continue
        for (condition_name, _) in LOSS_CONDITIONS:
            condition_df = dataset_seed_df[dataset_seed_df['condition'] == condition_name]
            auc_values = condition_df['test_auc'].to_numpy(dtype=np.float64)
            aupr_values = condition_df['test_aupr'].to_numpy(dtype=np.float64)
            if len(auc_values) == 0:
                continue
            summary_rows.append({'dataset': dataset_name, 'comparison': condition_name, 'auc_mean': np.nanmean(auc_values), 'auc_std_across_seeds': nanstd_ddof1(auc_values), 'aupr_mean': np.nanmean(aupr_values), 'aupr_std_across_seeds': nanstd_ddof1(aupr_values)})
        unweighted = dataset_seed_df[dataset_seed_df['condition'] == 'Unweighted BCE'][['seed', 'test_auc', 'test_aupr']].rename(columns={'test_auc': 'auc_unweighted', 'test_aupr': 'aupr_unweighted'})
        weighted = dataset_seed_df[dataset_seed_df['condition'] == 'Class-weighted BCE'][['seed', 'test_auc', 'test_aupr']].rename(columns={'test_auc': 'auc_weighted', 'test_aupr': 'aupr_weighted'})
        paired = unweighted.merge(weighted, on='seed', how='inner')
        if not paired.empty:
            auc_delta = (paired['auc_weighted'] - paired['auc_unweighted']).to_numpy(dtype=np.float64)
            aupr_delta = (paired['aupr_weighted'] - paired['aupr_unweighted']).to_numpy(dtype=np.float64)
            summary_rows.append({'dataset': dataset_name, 'comparison': 'Weighted - Unweighted (paired seed delta)', 'auc_mean': np.nanmean(auc_delta), 'auc_std_across_seeds': nanstd_ddof1(auc_delta), 'aupr_mean': np.nanmean(aupr_delta), 'aupr_std_across_seeds': nanstd_ddof1(aupr_delta)})
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_RESULTS_FILE, index=False, encoding='utf-8-sig')
    print('\n\n' + '=' * 96)
    print('PAIRED CLASS-IMBALANCE COMPARISON (mean +/- SD across 3 seed-level fold means)')
    print('=' * 96)
    for dataset_name in DATASETS:
        ds_rows = summary_df[summary_df['dataset'] == dataset_name]
        if ds_rows.empty:
            continue
        print(f'\nDataset: {dataset_name}')
        for row in ds_rows.itertuples(index=False):
            print(f'  {row.comparison:<45} AUC={row.auc_mean:.4f} +/- {row.auc_std_across_seeds:.4f} | AUPR={row.aupr_mean:.4f} +/- {row.aupr_std_across_seeds:.4f}')
    print(f'\nFold-level results saved to: {FOLD_RESULTS_FILE}')
    print(f'Summary results saved to:    {SUMMARY_RESULTS_FILE}')
if __name__ == '__main__':
    print('\n' + '=' * 96)
    print('PAIRED EXPERIMENT: UNWEIGHTED BCE VS FOLD-SPECIFIC CLASS-WEIGHTED BCE')
    print(f'Device: {device}')
    print('Only the loss weighting differs within each paired seed/fold run.')
    print('=' * 96)
    all_records = []
    for ds in DATASETS:
        print(f"\n\n{'=' * 34} Dataset: {ds} {'=' * 34}")
        for seed in SEEDS:
            try:
                seed_records = run_experiment(ds, seed)
                if seed_records is not None:
                    all_records.extend(seed_records)
            except Exception as e:
                print(f'[ERROR] Processing {ds} with seed {seed} failed: {e}')
    summarize_and_save(all_records)
