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
SCRIPT_DIR = Path(__file__).resolve().parent
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

    def __init__(self, dim_mrna, dim_mirna, gene_path_mask_df, mirna_path_mask_df, reg_matrix, driver_idx, path_prior, hidden_dim=128, depth=2, ff_dropout=0.2, classifier_dropout=0.3, use_path=True, use_micro=True):
        super().__init__()
        self.use_path = use_path
        self.use_micro = use_micro
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
        if use_path:
            self.pathway_block = PathwayCrossAttnBlock(n_genes=dim_mrna, n_mirnas=dim_mirna, n_pathways=n_pathways, gene_mask=gene_path_tensor, mirna_mask=mirna_path_tensor, path_embed_dim=16, out_dim=hidden_dim, path_prior=path_prior)
        if use_micro:
            self.micro_reg_block = MicroRegCrossAttnBlock(n_genes=len(driver_idx), n_mirnas=dim_mirna, reg_matrix=reg_tensor, embed_dim=16, out_dim=hidden_dim)
        fusion_dim = hidden_dim * (2 + int(use_path) + int(use_micro))
        self.classifier = nn.Sequential(nn.Linear(fusion_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(classifier_dropout), nn.Linear(hidden_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(classifier_dropout), nn.Linear(64, 1))

    def forward(self, x_mrna, x_mirna):
        h_mrna_glob = self.mrna_embedding(x_mrna)
        for blk in self.mrna_encoder:
            h_mrna_glob = blk(h_mrna_glob)
        h_mirna = self.mirna_embedding(x_mirna)
        for blk in self.mirna_encoder:
            h_mirna = blk(h_mirna)
        features = [h_mrna_glob, h_mirna]
        if self.use_path:
            features.append(self.pathway_block(x_mrna, x_mirna))
        if self.use_micro:
            x_mrna_driver = x_mrna.index_select(1, self.driver_idx)
            features.append(self.micro_reg_block(x_mrna_driver, x_mirna))
        h_concat = torch.cat(features, dim=1)
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
    res = checked_load_data(dataset_name)
    if res[0] is None:
        return None
    (df_mrna, df_mirna, gene_path_mask_df, mirna_path_mask_df, reg_matrix, driver_idx, path_prior, y) = res
    X_mrna = df_mrna.values.astype(np.float32)
    X_mirna = df_mirna.values.astype(np.float32)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_metrics = []
    for (fold_idx, (train_idx, test_idx)) in enumerate(skf.split(X_mrna, y)):
        fold_started = time.perf_counter()
        epoch_records = []
        (X_m_train, X_m_test) = (X_mrna[train_idx], X_mrna[test_idx])
        (X_mi_train, X_mi_test) = (X_mirna[train_idx], X_mirna[test_idx])
        (y_train, y_test) = (y[train_idx], y[test_idx])
        (sub_train_idx, sub_val_idx) = train_test_split(np.arange(len(y_train)), test_size=0.2, stratify=y_train, random_state=seed)
        record_split(df_mrna.index, y, train_idx, test_idx, sub_train_idx, sub_val_idx, fold_idx + 1)
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
        model = PathformerBio(dim_mrna=X_mrna.shape[1], dim_mirna=X_mirna.shape[1], gene_path_mask_df=gene_path_mask_df, mirna_path_mask_df=mirna_path_mask_df, reg_matrix=reg_matrix, driver_idx=driver_idx, path_prior=path_prior, hidden_dim=128, use_path=VARIANTS[RUN_CONTEXT['variant']][0], use_micro=VARIANTS[RUN_CONTEXT['variant']][1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
        criterion = nn.BCEWithLogitsLoss()
        best_val_score = float('-inf')
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
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
            epoch_records.append({'epoch': epoch + 1, 'validation_auc': float(val_auc)})
            if val_auc > best_val_score:
                best_epoch = epoch + 1
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
        record_fold(model, df_mrna.index[test_idx], test_true, test_probs, metrics_list, fold_idx + 1, epoch + 1, best_epoch, best_val_score, epoch_records, time.perf_counter() - fold_started)
    fold_metrics = np.array(fold_metrics, dtype=np.float64)
    seed_mean = np.nanmean(fold_metrics, axis=0)
    return seed_mean
import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from datetime import datetime, timezone, timedelta
VARIANTS = {'wo_microreg': (True, False), 'full': (True, True), 'wo_pathattn': (False, True), 'global_only': (False, False)}
RUN_CONTEXT = {}
PAPER_FULL = {'KIRC': (0.7646, 0.0226, 0.7003, 0.0094), 'MESO': (0.6467, 0.0376, 0.8531, 0.0152), 'STAD': (0.6004, 0.0066, 0.6441, 0.0042), 'THCA': (0.7087, 0.0164, 0.5941, 0.035)}

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda : stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def atomic_csv(frame, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False, encoding='utf-8-sig')
    temporary.replace(path)

def checked_load_data(dataset):
    for path in [GMT_GENE_FILE, GMT_MIRNA_FILE, INTERACTION_FILE, DRIVER_FILE]:
        if not path.is_file():
            raise FileNotFoundError('Required biological prior missing: ' + str(path))
    res = load_data(dataset)
    if res[0] is None:
        raise FileNotFoundError(BASE_DATA_DIR / dataset)
    (mrna, mirna, gm, mm, reg, drivers, prior, labels) = res
    assert mrna.index.is_unique and mirna.index.is_unique
    assert mrna.columns.is_unique and mirna.columns.is_unique
    assert mrna.index.equals(mirna.index)
    assert len(labels) == len(mrna) and set(np.unique(labels)) == {0, 1}
    assert np.isfinite(mrna.values).all() and np.isfinite(mirna.values).all()
    assert gm.shape[1] == mm.shape[1] == 237, 'Unexpected pathway space; no dummy priors allowed'
    assert len(drivers) > 0 and reg.sum() > 0 and (reg.sum(axis=0) > 0).all()
    assert prior.shape == (237, 237) and np.isfinite(prior).all()
    return res

def record_split(sample_ids, labels, train_idx, test_idx, sub_train_idx, sub_val_idx, fold):
    rows = []
    partitions = {'train': train_idx[sub_train_idx], 'validation': train_idx[sub_val_idx], 'test': test_idx}
    all_indices = np.concatenate(list(partitions.values()))
    assert len(all_indices) == len(sample_ids) == len(np.unique(all_indices))
    for (role, indices) in partitions.items():
        rows.extend(({'sample_id': str(sample_ids[i]), 'label': int(labels[i]), 'split': role} for i in indices))
    atomic_csv(pd.DataFrame(rows), RUN_CONTEXT['directory'] / ('fold_%d_split.csv' % fold))

def record_fold(model, sample_ids, labels, probs, metrics, fold, stopped, best_epoch, best_auc, epoch_records, seconds):
    assert np.isfinite(probs).all() and np.isfinite(metrics).all()
    directory = RUN_CONTEXT['directory']
    atomic_csv(pd.DataFrame({'sample_id': sample_ids, 'label': labels, 'probability': probs}), directory / ('fold_%d_predictions.csv' % fold))
    atomic_csv(pd.DataFrame(epoch_records), directory / ('fold_%d_training.csv' % fold))
    row = {'variant': RUN_CONTEXT['variant'], 'dataset': RUN_CONTEXT['dataset'], 'seed': RUN_CONTEXT['seed'], 'fold': fold, 'auc': metrics[0], 'aupr': metrics[1], 'test_samples': len(labels), 'epochs_trained': stopped, 'best_validation_epoch': best_epoch, 'best_validation_auc': best_auc, 'evaluated_epoch': stopped, 'trainable_parameters': sum((p.numel() for p in model.parameters() if p.requires_grad)), 'seconds': seconds}
    RUN_CONTEXT['rows'].append(row)
    atomic_csv(pd.DataFrame(RUN_CONTEXT['rows']), directory / 'fold_metrics.csv')
    print('{variant} {dataset} seed={seed} fold={fold}/5 AUC={auc:.6f} AUPR={aupr:.6f} epochs={epochs_trained} time={seconds:.1f}s'.format(**row), flush=True)

def valid_seed(directory):
    done = directory / 'complete.json'
    if not done.exists():
        return False
    marker = json.loads(done.read_text(encoding='utf-8'))
    files = marker['sha256']
    if not all(((directory / name).exists() and sha256(directory / name) == digest for (name, digest) in files.items())):
        raise RuntimeError('Completed seed artifacts changed: ' + str(directory))
    frame = pd.read_csv(directory / 'fold_metrics.csv')
    assert len(frame) == N_FOLDS and set(frame.fold) == set(range(1, N_FOLDS + 1))
    ids = []
    for row in frame.itertuples():
        prediction = pd.read_csv(directory / ('fold_%d_predictions.csv' % row.fold))
        assert np.allclose(compute_metrics(prediction.label, prediction.probability), [row.auc, row.aupr], atol=1e-12, rtol=0)
        ids.extend(prediction.sample_id.tolist())
    assert len(ids) == len(set(ids)), 'Outer test sets must not overlap within a seed'
    return True

def summarize(output, args):
    frames = []
    for variant in args.variants:
        for dataset in args.datasets:
            for seed in args.seeds:
                directory = output / variant / dataset / str(seed)
                if valid_seed(directory):
                    frames.append(pd.read_csv(directory / 'fold_metrics.csv'))
    if not frames:
        return
    folds = pd.concat(frames, ignore_index=True)
    atomic_csv(folds, output / 'stage_fold_metrics.csv')
    seeds = folds.groupby(['variant', 'dataset', 'seed'], sort=False)[['auc', 'aupr']].mean().reset_index()
    atomic_csv(seeds, output / 'stage_seed_means.csv')
    rows = []
    for ((variant, dataset), group) in seeds.groupby(['variant', 'dataset'], sort=False):
        rows.append({'variant': variant, 'dataset': dataset, 'n_seeds': len(group), 'auc_mean': group.auc.mean(), 'auc_std': group.auc.std(ddof=1), 'aupr_mean': group.aupr.mean(), 'aupr_std': group.aupr.std(ddof=1), 'complete': len(group) == len(args.seeds)})
    summary = pd.DataFrame(rows)
    atomic_csv(summary, output / 'stage_summary.csv')
    lines = ['# M3 CancerStagePrediction ablation', '', 'Fresh runs; each seed is averaged over five outer test folds. Standard deviation is computed over seed means (ddof=1).', '', '| Variant | Dataset | AUC | AUPR | Seeds complete |', '|---|---|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append('| %s | %s | %.4f ± %.4f | %.4f ± %.4f | %d/%d |' % (r.variant, r.dataset, r.auc_mean, r.auc_std, r.aupr_mean, r.aupr_std, r.n_seeds, len(args.seeds)))
    lines += ['', '## Protocol', '', '- wo_microreg removes the entire MicroReg module and its fusion features (512 -> 384 input units).', '- Both global encoders and PathAttn, including driver-derived regulatory topology, remain active.', '- Same seeds, outer folds, inner 20% validation splits, training-only StandardScaler, optimizer, loss and hyperparameters as the source.', '- Original shallow state_dict checkpoint behavior is retained: validation AUC controls stopping, but the evaluated weights are from the stopping epoch, not a cloned best-validation checkpoint.', '- Input feature selection is inherited from the supplied preprocessed CSV files; this run does not re-audit upstream feature selection.', '- AUPR uses sklearn average_precision_score, matching the source.', '- No cohort-specific tuning or selection of favorable runs.', '- Survival/Vital-StatusClassification is outside this run.', '']
    (output / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    comparison = []
    for r in summary.itertuples():
        if r.variant == 'full' and r.complete and (args.seeds == [1028, 2024, 3407]):
            values = [r.auc_mean, r.auc_std, r.aupr_mean, r.aupr_std]
            for (key, observed, expected) in zip(['auc_mean', 'auc_std', 'aupr_mean', 'aupr_std'], values, PAPER_FULL[r.dataset]):
                comparison.append({'dataset': r.dataset, 'metric': key, 'rerun': observed, 'paper': expected, 'match_4dp': '%.4f' % observed == '%.4f' % expected})
    if comparison:
        atomic_csv(pd.DataFrame(comparison), output / 'full_paper_comparison.csv')

def model_arguments(res):
    (a, b, gm, mm, reg, drivers, prior, _) = res
    return dict(dim_mrna=a.shape[1], dim_mirna=b.shape[1], gene_path_mask_df=gm, mirna_path_mask_df=mm, reg_matrix=reg, driver_idx=drivers, path_prior=prior, hidden_dim=128)

def verify_implementation():
    source_path = SCRIPT_DIR / '_reference_stage.py'
    spec = importlib.util.spec_from_file_location('m3_original_verification', source_path)
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    res = checked_load_data('KIRC')
    kwargs = model_arguments(res)
    set_seeds(1028)
    reference = original.PathformerBio(**kwargs).to(device).eval()
    set_seeds(1028)
    full = PathformerBio(**kwargs).to(device).eval()
    assert list(reference.state_dict()) == list(full.state_dict())
    assert all((torch.equal(v, full.state_dict()[k]) for (k, v) in reference.state_dict().items()))
    x = torch.tensor(StandardScaler().fit_transform(res[0].values[:8]).astype(np.float32), device=device)
    m = torch.tensor(StandardScaler().fit_transform(res[1].values[:8]).astype(np.float32), device=device)
    with torch.no_grad():
        assert torch.equal(reference(x, m), full(x, m)), 'Full-model inference parity failed'
    reference.train()
    full.train()
    set_seeds(72)
    reference_prediction = reference(x, m)
    set_seeds(72)
    full_prediction = full(x, m)
    assert torch.equal(reference_prediction, full_prediction), 'Train-mode/dropout parity failed'
    reference_prediction.sum().backward()
    full_prediction.sum().backward()
    for ((name, rp), (_, fp)) in zip(reference.named_parameters(), full.named_parameters()):
        assert torch.equal(rp.grad, fp.grad), 'Gradient parity failed: ' + name
    checks = []
    for (variant, (path, micro)) in VARIANTS.items():
        set_seeds(1028)
        model = PathformerBio(**kwargs, use_path=path, use_micro=micro).to(device)
        assert hasattr(model, 'pathway_block') == path
        assert hasattr(model, 'micro_reg_block') == micro
        assert model.classifier[0].in_features == 128 * (2 + int(path) + int(micro))
        if path:
            assert torch.equal(model.pathway_block.prior_norm_mask, full.pathway_block.prior_norm_mask)
        logits = model(x, m)
        loss = nn.BCEWithLogitsLoss()(logits.squeeze(-1), torch.tensor(res[-1][:8], dtype=torch.float32, device=device))
        loss.backward()
        assert torch.isfinite(logits).all()
        assert all((p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        if path:
            assert model.pathway_block.g_proj.weight.grad.abs().sum() > 0
        checks.append({'variant': variant, 'fusion_input': model.classifier[0].in_features, 'parameters': sum((p.numel() for p in model.parameters())), 'finite_backward': True})
    result = {'full_original_state_and_forward_and_gradient_equal': True, 'checks': checks, 'original_sha256': sha256(source_path), 'script_sha256': sha256(__file__)}
    print(json.dumps(result, indent=2), flush=True)
    return result

def main():
    global BASE_DATA_DIR, PATHWAY_DIR, GMT_GENE_FILE, GMT_MIRNA_FILE, INTERACTION_FILE, DRIVER_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=['wo_microreg', 'full'])
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--data-dir', type=Path, default=data_dir('stage'))
    parser.add_argument('--pathway-dir', type=Path, default=prior_dir())
    parser.add_argument('--output-dir', type=Path, default=output_dir('run_modules_stage'))
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    for values in [args.variants, args.datasets, args.seeds]:
        if len(values) != len(set(values)):
            parser.error('Repeated variants/datasets/seeds are not allowed')
    if len(args.seeds) < 2 and (not args.verify_only):
        parser.error('Use at least two seeds; the manuscript protocol uses three')
    (BASE_DATA_DIR, PATHWAY_DIR) = (args.data_dir.resolve(), args.pathway_dir.resolve())
    GMT_GENE_FILE = PATHWAY_DIR / '20230205_kegg_hsa.gmt'
    GMT_MIRNA_FILE = PATHWAY_DIR / 'kegg_anano.txt'
    INTERACTION_FILE = PATHWAY_DIR / 'miRTarBase_MTI_processed_cleaned.csv'
    DRIVER_FILE = driver_file()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.verify_only:
        result = verify_implementation()
        (output / 'implementation_checks.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return
    if args.summarize_only:
        summarize(output, args)
        return
    files = [Path(__file__).resolve(), GMT_GENE_FILE, GMT_MIRNA_FILE, INTERACTION_FILE, DRIVER_FILE]
    for dataset in args.datasets:
        files.extend((BASE_DATA_DIR / dataset / name for name in ['mRNA.csv', 'miRNA.csv', 'label.csv']))
    manifest = [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in files]
    config = {'variants': args.variants, 'datasets': args.datasets, 'seeds': args.seeds, 'folds': N_FOLDS, 'epochs': EPOCHS, 'patience': PATIENCE, 'batch_size': BATCH_SIZE, 'learning_rate': LEARNING_RATE, 'weight_decay': 0.0001, 'checkpoint': 'original_shallow_state_dict_stop_epoch', 'python_executable': sys.executable, 'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__, 'pandas': pd.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'platform': platform.platform(), 'inputs': manifest}
    config_path = output / 'run_config.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != config:
            raise RuntimeError('Run configuration or input files changed. Choose a new output directory.')
    else:
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    atomic_csv(pd.DataFrame(manifest), output / 'input_manifest.csv')
    print('START', datetime.now(timezone(timedelta(hours=8))).isoformat(), sys.executable, config['gpu'], flush=True)
    for dataset in args.datasets:
        for variant in args.variants:
            for seed in args.seeds:
                directory = output / variant / dataset / str(seed)
                directory.mkdir(parents=True, exist_ok=True)
                if valid_seed(directory):
                    print('RESUME completed:', variant, dataset, seed, flush=True)
                    continue
                RUN_CONTEXT.clear()
                RUN_CONTEXT.update(variant=variant, dataset=dataset, seed=seed, directory=directory, rows=[])
                run_experiment(dataset, seed)
                expected = ['fold_metrics.csv'] + ['fold_%d_%s.csv' % (fold, kind) for fold in range(1, 6) for kind in ['split', 'predictions', 'training']]
                marker = {'sha256': {name: sha256(directory / name) for name in expected}}
                (directory / 'complete.json').write_text(json.dumps(marker, indent=2), encoding='utf-8')
                assert valid_seed(directory)
                summarize(output, args)
    summarize(output, args)
    print('COMPLETE: ' + str(output / 'RESULTS.md'), flush=True)
if __name__ == '__main__':
    main()
