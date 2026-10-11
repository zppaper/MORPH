from _settings import data_dir, prior_dir, output_dir, external_root, driver_file
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random
import warnings
import re
from pathlib import Path
import shap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
KEGG_MAP = {'hsa01100': 'Metabolic pathways', 'hsa05200': 'Pathways in cancer', 'hsa04151': 'PI3K-Akt signaling', 'hsa04390': 'Hippo signaling', 'hsa04010': 'MAPK signaling', 'hsa04310': 'Wnt signaling', 'hsa05205': 'Proteoglycans in cancer', 'hsa04510': 'Focal adhesion', 'hsa04060': 'Cytokine receptor interaction', 'hsa04015': 'Rap1 signaling', 'hsa05202': 'Transcriptional misreg.', 'hsa04530': 'Tight junction', 'hsa05166': 'HTLV-I infection', 'hsa05206': 'MicroRNAs in cancer', 'hsa04145': 'Phagosome', 'hsa04360': 'Axon guidance', 'hsa04014': 'Ras signaling pathway', 'hsa04514': 'Cell adhesion molecules'}
SEED = 2024
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
BASE_DATA_DIR = data_dir('stage')
PATHWAY_DIR = prior_dir()
INTERACTION_FILE = PATHWAY_DIR / 'miRTarBase_MTI_processed_cleaned.csv'
DRIVER_FILE = driver_file()
GMT_GENE_FILE = PATHWAY_DIR / '20230205_kegg_hsa.gmt'
GMT_MIRNA_FILE = PATHWAY_DIR / 'kegg_anano.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seeds(SEED)

def clean_name_logic(name):
    return re.sub('-(3|5)p', '', str(name).lower()).strip()

def normalize_mirna_key(name):
    return re.sub('-(3|5)p', '', re.sub('^(hsa|mmu)-', '', str(name).lower())).strip()

def build_mirna_gene_interaction_matrix(gene_list, mirna_list):
    (n_genes, n_mirnas) = (len(gene_list), len(mirna_list))
    interaction = np.zeros((n_mirnas, n_genes), dtype=np.float32)
    if not INTERACTION_FILE.exists():
        return interaction
    try:
        df = pd.read_csv(INTERACTION_FILE, usecols=['miRNA', 'Target Gene'], low_memory=False)
        mirna_map = {normalize_mirna_key(m): i for (i, m) in enumerate(mirna_list)}
        gene_map = {str(g).strip().upper(): i for (i, g) in enumerate(gene_list)}
        mi_idx = df['miRNA'].astype(str).str.strip().str.lower().map(mirna_map)
        g_idx = df['Target Gene'].astype(str).str.strip().str.upper().map(gene_map)
        valid = mi_idx.notna() & g_idx.notna()
        interaction[mi_idx[valid].astype(np.int64).values, g_idx[valid].astype(np.int64).values] = 1.0
    except Exception:
        pass
    return interaction

def load_dual_pathway_structure(gene_list, mirna_list):
    (g_dict, m_dict) = ({}, {})
    if GMT_GENE_FILE.exists():
        with open(GMT_GENE_FILE, 'r', encoding='utf-8') as f:
            for l in f:
                p = l.strip().split('\t')
                if len(p) >= 3:
                    g_dict[p[0].split('_')[0].strip()] = [m.strip() for m in p[2:] if m.strip()]
    if GMT_MIRNA_FILE.exists():
        with open(GMT_MIRNA_FILE, 'r', encoding='utf-8') as f:
            for l in f:
                p = l.strip().split(',')
                if len(p) >= 2 and len(p[0].split('|')) >= 2:
                    m_dict[p[0].split('|')[1].strip()] = list(set([m.strip() for m in p[1:] if m.strip()]))
    common_paths = sorted(list(set(g_dict.keys()) & set(m_dict.keys())))
    if not common_paths:
        return (pd.DataFrame(np.random.randint(0, 2, (len(gene_list), 10)), index=gene_list), pd.DataFrame(np.random.randint(0, 2, (len(mirna_list), 10)), index=mirna_list))
    g_mask = np.zeros((len(gene_list), len(common_paths)))
    for (j, pid) in enumerate(common_paths):
        indices = [i for (i, g) in enumerate(gene_list) if g in g_dict.get(pid, [])]
        g_mask[indices, j] = 1
    m_mask = np.zeros((len(mirna_list), len(common_paths)))
    m_map = {clean_name_logic(m): i for (i, m) in enumerate(mirna_list)}
    for (j, pid) in enumerate(common_paths):
        for m in m_dict.get(pid, []):
            if clean_name_logic(m) in m_map:
                m_mask[m_map[clean_name_logic(m)], j] = 1
    return (pd.DataFrame(g_mask, index=gene_list, columns=common_paths), pd.DataFrame(m_mask, index=mirna_list, columns=common_paths))

def load_data(dataset_name):
    data_dir = BASE_DATA_DIR / dataset_name
    try:
        mrna_df = pd.read_csv(data_dir / 'mRNA.csv', index_col=0).T
        mirna_df = pd.read_csv(data_dir / 'miRNA.csv', index_col=0).T
        label_df = pd.read_csv(data_dir / 'label.csv', index_col=0)
    except FileNotFoundError:
        return [None] * 8
    common_samples = mrna_df.index.intersection(mirna_df.index).intersection(label_df.index)
    (mrna_df, mirna_df, label_df) = (mrna_df.loc[common_samples], mirna_df.loc[common_samples], label_df.loc[common_samples])
    (gene_list, mirna_list) = (list(mrna_df.columns), list(mirna_df.columns))
    driver_idx = list(range(len(gene_list)))
    reg_matrix = build_mirna_gene_interaction_matrix([gene_list[i] for i in driver_idx], mirna_list)
    edge_mask = reg_matrix.sum(axis=0) > 0
    if edge_mask.any():
        driver_idx = [idx for (idx, keep) in zip(driver_idx, edge_mask) if keep]
        reg_matrix = reg_matrix[:, edge_mask]
    (gene_path_mask_df, mirna_path_mask_df) = load_dual_pathway_structure(gene_list, mirna_list)
    return (mrna_df, mirna_df, gene_path_mask_df, mirna_path_mask_df, reg_matrix, driver_idx, None, label_df['label'].values)

class ResNetBlock(nn.Module):

    def __init__(self, dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, x):
        return x + self.ff(self.norm(x))

class PathwayCrossAttnBlock(nn.Module):

    def __init__(self, n_genes, n_mirnas, n_pathways, gene_mask, mirna_mask, path_embed_dim=16, out_dim=128):
        super().__init__()
        self.num_heads = 4
        self.head_dim = path_embed_dim // 4
        self.register_buffer('gene_mask', gene_mask / (gene_mask.sum(dim=0, keepdim=True) + 1e-08))
        self.register_buffer('mirna_mask', mirna_mask / (mirna_mask.sum(dim=0, keepdim=True) + 1e-08))
        (self.g_proj, self.m_proj) = (nn.Linear(1, path_embed_dim), nn.Linear(1, path_embed_dim))
        (self.to_q, self.to_k, self.to_v) = (nn.Linear(path_embed_dim, path_embed_dim, bias=False), nn.Linear(path_embed_dim, path_embed_dim, bias=False), nn.Linear(path_embed_dim, path_embed_dim, bias=False))
        self.scale = self.head_dim ** (-0.5)
        self.norm = nn.LayerNorm(path_embed_dim)
        self.tau = nn.Parameter(torch.tensor(0.35))
        self.final_proj = nn.Sequential(nn.Linear(path_embed_dim * 2, out_dim), nn.LayerNorm(out_dim), nn.ReLU())

    def forward(self, x_mrna, x_mirna):
        B = x_mrna.shape[0]
        path_g = torch.einsum('b g d, g p -> b p d', F.gelu(self.g_proj(x_mrna.unsqueeze(-1))), self.gene_mask)
        path_m = torch.einsum('b m d, m p -> b p d', F.gelu(self.m_proj(x_mirna.unsqueeze(-1))), self.mirna_mask)
        q = self.to_q(path_g).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(path_m).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(path_m).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale / torch.clamp(self.tau, min=0.01), dim=-1)
        update = torch.matmul(attn, v).transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
        path_g_refined = self.norm(path_g + update)
        return self.final_proj(torch.cat([path_g_refined.mean(dim=1), path_g_refined.max(dim=1)[0]], dim=-1))

class PathformerBio(nn.Module):

    def __init__(self, dim_mrna, dim_mirna, gene_path_mask_df, mirna_path_mask_df):
        super().__init__()
        hidden_dim = 128
        self.mrna_emb = nn.Sequential(nn.Linear(dim_mrna, hidden_dim), nn.ReLU())
        self.mirna_emb = nn.Sequential(nn.Linear(dim_mirna, hidden_dim), nn.ReLU())
        self.pathway_block = PathwayCrossAttnBlock(dim_mrna, dim_mirna, gene_path_mask_df.shape[1], torch.tensor(gene_path_mask_df.values, dtype=torch.float32), torch.tensor(mirna_path_mask_df.values, dtype=torch.float32), out_dim=hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, x_mrna, x_mirna):
        return self.classifier(torch.cat([self.mrna_emb(x_mrna), self.mirna_emb(x_mirna), self.pathway_block(x_mrna, x_mirna)], dim=1))

class SHAPWrapper(nn.Module):

    def __init__(self, model, dim_mrna):
        super().__init__()
        (self.model, self.dim_mrna) = (model, dim_mrna)

    def forward(self, x):
        return self.model(x[:, :self.dim_mrna], x[:, self.dim_mrna:])

def plot_pathway_signatures(model, X_mrna, X_mirna, gene_list, mirna_list, gene_path_mask_df, mirna_path_mask_df):
    model.eval()
    dim_mrna = X_mrna.shape[1]
    print('[SHAP] 正在计算 SHAP 值 (可能需要几分钟)...')
    bg_size = min(100, X_mrna.shape[0])
    X_comb = np.concatenate([X_mrna, X_mirna], axis=1)
    explainer = shap.GradientExplainer(SHAPWrapper(model, dim_mrna).to(device), torch.tensor(X_comb[:bg_size], dtype=torch.float32).to(device))
    shap_values = explainer.shap_values(torch.tensor(X_comb, dtype=torch.float32).to(device))
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    (shap_mrna, shap_mirna) = (shap_values[:, :dim_mrna], shap_values[:, dim_mrna:])
    (imp_mrna_raw, imp_mirna_raw) = (np.abs(shap_mrna).mean(axis=0), np.abs(shap_mirna).mean(axis=0))
    imp_mrna = imp_mrna_raw / (imp_mrna_raw.max() + 1e-08)
    imp_mirna = imp_mirna_raw / (imp_mirna_raw.max() + 1e-08)
    print('[SHAP] 提取特征，应用严格多样性过滤...')
    pathways = gene_path_mask_df.columns.tolist()
    path_data_raw = []
    for path in pathways:
        p_g_idx = np.where(gene_path_mask_df[path].values > 0)[0]
        p_m_idx = np.where(mirna_path_mask_df[path].values > 0)[0]
        score_m = np.sum(imp_mrna[p_g_idx]) if len(p_g_idx) > 0 else 0
        score_mi = np.sum(imp_mirna[p_m_idx]) if len(p_m_idx) > 0 else 0
        if score_m + score_mi > 0:
            path_data_raw.append({'name': KEGG_MAP.get(path, path), 'total': score_m + score_mi, 'score_m': score_m, 'score_mi': score_mi, 'all_m': [{'name': gene_list[i], 'score': imp_mrna[i], 'type': 'mRNA'} for i in p_g_idx], 'all_mi': [{'name': mirna_list[i], 'score': imp_mirna[i], 'type': 'miRNA'} for i in p_m_idx]})
    path_data_raw = sorted(path_data_raw, key=lambda x: x['total'], reverse=True)[:15]
    usage_counts = {}
    final_path_data = []
    global_max_score = 0
    for p in path_data_raw:

        def select_diverse(candidate_list, k):
            sorted_candidates = sorted(candidate_list, key=lambda x: x['score'], reverse=True)
            selected = []
            for feat in sorted_candidates:
                if len(selected) >= k:
                    break
                penalty = 0.05 ** usage_counts.get(feat['name'], 0)
                if feat['score'] * penalty > 0.001:
                    adj_feat = feat.copy()
                    adj_feat['score'] = feat['score'] * penalty
                    selected.append(adj_feat)
                    usage_counts[feat['name']] = usage_counts.get(feat['name'], 0) + 1
            return selected
        top_mi = select_diverse(p['all_mi'], 3)
        top_m = select_diverse(p['all_m'], 3)
        for f in top_mi + top_m:
            if f['score'] > global_max_score:
                global_max_score = f['score']
        final_path_data.append({'name': p['name'], 'score_m': p['score_m'], 'score_mi': p['score_mi'], 'top_mi': top_mi, 'top_m': top_m})
    (fig, ax) = plt.subplots(figsize=(18.0, 9.5), facecolor='white')
    ax.axis('off')
    color_m_base = '#2D728F'
    color_mi_base = '#C15C53'
    cmap_m = LinearSegmentedColormap.from_list('m', ['#E6F0F2', color_m_base])
    cmap_mi = LinearSegmentedColormap.from_list('mi', ['#F7E6E5', color_mi_base])
    scores_m = np.array([p['score_m'] for p in final_path_data])
    scores_mi = np.array([p['score_mi'] for p in final_path_data])
    scores_m = scores_m / (scores_m.max() + 1e-08)
    scores_mi = scores_mi / (scores_mi.max() + 1e-08)
    y_start = 15
    x_path_name = 3.6
    x_heatmap_m = 3.9
    x_heatmap_mi = 4.35
    col_x_miRNA = [5.3, 7.0, 8.7]
    col_x_mRNA = [10.8, 12.3, 13.8]
    for i in range(len(final_path_data)):
        if i % 2 == 0:
            ax.axhspan(y_start - i - 0.45, y_start - i + 0.45, color='#F8F9FA', zorder=0)
    ax.text((col_x_miRNA[0] + col_x_miRNA[-1]) / 2 + 0.5, y_start + 1.2, 'Top Regulators (miRNA)', ha='center', va='center', fontsize=14, fontweight='bold', color=color_mi_base)
    ax.text((col_x_mRNA[0] + col_x_mRNA[-1]) / 2 + 0.5, y_start + 1.2, 'Top Targets (mRNA)', ha='center', va='center', fontsize=14, fontweight='bold', color=color_m_base)
    ax.plot([col_x_miRNA[0] - 0.2, col_x_miRNA[-1] + 1.2], [y_start + 0.8, y_start + 0.8], color=color_mi_base, lw=2)
    ax.plot([col_x_mRNA[0] - 0.2, col_x_mRNA[-1] + 1.2], [y_start + 0.8, y_start + 0.8], color=color_m_base, lw=2)
    for (i, p) in enumerate(final_path_data):
        y = y_start - i
        ax.text(x_path_name, y, p['name'], ha='right', va='center', fontsize=13, fontweight='bold', color='#333')
        ax.add_patch(patches.Rectangle((x_heatmap_m, y - 0.5), 0.35, 1.0, color=cmap_m(scores_m[i])))
        ax.add_patch(patches.Rectangle((x_heatmap_mi, y - 0.5), 0.35, 1.0, color=cmap_mi(scores_mi[i])))
        for (j, f) in enumerate(p['top_mi']):
            if j < len(col_x_miRNA):
                pt_size = 50 + f['score'] / global_max_score * 600
                ax.scatter(col_x_miRNA[j], y, s=pt_size, color=color_mi_base, alpha=0.85, zorder=3)
                ax.text(col_x_miRNA[j] + 0.18, y, f['name'], color=color_mi_base, va='center', ha='left', fontsize=12, fontweight='bold')
        for (j, f) in enumerate(p['top_m']):
            if j < len(col_x_mRNA):
                pt_size = 50 + f['score'] / global_max_score * 600
                ax.scatter(col_x_mRNA[j], y, s=pt_size, color=color_m_base, alpha=0.85, zorder=3)
                ax.text(col_x_mRNA[j] + 0.18, y, f['name'], color=color_m_base, va='center', ha='left', fontsize=12, fontweight='bold')
    legend_x = 16.0
    ax.text(8.5, y_start + 2.8, 'Pathway Signatures & Key Multi-omic Drivers on STAD', ha='center', va='center', fontsize=22, fontweight='bold')
    ax.text(legend_x, y_start + 0.5, 'Pathway Activity', fontsize=13, fontweight='bold')
    ax.text(legend_x, y_start - 0.5, 'mRNA Module', fontsize=12)
    ax.imshow(np.linspace(0, 1, 100).reshape(1, -1), cmap=cmap_m, extent=(legend_x, legend_x + 1.8, y_start - 1.2, y_start - 0.9), aspect='auto')
    ax.text(legend_x, y_start - 2.0, 'miRNA Module', fontsize=12)
    ax.imshow(np.linspace(0, 1, 100).reshape(1, -1), cmap=cmap_mi, extent=(legend_x, legend_x + 1.8, y_start - 2.7, y_start - 2.4), aspect='auto')
    ax.text(legend_x, y_start - 3.2, '0', fontsize=11)
    ax.text(legend_x + 1.8, y_start - 3.2, '1', fontsize=11, ha='right')
    ax.text(legend_x, y_start - 4.8, 'Driver Impact (SHAP)', fontsize=13, fontweight='bold')
    ax.scatter([legend_x + 0.3], [y_start - 6.0], s=50, color='gray', alpha=0.7)
    ax.scatter([legend_x + 0.8], [y_start - 6.0], s=300, color='gray', alpha=0.7)
    ax.scatter([legend_x + 1.5], [y_start - 6.0], s=650, color='gray', alpha=0.7)
    ax.text(legend_x + 0.3, y_start - 6.8, 'Low', ha='center', fontsize=11)
    ax.text(legend_x + 1.5, y_start - 6.8, 'High', ha='center', fontsize=11)
    ax.text(legend_x, y_start - 8.5, 'Modality', fontsize=13, fontweight='bold')
    ax.scatter([legend_x + 0.2], [y_start - 9.5], s=200, color=color_m_base)
    ax.text(legend_x + 0.5, y_start - 9.5, 'mRNA', va='center', fontsize=12)
    ax.scatter([legend_x + 0.2], [y_start - 10.5], s=200, color=color_mi_base)
    ax.text(legend_x + 0.5, y_start - 10.5, 'miRNA', va='center', fontsize=12)
    ax.set_xlim(1.0, legend_x + 2.5)
    ax.set_ylim(-0.5, y_start + 3.5)
    plt.subplots_adjust(left=0.01, right=0.98, top=0.95, bottom=0.02)
    plt.savefig(output_dir('legacy_pathway_signatures') / Path('STAD_pathway_signatures.png').name, dpi=400, bbox_inches='tight')
    plt.savefig(output_dir('legacy_pathway_signatures') / Path('STAD_pathway_signatures.pdf').name, format='pdf', bbox_inches='tight')
    print('[SHAP] Image generated successfully: STAD_pathway_signatures.png & STAD_pathway_signatures.pdf')
    plt.show()
if __name__ == '__main__':
    res = load_data('STAD')
    if res[0] is None:
        print('未找到数据，请检查路径配置。')
    else:
        (df_mrna, df_mirna, gene_path_mask_df, mirna_path_mask_df, _, _, _, y) = res
        (gene_list, mirna_list) = (df_mrna.columns.tolist(), df_mirna.columns.tolist())
        (X_m_tr, X_m_te, X_mi_tr, X_mi_te, y_tr, y_te) = train_test_split(df_mrna.values.astype(np.float32), df_mirna.values.astype(np.float32), y, test_size=0.2, random_state=SEED)
        (scaler_m, scaler_mi) = (StandardScaler(), StandardScaler())
        (X_m_tr, X_mi_tr) = (scaler_m.fit_transform(X_m_tr), scaler_mi.fit_transform(X_mi_tr))
        (X_m_te, X_mi_te) = (scaler_m.transform(X_m_te), scaler_mi.transform(X_mi_te))
        model = PathformerBio(dim_mrna=X_m_tr.shape[1], dim_mirna=X_mi_tr.shape[1], gene_path_mask_df=gene_path_mask_df, mirna_path_mask_df=mirna_path_mask_df).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.BCEWithLogitsLoss()
        model.train()
        loader = DataLoader(TensorDataset(torch.tensor(X_m_tr), torch.tensor(X_mi_tr), torch.tensor(y_tr).float()), batch_size=BATCH_SIZE, shuffle=True)
        for _ in range(5):
            for (bx_m, bx_mi, by) in loader:
                loss = criterion(model(bx_m.to(device), bx_mi.to(device)).squeeze(), by.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        plot_pathway_signatures(model, X_m_te, X_mi_te, gene_list, mirna_list, gene_path_mask_df, mirna_path_mask_df)
