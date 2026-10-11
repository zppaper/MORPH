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
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
KEGG_MAP = {'hsa01100': 'Metabolic pathways', 'hsa05200': 'Pathways in cancer', 'hsa04151': 'PI3K-Akt signaling', 'hsa04390': 'Hippo signaling', 'hsa04010': 'MAPK signaling', 'hsa04310': 'Wnt signaling', 'hsa05205': 'Proteoglycans in cancer', 'hsa04510': 'Focal adhesion', 'hsa04060': 'Cytokine receptor interaction', 'hsa04015': 'Rap1 signaling', 'hsa05202': 'Transcriptional misreg.', 'hsa04530': 'Tight junction', 'hsa05166': 'HTLV-I infection', 'hsa05206': 'MicroRNAs in cancer', 'hsa04145': 'Phagosome', 'hsa04360': 'Axon guidance', 'hsa04014': 'Ras signaling pathway', 'hsa04514': 'Cell adhesion molecules', 'hsa04910': 'Insulin signaling', 'hsa04919': 'Thyroid hormone', 'hsa04550': 'Signaling pathways', 'hsa04350': 'TGF-beta signaling', 'hsa04068': 'FoxO signaling', 'hsa04120': 'Ubiquitin mediated proteolysis', 'hsa03040': 'Spliceosome', 'hsa04066': 'HIF-1 signaling'}
SEED = 2024
BATCH_SIZE = 32
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

def build_pathway_prior(gene_list, mirna_list, gene_mask_df, mirna_mask_df):
    G = gene_mask_df.values.astype(np.float32)
    R = build_mirna_gene_interaction_matrix(gene_list, mirna_list)
    M = mirna_mask_df.values.astype(np.float32)
    A_raw = M.T @ R @ G
    if A_raw.size == 0:
        return np.zeros((G.shape[1], M.shape[1]), dtype=np.float32)
    A_raw = A_raw.T + np.eye(A_raw.shape[1], dtype=np.float32)
    row_sum = A_raw.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return (A_raw / row_sum).astype(np.float32)

def load_data(dataset_name):
    data_dir = BASE_DATA_DIR / dataset_name
    try:
        mrna_df = pd.read_csv(data_dir / 'mRNA.csv', index_col=0).T
        mirna_df = pd.read_csv(data_dir / 'miRNA.csv', index_col=0).T
        label_df = pd.read_csv(data_dir / 'label.csv', index_col=0)
    except FileNotFoundError:
        return [None] * 7
    common_samples = mrna_df.index.intersection(mirna_df.index).intersection(label_df.index)
    (mrna_df, mirna_df, label_df) = (mrna_df.loc[common_samples], mirna_df.loc[common_samples], label_df.loc[common_samples])
    (gene_list, mirna_list) = (list(mrna_df.columns), list(mirna_df.columns))
    (gene_path_mask_df, mirna_path_mask_df) = load_dual_pathway_structure(gene_list, mirna_list)
    path_prior = build_pathway_prior(gene_list, mirna_list, gene_path_mask_df, mirna_path_mask_df)
    return (mrna_df, mirna_df, gene_path_mask_df, mirna_path_mask_df, path_prior, label_df['label'].values)

class PathwayCrossAttnBlock(nn.Module):

    def __init__(self, n_genes, n_mirnas, n_pathways, gene_mask, mirna_mask, path_embed_dim=16, path_prior=None):
        super().__init__()
        self.num_heads = 4
        self.head_dim = path_embed_dim // 4
        self.register_buffer('gene_mask', gene_mask / (gene_mask.sum(dim=0, keepdim=True) + 1e-08))
        self.register_buffer('mirna_mask', mirna_mask / (mirna_mask.sum(dim=0, keepdim=True) + 1e-08))
        (self.g_proj, self.m_proj) = (nn.Linear(1, path_embed_dim), nn.Linear(1, path_embed_dim))
        (self.to_q, self.to_k, self.to_v) = (nn.Linear(path_embed_dim, path_embed_dim, bias=False), nn.Linear(path_embed_dim, path_embed_dim, bias=False), nn.Linear(path_embed_dim, path_embed_dim, bias=False))
        self.scale = self.head_dim ** (-0.5)
        self.norm = nn.LayerNorm(path_embed_dim)
        (self.tau, self.alpha) = (nn.Parameter(torch.tensor(0.35)), nn.Parameter(torch.tensor(0.4)))
        if path_prior is not None:
            prior_tensor = torch.tensor(path_prior, dtype=torch.float32)
            noise = torch.rand_like(prior_tensor) * 1e-05
            (_, topk_idx) = torch.topk(prior_tensor + noise, k=min(15, prior_tensor.shape[-1]), dim=-1)
            sparse_mask = torch.zeros_like(prior_tensor).scatter_(-1, topk_idx, 1.0)
            sparse_mask.fill_diagonal_(1.0)
            self.register_buffer('prior_norm_mask', sparse_mask / (sparse_mask.sum(dim=-1, keepdim=True) + 1e-08))
        else:
            self.register_buffer('prior_norm_mask', None)
        self.gate_proj = nn.Linear(path_embed_dim, path_embed_dim)
        self.final_proj = nn.Sequential(nn.Linear(path_embed_dim * 2, 128), nn.LayerNorm(128), nn.ReLU())
        self.last_attn = None

    def forward(self, x_mrna, x_mirna):
        B = x_mrna.shape[0]
        (h_g_local, h_m_local) = (F.gelu(self.g_proj(x_mrna.unsqueeze(-1))), F.gelu(self.m_proj(x_mirna.unsqueeze(-1))))
        path_g = torch.einsum('b g d, g p -> b p d', h_g_local, self.gene_mask)
        path_m = torch.einsum('b m d, m p -> b p d', h_m_local, self.mirna_mask)
        q = self.to_q(path_g).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(path_m).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(path_m).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        raw_attn = F.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale / torch.clamp(self.tau, min=0.01), dim=-1)
        if self.prior_norm_mask is not None:
            alpha = torch.sigmoid(self.alpha)
            attn = alpha * self.prior_norm_mask.unsqueeze(0).unsqueeze(0) + (1.0 - alpha) * raw_attn
        else:
            attn = raw_attn
        self.last_attn = attn.detach()
        update = torch.matmul(attn, v).transpose(1, 2).reshape(B, -1, self.num_heads * self.head_dim)
        path_g_refined = self.norm(path_g + update)
        path_g_gated = path_g_refined * torch.sigmoid(self.gate_proj(path_g_refined))
        return self.final_proj(torch.cat([path_g_gated.mean(dim=1), path_g_gated.max(dim=1)[0]], dim=-1))

class PathformerBio(nn.Module):

    def __init__(self, dim_mrna, dim_mirna, gene_path_mask_df, mirna_path_mask_df, path_prior):
        super().__init__()
        self.pathway_block = PathwayCrossAttnBlock(dim_mrna, dim_mirna, gene_path_mask_df.shape[1], torch.tensor(gene_path_mask_df.values, dtype=torch.float32), torch.tensor(mirna_path_mask_df.values, dtype=torch.float32), path_prior=path_prior)
        self.classifier = nn.Linear(128, 1)

    def forward(self, x_mrna, x_mirna):
        return self.classifier(self.pathway_block(x_mrna, x_mirna))

def wrap_text(text, width=22):
    if len(text) <= width:
        return text
    words = text.split(' ')
    (lines, current) = ([], [])
    for w in words:
        if sum((len(x) for x in current)) + len(current) + len(w) > width:
            lines.append(' '.join(current))
            current = [w]
        else:
            current.append(w)
    if current:
        lines.append(' '.join(current))
    return '\n'.join(lines)

def plot_pathway_network(model, test_loader, pathways, original_path_prior):
    model.eval()
    print('[Network] 正在计算通路交叉注意力矩阵...')
    attn_list = []
    with torch.no_grad():
        for (bx_m, bx_mi, _) in test_loader:
            model(bx_m.to(device), bx_mi.to(device))
            attn_list.append(model.pathway_block.last_attn.cpu().numpy())
    avg_attn = np.concatenate(attn_list, axis=0).mean(axis=(0, 1))
    prior_matrix = original_path_prior
    symmetric_attn = np.maximum(avg_attn, avg_attn.T)
    symmetric_prior = np.maximum(prior_matrix, prior_matrix.T)
    np.fill_diagonal(symmetric_attn, 0)
    np.fill_diagonal(symmetric_prior, 0)
    N_EDGES = 75
    N_HUBS = 4
    flat_indices = np.argsort(symmetric_attn, axis=None)[::-1]
    top_edges = []
    added_pairs = set()
    for idx in flat_indices:
        (u, v) = np.unravel_index(idx, symmetric_attn.shape)
        if u != v and frozenset((u, v)) not in added_pairs:
            top_edges.append((u, v))
            added_pairs.add(frozenset((u, v)))
        if len(top_edges) >= N_EDGES:
            break
    G = nx.Graph()
    for (u, v) in top_edges:
        G.add_edge(u, v, weight=symmetric_attn[u, v], prior_weight=symmetric_prior[u, v])
    prior_weights = [d['prior_weight'] for (u, v, d) in G.edges(data=True)]
    non_zero_priors = [p for p in prior_weights if p > 0]
    if len(non_zero_priors) > 0:
        thresh_prior = np.percentile(non_zero_priors, 50)
    else:
        thresh_prior = 0.0
    for (u, v, d) in G.edges(data=True):
        if d['prior_weight'] > thresh_prior and d['prior_weight'] > 0.0001:
            d['type'] = 'prior_driven'
        else:
            d['type'] = 'data_driven'
    for n in G.nodes():
        p_name = KEGG_MAP.get(pathways[n], pathways[n])
        G.nodes[n]['name'] = wrap_text(p_name)
    degrees = dict(G.degree(weight='weight'))
    sorted_nodes = sorted(degrees, key=degrees.get, reverse=True)
    hub_nodes = sorted_nodes[:N_HUBS]
    peri_nodes = sorted_nodes[N_HUBS:]
    pos = {}
    r_inner = 0.45
    for (i, n) in enumerate(hub_nodes):
        angle = i * (2 * np.pi / len(hub_nodes)) + np.pi / 4
        pos[n] = np.array([r_inner * np.cos(angle), r_inner * np.sin(angle)])
    r_outer_base = 1.55
    for (i, n) in enumerate(peri_nodes):
        angle = i * (2 * np.pi / len(peri_nodes))
        r = r_outer_base + random.uniform(-0.2, 0.2)
        pos[n] = np.array([r * np.cos(angle), r * np.sin(angle)])
    node_colors = ['#005B82' if n in hub_nodes else '#89B6D5' for n in G.nodes()]
    node_sizes = [1100 if n in hub_nodes else 400 for n in G.nodes()]
    (fig, ax) = plt.subplots(figsize=(12, 10), facecolor='white')
    ax.axis('off')
    prior_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['type'] == 'prior_driven']
    data_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d['type'] == 'data_driven']
    edge_kwargs = dict(width=1.5, alpha=0.6, connectionstyle='arc3,rad=0.15', arrows=True, arrowstyle='-', node_size=0, ax=ax)
    if prior_edges:
        edges_k = nx.draw_networkx_edges(G, pos, edgelist=prior_edges, edge_color='#4CB385', **edge_kwargs)
        for e in edges_k:
            e.set_zorder(1)
    if data_edges:
        edges_n = nx.draw_networkx_edges(G, pos, edgelist=data_edges, edge_color='#F2848F', **edge_kwargs)
        for e in edges_n:
            e.set_zorder(1)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, edgecolors='#222222', linewidths=1.5, ax=ax)
    nodes.set_zorder(2)
    for n in G.nodes():
        (x, y) = pos[n]
        text = G.nodes[n]['name']
        if n in hub_nodes:
            ax.text(x, y - 0.14, text, ha='center', va='top', fontsize=12, fontweight='bold', color='black', zorder=3)
        else:
            angle = np.arctan2(y, x)
            dist_factor = 1.15
            x_text = x * dist_factor
            y_text = y * dist_factor
            ha = 'left' if np.cos(angle) > 0.1 else 'right' if np.cos(angle) < -0.1 else 'center'
            va = 'bottom' if np.sin(angle) > 0.1 else 'top' if np.sin(angle) < -0.1 else 'center'
            ax.text(x_text, y_text, text, ha=ha, va=va, fontsize=11, fontweight='bold', color='black', zorder=3)
    fig.text(0.5, 0.96, 'Hub Module of Pathway Crosstalk Network\nIntegrating Prior Knowledge & Data-Driven Interactions', fontsize=18, fontweight='bold', ha='center', va='center')
    legend_elements = [Line2D([0], [0], color='#4CB385', lw=3, label='Prior-Driven Crosstalk'), Line2D([0], [0], color='#F2848F', lw=3, label='Data-Driven Crosstalk (Novel)')]
    ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(-0.1, -0.05), title='Links identified by\nModel Attention', title_fontproperties={'weight': 'bold', 'size': 14}, frameon=False, fontsize=13, handlelength=1.5)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_dir('legacy_pathway_network') / Path('STAD_pathway_network.png').name, dpi=400, bbox_inches='tight')
    plt.savefig(output_dir('legacy_pathway_network') / Path('STAD_pathway_network.pdf').name, format='pdf', bbox_inches='tight')
    print('[Network] 通路网络已保存: STAD_pathway_network.png & STAD_pathway_network.pdf')
    plt.show()
if __name__ == '__main__':
    res = load_data('STAD')
    if res[0] is None:
        print('未找到数据，请检查路径。')
    else:
        (df_mrna, df_mirna, gene_path_mask_df, mirna_path_mask_df, path_prior, y) = res
        pathways = gene_path_mask_df.columns.tolist()
        (X_m_tr, X_m_te, X_mi_tr, X_mi_te, y_tr, y_te) = train_test_split(df_mrna.values.astype(np.float32), df_mirna.values.astype(np.float32), y, test_size=0.2, random_state=SEED)
        (scaler_m, scaler_mi) = (StandardScaler(), StandardScaler())
        (X_m_tr, X_mi_tr) = (scaler_m.fit_transform(X_m_tr), scaler_mi.fit_transform(X_mi_tr))
        (X_m_te, X_mi_te) = (scaler_m.transform(X_m_te), scaler_mi.transform(X_mi_te))
        model = PathformerBio(dim_mrna=X_m_tr.shape[1], dim_mirna=X_mi_tr.shape[1], gene_path_mask_df=gene_path_mask_df, mirna_path_mask_df=mirna_path_mask_df, path_prior=path_prior).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCEWithLogitsLoss()
        model.train()
        train_loader = DataLoader(TensorDataset(torch.tensor(X_m_tr), torch.tensor(X_mi_tr), torch.tensor(y_tr).float()), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TensorDataset(torch.tensor(X_m_te), torch.tensor(X_mi_te), torch.tensor(y_te).float()), batch_size=BATCH_SIZE, shuffle=False)
        print('正在训练模型以激活注意力权重...')
        for _ in range(5):
            for (bx_m, bx_mi, by) in train_loader:
                loss = criterion(model(bx_m.to(device), bx_mi.to(device)).squeeze(), by.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        plot_pathway_network(model, test_loader, pathways, path_prior)
