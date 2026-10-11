from _settings import data_dir, prior_dir, output_dir, external_root
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from pathlib import Path
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'figure.dpi': 300})
EXPORT_DIR = external_root() / 'case_analysis_exports' / 'STAD' / 'seed_1028' / 'fold_0'
PAYLOAD_FILE = EXPORT_DIR / 'test_case_payload.pt'
META_FILE = EXPORT_DIR / 'test_meta.json'
REG_MATRIX_FILE = EXPORT_DIR / 'reg_matrix.npy'
NUM_PRIOR = 15
NUM_NOVEL = 30
COLOR_MIRNA_BASE = '#FF8C00'
COLOR_GENE_BASE = '#005BB5'
COLOR_PRIOR_EDGE = '#D32F2F'
COLOR_NOVEL_EDGE = '#5DADE2'
BG_COLOR = '#FDFDFD'

def load_and_process_network_data():
    if PAYLOAD_FILE.exists() and META_FILE.exists():
        with open(META_FILE, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        gene_names = meta['driver_gene_names']
        mirna_names = meta['mirna_list']
        payload = torch.load(PAYLOAD_FILE, map_location='cpu', weights_only=False)
        attn_tensor = payload['aux']['micro_attn_final']
        if isinstance(attn_tensor, torch.Tensor):
            attn_tensor = attn_tensor.detach().cpu().numpy()
        if attn_tensor.ndim == 3:
            mean_attn = np.mean(attn_tensor, axis=0)
        elif attn_tensor.ndim == 4:
            mean_attn = np.mean(attn_tensor, axis=(0, 1))
        else:
            raise ValueError('Expected attention with batch, optional head, driver and miRNA axes')
        if mean_attn.shape != (len(gene_names), len(mirna_names)):
            raise ValueError('Attention dimensions do not match metadata')
        if REG_MATRIX_FILE.exists():
            reg_matrix = np.load(REG_MATRIX_FILE)
        else:
            raise FileNotFoundError(REG_MATRIX_FILE)
        if reg_matrix.shape not in [(len(mirna_names), len(gene_names)), (len(gene_names), len(mirna_names))]:
            raise ValueError('Regulatory matrix dimensions do not match metadata')
        if not np.isfinite(mean_attn).all() or not np.isfinite(reg_matrix).all():
            raise ValueError('Non-finite attention or regulatory matrix')
        (prior_edges_all, novel_edges_all) = ([], [])
        for (g_idx, g_name) in enumerate(gene_names):
            for (m_idx, m_name) in enumerate(mirna_names):
                weight = float(mean_attn[g_idx, m_idx])
                if reg_matrix.shape == (len(mirna_names), len(gene_names)):
                    has_prior = bool(reg_matrix[m_idx, g_idx] > 0)
                elif reg_matrix.shape == (len(gene_names), len(mirna_names)):
                    has_prior = bool(reg_matrix[g_idx, m_idx] > 0)
                else:
                    has_prior = False
                edge_info = {'mirna': m_name, 'gene': g_name, 'weight': weight, 'has_prior': has_prior}
                if has_prior:
                    prior_edges_all.append(edge_info)
                else:
                    novel_edges_all.append(edge_info)
        prior_edges_all.sort(key=lambda x: x['weight'], reverse=True)
        novel_edges_all.sort(key=lambda x: x['weight'], reverse=True)
        selected_edges = prior_edges_all[:NUM_PRIOR] + novel_edges_all[:NUM_NOVEL]
        return (selected_edges, set((e['mirna'] for e in selected_edges)), set((e['gene'] for e in selected_edges)))
    else:
        raise FileNotFoundError('Real attention exports are required: test_case_payload.pt, test_meta.json, reg_matrix.npy')

def draw_node(ax, x, y, r, color_base, zorder=5):
    ax.add_patch(plt.Circle((x, y - r * 0.15), r * 1.05, color='black', alpha=0.15, zorder=zorder - 1))
    c_base = np.array(mcolors.to_rgb(color_base))
    c_high = np.array([1.0, 1.0, 1.0])
    c_edge = np.clip(c_base * 0.6, 0, 1)
    ax.add_patch(plt.Circle((x, y), r, facecolor=c_base, edgecolor=c_edge, lw=1.2, zorder=zorder))
    n_steps = 15
    for i in range(n_steps):
        frac = i / (n_steps - 1)
        rad = r * (1 - frac) * 0.85
        cx = x - r * 0.25 * frac
        cy = y + r * 0.25 * frac
        interp = frac ** 2.5
        cur_color = c_base * (1 - interp) + c_high * interp
        ax.add_patch(plt.Circle((cx, cy), rad, facecolor=cur_color, edgecolor='none', zorder=zorder + 1))

def plot_network():
    (edges_data, all_mirnas, all_genes) = load_and_process_network_data()
    if not edges_data:
        return
    node_strength = {}
    for e in edges_data:
        node_strength[e['mirna']] = node_strength.get(e['mirna'], 0) + e['weight']
        node_strength[e['gene']] = node_strength.get(e['gene'], 0) + e['weight']
    mirna_list = sorted(all_mirnas, key=lambda x: node_strength.get(x, 0), reverse=True)
    gene_list = sorted(all_genes, key=lambda x: node_strength.get(x, 0), reverse=True)
    (outer_r, inner_r) = (4.5, 2.2)
    pos = {}
    for (i, m) in enumerate(mirna_list):
        angle = 2 * np.pi * i / len(mirna_list) - np.pi / 2
        pos[m] = (outer_r * np.cos(angle), outer_r * np.sin(angle))
    for (i, g) in enumerate(gene_list):
        angle = 2 * np.pi * i / len(gene_list) - np.pi / 2 + np.pi / len(gene_list)
        pos[g] = (inner_r * np.cos(angle), inner_r * np.sin(angle))
    weights = [e['weight'] for e in edges_data]
    (max_w, min_w) = (max(weights), min(weights))
    norm = lambda w: (w - min_w) / (max_w - min_w + 1e-09)
    (fig, ax) = plt.subplots(figsize=(12, 13), dpi=300)
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-6.0, 6.0)
    ax.set_ylim(-7.0, 6.0)
    ax.add_patch(plt.Circle((0, 0), outer_r, fill=False, color='#B0BEC5', alpha=0.5, ls='--', lw=1.2, zorder=0))
    ax.add_patch(plt.Circle((0, 0), inner_r, fill=False, color='#B0BEC5', alpha=0.5, ls='--', lw=1.2, zorder=0))
    for (has_prior_flag, edge_color, alpha_base, lw_base, lw_scale, zorder) in [(False, COLOR_NOVEL_EDGE, 0.7, 1.0, 1.5, 1), (True, COLOR_PRIOR_EDGE, 0.9, 2.5, 1.5, 2)]:
        for e in edges_data:
            if e['has_prior'] != has_prior_flag:
                continue
            (x1, y1) = pos[e['mirna']]
            (x2, y2) = pos[e['gene']]
            nw = norm(e['weight'])
            lw = lw_base + lw_scale * nw
            ax.plot([x1, x2], [y1, y2], color=edge_color, lw=lw, alpha=alpha_base, zorder=zorder)
    max_str = max(node_strength.values()) if node_strength else 1
    for m in mirna_list:
        r = 0.16 + 0.08 * (node_strength.get(m, 0) / max_str)
        draw_node(ax, pos[m][0], pos[m][1], r, COLOR_MIRNA_BASE, zorder=10)
    for g in gene_list:
        r = 0.14 + 0.08 * (node_strength.get(g, 0) / max_str)
        draw_node(ax, pos[g][0], pos[g][1], r, COLOR_GENE_BASE, zorder=10)
    for (i, m) in enumerate(mirna_list):
        angle = 2 * np.pi * i / len(mirna_list) - np.pi / 2
        (ux, uy) = (np.cos(angle), np.sin(angle))
        (tx, ty) = ((outer_r + 0.35) * ux, (outer_r + 0.35) * uy)
        ha = 'left' if ux > 0 else 'right'
        if abs(ux) < 0.15:
            (ha, ty) = ('center', (outer_r + 0.45) * uy)
        txt = ax.text(tx, ty, m, fontsize=13, color='#2C3E50', ha=ha, va='center', zorder=20)
        txt.set_path_effects([pe.withStroke(linewidth=3.5, foreground=BG_COLOR)])
    for (i, g) in enumerate(gene_list):
        angle = 2 * np.pi * i / len(gene_list) - np.pi / 2 + np.pi / len(gene_list)
        (ux, uy) = (np.cos(angle), np.sin(angle))
        (tx, ty) = ((inner_r + 0.35) * ux, (inner_r + 0.35) * uy)
        ha = 'left' if ux > 0 else 'right'
        if abs(ux) < 0.15:
            (ha, ty) = ('center', (inner_r + 0.45) * uy)
        txt = ax.text(tx, ty, g, fontsize=12, fontweight='bold', color='#154360', ha=ha, va='center', zorder=20)
        txt.set_path_effects([pe.withStroke(linewidth=3.5, foreground=BG_COLOR)])
    fig.text(0.5, 0.94, 'Key miRNA–Driver Gene Interaction Landscape', fontsize=24, fontweight='bold', color='#17202A', ha='center')
    legend_elements = [Line2D([0], [0], color=COLOR_PRIOR_EDGE, lw=3.5, label='Prior Validated Interaction (miRTarBase)'), Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_MIRNA_BASE, markersize=15, label='Regulatory miRNA'), Line2D([0], [0], color=COLOR_NOVEL_EDGE, lw=2.0, label='Model-highlighted non-prior miRNA–gene pair'), Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_GENE_BASE, markersize=15, label='Driver Gene')]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, frameon=False, fontsize=13, handletextpad=0.8, columnspacing=4.0)
    arrow_y = -6.2
    ax.annotate('', xy=(1.5, arrow_y), xytext=(-1.5, arrow_y), arrowprops=dict(arrowstyle='<|-|>', color='#3498DB', lw=2.5, mutation_scale=20))
    ax.text(-1.7, arrow_y, 'Lower', fontsize=12, fontweight='bold', color='#555555', ha='right', va='center')
    ax.text(1.7, arrow_y, 'Higher', fontsize=12, fontweight='bold', color='#555555', ha='left', va='center')
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    save_pdf = 'STAD_regulatory_network.pdf'
    save_png = 'STAD_regulatory_network.png'
    plt.savefig(output_dir('plot_regulatory_network') / Path(save_pdf).name, format='pdf', bbox_inches='tight', facecolor=BG_COLOR)
    plt.savefig(output_dir('plot_regulatory_network') / Path(save_png).name, format='png', bbox_inches='tight', facecolor=BG_COLOR)
    plt.show()
    print(f'\n 图表已保存至：\n - {save_pdf}\n - {save_png}')
if __name__ == '__main__':
    plot_network()
