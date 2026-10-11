import re
import numpy as np
import pandas as pd

def clean_name_logic(name):
    name = str(name).lower()
    name = re.sub('-(3|5)p', '', name)
    return name.strip()

def normalize_mirna_key(name):
    name = str(name).lower()
    name = re.sub('^(hsa|mmu)-', '', name)
    name = re.sub('-(3|5)p', '', name)
    return name.strip()

def build_mirna_gene_interaction_matrix(gene_list, mirna_list, *, interaction_file):
    (n_genes, n_mirnas) = (len(gene_list), len(mirna_list))
    interaction = np.zeros((n_mirnas, n_genes), dtype=np.float32)
    if not interaction_file.exists():
        return interaction
    try:
        df = pd.read_csv(interaction_file, usecols=['miRNA', 'Target Gene'], low_memory=False)
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

def load_driver_genes(gene_list, *, driver_file):
    if not driver_file.exists():
        return gene_list
    try:
        with open(driver_file, 'r', encoding='utf-8') as f:
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

def load_dual_pathway_structure(gene_list, mirna_list, *, gene_pathway_file, mirna_pathway_file):
    gene_path_dict = parse_gmt(gene_pathway_file)
    mirna_path_dict = parse_mirna_pathway_file(mirna_pathway_file)
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

def build_pathway_prior(gene_list, mirna_list, gene_mask_df, mirna_mask_df, use_driver_only=True, driver_idx=None, *, interaction_file):
    if use_driver_only and driver_idx is not None and (len(driver_idx) > 0):
        gene_list_use = [gene_list[i] for i in driver_idx]
        G = gene_mask_df.values.astype(np.float32)[driver_idx]
    else:
        gene_list_use = gene_list
        G = gene_mask_df.values.astype(np.float32)
    R = build_mirna_gene_interaction_matrix(gene_list_use, mirna_list, interaction_file=interaction_file)
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
