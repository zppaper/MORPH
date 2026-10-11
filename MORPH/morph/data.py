from pathlib import Path
import pandas as pd
from .priors import load_driver_genes, build_mirna_gene_interaction_matrix, load_dual_pathway_structure, build_pathway_prior

def load_data(dataset_name, data_root, prior_dir, driver_file):
    data_root = Path(data_root)
    prior_dir = Path(prior_dir)
    driver_file = Path(driver_file)
    gene_pathway_file = prior_dir / '20230205_kegg_hsa.gmt'
    mirna_pathway_file = prior_dir / 'kegg_anano.txt'
    interaction_file = prior_dir / 'miRTarBase_MTI_processed_cleaned.csv'
    data_dir = data_root / dataset_name
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
    driver_genes = load_driver_genes(gene_list, driver_file=driver_file)
    driver_idx = [i for (i, g) in enumerate(gene_list) if g in driver_genes]
    gene_list_driver = [gene_list[i] for i in driver_idx]
    if len(driver_idx) == 0:
        driver_idx = list(range(len(gene_list)))
        gene_list_driver = gene_list
    reg_matrix = build_mirna_gene_interaction_matrix(gene_list_driver, mirna_list, interaction_file=interaction_file)
    edge_mask = reg_matrix.sum(axis=0) > 0
    if edge_mask.any():
        driver_idx = [idx for (idx, keep) in zip(driver_idx, edge_mask) if keep]
        reg_matrix = reg_matrix[:, edge_mask]
    (gene_path_mask_df, mirna_path_mask_df) = load_dual_pathway_structure(gene_list, mirna_list, gene_pathway_file=gene_pathway_file, mirna_pathway_file=mirna_pathway_file)
    path_prior = build_pathway_prior(gene_list, mirna_list, gene_path_mask_df, mirna_path_mask_df, use_driver_only=True, driver_idx=driver_idx, interaction_file=interaction_file)
    return (mrna_df, mirna_df, gene_path_mask_df, mirna_path_mask_df, reg_matrix, driver_idx, path_prior, label_df['label'].values)
