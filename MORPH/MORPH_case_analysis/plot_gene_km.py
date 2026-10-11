from _settings import data_dir, prior_dir, output_dir, external_root
from pathlib import Path
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.plotting import add_at_risk_counts
import warnings
warnings.filterwarnings('ignore')
COLOR_HIGH = '#E64B35'
COLOR_LOW = '#4DBBD5'

def find_best_match(target, columns):
    target_clean = target.lower().strip()
    for col in columns:
        if target_clean == col.lower().strip():
            return col
    for col in columns:
        if target_clean in col.lower().strip() or str(col).lower().strip() in target_clean:
            return col
    return None

def main():
    base_dir = str(external_root())
    data_dir = str(external_root() / 'stage' / 'STAD')
    out_dir = str(output_dir('gene_km'))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    clinical_file = os.path.join(data_dir, 'km_clinical.csv')
    mrna_file = os.path.join(data_dir, 'mRNA.csv')
    mirna_file = os.path.join(data_dir, 'miRNA.csv')
    print('==== 加载数据用于作图 ====')
    df_clin = pd.read_csv(clinical_file)
    df_clin = df_clin.dropna(subset=['OS.time', 'OS'])
    df_clin['time'] = pd.to_numeric(df_clin['OS.time'], errors='coerce')
    df_clin['status'] = pd.to_numeric(df_clin['OS'], errors='coerce')
    df_clin = df_clin.dropna(subset=['time', 'status'])
    df_clin = df_clin[df_clin['time'] > 0]
    df_mrna = pd.read_csv(mrna_file, index_col=0).T
    df_mirna = pd.read_csv(mirna_file, index_col=0).T
    targets = {'PTPN20': df_mrna}
    for (target_name, df_expr) in targets.items():
        actual_col = find_best_match(target_name, df_expr.columns)
        if actual_col is None:
            continue
        df_target = df_expr[[actual_col]].reset_index()
        df_target.columns = ['sample_id', 'expression']
        df_merged = pd.merge(df_clin, df_target, on='sample_id', how='inner')
        df_merged = df_merged.dropna(subset=['expression'])
        df_merged['expression'] = pd.to_numeric(df_merged['expression'], errors='coerce')
        median_expr = df_merged['expression'].median()
        df_high = df_merged[df_merged['expression'] >= median_expr]
        df_low = df_merged[df_merged['expression'] < median_expr]
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        (fig, ax) = plt.subplots(figsize=(8, 6))
        kmf_high.fit(df_high['time'], event_observed=df_high['status'], label=f'High (n={len(df_high)})')
        ax = kmf_high.plot_survival_function(ax=ax, color=COLOR_HIGH, linewidth=2.5, ci_alpha=0.15)
        kmf_low.fit(df_low['time'], event_observed=df_low['status'], label=f'Low (n={len(df_low)})')
        ax = kmf_low.plot_survival_function(ax=ax, color=COLOR_LOW, linewidth=2.5, ci_alpha=0.15)
        results = logrank_test(df_high['time'], df_low['time'], event_observed_A=df_high['status'], event_observed_B=df_low['status'])
        p_val = results.p_value
        if p_val < 0.001:
            p_text = 'Log-rank $P$ < 0.001'
        else:
            p_text = f'Log-rank $P$ = {p_val:.3f}'
        ax.set_title(f'Kaplan-Meier Survival Curve: {target_name}', fontsize=15, fontweight='bold', pad=15)
        ax.set_ylabel('Survival Probability', fontsize=13, fontweight='bold')
        ax.set_xlabel('Time (Days)', fontsize=13, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.text(0.05, 0.1, p_text, transform=ax.transAxes, fontsize=12, fontweight='bold', bbox=dict(boxstyle='square,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.8))
        (handles, labels) = ax.get_legend_handles_labels()
        ax.legend(handles=[handles[0], handles[2]] if len(handles) > 2 else handles, labels=[labels[0], labels[2]] if len(labels) > 2 else labels, loc='upper right', frameon=False, fontsize=11)
        plt.subplots_adjust(bottom=0.3)
        add_at_risk_counts(kmf_high, kmf_low, ax=ax, fig=fig, fontsize=11)
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f'KM_Curve_{target_name}.pdf')
        png_path = os.path.join(out_dir, f'KM_Curve_{target_name}.png')
        plt.savefig(output_dir('plot_gene_km') / Path(plot_path).name, dpi=300, bbox_inches='tight')
        plt.savefig(output_dir('plot_gene_km') / Path(png_path).name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'--> Saved KM plot to: {png_path} and .pdf')
if __name__ == '__main__':
    main()
