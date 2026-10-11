import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('results/sensitivity_figures'))
    args = parser.parse_args()
    frame = pd.read_csv(args.summary)
    cohorts = ['LAML','SKCM','UVM']
    groups = [(['path_topk_2','path_topk_10','path_topk_20','path_topk_40','path_dense_weighted','path_random_matched'], ['k=2','k=10','k=20','k=40','Dense','Random']), (['temperature_fixed_0_5','temperature_fixed_1','temperature_fixed_2'], ['0.5','1','2'])]
    if frame.duplicated(['dataset','variant']).any():
        raise ValueError('Summary must have one row per cohort and variant')
    frame = frame.set_index(['dataset','variant'])
    fig, axes = plt.subplots(2,2,figsize=(12,7),layout='constrained')
    rows = []
    for row,metric in enumerate(['auc_mean','aupr_mean']):
        for col,(variants,labels) in enumerate(groups):
            ax = axes[row,col]
            x = np.arange(len(variants))
            for j,cohort in enumerate(cohorts):
                baseline = float(frame.loc[(cohort,'full_original'),metric])
                delta = [100*(float(frame.loc[(cohort,v),metric])-baseline) for v in variants]
                ax.bar(x+(j-1)*0.24,delta,width=0.24,label=cohort)
                rows += [dict(dataset=cohort,variant=v,metric=metric,delta_percentage_points=d) for v,d in zip(variants,delta)]
            ax.axhline(0,color='black',linewidth=0.8)
            ax.set_xticks(x,labels)
            ax.set_ylabel(('AUC' if row==0 else 'AUPR')+' change (percentage points)')
            ax.set_xlabel('Pathway prior' if col==0 else 'Fixed temperature')
            ax.legend(frameon=False)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output_dir/'sensitivity_deltas.csv',index=False)
    for suffix in ['pdf','png']:
        fig.savefig(args.output_dir/f'sensitivity_analysis_vital_status.{suffix}',dpi=300)

if __name__ == '__main__':
    main()
