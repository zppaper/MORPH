import argparse
from pathlib import Path
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--clinical', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    clinical = pd.read_csv(args.clinical, sep='\t')
    labels = pd.read_csv(args.labels)
    clinical = clinical.rename(columns={'bcr_patient_barcode':'sample_id','survival_time':'OS.time','survival_status':'OS'})
    required = ['sample_id','OS.time','OS']
    if not set(required).issubset(clinical):
        raise ValueError('Use the clinical.txt produced by assemble_tcga_clinical.py')
    clinical = clinical[required].copy()
    if clinical.sample_id.duplicated().any():
        raise ValueError('Clinical patient identifiers are not unique')
    out = labels[['sample_id']].merge(clinical,on='sample_id',how='inner',validate='one_to_one')
    out['OS.time'] = pd.to_numeric(out['OS.time'],errors='coerce')
    out['OS'] = pd.to_numeric(out['OS'],errors='coerce')
    out = out[(out['OS.time'] > 0) & out['OS'].isin([0,1])]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(args.output,index=False)

if __name__ == '__main__':
    main()
