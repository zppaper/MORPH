import argparse
import os
import runpy
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['stage', 'vital_status', 'both'], default='both')
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    os.environ['MORPH_DATA_ROOT'] = str(args.data_root.resolve())
    root = Path(__file__).resolve().parent
    tasks = ['stage', 'vital_status'] if args.task == 'both' else [args.task]
    cohorts = {'stage': ['KIRC','MESO','STAD','THCA'], 'vital_status': ['LAML','SKCM','UVM']}
    for task in tasks:
        for ds in cohorts[task]:
            for name in ['clinical.txt', 'TPM.txt', 'miRNA_RPM.txt']:
                path = args.data_root / 'raw' / 'TCGA' / ds / name
                if not path.is_file():
                    raise FileNotFoundError(path)
            target = args.data_root / task / ds
            if target.exists() and any(target.iterdir()):
                raise FileExistsError(f'Choose a data root with empty task outputs: {target}')
        for operation in ['labels', 'align']:
            runpy.run_path(str(root / f'tcga_{task}_{operation}.py'), run_name='__main__')
        for ds in cohorts[task]:
            for name in ['mRNA.csv','miRNA.csv','label.csv']:
                path = args.data_root / task / ds / name
                if not path.is_file():
                    raise RuntimeError(f'Preprocessing did not produce {path}')

if __name__ == '__main__':
    main()
