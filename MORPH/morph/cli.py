import argparse
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
COHORTS = {'stage': ['KIRC', 'MESO', 'STAD', 'THCA'], 'vital_status': ['LAML', 'SKCM', 'UVM']}

def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()

def main(task):
    parser = argparse.ArgumentParser(description=f'MORPH {task} classification')
    parser.add_argument('--config', type=Path, default=REPO_ROOT / 'configs' / f'{task}.yaml')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--prior-dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--datasets', nargs='+', choices=COHORTS[task])
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    import yaml
    import numpy as np
    import pandas as pd
    import torch
    from .model import StageMORPH, VitalStatusMORPH
    from .train import run_experiment
    with args.config.open(encoding='utf-8') as handle:
        config = yaml.safe_load(handle)
    if config['task'] != task:
        raise ValueError('The configuration does not match the selected task')
    for key in ('data_root', 'prior_dir', 'output_dir'):
        override = getattr(args, key)
        config[key] = resolve_path(override if override is not None else config[key])
    config['driver_file'] = resolve_path(config['driver_file'])
    datasets = args.datasets if args.datasets is not None else config['datasets']
    seeds = args.seeds if args.seeds is not None else config['seeds']
    if not datasets or len(set(datasets)) != len(datasets) or (not set(datasets).issubset(COHORTS[task])):
        raise ValueError('Invalid or duplicate datasets')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Invalid or duplicate seeds')
    (config['datasets'], config['seeds']) = (datasets, seeds)
    required = [config['driver_file']] + [config['prior_dir'] / name for name in ('20230205_kegg_hsa.gmt', 'kegg_anano.txt', 'miRTarBase_MTI_processed_cleaned.csv')]
    required += [config['data_root'] / ds / name for ds in datasets for name in ('mRNA.csv', 'miRNA.csv', 'label.csv')]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing inputs:\n' + '\n'.join(missing))
    selected_device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    device = torch.device(selected_device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available; install the appropriate PyTorch build or use --device cpu')
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        device = torch.device('cuda', device_index)
        torch.cuda.set_device(device_index)
    config['device'] = str(device)
    output = config['output_dir']
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {output}. Use --output-dir for a new run.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2, default=str) + '\n', encoding='utf-8')
    environment = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': str(device), 'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None}
    for package in ('numpy', 'pandas', 'scikit-learn', 'PyYAML'):
        environment[package] = importlib.metadata.version(package)
    (output / 'environment.json').write_text(json.dumps(environment, indent=2) + '\n', encoding='utf-8')
    model_class = StageMORPH if task == 'stage' else VitalStatusMORPH
    (folds, seed_rows, summaries) = ([], [], [])
    for dataset in datasets:
        values = []
        for seed in seeds:
            (mean, records) = run_experiment(dataset, seed, config, device, model_class)
            folds.extend(records)
            values.append(mean)
            seed_rows.append({'dataset': dataset, 'seed': seed, 'auc': float(mean[0]), 'aupr': float(mean[1])})
            pd.DataFrame(folds).to_csv(output / 'fold_metrics.csv', index=False)
            pd.DataFrame(seed_rows).to_csv(output / 'seed_metrics.csv', index=False)
        array = np.asarray(values, dtype=float)
        averages = np.nanmean(array, axis=0)
        from .metrics import nanstd_ddof1
        deviations = nanstd_ddof1(array, axis=0)
        summaries.append({'dataset': dataset, 'n_seeds': len(seeds), 'auc_mean': float(averages[0]), 'auc_std': float(deviations[0]), 'aupr_mean': float(averages[1]), 'aupr_std': float(deviations[1])})
        pd.DataFrame(summaries).to_csv(output / 'summary.csv', index=False)
        print(f'{dataset}: AUC={averages[0]:.4f} +/- {deviations[0]:.4f}; AUPR={averages[1]:.4f} +/- {deviations[1]:.4f}', flush=True)
    print(f'Results: {output}', flush=True)
