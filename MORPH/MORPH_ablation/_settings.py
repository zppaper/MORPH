import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def external_root():
    return Path(os.environ.get('MORPH_DATA_ROOT', str(ROOT.parent / 'MORPH_main' / 'datasets'))).expanduser().resolve()

def data_dir(task):
    return Path(os.environ.get('MORPH_DATA_DIR', str(external_root() / task))).expanduser().resolve()

def prior_dir():
    return Path(os.environ.get('MORPH_PRIOR_DIR', str(external_root() / 'priors'))).expanduser().resolve()

def output_dir(name):
    path = Path(os.environ.get('MORPH_OUTPUT_DIR', str(ROOT / 'results' / name))).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def driver_file():
    return Path(os.environ.get('MORPH_DRIVER_FILE', str(ROOT / 'resources' / 'driver_genes_list.txt'))).expanduser().resolve()
