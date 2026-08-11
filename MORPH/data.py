from pathlib import Path
import pandas as pd


DATASETS = ("KIRC", "MESO", "STAD", "THCA")


def load_dataset(data_root, dataset_name):
    """Load and align mRNA, miRNA, and stage labels by sample ID."""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data_dir = Path(data_root) / dataset_name
    mrna = pd.read_csv(data_dir / "mRNA.csv", index_col=0).T
    mirna = pd.read_csv(data_dir / "miRNA.csv", index_col=0).T
    labels = pd.read_csv(data_dir / "label.csv", index_col=0)

    sample_ids = mrna.index.intersection(mirna.index).intersection(labels.index)
    if sample_ids.empty:
        raise ValueError(f"No common samples found in {data_dir}")

    return (
        mrna.loc[sample_ids],
        mirna.loc[sample_ids],
        labels.loc[sample_ids, "label"].to_numpy(),
    )
