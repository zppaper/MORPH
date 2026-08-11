# MORPH

**MORPH: Integrating Multi-Scale Priors from Driver-Centric Targeted Regulation to Pathway Crosstalk for Interpretable Cancer Prognosis**

MORPH is an interpretable multi-omics framework for cancer prognosis. It jointly models mRNA and miRNA expression while incorporating biological priors at complementary scales.

![Overview of the MORPH framework](assets/pathduet_overview.png)

## Model overview

MORPH contains three main components:

1. **Driver-centric cross-omics regulation:** models targeted interactions between selected driver mRNAs and regulatory miRNAs by combining data-driven and prior-guided attention.
2. **Regulation-derived pathway-level associations:** projects mRNA and miRNA features into pathway representations and captures pathway crosstalk using biological knowledge.
3. **Multi-scale feature fusion:** integrates global, micro-level, and pathway-level representations for downstream cancer prognosis tasks, including cancer stage prediction and survival analysis.

## Repository contents

```text
MORPH/
|-- MORPH/
|   |-- data.py                 # Multi-omics data loading and sample alignment
|   `-- utils.py                # Reproducibility and evaluation utilities
|-- datasets/
|   `-- KIRC/                   # Example mRNA, miRNA, and label files
|-- assets/
|   `-- pathduet_overview.png   # Framework overview
|-- 20230205_kegg_hsa.gmt       # KEGG gene-pathway annotations
|-- kegg_anano.txt              # miRNA-pathway annotations
|-- driver_genes_list.txt       # Driver-gene prior
`-- requirements.txt
```

## Quick start

Install the dependencies:

```bash
pip install -r requirements.txt
```

Load the example KIRC dataset:

```python
from MORPH.data import load_dataset

mrna, mirna, labels = load_dataset("datasets", "KIRC")
print(mrna.shape, mirna.shape, labels.shape)
```

## Code availability

This repository currently provides a preview of the data-processing and evaluation utilities, example data, biological annotations, and the MORPH framework overview.

**The full implementation will be publicly released on GitHub upon acceptance of this manuscript.**

## Citation

Citation information will be added upon publication.
