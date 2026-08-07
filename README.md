# GeneCurator

A human-in-the-loop web application for constructing and curating gene panels
from expression data.

GeneCurator builds a gene panel from an expression matrix, lets a domain expert
curate it (add, remove, or reinstate genes) with supporting evidence, and
recomputes the downstream analyses after every change — functional enrichment,
KEGG pathway maps, protein-protein interaction networks, and ligand-receptor
annotation.

This repository accompanies the paper:

> Papadaki, E. and Vrahatis, A.G. *GeneCurator: a human-in-the-loop web
> application for constructing and curating gene panels from expression data.*
> Computational Biology and Chemistry. [DOI when available]

---

## Features

- Builds a gene panel directly from an expression matrix (no gene list needed).
- Six feature-selection methods with a plain-language guide: threshold-based
  DEG, random forest, gradient boosting, LASSO, elastic net, and SVM-RFE.
- Interactive curation with per-gene evidence from a local UniProt reference;
  changes are logged and reversible.
- Downstream analyses recomputed on the current panel:
  - Functional over-representation analysis against 12 gene-set libraries.
  - KEGG pathway maps with panel genes highlighted and coloured by log2 fold change.
  - Protein-protein interaction networks from a local STRING v12.0 snapshot.
  - Ligand-receptor annotation from CellTalkDB.
- All reference resources are held as local snapshots for reproducibility.

## Requirements

- Python >= 3.10
- Dependencies listed in `requirements.txt` / `pyproject.toml`
  (FastAPI, uvicorn, pandas, numpy, scikit-learn, shap, plotly, networkx,
  graphviz, openpyxl, jinja2).

## Installation

```bash
git clone https://github.com/<username>/GeneCurator.git
cd GeneCurator

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Reference data (one-time download)

The large reference databases are **not** stored in this repository. Download
them once with the provided scripts:

```bash
# STRING v12.0 protein-protein interactions (high-confidence, score >= 700)
python download_ppi_assets.py

# KEGG pathway maps (PNG + KGML), ~50 MB, ~15 min
python download_kegg_assets.py
```

The enrichment gene-set libraries (KEGG, Reactome, WikiPathways, BioPlanet,
MSigDB Hallmark, GO BP/MF/CC, DisGeNET, Disease Perturbations, PanglaoDB) and
the UniProt human reference proteome (UP000005640) are expected in the project
root as Enrichr-format `.txt` files and a `.tsv`, respectively. See
Data sources below for where to obtain them.

## Running

```bash
python run.py
```

This starts the FastAPI server on `http://localhost:8000` and opens the
browser. Then follow the four-step workflow:

1. **Load** an expression matrix and a sample-metadata table.
2. **Select** a feature-selection method to produce a ranked panel.
3. **Curate** the panel - remove or reinstate genes with supporting evidence.
4. **Analyse** - the downstream results recompute on the current panel.

## Input format

- **Expression matrix**: rows are samples, columns are genes (CSV or TSV).
- **Sample metadata**: assigns each sample to a group, with identifiers
  matching the expression matrix.

## Data sources

The case study uses publicly available data from The Cancer Genome Atlas
Breast Invasive Carcinoma (TCGA-BRCA) project, obtained through the NCI Genomic
Data Commons Data Portal (https://portal.gdc.cancer.gov).

Reference resources and their citations:

- UniProt human reference proteome UP000005640 - https://www.uniprot.org/proteomes/UP000005640
- STRING v12.0 (Szklarczyk et al., 2023) - https://string-db.org
- KEGG (Kanehisa & Goto, 2000) - https://www.kegg.jp
- Enrichr gene-set libraries (Kuleshov et al., 2016) - https://maayanlab.cloud/Enrichr
- CellTalkDB (Shao et al., 2021) - http://tcm.zju.edu.cn/celltalkdb

## Citation

If you use GeneCurator, please cite the paper above. A `CITATION.cff` file is
included so GitHub shows a "Cite this repository" button.

## License

Released under the MIT License. See `LICENSE`.

## Contact

Eugenia Papadaki - epapadaki@ionio.gr
Bioinformatics and Human Electrophysiology Laboratory, Department of
Informatics, Ionian University, Corfu, Greece.
