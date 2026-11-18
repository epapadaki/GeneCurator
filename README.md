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
> Computational Biology and Chemistry 126, 109471.
> https://doi.org/10.1016/j.compbiolchem.2026.109471
>
> The version described in the paper is release
> [v1.0.0](https://github.com/epapadaki/GeneCurator/releases/tag/v1.0.0).

---

## Features

- Builds a gene panel directly from an expression matrix (no gene list needed).
- Six feature-selection methods with a plain-language guide: threshold-based
  DEG, random forest, gradient boosting, LASSO, elastic net, and SVM-RFE.
- Interactive curation with per-gene evidence from a local UniProt reference;
  changes are logged and reversible.
- Downstream analyses recomputed on the current panel:
  - Functional over-representation analysis against 15 gene-set libraries.
  - KEGG pathway maps with panel genes highlighted and coloured by log2 fold change.
  - Protein-protein interaction networks from a local STRING v12.0 snapshot.
  - Ligand-receptor annotation from CellTalkDB.
- All reference resources are held as local snapshots for reproducibility.

## Requirements

- Python >= 3.10
- Graphviz (the `dot` program), used to draw the pipeline diagrams
- The Python packages in `requirements.txt` (FastAPI, uvicorn, pandas, numpy,
  scikit-learn, shap, plotly, networkx, graphviz, openpyxl, jinja2)

## Installation

With conda (Linux, macOS, Windows), which also installs Graphviz:

```bash
git clone https://github.com/epapadaki/GeneCurator.git
cd GeneCurator
conda create -n genecurator -c conda-forge python=3.12 graphviz
conda activate genecurator
pip install -r requirements.txt
```

Without conda, install Graphviz from <https://graphviz.org/download/>
(Linux: `sudo apt install graphviz`) and then, in a virtual environment,
`pip install -r requirements.txt`.

## Reference data

Included in the repository (`data/` and the project folder): the TCGA-BRCA
case study (expression matrix, sample groups, DEG table), the CellTalkDB human
ligand–receptor pairs, the STRING v12.0 interactions (combined score ≥ 700)
and the UniProt human reference proteome (UP000005640, release 2026_03_30).

The rest is downloaded once with one command:

```bash
python setup_data.py              # about 20 minutes, mostly the KEGG maps
python setup_data.py --skip-kegg  # without the KEGG pathway maps
```

| Data | Source |
|---|---|
| 15 gene-set libraries (KEGG, Reactome, WikiPathways, BioPlanet, MSigDB Hallmark, GO BP/MF/CC, DisGeNET, Disease/Drug Perturbations from GEO, PanglaoDB, DSigDB) | Enrichr |
| KEGG pathway maps (PNG + KGML) | KEGG REST API |
| IBM Plex Sans font | @fontsource (SIL Open Font License) |

Re-running the script skips files that are already present. The gene-set
libraries and the KEGG maps are downloaded from their providers rather than
redistributed, because some of their licences do not allow redistribution.
`download_ppi_assets.py` rebuilds the STRING snapshot from STRING itself.

## Running

```bash
python run.py
```

This starts GeneCurator on <http://127.0.0.1:8000> and opens the browser.
To serve it to other computers (e.g. on a server):

```bash
HOST=0.0.0.0 PORT=8020 python run.py
```

Then follow the four-step workflow:

1. **Load** an expression matrix and a sample-metadata table, or the TCGA-BRCA case study.
2. **Select** a feature-selection method to produce a ranked panel.
3. **Curate** the panel - remove or reinstate genes with supporting evidence.
4. **Analyse** - the downstream results recompute on the current panel.

The optional AI assistant needs a Gemini API key: copy `.env.example` to
`.env` and fill in `GEMINI_API_KEY`. The `.env` file is never committed.

## Input format

- **Expression matrix**: rows are samples, columns are genes (CSV or TSV).
- **Sample metadata**: assigns each sample to a group, with identifiers
  matching the expression matrix.

## Data sources

The case study uses publicly available data from The Cancer Genome Atlas
Breast Invasive Carcinoma (TCGA-BRCA) project, obtained through the NCI Genomic
Data Commons Data Portal (https://portal.gdc.cancer.gov).
`data/prepare_brca_deg.R` reproduces the case-study files from the GDC
download.

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
