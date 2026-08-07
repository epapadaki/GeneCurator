"""UniProt lookup. Loads the proteome TSV once on first call."""

import os
import re
from functools import lru_cache
from typing import Optional

import pandas as pd


UNIPROT_PATH = "uniprotkb_proteome_UP000005640_2026_03_30.tsv"


@lru_cache(maxsize=1)
def _load_uniprot() -> Optional[pd.DataFrame]:
    if not os.path.exists(UNIPROT_PATH):
        return None
    return pd.read_csv(UNIPROT_PATH, sep="\t").astype(str)


def lookup(gene_symbol: str) -> Optional[dict]:
    """Returns dict with protein_name/length/function/disease or None."""
    db = _load_uniprot()
    if db is None or not gene_symbol:
        return None
    match = db[db["Gene Names"].str.contains(
        fr"\b{re.escape(str(gene_symbol))}\b", na=False, case=False
    )]
    if match.empty:
        return None
    row = match.iloc[0]
    return {
        "protein_name": (str(row.get("Protein names", "")).replace("nan", "").strip() or "—"),
        "length": (str(row.get("Length", "")).replace("nan", "").strip() or "—"),
        "function": (str(row.get("Function [CC]", "")).replace("nan", "").strip()
                     or "No data available."),
        "disease": (str(row.get("Involvement in disease", "")).replace("nan", "").strip()
                    or "No clinical record."),
        "records": int(len(match)),
    }


def is_loaded() -> bool:
    return _load_uniprot() is not None
