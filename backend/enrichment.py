"""
Functional enrichment analysis using Enrichr gene set libraries.

Parses .txt files in the Enrichr GMT-like format:
    PATHWAY_NAME<TAB><TAB>GENE1<TAB>GENE2<TAB>...

For each pathway, runs Fisher's exact test against the user's gene list,
then applies Benjamini-Hochberg FDR correction.

Libraries are loaded lazily on first access and cached in memory.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scipy import stats
import numpy as np


# ---- Library catalog -----------------------------------------------------
# Group libraries by category so the UI can show subtabs.
LIBRARY_CATALOG = {
    "pathways": [
        ("KEGG_2026",               "KEGG 2026"),
        ("Reactome_Pathways_2024",  "Reactome 2024"),
        ("WikiPathways_2024_Human", "WikiPathways 2024"),
        ("BioPlanet_2019",          "BioPlanet"),
        ("MSigDB_Hallmark_2020",    "MSigDB Hallmark"),
    ],
    "ontology": [
        ("GO_Biological_Process_2026", "GO: Biological Process"),
        ("GO_Molecular_Function_2026", "GO: Molecular Function"),
        ("GO_Cellular_Component_2026", "GO: Cellular Component"),
    ],
    "diseases": [
        ("DisGeNET",                             "DisGeNET"),
        ("Disease_Perturbations_from_GEO_up",    "Disease Perturbations (Up)"),
        ("Disease_Perturbations_from_GEO_down",  "Disease Perturbations (Down)"),
    ],
    "cell_types": [
        ("PanglaoDB_Augmented_2021", "PanglaoDB"),
    ],
    "drugs": [
        ("DSigDB",                            "DSigDB"),
        ("Drug_Perturbations_from_GEO_up",    "Drug Perturbations (Up)"),
        ("Drug_Perturbations_from_GEO_down",  "Drug Perturbations (Down)"),
    ],
}


# Flatten the catalog for quick lookup
_ALL_LIBS = {lib_id: label for group in LIBRARY_CATALOG.values() for lib_id, label in group}


# ---- Loading -------------------------------------------------------------
_CACHE: Dict[str, Dict[str, List[str]]] = {}
_GENE_UNION_CACHE: Dict[str, set] = {}


def _library_path(lib_id: str, search_dir: Path) -> Optional[Path]:
    """Return the .txt file path for a library, or None if missing."""
    p = search_dir / f"{lib_id}.txt"
    return p if p.exists() else None


def load_library(lib_id: str, search_dir: Path) -> Dict[str, List[str]]:
    """Load a library from disk (cached).

    Returns: {pathway_name: [gene1, gene2, ...], ...}
    """
    if lib_id in _CACHE:
        return _CACHE[lib_id]

    path = _library_path(lib_id, search_dir)
    if path is None:
        _CACHE[lib_id] = {}
        return {}

    result: Dict[str, List[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            # Enrichr GMT: name, description (often empty), then genes.
            # Drop empty strings and dedup.
            genes = [g.strip().upper() for g in parts[1:] if g.strip()]
            genes = list(dict.fromkeys(genes))  # preserve order, unique
            if name and genes:
                result[name] = genes

    _CACHE[lib_id] = result
    return result


def library_gene_union(lib_id: str, search_dir: Path) -> set:
    """All genes that appear anywhere in the library. Used as the background
    universe for Fisher's exact test — this is what Enrichr / gseapy do."""
    if lib_id in _GENE_UNION_CACHE:
        return _GENE_UNION_CACHE[lib_id]
    lib = load_library(lib_id, search_dir)
    union = set()
    for genes in lib.values():
        union.update(genes)
    _GENE_UNION_CACHE[lib_id] = union
    return union


def available_libraries(search_dir: Path) -> Dict[str, bool]:
    """Which libraries from the catalog are actually present on disk."""
    return {lib_id: _library_path(lib_id, search_dir) is not None for lib_id in _ALL_LIBS}


# ---- Enrichment computation ---------------------------------------------
@dataclass
class EnrichmentRow:
    term: str
    p_value: float
    q_value: float          # BH-adjusted
    odds_ratio: float
    overlap_size: int
    term_size: int
    overlap_genes: List[str]

    def as_dict(self) -> dict:
        return {
            "term": self.term,
            "p_value": self.p_value,
            "q_value": self.q_value,
            "odds_ratio": self.odds_ratio,
            "overlap": self.overlap_size,
            "term_size": self.term_size,
            "combined_score": self._combined_score(),
            "overlap_genes": self.overlap_genes,
        }

    def _combined_score(self) -> float:
        """Enrichr's combined score: log(p) * z-score of odds ratio.
        We use a simplified but consistent version: -log10(p) * log2(odds_ratio+1).
        """
        p = max(self.p_value, 1e-300)
        or_term = np.log2(max(self.odds_ratio, 0) + 1)
        return float(-np.log10(p) * or_term)


def bh_correction(pvals: List[float]) -> List[float]:
    """Benjamini-Hochberg FDR."""
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    ranked = np.array(pvals)[order]
    q = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out.tolist()


def enrich(gene_list: List[str],
           lib_id: str,
           search_dir: Path,
           top_n: int = 50) -> List[EnrichmentRow]:
    """Fisher's exact enrichment of gene_list against every term in lib_id.

    Background universe = union of all genes across all terms in this library.
    This is standard practice in Enrichr and gseapy: we can only detect
    enrichment for genes we have annotations for.
    """
    lib = load_library(lib_id, search_dir)
    if not lib:
        return []

    universe = library_gene_union(lib_id, search_dir)
    query = {g.upper() for g in gene_list if g}
    query_in_universe = query & universe

    N = len(universe)                     # total genes in universe
    K = len(query_in_universe)            # user's genes that are in universe

    if N == 0 or K == 0:
        return []

    # For each term:
    #   n = size of term (in universe)
    #   k = overlap between user's genes and term
    #
    # 2x2 contingency table for Fisher's exact:
    #                  In term    Not in term    Total
    #   In query         k          K - k          K
    #   Not in query     n - k      N - K - n + k  N - K
    #   Total            n          N - n          N
    rows_raw = []
    for term, term_genes in lib.items():
        term_set = set(term_genes) & universe
        n = len(term_set)
        overlap = query_in_universe & term_set
        k = len(overlap)
        if k == 0:
            continue  # zero-overlap terms are unnecessary noise

        # Contingency table
        a = k
        b = K - k
        c = n - k
        d = N - K - n + k
        if a < 0 or b < 0 or c < 0 or d < 0:
            continue

        # scipy's fisher_exact uses (odds, p); alternative="greater" is
        # over-representation (one-sided), which is what pathway enrichment
        # standardly reports.
        try:
            odds_ratio, pval = stats.fisher_exact([[a, b], [c, d]],
                                                  alternative="greater")
        except Exception:
            continue

        rows_raw.append((term, pval, float(odds_ratio), k, n, sorted(overlap)))

    if not rows_raw:
        return []

    # BH correction
    pvals = [r[1] for r in rows_raw]
    qvals = bh_correction(pvals)

    enriched = [
        EnrichmentRow(
            term=r[0], p_value=r[1], q_value=q,
            odds_ratio=r[2], overlap_size=r[3], term_size=r[4],
            overlap_genes=r[5],
        )
        for r, q in zip(rows_raw, qvals)
    ]

    enriched.sort(key=lambda x: x.p_value)
    return enriched[:top_n]
