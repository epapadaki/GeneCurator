"""
Protein-protein interaction lookup from a local STRING snapshot.

Loads data/ppi_edges.tsv (built once by download_ppi_assets.py) with columns:
    geneA <TAB> geneB <TAB> score
Edges are undirected; gene symbols are upper-cased. The file is already
filtered to a confidence threshold at build time, but `subnetwork` also
accepts a runtime `min_score` for further tightening.

If the file is missing, load_edges() returns None so the caller can fall
back to its built-in curated interaction list rather than crashing.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

# Project root is one level up from backend/, same convention as the other
# local databases (lr_annotation.py, uniprot_local.py).
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "ppi_edges.tsv"


@lru_cache(maxsize=1)
def load_edges() -> Optional[dict]:
    """
    Load the edge list into an adjacency map: symbol -> {neighbor: score, ...}.
    Cached for the process lifetime. Returns None if the file is absent.
    """
    if not DATA_PATH.exists():
        return None

    adj: dict[str, dict[str, int]] = {}
    with open(DATA_PATH, "r", encoding="utf-8") as fh:
        header = fh.readline()  # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            a = parts[0].strip().upper()
            b = parts[1].strip().upper()
            if not a or not b or a == b:
                continue
            try:
                score = int(parts[2]) if len(parts) >= 3 and parts[2] else 0
            except ValueError:
                score = 0
            adj.setdefault(a, {})[b] = score
            adj.setdefault(b, {})[a] = score
    return adj


def is_available() -> bool:
    """Whether the local STRING snapshot was found."""
    return load_edges() is not None


def subnetwork(panel, min_score: int = 0, max_edges: int = 400):
    """
    Return undirected edges (a, b) where BOTH endpoints are in `panel`.

    Parameters
    ----------
    panel : iterable of gene symbols (case-insensitive)
    min_score : keep only edges with confidence >= this value
    max_edges : safety cap for very connected panels

    Returns None if the local DB is unavailable, else a list of (A, B) tuples
    with upper-cased symbols, deduplicated.
    """
    adj = load_edges()
    if not adj:
        return None

    P = {str(g).upper() for g in panel if g}
    edges = []
    seen = set()
    for a in P:
        for b, score in adj.get(a, {}).items():
            if b in P and score >= min_score:
                key = (a, b) if a < b else (b, a)
                if key not in seen:
                    seen.add(key)
                    edges.append(key)
                    if len(edges) >= max_edges:
                        return edges
    return edges


def stats() -> dict:
    """Diagnostics for /api/state."""
    adj = load_edges()
    if not adj:
        return {"available": False, "n_nodes": 0, "n_edges": 0}
    n_edges = sum(len(v) for v in adj.values()) // 2
    return {"available": True, "n_nodes": len(adj), "n_edges": n_edges}
