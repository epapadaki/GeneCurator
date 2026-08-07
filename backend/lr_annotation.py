"""
Ligand-Receptor annotation for a curated gene panel.

Uses CellTalkDB (Shao et al., Brief Bioinform 2020) via the LewisLabUCSD
mirror. Loaded from data/human_lr_pairs.tsv at server startup.

NOTE: This module provides ANNOTATION only. It does not infer cell-cell
communication, which requires single-cell resolution. It answers three
questions for a curated bulk gene panel:
  1. Which panel genes are known ligands or receptors?
  2. Which known LR pairs are present entirely inside the panel?
  3. Which LR pairs are only half-present, with the partner in the dataset
     but excluded from the panel (candidates for the researcher to add)?
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "human_lr_pairs.tsv"


# ---------- data structures --------------------------------------------------

@dataclass
class LRDatabase:
    """Parsed LR database with the lookups the annotation logic needs."""
    ligands: set[str]                        # all ligand symbols
    receptors: set[str]                      # all receptor symbols
    pairs: list[tuple[str, str, str]]        # (ligand, receptor, evidence)
    pairs_by_ligand: dict[str, list[str]]    # ligand -> [receptors]
    pairs_by_receptor: dict[str, list[str]]  # receptor -> [ligands]

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)


# ---------- loader (cached at startup) --------------------------------------

@lru_cache(maxsize=1)
def load_db() -> Optional[LRDatabase]:
    """
    Load the LR database from disk. Cached for the lifetime of the process.
    Returns None if the file is missing (feature will be reported as
    unavailable rather than crashing).
    """
    if not DATA_PATH.exists():
        return None

    ligands: set[str] = set()
    receptors: set[str] = set()
    pairs: list[tuple[str, str, str]] = []
    pairs_by_ligand: dict[str, list[str]] = {}
    pairs_by_receptor: dict[str, list[str]] = {}

    with open(DATA_PATH, "r", encoding="utf-8") as fh:
        header = fh.readline()  # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            lig = parts[0].strip().upper()
            rec = parts[1].strip().upper()
            ev = parts[2].strip() if len(parts) >= 3 else ""
            if not lig or not rec:
                continue
            ligands.add(lig)
            receptors.add(rec)
            pairs.append((lig, rec, ev))
            pairs_by_ligand.setdefault(lig, []).append(rec)
            pairs_by_receptor.setdefault(rec, []).append(lig)

    return LRDatabase(
        ligands=ligands,
        receptors=receptors,
        pairs=pairs,
        pairs_by_ligand=pairs_by_ligand,
        pairs_by_receptor=pairs_by_receptor,
    )


# ---------- classification helpers -----------------------------------------

def _classify(symbol: str, db: LRDatabase) -> str:
    """Return 'ligand', 'receptor', 'both', or 'neither'."""
    sym = symbol.upper()
    is_l = sym in db.ligands
    is_r = sym in db.receptors
    if is_l and is_r:
        return "both"
    if is_l:
        return "ligand"
    if is_r:
        return "receptor"
    return "neither"


# ---------- main annotation entrypoint --------------------------------------

def annotate_panel(
    panel: list[str],
    all_dataset_symbols: list[str],
    max_partial_per_side: int = 30,
) -> dict:
    """
    Annotate a curated gene panel against the LR database.

    Parameters
    ----------
    panel : list of gene symbols in the current final panel
    all_dataset_symbols : all symbols in the user's uploaded dataset
        Used to identify partial pairs — cases where one partner is in
        the panel and the other is present in the dataset but excluded.
    max_partial_per_side : cap on partial pairs shown per direction
        (ligand-in-panel and receptor-in-panel), to avoid overwhelming UI.

    Returns
    -------
    dict with keys:
        ok, error, n_ligands, n_receptors, n_both, n_complete_pairs,
        n_partial_ligand_side, n_partial_receptor_side,
        classifications (list of {symbol, role}),
        complete_pairs (list of {ligand, receptor, evidence, both_in_panel: True}),
        partial_ligand_side (list of {ligand, receptor, evidence, in_dataset}),
        partial_receptor_side (list of {receptor, ligand, evidence, in_dataset})
    """
    db = load_db()
    if db is None:
        return {
            "ok": False,
            "error": (
                "LR database not available: data/human_lr_pairs.tsv not found. "
                "See data/human_lr_pairs.README.md for setup."
            ),
        }

    panel_set = {s.upper() for s in panel}
    dataset_set = {s.upper() for s in all_dataset_symbols}

    # 1. Panel classification — only include genes with a known role
    classifications = []
    n_ligands = n_receptors = n_both = 0
    for sym in sorted(panel_set):
        role = _classify(sym, db)
        if role != "neither":
            classifications.append({"symbol": sym, "role": role})
            if role == "ligand":
                n_ligands += 1
            elif role == "receptor":
                n_receptors += 1
            else:
                n_both += 1

    # 2. Complete pairs — both ligand and receptor in panel
    complete_pairs = []
    seen = set()
    for lig, rec, ev in db.pairs:
        if lig in panel_set and rec in panel_set:
            key = (lig, rec)
            if key in seen:
                continue
            seen.add(key)
            complete_pairs.append({
                "ligand": lig,
                "receptor": rec,
                "evidence": ev,
            })

    # 3. Partial pairs — ligand in panel, receptor in dataset but not panel
    partial_ligand_side = []
    seen_pl = set()
    for lig, rec, ev in db.pairs:
        if lig in panel_set and rec not in panel_set and rec in dataset_set:
            key = (lig, rec)
            if key in seen_pl:
                continue
            seen_pl.add(key)
            partial_ligand_side.append({
                "ligand": lig,
                "receptor": rec,
                "evidence": ev,
            })

    # 4. Partial pairs — receptor in panel, ligand in dataset but not panel
    partial_receptor_side = []
    seen_pr = set()
    for lig, rec, ev in db.pairs:
        if rec in panel_set and lig not in panel_set and lig in dataset_set:
            key = (lig, rec)
            if key in seen_pr:
                continue
            seen_pr.add(key)
            partial_receptor_side.append({
                "ligand": lig,
                "receptor": rec,
                "evidence": ev,
            })

    # Sort partials by symbol for stable output
    partial_ligand_side.sort(key=lambda x: (x["ligand"], x["receptor"]))
    partial_receptor_side.sort(key=lambda x: (x["receptor"], x["ligand"]))

    # Total before capping (for the summary stats)
    n_partial_l_total = len(partial_ligand_side)
    n_partial_r_total = len(partial_receptor_side)

    # Cap for UI performance / readability
    partial_ligand_side = partial_ligand_side[:max_partial_per_side]
    partial_receptor_side = partial_receptor_side[:max_partial_per_side]

    return {
        "ok": True,
        "error": None,
        "n_ligands": n_ligands,
        "n_receptors": n_receptors,
        "n_both": n_both,
        "n_complete_pairs": len(complete_pairs),
        "n_partial_ligand_side_total": n_partial_l_total,
        "n_partial_receptor_side_total": n_partial_r_total,
        "database_info": {
            "source": "CellTalkDB (Shao et al., Brief Bioinform 2020)",
            "n_pairs_total": db.n_pairs,
            "n_ligands_total": len(db.ligands),
            "n_receptors_total": len(db.receptors),
        },
        "classifications": classifications,
        "complete_pairs": complete_pairs,
        "partial_ligand_side": partial_ligand_side,
        "partial_receptor_side": partial_receptor_side,
    }
