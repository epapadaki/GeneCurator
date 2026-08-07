"""
Local UniProt lookup.

Replaces the previous live UniProt API calls with a fast in-memory
lookup over a downloaded reference proteome file. Loaded once at
server startup and cached for the process lifetime.

Data source:
    UniProt human reference proteome (UP000005640)
    File: uniprotkb_proteome_UP000005640_*.tsv
    Columns:
        Entry, Entry Name, Gene Names, Developmental stage,
        Tissue specificity, Function [CC], Gene Ontology (biological
        process), Biotechnological use, Involvement in disease,
        Subcellular location [CC], KEGG

Design notes:
- Gene symbol matching handles the fact that "Gene Names" is a
  space-separated list of aliases; we index every alias to the same
  record so lookup by any symbol succeeds.
- Text fields are lightly cleaned: the UniProt "SUBCELLULAR LOCATION:"
  prefix and the "{ECO:...}" evidence codes are stripped so the text
  reads cleanly in a UI card.
- The loader is deliberately synchronous and simple. The reference
  file is ~42 MB and loads in well under a second.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


# ---- config ---------------------------------------------------------------

# Look for the file at the project root. If the project rearranges files
# later, this glob makes it trivial to move the file without editing code.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Cached path to the reference TSV — resolved lazily so tests can override.
_FILE_GLOB = "uniprotkb_proteome_UP000005640_*.tsv"


def _find_reference_file() -> Optional[Path]:
    """Locate the UniProt reference TSV at the project root."""
    matches = sorted(_PROJECT_ROOT.glob(_FILE_GLOB))
    return matches[-1] if matches else None


# ---- data class -----------------------------------------------------------

@dataclass
class UniProtRecord:
    """
    A single UniProt entry, cleaned and ready for display.

    All string fields are pre-cleaned (evidence codes stripped, subcellular
    location prefix removed). Missing fields are empty strings, never None.
    """
    entry: str            # UniProt accession (e.g. "P00533")
    entry_name: str       # UniProt entry name (e.g. "EGFR_HUMAN")
    gene_names: str       # space-separated list of symbols
    function: str
    disease: str
    subcellular_location: str
    tissue_specificity: str
    kegg: str             # semicolon-separated KEGG IDs


# ---- text cleanup ---------------------------------------------------------

# UniProt evidence codes look like "{ECO:0000255|HAMAP-Rule:MF_00161}" and
# appear inline. They're helpful for provenance but noisy in a UI card.
_EVIDENCE_RE = re.compile(r"\s*\{ECO:[^}]*\}")

# "FUNCTION: ..." and "SUBCELLULAR LOCATION: ..." prefixes are redundant
# when the card already has a labelled section.
_PREFIX_FUNCTION = re.compile(r"^FUNCTION:\s*", re.IGNORECASE)
_PREFIX_LOCATION = re.compile(r"^SUBCELLULAR LOCATION:\s*", re.IGNORECASE)
_PREFIX_TISSUE = re.compile(r"^TISSUE SPECIFICITY:\s*", re.IGNORECASE)
_PREFIX_DISEASE = re.compile(r"^DISEASE:\s*", re.IGNORECASE)


def _clean_text(s: str, strip_prefix: re.Pattern | None = None) -> str:
    """Remove evidence codes and optional leading section prefix."""
    if not s:
        return ""
    s = _EVIDENCE_RE.sub("", s)
    if strip_prefix is not None:
        s = strip_prefix.sub("", s)
    # UniProt sometimes wraps multiple sections in the same cell with
    # `--`; keep the first section, which is the most general.
    s = s.split("--")[0]
    # Collapse whitespace.
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ---- loader (cached at startup) -------------------------------------------

@lru_cache(maxsize=1)
def load_uniprot() -> dict[str, UniProtRecord]:
    """
    Load the reference TSV into a symbol -> record index.

    Every alias in "Gene Names" is indexed to the same record, so
    lookup by any symbol (primary or synonym) succeeds. Symbols are
    upper-cased to match TCGA / HGNC convention.

    Returns an empty dict if the reference file is missing — callers
    should treat this as "annotation unavailable" rather than crashing.
    """
    path = _find_reference_file()
    if path is None or not path.exists():
        return {}

    index: dict[str, UniProtRecord] = {}

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            gene_names = (row.get("Gene Names") or "").strip()
            if not gene_names:
                continue

            record = UniProtRecord(
                entry=(row.get("Entry") or "").strip(),
                entry_name=(row.get("Entry Name") or "").strip(),
                gene_names=gene_names,
                function=_clean_text(row.get("Function [CC]", ""), _PREFIX_FUNCTION),
                disease=_clean_text(row.get("Involvement in disease", ""), _PREFIX_DISEASE),
                subcellular_location=_clean_text(
                    row.get("Subcellular location [CC]", ""), _PREFIX_LOCATION
                ),
                tissue_specificity=_clean_text(
                    row.get("Tissue specificity", ""), _PREFIX_TISSUE
                ),
                kegg=(row.get("KEGG") or "").strip(),
            )

            # Index every symbol (upper-cased) to this record.
            for symbol in gene_names.split():
                # Prefer the first occurrence per symbol (matches primary
                # reviewed entries when duplicates exist).
                key = symbol.upper()
                if key not in index:
                    index[key] = record

    return index


# ---- public API -----------------------------------------------------------

def lookup(symbol: str) -> Optional[dict]:
    """
    Look up a gene symbol and return a UI-ready dict.

    The dict shape matches the fields the frontend Step 3 card expects.
    Returns None if the symbol is not in the reference file.
    """
    if not symbol:
        return None
    index = load_uniprot()
    if not index:
        return None

    record = index.get(symbol.upper())
    if record is None:
        return None

    return {
        "entry": record.entry,
        "entry_name": record.entry_name,
        "gene_names": record.gene_names,
        "function": record.function,
        "disease": record.disease,
        "subcellular_location": record.subcellular_location,
        "tissue_specificity": record.tissue_specificity,
        "kegg": record.kegg,
        # Direct link to UniProt entry page — useful for the researcher
        # who wants to drill deeper.
        "uniprot_url": (
            f"https://www.uniprot.org/uniprotkb/{record.entry}/entry"
            if record.entry else ""
        ),
        # Direct KEGG link, using the first KEGG ID if multiple.
        "kegg_url": (
            f"https://www.kegg.jp/entry/{record.kegg.split(';')[0].strip()}"
            if record.kegg else ""
        ),
    }


def is_available() -> bool:
    """Whether the reference file was found at startup."""
    return bool(load_uniprot())


def summary() -> dict:
    """Basic status info — useful for diagnostics on /api/state."""
    index = load_uniprot()
    return {
        "available": bool(index),
        "n_symbols_indexed": len(index),
    }
