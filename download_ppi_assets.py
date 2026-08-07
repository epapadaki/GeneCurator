#!/usr/bin/env python3
"""
Build a local STRING protein-protein interaction snapshot for GeneCurator.

Downloads two STRING v12.0 files for human (taxon 9606):
  * protein.info  — maps STRING protein IDs to gene symbols (preferred_name)
  * protein.links — undirected interactions with a combined_score (0–1000)

Filters links by confidence, maps both endpoints to gene symbols, deduplicates
the undirected pairs, and writes:

    data/ppi_edges.tsv     (geneA <TAB> geneB <TAB> score)

which backend/ppi.py loads at runtime. Run this once from the project root:

    python download_ppi_assets.py                 # default score >= 700
    python download_ppi_assets.py --min-score 900 # leaner, highest-confidence

Notes
-----
* score >= 700 is STRING's "high confidence" cut-off (recommended default).
  >= 900 ("highest confidence") produces a much smaller file and uses far
  less memory at load time — pick it if RAM is tight.
* The download is a few hundred MB compressed; the final TSV is far smaller.
* No API key required. STRING data is CC BY 4.0 — cite Szklarczyk et al.,
  Nucleic Acids Res, when you use it in the paper.
"""
from __future__ import annotations

import argparse
import gzip
import sys
import urllib.request
from pathlib import Path

STRING_VERSION = "12.0"
TAXON = "9606"  # Homo sapiens
BASE = "https://stringdb-downloads.org/download"
INFO_URL = f"{BASE}/protein.info.v{STRING_VERSION}/{TAXON}.protein.info.v{STRING_VERSION}.txt.gz"
LINKS_URL = f"{BASE}/protein.links.v{STRING_VERSION}/{TAXON}.protein.links.v{STRING_VERSION}.txt.gz"

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = OUT_DIR / "ppi_edges.tsv"


def _stream(url: str):
    """Yield decoded text lines from a gzipped remote file, streaming."""
    print(f"  → {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "GeneCurator/ppi-builder"})
    with urllib.request.urlopen(req) as resp, gzip.open(resp, "rt", encoding="utf-8") as fh:
        for line in fh:
            yield line


def build_id_map() -> dict:
    """STRING protein ID -> gene symbol (preferred_name)."""
    print("Downloading protein.info (ID → symbol map)...")
    id2sym: dict[str, str] = {}
    it = _stream(INFO_URL)
    next(it, None)  # header
    for line in it:
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 2:
            id2sym[parts[0]] = parts[1]
    print(f"  mapped {len(id2sym):,} proteins")
    return id2sym


def build_edges(id2sym: dict, min_score: int) -> int:
    """Stream links, filter, map to symbols, dedupe, write TSV. Returns edge count."""
    print(f"Downloading protein.links and filtering (combined_score >= {min_score})...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, str]] = set()
    n_written = 0

    it = _stream(LINKS_URL)
    next(it, None)  # header: "protein1 protein2 combined_score"
    with open(OUT_PATH, "w", encoding="utf-8") as out:
        out.write("geneA\tgeneB\tscore\n")
        for i, line in enumerate(it):
            p1, p2, score = line.split()
            s = int(score)
            if s < min_score:
                continue
            a = id2sym.get(p1)
            b = id2sym.get(p2)
            if not a or not b or a == b:
                continue
            key = (a, b) if a < b else (b, a)
            if key in seen:
                continue  # links file lists each undirected pair twice
            seen.add(key)
            out.write(f"{key[0]}\t{key[1]}\t{s}\n")
            n_written += 1
            if i and i % 2_000_000 == 0:
                print(f"    scanned {i:,} links, kept {n_written:,} edges...")
    return n_written


def main() -> int:
    ap = argparse.ArgumentParser(description="Build local STRING PPI snapshot.")
    ap.add_argument("--min-score", type=int, default=700,
                    help="STRING combined_score cut-off (0–1000). Default 700.")
    args = ap.parse_args()

    if not (0 <= args.min_score <= 1000):
        print("min-score must be between 0 and 1000", file=sys.stderr)
        return 2

    try:
        id2sym = build_id_map()
        n = build_edges(id2sym, args.min_score)
    except Exception as e:  # network / parse failure
        print(f"\nFailed: {type(e).__name__}: {e}", file=sys.stderr)
        print("Check your connection; STRING downloads can be large and occasionally "
              "rate-limit. You can rerun this script safely.", file=sys.stderr)
        return 1

    print(f"\nDone. Wrote {n:,} edges to {OUT_PATH}")
    print("Restart the app — the PPI network will now use STRING for any panel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
