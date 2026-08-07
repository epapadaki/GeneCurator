"""
Download all human KEGG pathway PNG maps and KGML XML files from KEGG REST API.

Run this ONCE from your project folder:

    python download_kegg_assets.py

It creates a folder `kegg_data/` with two subfolders:
    kegg_data/png/   — pathway PNG images
    kegg_data/kgml/  — KGML XML with gene positions

Takes ~15-20 minutes. Respects KEGG's 3-req/sec rate limit.
Safe to re-run: skips files that already exist. Ctrl+C is fine, it resumes.

Cites:
    Kanehisa, M. and Goto, S.; KEGG: Kyoto Encyclopedia of Genes and Genomes.
    Nucleic Acids Res. 28, 27-30 (2000).
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


BASE = "https://rest.kegg.jp"
ORG = "hsa"                # Homo sapiens
OUT = Path("kegg_data")
DELAY = 0.4                # ~2.5 req/sec, well under KEGG's 3/sec limit
UA = "GeneCurator-downloader/1.0 (academic use)"


def fetch(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            req = Request(url, headers={"User-Agent": UA})
            with urlopen(req, timeout=30) as r:
                return r.read()
        except HTTPError as e:
            if e.code in (429, 503):
                wait = 2 ** attempt
                print(f"    rate-limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if e.code == 404:
                return b""     # not-found is not fatal, some pathways lack PNG/KGML
            raise
        except URLError as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return b""


def list_pathways() -> list[tuple[str, str]]:
    """Returns [(pathway_id, name), ...] for all human pathways."""
    data = fetch(f"{BASE}/list/pathway/{ORG}").decode("utf-8", errors="replace")
    out = []
    for line in data.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            pid = parts[0].strip()
            # Sometimes prefixed 'path:', sometimes not
            if pid.startswith("path:"):
                pid = pid[5:]
            name = parts[1].strip()
            # Trim "- Homo sapiens (human)" suffix that KEGG appends
            for suf in [" - Homo sapiens (human)", " - Homo sapiens"]:
                if name.endswith(suf):
                    name = name[:-len(suf)]
            out.append((pid, name))
    return out


def main() -> int:
    OUT.mkdir(exist_ok=True)
    (OUT / "png").mkdir(exist_ok=True)
    (OUT / "kgml").mkdir(exist_ok=True)

    print(f"Fetching pathway list from {BASE}/list/pathway/{ORG}...")
    try:
        pathways = list_pathways()
    except Exception as e:
        print(f"ERROR: could not fetch pathway list: {e}")
        print("Check your internet connection and try again.")
        return 1
    print(f"  → {len(pathways)} human pathways found\n")

    manifest = {
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
        "source": "KEGG REST API (rest.kegg.jp)",
        "organism": ORG,
        "citation": ("Kanehisa M, Goto S. KEGG: Kyoto Encyclopedia of Genes "
                     "and Genomes. Nucleic Acids Res. 28, 27-30 (2000)."),
        "pathways": {},
    }

    total = len(pathways)
    for i, (pid, name) in enumerate(pathways, 1):
        png_path = OUT / "png" / f"{pid}.png"
        kgml_path = OUT / "kgml" / f"{pid}.xml"

        # Skip if both already present (resumable)
        if png_path.exists() and kgml_path.exists():
            manifest["pathways"][pid] = {"name": name, "png": True, "kgml": True}
            if i % 20 == 0:
                print(f"  [{i}/{total}] {pid} — cached")
            continue

        print(f"  [{i}/{total}] {pid} — {name[:50]}")

        # PNG
        if not png_path.exists():
            try:
                png = fetch(f"{BASE}/get/{pid}/image")
                if png:
                    png_path.write_bytes(png)
                time.sleep(DELAY)
            except Exception as e:
                print(f"    PNG failed: {e}")

        # KGML
        if not kgml_path.exists():
            try:
                kgml = fetch(f"{BASE}/get/{pid}/kgml")
                if kgml:
                    kgml_path.write_bytes(kgml)
                time.sleep(DELAY)
            except Exception as e:
                print(f"    KGML failed: {e}")

        manifest["pathways"][pid] = {
            "name": name,
            "png":  png_path.exists(),
            "kgml": kgml_path.exists(),
        }

        # Save manifest periodically so a Ctrl+C doesn't lose progress info
        if i % 25 == 0:
            (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nDONE. Snapshot dated {manifest['downloaded_at']}")
    print(f"     PNG maps:  {sum(1 for p in manifest['pathways'].values() if p['png'])}/{total}")
    print(f"     KGML XML:  {sum(1 for p in manifest['pathways'].values() if p['kgml'])}/{total}")
    print(f"     Total size on disk: {sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()) // (1024*1024)} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
