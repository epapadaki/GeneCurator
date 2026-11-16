"""
Download the reference files that are not stored in the repository, in one step.

    python setup_data.py              # everything
    python setup_data.py --skip-kegg  # without the KEGG maps (~15 min, ~50 MB)

The case-study data, the ligand-receptor pairs, the STRING snapshot and the
UniProt reference are part of the repository. This script fetches the rest,
into the project folder:

  From Enrichr (the gene-set libraries are not redistributed here; some of
  them have licences that do not allow it)
    <library>.txt for every library listed in backend/enrichment.py

  From KEGG REST (download_kegg_assets.py; KEGG maps may not be redistributed)
    kegg_data/png, kegg_data/kgml

  From the @fontsource package (SIL Open Font License)
    frontend/static/css/fonts/ibm-plex-sans-latin-*-normal.woff2

Files that already exist are skipped, so the script can be re-run safely.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

ENRICHR_URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={}"
FONT_URL = ("https://cdn.jsdelivr.net/npm/@fontsource/ibm-plex-sans@5.0.20/files/"
            "ibm-plex-sans-latin-{}-normal.woff2")
UA = "GeneCurator-setup/1.0 (academic use)"

def fetch(url: str, retries: int = 3) -> bytes:
    for attempt in range(retries):
        try:
            with urlopen(Request(url, headers={"User-Agent": UA}), timeout=120) as r:
                return r.read()
        except (HTTPError, URLError) as e:
            if attempt == retries - 1:
                raise RuntimeError(f"Could not download {url}: {e}") from e
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def save(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(content)
    tmp.replace(path)


def enrichr_libraries() -> None:
    from backend.enrichment import LIBRARY_CATALOG
    libs = [lib_id for group in LIBRARY_CATALOG.values() for lib_id, _ in group]
    print(f"\n[1/3] {len(libs)} gene-set libraries from Enrichr")
    for lib in libs:
        dest = ROOT / f"{lib}.txt"
        if dest.exists():
            print(f"  {lib}: already present")
            continue
        print(f"  {lib}: downloading")
        content = fetch(ENRICHR_URL.format(quote(lib)))
        if not content.strip() or b"\t" not in content[:5000]:
            raise RuntimeError(f"Enrichr returned no data for {lib}; the library name may have changed.")
        save(dest, content)
        time.sleep(0.5)


def fonts() -> None:
    print("\n[2/3] Fonts (IBM Plex Sans)")
    for w in (400, 500, 600, 700):
        dest = ROOT / "frontend" / "static" / "css" / "fonts" / f"ibm-plex-sans-latin-{w}-normal.woff2"
        if dest.exists():
            continue
        save(dest, fetch(FONT_URL.format(w)))
    print("  done")


def kegg(skip: bool) -> None:
    print("\n[3/3] KEGG pathway maps")
    if skip:
        print("  skipped (--skip-kegg)")
        return
    subprocess.run([sys.executable, str(ROOT / "download_kegg_assets.py")], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-kegg", action="store_true", help="do not download the KEGG maps")
    args = ap.parse_args()
    enrichr_libraries()
    fonts()
    kegg(args.skip_kegg)
    print("\nAll reference data are in place. Start GeneCurator with: python run.py")


if __name__ == "__main__":
    main()
