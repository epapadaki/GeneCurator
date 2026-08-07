"""
Serve KEGG pathway maps with user gene highlighting.

Reads locally cached PNG + KGML files (created by download_kegg_assets.py)
and produces an SVG that layers a color-coded highlight rectangle over each
of the user's genes in the pathway.

Log2FC-based coloring:
    strongly up   → red     (log2FC ≥ +2)
    weakly up     → light red
    weakly down   → light blue
    strongly down → blue    (log2FC ≤ -2)

Genes NOT in the user's list get a subtle grey border. Genes IN the list get
a coloured semi-transparent overlay + coloured border.
"""

import base64
import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Data folder relative to project root (the download script writes here).
DATA_DIR = Path("kegg_data")


# ---- Manifest ------------------------------------------------------------
@lru_cache(maxsize=1)
def _manifest(data_dir: Optional[Path] = None) -> dict:
    d = Path(data_dir) if data_dir else DATA_DIR
    mp = d / "manifest.json"
    if not mp.exists():
        return {}
    try:
        return json.loads(mp.read_text())
    except Exception:
        return {}


def available(data_dir: Optional[Path] = None) -> bool:
    return _manifest(data_dir).get("pathways") is not None


def snapshot_date(data_dir: Optional[Path] = None) -> Optional[str]:
    return _manifest(data_dir).get("downloaded_at")


# ---- KEGG pathway ID from name ------------------------------------------
def find_pathway_id(term_name: str, data_dir: Optional[Path] = None) -> Optional[str]:
    """Given a KEGG library term (e.g. 'HEPATITIS B'), return 'hsa05161'.

    Enrichr KEGG library names are stored uppercase; KEGG's own names are
    mixed case. We do a case-insensitive match, ignoring punctuation.
    """
    def norm(s: str) -> str:
        return "".join(ch for ch in s.lower() if ch.isalnum())

    target = norm(term_name)
    manifest = _manifest(data_dir)
    for pid, meta in manifest.get("pathways", {}).items():
        if norm(meta.get("name", "")) == target:
            return pid
    # try partial (some Enrichr terms are truncated)
    for pid, meta in manifest.get("pathways", {}).items():
        n = norm(meta.get("name", ""))
        if target and (target in n or n in target):
            return pid
    return None


# ---- KGML parsing --------------------------------------------------------
@lru_cache(maxsize=32)
def _parse_kgml(kgml_path: str) -> List[Dict]:
    """Return list of gene-boxes: [{'genes': [...], 'x','y','width','height','label'}, ...]

    Each <entry type="gene"> in KGML has a <graphics> child with pixel
    coordinates matching the PNG image.
    """
    try:
        tree = ET.parse(kgml_path)
    except Exception:
        return []
    root = tree.getroot()
    out = []
    for entry in root.findall("entry"):
        if entry.get("type") != "gene":
            continue
        graphics = entry.find("graphics")
        if graphics is None:
            continue
        # `name` in the entry looks like "hsa:3454 hsa:3455 hsa:3717"
        # `name` in the graphics is the display label like "IFNAR1..."
        entry_names = entry.get("name", "").split()
        display = (graphics.get("name") or "").strip()
        # Display label may be "IFNAR1, JAK1, STAT1..." for merged nodes.
        gene_symbols = [
            g.strip().rstrip("...").upper()
            for g in display.replace(",", " ").split()
            if g.strip()
        ]
        try:
            x = int(graphics.get("x", "0"))
            y = int(graphics.get("y", "0"))
            w = int(graphics.get("width", "46"))
            h = int(graphics.get("height", "17"))
        except (TypeError, ValueError):
            continue
        out.append({
            "genes": gene_symbols,
            "kegg_ids": entry_names,
            "x": x, "y": y, "width": w, "height": h,
            "label": display,
        })
    return out


# ---- Rendering ----------------------------------------------------------
def _color_for_log2fc(log2fc: float) -> Tuple[str, str]:
    """Return (fill, stroke) colors for a gene at this log2FC.

    Colors span red (up) → white → blue (down), with intensity by |log2FC|.
    Semi-transparent fill so text is still readable through it.
    """
    # Clip to +/-3
    x = max(-3.0, min(3.0, log2fc))
    if x > 0:
        # red family: intensify with x
        intensity = min(1.0, x / 3.0)
        fill = f"rgba(220, 38, 38, {0.25 + 0.45 * intensity:.2f})"
        stroke = "#b91c1c"
    else:
        # blue family
        intensity = min(1.0, -x / 3.0)
        fill = f"rgba(37, 99, 235, {0.25 + 0.45 * intensity:.2f})"
        stroke = "#1d4ed8"
    return fill, stroke


def render_map(
    pathway_id: str,
    gene_log2fc: Dict[str, float],
    data_dir: Optional[Path] = None,
) -> dict:
    """Build an SVG showing the KEGG map with user genes highlighted.

    Args:
        pathway_id:   like 'hsa04630'
        gene_log2fc:  {'STAT1': 2.3, 'JAK1': 1.8, ...} — user's final panel

    Returns dict with keys:
        svg          str    — the composed SVG markup
        width, height int    — image dimensions
        highlighted  int    — number of gene-boxes with a hit
        total_boxes  int    — total gene-boxes in the pathway
        hit_genes    list   — user genes actually found in the map
    """
    d = Path(data_dir) if data_dir else DATA_DIR
    png_path = d / "png" / f"{pathway_id}.png"
    kgml_path = d / "kgml" / f"{pathway_id}.xml"

    if not png_path.exists():
        return {"error": f"PNG for {pathway_id} not found. "
                         f"Run download_kegg_assets.py to fetch it."}
    if not kgml_path.exists():
        return {"error": f"KGML for {pathway_id} not found. "
                         f"Run download_kegg_assets.py to fetch it."}

    # Base64 embed the PNG so the SVG is self-contained (no separate <img>).
    png_b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")

    # Determine image size from PNG header (chunkless — just read IHDR)
    header = png_path.read_bytes()[:24]
    if len(header) >= 24 and header[:8] == b"\x89PNG\r\n\x1a\n":
        width = int.from_bytes(header[16:20], "big")
        height = int.from_bytes(header[20:24], "big")
    else:
        width, height = 900, 700

    entries = _parse_kgml(str(kgml_path))

    # Uppercase the user genes for case-insensitive matching
    user_set_upper = {g.upper(): float(v) for g, v in gene_log2fc.items()}

    overlays = []
    hit_boxes = 0
    hit_genes: set = set()

    for e in entries:
        # KGML gene entries store box top-left coords for some releases and
        # box center for others. Modern KGML (post-2014) uses CENTER — we
        # convert to top-left.
        cx, cy = e["x"], e["y"]
        w, h = e["width"], e["height"]
        x0, y0 = cx - w // 2, cy - h // 2

        # Does any gene in this box match the user's list?
        matches = [g for g in e["genes"] if g in user_set_upper]
        if matches:
            hit_boxes += 1
            # Use the biggest-magnitude log2FC among matches for coloring
            best_fc = max((user_set_upper[g] for g in matches), key=abs)
            fill, stroke = _color_for_log2fc(best_fc)
            for g in matches:
                hit_genes.add(g)
            overlays.append(
                f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" '
                f'fill="{fill}" stroke="{stroke}" stroke-width="2" '
                f'rx="2" ry="2"><title>{", ".join(matches)} '
                f'(log2FC={best_fc:+.2f})</title></rect>'
            )
        else:
            # Faint outline on other gene boxes for orientation
            overlays.append(
                f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" '
                f'fill="none" stroke="#94a3b8" stroke-width="0.5" '
                f'stroke-opacity="0.4"/>'
            )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {width} {height}" '
        f'style="max-width:100%; height:auto; background:#fff;">'
        f'<image href="data:image/png;base64,{png_b64}" '
        f'x="0" y="0" width="{width}" height="{height}"/>'
        f'<g class="overlays">{"".join(overlays)}</g>'
        f'</svg>'
    )

    return {
        "svg": svg,
        "width": width,
        "height": height,
        "highlighted": hit_boxes,
        "total_boxes": len(entries),
        "hit_genes": sorted(hit_genes),
        "pathway_id": pathway_id,
    }


def list_pathways(data_dir: Optional[Path] = None) -> List[Dict]:
    """All pathways available in the local snapshot."""
    manifest = _manifest(data_dir)
    return [
        {"id": pid, "name": meta.get("name", pid),
         "png": meta.get("png", False), "kgml": meta.get("kgml", False)}
        for pid, meta in manifest.get("pathways", {}).items()
    ]
