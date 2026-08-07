"""
Plotly figure builders. Each function returns a dict that Plotly.js can render
directly via Plotly.newPlot(div, data, layout).

The chart definitions match the Streamlit app's exactly. Only the rendering
surface changed (was: st.plotly_chart; now: serialized JSON sent to Plotly.js).
"""

import glob
import os
import shutil
from typing import List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---- Graphviz auto-detection ---------------------------------------------
# The KEGG diagram needs the Graphviz `dot` executable. On Windows the
# installer often does NOT add itself to PATH, so we look in the standard
# install locations before giving up. This means a reviewer only has to
# *install* Graphviz — they don't have to fiddle with environment variables.
def _find_dot() -> Optional[str]:
    """Return a path to the Graphviz `dot` executable, or None if not found."""
    # 1. Already on PATH?
    on_path = shutil.which("dot")
    if on_path:
        return on_path
    # 2. Standard Windows install locations.
    candidates = []
    for base in (
        r"C:\Program Files\Graphviz",
        r"C:\Program Files (x86)\Graphviz",
        os.path.expanduser(r"~\AppData\Local\Programs\Graphviz"),
    ):
        candidates += glob.glob(os.path.join(base, "**", "dot.exe"), recursive=True)
    # 3. Standard macOS / Linux locations.
    for p in ("/usr/bin/dot", "/usr/local/bin/dot", "/opt/homebrew/bin/dot"):
        if os.path.exists(p):
            candidates.append(p)
    return candidates[0] if candidates else None


_DOT_PATH = _find_dot()
# If we found dot in a non-PATH location, prepend its directory to PATH so the
# `graphviz` Python package (which shells out to `dot`) can locate it.
if _DOT_PATH and not shutil.which("dot"):
    os.environ["PATH"] = os.path.dirname(_DOT_PATH) + os.pathsep + os.environ.get("PATH", "")


def graphviz_available() -> bool:
    return _DOT_PATH is not None


# ---- Shared style ---------------------------------------------------------
PLOTLY_LAYOUT = dict(
    font=dict(family="Inter, sans-serif", size=12, color="#1f2937"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    colorway=["#6366f1", "#ec4899", "#10b981", "#f59e0b", "#8b5cf6", "#0ea5e9", "#ef4444"],
    xaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb", linecolor="#d1d5db"),
    yaxis=dict(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb", linecolor="#d1d5db"),
    margin=dict(l=40, r=20, t=30, b=40),
)


def _decode_bdata(obj):
    """Recursively convert Plotly's compact {"dtype","bdata"} arrays back into
    plain Python lists. Recent Plotly emits this base64 form by default and
    Plotly.js in the browser does not reliably render it — so we decode it
    server-side and ship plain JSON arrays."""
    import base64
    if isinstance(obj, dict):
        # A typed-array node looks like {"dtype": "f8", "bdata": "....="}
        if set(obj.keys()) >= {"dtype", "bdata"} and isinstance(obj["bdata"], str):
            dtype_map = {
                "f8": np.float64, "f4": np.float32,
                "i8": np.int64, "i4": np.int32, "i2": np.int16, "i1": np.int8,
                "u8": np.uint64, "u4": np.uint32, "u2": np.uint16, "u1": np.uint8,
            }
            np_dtype = dtype_map.get(obj["dtype"])
            if np_dtype is not None:
                raw = base64.b64decode(obj["bdata"])
                arr = np.frombuffer(raw, dtype=np_dtype)
                # Plotly may also nest a 'shape' for 2-D; 1-D is the common case
                if "shape" in obj:
                    try:
                        shape = tuple(int(s) for s in str(obj["shape"]).split(","))
                        arr = arr.reshape(shape)
                    except Exception:
                        pass
                return arr.tolist()
        return {k: _decode_bdata(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_bdata(v) for v in obj]
    return obj


def _to_json_dict(fig) -> dict:
    """Serialize a Plotly figure to a dict the JS library can consume.

    Plotly encodes numpy arrays in a compact base64 form
    ({"dtype":..., "bdata":...}) which Plotly.js does not reliably render.
    We serialize, then walk the structure and decode every such array back
    into a plain JSON list so the browser always gets normal arrays.
    """
    fig.update_layout(**PLOTLY_LAYOUT)
    import json
    raw = json.loads(pio.to_json(fig))
    return _decode_bdata(raw)


# ---- 1. Volcano / Discovery Summary --------------------------------------
def volcano(df: pd.DataFrame, g: str, p: str, fc: str,
            algo_picks: List[str], final: List[str]) -> dict:
    vdf = df.copy()
    # Coerce BOTH axes to numeric. Uploaded CSVs frequently carry these as
    # strings (stray whitespace, locale formatting); without this the scatter
    # gets non-numeric x-values and renders an empty plot.
    vdf[fc] = pd.to_numeric(vdf[fc], errors="coerce")
    vdf["_p_num"] = pd.to_numeric(vdf[p], errors="coerce")
    vdf = vdf.dropna(subset=[fc, "_p_num"])
    if vdf.empty:
        return {"empty": True}
    vdf["-log10(p)"] = -np.log10(vdf["_p_num"].clip(lower=1e-300))
    vdf["status"] = "Other"
    vdf.loc[vdf[g].astype(str).isin([str(x) for x in algo_picks]), "status"] = "Algo pick"
    vdf.loc[vdf[g].astype(str).isin(final), "status"] = "Final (HITL)"
    fig = px.scatter(
        vdf, x=fc, y="-log10(p)", color="status", hover_name=g,
        color_discrete_map={"Other": "#cbd5e1", "Algo pick": "#0ea5e9", "Final (HITL)": "#6366f1"},
        opacity=0.75,
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    fig.update_layout(height=420,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return _to_json_dict(fig)


# ---- 2. PPI Network -------------------------------------------------------
# Fallback interaction set, used only when the local STRING snapshot
# (data/ppi_edges.tsv, built by download_ppi_assets.py) is not present.
# Kept so the embedded demo still shows a small network on a fresh checkout.
PPI_INTERACTIONS = [
    ("FMR1", "FXR1"), ("STAT1", "IFNG"), ("IL6", "STAT3"), ("TNF", "NFKB1"),
    ("ACE2", "TMPRSS2"), ("MX1", "OAS1"), ("IFIT1", "IFIT2"), ("CXCL10", "STAT1"),
    ("TP53", "MDM2"), ("PTEN", "AKT1"), ("APP", "ROCK1"),
]


def ppi_network(selected_genes: List[str], min_score: int = 700) -> dict:
    """Build a PPI subnetwork among the selected genes.

    Prefers a local STRING snapshot (data/ppi_edges.tsv). When that file is
    absent, falls back to the small curated PPI_INTERACTIONS set above.
    Matching is case-insensitive in both paths; original symbol casing is
    preserved for display.

    `min_score` applies only to the STRING path (STRING combined_score,
    0–1000; 700 = high confidence).
    """
    selected_genes = [str(x) for x in selected_genes]
    # upper-cased -> original casing, for display after case-insensitive match
    disp = {}
    for g in selected_genes:
        disp.setdefault(g.upper(), g)

    from . import ppi as ppi_db  # local STRING snapshot loader

    G = nx.Graph()
    source = "curated demo set"

    string_edges = ppi_db.subnetwork(selected_genes, min_score=min_score) \
        if ppi_db.is_available() else None

    if string_edges is not None:
        source = "STRING (local snapshot)"
        for a, b in string_edges:
            G.add_edge(disp.get(a, a), disp.get(b, b))
    else:
        sel_upper = set(disp.keys())
        for a, b in PPI_INTERACTIONS:
            if a.upper() in sel_upper and b.upper() in sel_upper:
                G.add_edge(disp.get(a.upper(), a), disp.get(b.upper(), b))

    edges = G.number_of_edges()
    if edges == 0:
        for n in selected_genes[:12]:
            G.add_node(n)
    if not G.nodes():
        return {"empty": True}

    pos = nx.spring_layout(G, k=0.6, seed=42)
    ex, ey = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        ex.extend([x0, x1, None])
        ey.extend([y0, y1, None])

    edge_trace = go.Scatter(x=ex, y=ey, line=dict(width=1.5, color="#cbd5e1"),
                            mode="lines", hoverinfo="none")
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes()],
        y=[pos[n][1] for n in G.nodes()],
        mode="markers+text",
        text=list(G.nodes()),
        textposition="top center",
        textfont=dict(family="Inter", size=12, color="#1f2937"),
        marker=dict(size=28, color="#6366f1", line=dict(width=2.5, color="white")),
    )
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            margin=dict(b=10, l=10, r=10, t=10),
            height=480,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        ),
    )
    return _to_json_dict(fig) | {"has_edges": edges > 0, "node_count": len(G.nodes()),
                                  "edge_count": edges, "source": source}


# ---- 3. KEGG Pathway (rendered as graphviz on server, returned as SVG) ---
def kegg_pathway_svg(selected_genes: List[str]) -> dict:
    """KEGG is a static schematic. Graphviz emits SVG which the page embeds.
    If Graphviz isn't installed, returns a clean {unavailable: ...} dict
    rather than throwing — the frontend shows a friendly notice."""
    if not graphviz_available():
        return {
            "unavailable": True,
            "reason": (
                "Graphviz is not installed. The KEGG pathway diagram needs the "
                "Graphviz system package. Install it (Windows: "
                "`winget install Graphviz.Graphviz`, macOS: `brew install graphviz`, "
                "Linux: `sudo apt-get install graphviz`) and restart the app."
            ),
        }

    import graphviz

    selected_genes = [str(x) for x in selected_genes]
    dot = graphviz.Digraph(graph_attr={
        "rankdir": "LR", "bgcolor": "transparent",
        "pad": "0.3", "nodesep": "0.5", "ranksep": "0.7",
    })
    dot.attr("node", shape="box", style="filled,rounded", fontname="Inter",
             fontsize="11", margin="0.2,0.12", penwidth="1.5")
    dot.attr("edge", color="#cbd5e1", penwidth="1.5", arrowsize="0.7")

    nodes = ["Cytokine_Receptor", "JAK/STAT", "IFN_Signaling", "Immune_Response", "Apoptosis"]
    edges = [
        ("Cytokine_Receptor", "JAK/STAT"),
        ("JAK/STAT", "IFN_Signaling"),
        ("IFN_Signaling", "Immune_Response"),
        ("IFN_Signaling", "Apoptosis"),
    ]
    hit_count = 0
    for n in nodes:
        hit = any(n.split("_")[0].lower() in gene.lower() for gene in selected_genes)
        if hit:
            hit_count += 1
        dot.node(
            n, n.replace("_", " "),
            fillcolor="#6366f1" if hit else "#f8fafc",
            fontcolor="white" if hit else "#6b7280",
            color="#6366f1" if hit else "#e5e7eb",
        )
    for a, b in edges:
        dot.edge(a, b)

    try:
        svg = dot.pipe(format="svg").decode("utf-8")
    except Exception as e:
        # dot was found but failed to execute for some other reason
        return {"unavailable": True, "reason": f"Graphviz failed to render: {e}"}

    return {"svg": svg, "hits": hit_count, "node_count": len(nodes)}


# ---- 4. PCA Space ---------------------------------------------------------
def pca_scatter(df: pd.DataFrame, g: str, p: str, fc: str, final: List[str]) -> dict:
    # Coerce both stat columns to numeric and drop rows that can't be parsed.
    work = df.copy()
    work["_p_num"] = pd.to_numeric(work[p], errors="coerce")
    work["_fc_num"] = pd.to_numeric(work[fc], errors="coerce")
    work = work.dropna(subset=["_p_num", "_fc_num"])
    if len(work) < 2:
        return {"empty": True}

    pca_input = work[["_p_num", "_fc_num"]]
    scaled = StandardScaler().fit_transform(pca_input)
    pca_res = PCA(2).fit_transform(scaled)
    gnames = work[g].astype(str).values

    pca_df = pd.DataFrame({
        "PC1": pca_res[:, 0],
        "PC2": pca_res[:, 1],
        "Gene": gnames,
        "Log2FC": work["_fc_num"].values,
        "Selected": pd.Series(gnames).isin(final).values,
    })

    fig = px.scatter(
        pca_df, x="PC1", y="PC2", color="Log2FC", hover_name="Gene",
        color_continuous_scale="RdBu_r", color_continuous_midpoint=0, opacity=0.7,
    )
    fig.update_traces(marker=dict(size=6, line=dict(width=0)))
    sel = pca_df[pca_df["Selected"]]
    if len(sel) > 0:
        fig.add_trace(go.Scatter(
            x=sel["PC1"], y=sel["PC2"], mode="markers", name="Selected",
            marker=dict(size=11, color="rgba(0,0,0,0)",
                        line=dict(width=2, color="#6366f1")),
            text=sel["Gene"],
            hovertemplate="<b>%{text}</b><br>PC1=%{x:.2f}<br>PC2=%{y:.2f}",
        ))
    fig.update_layout(height=520)
    return _to_json_dict(fig)


# ---- 5. Log2FC Profile ----------------------------------------------------
def log2fc_profile(df: pd.DataFrame, g: str, p: str, fc: str, final: List[str]) -> dict:
    fc_df = (
        df[df[g].astype(str).isin(final)][[g, fc, p]]
        .copy()
        .assign(fc_num=lambda d: pd.to_numeric(d[fc], errors="coerce"))
        .sort_values("fc_num")
        .head(50)
    )
    if len(fc_df) == 0:
        return {"empty": True}

    fig = px.bar(
        fc_df, x="fc_num", y=g, orientation="h",
        color="fc_num", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
    )
    fig.update_layout(
        height=max(420, 22 * len(fc_df)),
        yaxis=dict(title=""),
        xaxis=dict(title="log2 fold change"),
        coloraxis_colorbar=dict(title="log2FC"),
    )
    return _to_json_dict(fig)
