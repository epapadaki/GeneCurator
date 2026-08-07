"""
Per-session state. Replaces Streamlit's st.session_state.

For Q1=local single-user, in-memory storage keyed by session ID (set via cookie)
is sufficient. The Session dataclass holds exactly the fields the original
Streamlit app's st.session_state held.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from uuid import uuid4

import pandas as pd


@dataclass
class Session:
    # Wizard navigation
    step: int = 1

    # Data layer
    df_raw: Optional[pd.DataFrame] = None
    df_clean: Optional[pd.DataFrame] = None
    data_source: Optional[str] = None       # "upload" | "demo" | "case_study"
    uploaded_filename: Optional[str] = None

    # Which shape of input data this session has: "deg_table" means only a
    # gene/p-value/log2FC summary table exists (no per-sample info), so
    # only "Traditional DEG (Volcano)" is a valid strategy.
    # "expression_matrix" means a real samples x genes matrix + group
    # labels were uploaded, so the 5 model-based algorithms can genuinely
    # select features. See algorithms.py / pipeline.py / data.py.
    data_mode: str = "deg_table"

    # Expression-matrix mode only: samples (rows) x genes (columns).
    expr_matrix: Optional[pd.DataFrame] = None
    # sample_id -> group label, restricted to samples used in the comparison.
    sample_groups: Optional[Dict[str, str]] = None
    # The two group labels being compared (e.g. "tumor", "normal").
    group_a: Optional[str] = None
    group_b: Optional[str] = None

    # Column mapping (always populated: for expression_matrix mode these
    # are the fixed names produced by data.compute_deg_stats — "gene_symbol",
    # "p_value", "log2FC" — so all downstream plotting/export code that was
    # written against a DEG table keeps working unchanged.)
    g_col: Optional[str] = None
    p_col: Optional[str] = None
    fc_col: Optional[str] = None

    # Algorithm config
    strategy: str = "Random Forest (Ensemble)"
    top_n: int = 50
    p_thr: float = 0.05
    fc_thr: float = 1.0

    # Pipeline output
    selected_by_algo: Optional[List[str]] = None
    rank_scores: Optional[Dict[str, float]] = None
    global_impact: Optional[list] = None
    ran_pipeline: bool = False
    # One-line human-readable note on what the last pipeline run actually
    # did (e.g. "LASSO selected 42/18000 genes with non-zero coefficients").
    # None for Traditional DEG, which is self-explanatory from p_thr/fc_thr.
    selection_detail: Optional[str] = None

    # HITL state
    rejected_genes: Set[str] = field(default_factory=set)
    added_genes: Set[str] = field(default_factory=set)


# Module-level store — single-process, in-memory. One entry per browser session.
_SESSIONS: Dict[str, Session] = {}


def get_or_create(session_id: Optional[str]) -> tuple[str, Session]:
    """Return (session_id, Session). Creates a new session if needed."""
    if session_id and session_id in _SESSIONS:
        return session_id, _SESSIONS[session_id]
    new_id = str(uuid4())
    _SESSIONS[new_id] = Session()
    return new_id, _SESSIONS[new_id]


def reset(session_id: str) -> None:
    """Drop a session — used by the 'New analysis' / restart action."""
    _SESSIONS.pop(session_id, None)
