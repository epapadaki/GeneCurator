"""
Per-session state. Replaces Streamlit's st.session_state.

For Q1=local single-user, in-memory storage keyed by session ID (set via cookie)
is sufficient. The Session dataclass holds exactly the fields the original
Streamlit app's st.session_state held.
"""

import threading
import time
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
# Each session can hold a full expression matrix, so on a shared server idle
# sessions are dropped after SESSION_TTL seconds and at most MAX_SESSIONS are
# kept (empty ones, then the least recently used, go first). The browser cookie lasts 24 h.
SESSION_TTL = 24 * 60 * 60
MAX_SESSIONS = 200

_SESSIONS: Dict[str, Session] = {}
_LAST_SEEN: Dict[str, float] = {}
_LOCK = threading.Lock()


def _evict(now: float) -> None:
    for sid in [s for s, t in _LAST_SEEN.items() if now - t > SESSION_TTL]:
        _SESSIONS.pop(sid, None)
        _LAST_SEEN.pop(sid, None)
    while len(_SESSIONS) > MAX_SESSIONS:
        # Drop sessions that never loaded data first (e.g. visits by crawlers),
        # so they cannot push out someone who is in the middle of an analysis.
        empty = [s for s in _SESSIONS if _SESSIONS[s].df_raw is None and _SESSIONS[s].expr_matrix is None]
        pool = empty or list(_SESSIONS)
        oldest = min(pool, key=lambda s: _LAST_SEEN.get(s, 0))
        _SESSIONS.pop(oldest, None)
        _LAST_SEEN.pop(oldest, None)


def get_or_create(session_id: Optional[str]) -> tuple[str, Session]:
    """Return (session_id, Session). Creates a new session if needed."""
    now = time.time()
    with _LOCK:
        if session_id and session_id in _SESSIONS:
            _LAST_SEEN[session_id] = now
            return session_id, _SESSIONS[session_id]
        new_id = str(uuid4())
        _SESSIONS[new_id] = Session()
        _LAST_SEEN[new_id] = now
        _evict(now)
        return new_id, _SESSIONS[new_id]


def reset(session_id: str) -> None:
    """Drop a session — used by the 'New analysis' / restart action."""
    with _LOCK:
        _SESSIONS.pop(session_id, None)
        _LAST_SEEN.pop(session_id, None)
