"""
Pipeline orchestration.

Branches on `session.data_mode`:

  "deg_table"          -> only "Traditional DEG (Volcano)" is valid. Same
                          threshold logic as before, operating directly on
                          the uploaded DEG summary table.

  "expression_matrix"  -> "Traditional DEG (Volcano)" computes DEG stats
                          (log2FC/p-value via Welch's t-test) from the raw
                          matrix first, then thresholds as usual. The other
                          five strategies run genuine feature selection via
                          algorithms.select_features(), treating genes as
                          columns/features and samples as rows.

In both cases `session.df_clean` ends up as a (gene_symbol, p_value,
log2FC) table so every downstream consumer (plots, export, gene lookup,
enrichment) keeps working unchanged regardless of which mode produced it.
"""

from typing import Dict, List, Set

import numpy as np
import pandas as pd

from .algorithms import REQUIRES_EXPRESSION_MATRIX, select_features
from .data import compute_deg_stats


class PipelineError(Exception):
    """Raised for pipeline-configuration problems the caller (main.py)
    should surface to the user as a 400, rather than a 500 crash."""


def _run_deg_table_mode(session) -> None:
    df = session.df_raw.copy()
    g, p, fc = session.g_col, session.p_col, session.fc_col
    df[p] = pd.to_numeric(df[p], errors="coerce")
    df[fc] = pd.to_numeric(df[fc], errors="coerce")
    df = df.dropna(subset=[p, fc, g]).reset_index(drop=True)
    df[g] = df[g].astype(str)
    session.df_clean = df

    if session.strategy != "Traditional DEG (Volcano)":
        raise PipelineError(
            f"{session.strategy!r} requires an expression matrix with "
            "per-sample data to genuinely select features. Only "
            "'Traditional DEG (Volcano)' is available for a DEG summary "
            "table upload. Upload an expression matrix + sample metadata "
            "in Step 1 to unlock the other algorithms."
        )

    _apply_volcano_threshold(session, df, g, p, fc)


def _apply_volcano_threshold(session, df: pd.DataFrame, g: str, p: str, fc: str) -> None:
    p_thr = session.p_thr
    fc_thr = session.fc_thr
    df = df.copy()
    # Prefer the Benjamini-Hochberg adjusted p-value ("padj") for the
    # threshold, so the selection corrects for multiple testing. This
    # column is present when stats are computed from an expression matrix.
    # For a legacy DEG-table upload that has no padj, fall back to the
    # raw p-value column.
    sig_col = "padj" if "padj" in df.columns else p
    df["DEG_Status"] = "NS"
    df.loc[(df[sig_col] < p_thr) & (df[fc] > fc_thr), "DEG_Status"] = "UP"
    df.loc[(df[sig_col] < p_thr) & (df[fc] < -fc_thr), "DEG_Status"] = "DOWN"
    sig = df[df["DEG_Status"] != "NS"].copy()
    sig["_rank_score"] = -np.log10(sig[sig_col].clip(lower=1e-300))
    sel = (
        sig.sort_values("_rank_score", ascending=False)
           .head(session.top_n)[g]
           .astype(str)
           .tolist()
    )
    scores = {
        str(row[g]): -float(np.log10(max(row[sig_col], 1e-300)))
        for _, row in df.iterrows()
    }
    session.selected_by_algo = sel
    session.rank_scores = scores
    session.global_impact = None


def _run_expression_matrix_mode(session) -> None:
    if session.expr_matrix is None or session.sample_groups is None:
        raise PipelineError(
            "No expression matrix loaded for this session. Upload an "
            "expression matrix and sample metadata in Step 1."
        )
    if not session.group_a or not session.group_b:
        raise PipelineError("Group comparison (group_a/group_b) is not set.")

    expr = session.expr_matrix
    groups = session.sample_groups  # sample_id -> label, already restricted to the 2 groups
    samples = [s for s in expr.index if s in groups]
    if len(samples) < 4:
        raise PipelineError(
            "Not enough samples with group labels to run feature selection "
            "(need at least 2 per group)."
        )
    expr_aligned = expr.loc[samples]
    y_labels = np.array([groups[s] for s in samples])
    gene_names = list(expr_aligned.columns.astype(str))

    # Always compute honest DEG stats from the matrix — used for plots,
    # export, and (for Traditional DEG) the threshold itself.
    deg_df = compute_deg_stats(
        expr_aligned,
        pd.Series(y_labels, index=expr_aligned.index),
        session.group_a,
        session.group_b,
    )
    session.g_col, session.p_col, session.fc_col = "gene_symbol", "p_value", "log2FC"
    session.df_clean = deg_df

    if session.strategy == "Traditional DEG (Volcano)":
        _apply_volcano_threshold(session, deg_df, "gene_symbol", "p_value", "log2FC")
        return

    if session.strategy not in REQUIRES_EXPRESSION_MATRIX:
        raise PipelineError(f"Unknown strategy: {session.strategy!r}")

    result = select_features(
        X=expr_aligned.values,
        y_labels=y_labels,
        gene_names=gene_names,
        group_a=session.group_a,
        group_b=session.group_b,
        strategy=session.strategy,
    )

    # Respect top_n as a display cap, but never invent genes beyond what
    # the algorithm actually judged non-trivially important: for sparse
    # methods (LASSO/Elastic Net/SVM-RFE) cap at whichever is smaller —
    # the algorithm's own selection size or top_n — instead of padding the
    # panel out with zero-weight genes just to hit a round number.
    n_take = min(session.top_n, result.n_nonzero) if result.n_nonzero > 0 else session.top_n
    n_take = max(n_take, 1)
    session.selected_by_algo = result.gene_order[:n_take]
    session.rank_scores = result.scores
    session.global_impact = None
    session.selection_detail = result.method_detail  # informational, read by /api/state


def run_pipeline(session) -> None:
    """Compute selected genes for the session's current data + strategy.
    Mutates session in place. Raises PipelineError for user-facing
    configuration problems (e.g. wrong strategy for the data mode)."""
    session.selection_detail = None
    if session.data_mode == "expression_matrix":
        _run_expression_matrix_mode(session)
    else:
        _run_deg_table_mode(session)

    session.rejected_genes = set()
    session.added_genes = set()
    session.ran_pipeline = True


def final_selection(session) -> List[str]:
    base = {str(x) for x in (session.selected_by_algo or [])}
    rejected = {str(x) for x in session.rejected_genes}
    added = {str(x) for x in session.added_genes}
    return sorted((base - rejected) | added)
