"""
Data loading and the embedded toy dataset.

The toy dataset is the same 30-gene immune/apoptosis panel we shipped in the
Streamlit app, embedded directly in source so no external file is required.

Two upload modes are supported:

1. DEG summary table (gene, p_value, log2FC) — one row per gene. This is
   the output of DESeq2/edgeR/limma. It carries no per-sample information,
   so genuine feature selection (LASSO/RF/GB/ElasticNet/SVM choosing which
   genes matter) is mathematically impossible on it — there is nothing to
   select *from* except two already-aggregated numbers. In this mode only
   the "Traditional DEG (Volcano)" threshold method is offered.

2. Expression matrix (samples x genes) + sample metadata (sample -> group
   label). This carries real per-sample, per-gene values, so the five
   ML-based algorithms can genuinely treat each gene as a feature and
   select/rank which ones best separate the two groups. See
   `parse_expression_matrix`, `parse_sample_metadata`, and
   `compute_deg_stats` below.
"""

from io import BytesIO, StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ----------------------------------------------------------------------------
# TOY DATASET — embedded so the demo works on any fresh checkout.
# 30 genes: 8 strong up (immune/inflammation), 5 strong down (apoptosis/tumor
# suppression), 6 modest receptors/cytoskeleton, 5 quiet housekeepers, 6 noise.
# ----------------------------------------------------------------------------
_TOY_DATA: List[Tuple[str, float, float]] = [
    # gene_symbol, log2FC, p_value
    ("STAT1",   2.191,  4.080e-08),
    ("STAT3",   3.950,  9.984e-07),
    ("IFNG",    3.096,  1.190e-05),
    ("IL6",     3.809,  6.644e-05),
    ("TNF",     4.445,  1.930e-07),
    ("NFKB1",   3.346,  2.487e-09),
    ("CXCL10",  3.253,  5.356e-08),
    ("IRF7",    2.180,  2.868e-04),
    ("TP53",   -2.533,  5.021e-04),
    ("BCL2",   -3.130,  2.409e-08),
    ("BAX",    -4.328,  1.954e-06),
    ("CASP3",  -2.062,  2.850e-04),
    ("PTEN",   -3.501,  6.295e-09),
    ("ACE2",    1.468,  2.845e-02),
    ("TMPRSS2", 1.205,  2.465e-02),
    ("TLR4",   -1.491,  9.607e-03),
    ("MYD88",   1.372,  1.642e-02),
    ("ROCK1",  -1.477,  8.881e-03),
    ("RHOA",    1.366,  1.281e-02),
    ("GAPDH",   0.201,  8.345e-01),
    ("ACTB",   -0.188,  3.263e-01),
    ("HPRT1",  -0.103,  7.188e-01),
    ("B2M",    -0.326,  4.957e-01),
    ("PPIA",   -0.096,  4.985e-01),
    ("FMR1",    0.967,  6.711e-01),
    ("FXR1",    0.230,  4.939e-01),
    ("APP",     1.349,  8.453e-02),
    ("MYC",     0.162,  2.508e-01),
    ("EGFR",   -0.315,  4.352e-01),
    ("VEGFA",   1.147,  2.028e-01),
]


def load_toy_dataset() -> pd.DataFrame:
    """Always available — no external file required."""
    return pd.DataFrame(_TOY_DATA, columns=["gene_symbol", "log2FC", "p_value"])


# Which biological group each toy gene belongs to, and roughly how far apart
# the two synthetic sample groups should be for that gene. Mirrors the
# design note at the top of this section (8 strong up / 5 strong down / 6
# modest / 5 housekeeping / 6 noise) so the *sample-level* twin below
# reproduces the same qualitative story as _TOY_DATA, not just its p/fc
# numbers in isolation.
_TOY_GENE_EFFECT: Dict[str, float] = {sym: fc for sym, fc, _p in _TOY_DATA}


def load_toy_expression_matrix(
    n_per_group: int = 10,
    seed: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic samples x genes twin of the embedded toy DEG panel.

    The original toy dataset (_TOY_DATA above) only ever existed as
    hand-picked (log2FC, p_value) pairs — there was never a real sample
    behind it, so genuine feature selection (which needs genes as
    features across real samples) had nothing to run on. This generates
    a small, deterministic "stimulated vs control" expression matrix
    whose per-gene group means roughly reproduce the same signal
    (immune/inflammation genes up, apoptosis genes down, housekeeping
    genes flat, a few noise genes) so the demo can showcase all six
    algorithms, not just the threshold-based one.

    Returns (expr_df, meta_df) in the same shapes `align_matrix_and_groups`
    expects: expr_df is samples x genes (index=sample_id), meta_df has
    columns ['sample_id', 'condition'] with values 'stimulated'/'control'.
    """
    rng = np.random.default_rng(seed)
    genes = [sym for sym, _fc, _p in _TOY_DATA]
    n_genes = len(genes)
    n_samples = n_per_group * 2
    sample_ids = [f"toy_{i:02d}" for i in range(n_samples)]
    conditions = ["stimulated"] * n_per_group + ["control"] * n_per_group

    # Baseline expression level per gene (log2 scale) so values look like
    # normalized RNA-seq (log2(CPM+1)) rather than centered at zero.
    baseline = rng.uniform(4.0, 9.0, size=n_genes)
    # Modest per-gene noise — large enough that the algorithms have real
    # work to do, small enough that the intended signal still comes through.
    noise_sd = rng.uniform(0.5, 1.0, size=n_genes)

    X = np.empty((n_samples, n_genes))
    for j, gene in enumerate(genes):
        target_fc = _TOY_GENE_EFFECT[gene]
        # Split the target log2FC symmetrically around the gene's baseline.
        mean_stim = baseline[j] + target_fc / 2
        mean_ctrl = baseline[j] - target_fc / 2
        X[:n_per_group, j] = rng.normal(mean_stim, noise_sd[j], size=n_per_group)
        X[n_per_group:, j] = rng.normal(mean_ctrl, noise_sd[j], size=n_per_group)

    expr_df = pd.DataFrame(X, columns=genes, index=sample_ids)
    expr_df.index.name = "sample_id"
    meta_df = pd.DataFrame({"sample_id": sample_ids, "condition": conditions})
    return expr_df, meta_df


def parse_uploaded(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded CSV/TSV/TXT into a DataFrame."""
    sep = "\t" if filename.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(BytesIO(file_bytes), sep=sep)
    df.columns = df.columns.str.strip()
    return df


def parse_expression_matrix(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse an uploaded expression matrix: samples x genes.

    Expected shape: first column is the sample identifier, every other
    column is a gene (numeric expression values). Returns a DataFrame
    indexed by sample id, columns = gene symbols.

        sample_id, GENE1, GENE2, GENE3, ...
        sample_01,   4.2,   0.1,   9.8, ...
        sample_02,   3.9,   0.3,  10.1, ...

    This is the transpose of what bioinformaticians often export from
    DESeq2's counts matrix (genes x samples) — if a matrix comes in that
    orientation, `looks_like_genes_by_samples` below can flag it so the
    caller can offer to transpose it rather than silently misreading it.
    """
    sep = "\t" if filename.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(BytesIO(file_bytes), sep=sep)
    df.columns = df.columns.str.strip()
    id_col = df.columns[0]
    df = df.set_index(id_col)
    df.index = df.index.astype(str)
    df.index.name = "sample_id"
    # Coerce every expression column to numeric; anything that fails becomes
    # NaN. Columns that end up entirely NaN are almost certainly not gene
    # expression columns (e.g. a stray text column) and are dropped.
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    return df


def looks_like_genes_by_samples(df: pd.DataFrame, known_gene_symbols: Optional[set] = None) -> bool:
    """Heuristic: warn the caller if a matrix looks transposed.

    If most of the *row labels* (index) match common gene-symbol shape
    (short, uppercase, alnum) while most *column labels* look like sample
    names (contain things like 'sample', 'S1', 'rep', a number suffix),
    the matrix was probably uploaded as genes x samples instead of the
    expected samples x genes.
    """
    def gene_like(tokens) -> float:
        if len(tokens) == 0:
            return 0.0
        hits = sum(1 for t in tokens if 2 <= len(str(t)) <= 15 and str(t).isupper())
        return hits / len(tokens)

    row_score = gene_like(list(df.index)[:200])
    col_score = gene_like(list(df.columns)[:200])
    return row_score > 0.6 and col_score < row_score


def parse_sample_metadata(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse a sample metadata / phenotype table.

    Expected shape: one row per sample, with a sample-id column and one or
    more categorical columns (e.g. condition, tissue, batch).

        sample_id, condition
        sample_01, tumor
        sample_02, normal

    Returns the raw DataFrame (not indexed) — callers pick the id and
    group columns explicitly via `autodetect_metadata_columns` or a user
    selection, since which column is "the" grouping variable is often
    ambiguous (condition vs batch vs tissue).
    """
    sep = "\t" if filename.endswith((".tsv", ".txt")) else ","
    df = pd.read_csv(BytesIO(file_bytes), sep=sep)
    df.columns = df.columns.str.strip()
    return df


def autodetect_metadata_columns(meta_df: pd.DataFrame) -> dict:
    """Suggest a sample-id column and a group/condition column.

    The id column is whichever column's values best match the expression
    matrix isn't known at this point, so we just look for common names.
    The group column is the first non-id categorical column with between
    2 and 10 unique values (a good proxy for "condition"/"group"/"tissue").
    """
    cols = list(meta_df.columns)
    low = {c: c.lower() for c in cols}

    id_col = None
    for want in ("sample_id", "sampleid", "sample", "sample_name", "id"):
        for c in cols:
            if low[c] == want:
                id_col = c
                break
        if id_col:
            break
    if id_col is None:
        id_col = cols[0]

    group_col = None
    for want in ("condition", "group", "phenotype", "status", "class", "label"):
        for c in cols:
            if low[c] == want:
                group_col = c
                break
        if group_col:
            break
    if group_col is None:
        for c in cols:
            if c == id_col:
                continue
            nunique = meta_df[c].nunique(dropna=True)
            if 2 <= nunique <= 10:
                group_col = c
                break
    if group_col is None and len(cols) > 1:
        group_col = cols[1] if cols[1] != id_col else (cols[0] if cols[0] != id_col else None)

    return {"id_col": id_col, "group_col": group_col}


def align_matrix_and_groups(
    expr_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    id_col: str,
    group_col: str,
    group_a: Optional[str] = None,
    group_b: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.Series, str, str]:
    """Align the expression matrix and metadata on sample id, and resolve
    which two group labels are being compared.

    Returns (expr_aligned, group_series, group_a, group_b) where
    expr_aligned is samples x genes restricted to samples present in both
    inputs, ordered consistently with group_series.

    Raises ValueError with a user-facing message if fewer than 2 samples
    per group are available (not enough replicates for a t-test / model)
    or if the requested group labels aren't present.
    """
    meta = meta_df.copy()
    meta[id_col] = meta[id_col].astype(str)
    meta = meta.set_index(id_col)

    common = expr_df.index.intersection(meta.index)
    if len(common) == 0:
        raise ValueError(
            "No matching sample IDs between the expression matrix and the "
            "sample metadata file. Check that the first column of each "
            "file uses the same sample identifiers."
        )

    groups_all = meta.loc[common, group_col].astype(str)

    if group_a is None or group_b is None:
        counts = groups_all.value_counts()
        if len(counts) < 2:
            raise ValueError(
                f"Column {group_col!r} has fewer than 2 distinct groups "
                "among the matched samples."
            )
        group_a, group_b = counts.index[0], counts.index[1]

    mask = groups_all.isin([group_a, group_b])
    n_a = int((groups_all[mask] == group_a).sum())
    n_b = int((groups_all[mask] == group_b).sum())
    if n_a < 2 or n_b < 2:
        raise ValueError(
            f"Need at least 2 samples per group to run feature selection "
            f"(found {n_a} '{group_a}' and {n_b} '{group_b}')."
        )

    kept_samples = groups_all[mask].index
    expr_aligned = expr_df.loc[kept_samples]
    group_series = groups_all.loc[kept_samples]
    return expr_aligned, group_series, group_a, group_b


def compute_deg_stats(
    expr_df: pd.DataFrame,
    group_series: pd.Series,
    group_a: str,
    group_b: str,
) -> pd.DataFrame:
    """Compute per-gene log2FC and p-value (Welch's t-test) between two
    groups from an expression matrix.

    This produces the same shape of table the rest of the app already
    knows how to work with (gene_symbol, log2FC, p_value) — used for the
    volcano/PCA/log2FC-profile plots and for the "Traditional DEG"
    threshold method — regardless of whether the underlying upload was a
    DEG table or a raw expression matrix.

    log2FC is computed as mean(group_a) - mean(group_b) on the assumption
    the matrix is already log-scale (standard for microarray/normalized
    RNA-seq such as log2(CPM+1) or log2(TPM+1)). If the values are clearly
    not log-scale (e.g. raw counts, all >= 0 with large dynamic range),
    the app should log2(x + 1) transform before calling this — see
    `looks_like_raw_counts`.
    """
    a_mask = group_series == group_a
    b_mask = group_series == group_b
    X_a = expr_df.loc[a_mask.index[a_mask]].values
    X_b = expr_df.loc[b_mask.index[b_mask]].values

    mean_a = np.nanmean(X_a, axis=0)
    mean_b = np.nanmean(X_b, axis=0)
    log2fc = mean_a - mean_b

    # Welch's t-test (does not assume equal variance) per gene/column.
    with np.errstate(invalid="ignore", divide="ignore"):
        _, pvals = stats.ttest_ind(X_a, X_b, axis=0, equal_var=False, nan_policy="omit")
    pvals = np.nan_to_num(pvals, nan=1.0)

    # Benjamini-Hochberg adjusted p-value (FDR), computed across all genes,
    # so the Traditional DEG threshold can correct for multiple testing.
    padj = _bh_adjust(pvals)

    return pd.DataFrame({
        "gene_symbol": expr_df.columns.astype(str),
        "log2FC": log2fc,
        "p_value": pvals,
        "padj": padj,
    })


def _bh_adjust(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment for an array of p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    # Enforce monotonicity from the largest p-value downwards.
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def looks_like_raw_counts(expr_df: pd.DataFrame) -> bool:
    """Heuristic: flag matrices that look like raw integer counts rather
    than log-normalized values, so the caller can offer a log2(x+1)
    transform before running t-tests / models."""
    sample = expr_df.values[:200, :200] if expr_df.size else expr_df.values
    if sample.size == 0:
        return False
    finite = sample[np.isfinite(sample)]
    if finite.size == 0:
        return False
    all_nonneg = bool((finite >= 0).all())
    integer_like = bool(np.allclose(finite, np.round(finite), atol=1e-6))
    wide_range = bool(finite.max() > 50) if finite.size else False
    return all_nonneg and integer_like and wide_range


def autodetect_columns(df: pd.DataFrame) -> dict:
    """Suggest gene / p-value / log2FC columns by name.

    Tuned for standard DESeq2 / edgeR / limma output, which is the most common
    upload format. Key subtleties handled:
      * `Symbol` is preferred over `GeneID` for the gene column (names, not IDs).
      * `lfcSE` and `stat` contain "fc"/"FC"-ish substrings but are NOT the
        fold-change column — they're explicitly excluded.
      * Both `pvalue` and `padj` are valid; we default to raw `pvalue`.
    """
    cols = list(df.columns)
    low = {c: c.lower() for c in cols}

    # --- Gene column: prefer an explicit symbol-like name -------------------
    gene_priority = ("gene_symbol", "genesymbol", "symbol", "gene_name",
                     "genename", "gene", "hgnc_symbol")
    g_col = None
    for want in gene_priority:
        for c in cols:
            if low[c] == want:
                g_col = c
                break
        if g_col:
            break
    if g_col is None:
        # last resort: a non-numeric column, else the first column
        non_numeric = [c for c in cols if df[c].dtype == object or str(df[c].dtype) == "string"]
        g_col = non_numeric[0] if non_numeric else cols[0]

    # --- P-value column: prefer raw pvalue, then padj/adjusted -------------
    p_col = None
    for want in ("pvalue", "p_value", "pval", "p.value", "p"):
        for c in cols:
            if low[c] == want:
                p_col = c
                break
        if p_col:
            break
    if p_col is None:  # adjusted p-value as fallback
        for c in cols:
            if low[c] in ("padj", "adj_pvalue", "adj.p.val", "fdr", "qvalue", "q_value"):
                p_col = c
                break
    if p_col is None:
        # loose match, but never pick the SE or stat columns
        for c in cols:
            if "p" in low[c] and "value" in low[c] and "se" not in low[c]:
                p_col = c
                break
    if p_col is None:
        p_col = cols[1] if len(cols) > 1 else cols[0]

    # --- Log2FC column: explicit names, EXCLUDING lfcSE / stat / SE --------
    EXCLUDE_FC = ("lfcse", "se", "stat", "stderr", "std")
    fc_col = None
    for want in ("log2foldchange", "log2fc", "logfc", "log2_fold_change",
                 "fold_change", "foldchange", "lfc"):
        for c in cols:
            if low[c] == want:
                fc_col = c
                break
        if fc_col:
            break
    if fc_col is None:
        # loose match on "fc"/"fold" but skip anything that looks like a
        # standard-error or test-statistic column
        for c in cols:
            lc = low[c]
            if lc in EXCLUDE_FC:
                continue
            if ("fold" in lc) or ("fc" in lc and "se" not in lc):
                fc_col = c
                break
    if fc_col is None:
        fc_col = cols[-1]

    return {"g_col": g_col, "p_col": p_col, "fc_col": fc_col}
