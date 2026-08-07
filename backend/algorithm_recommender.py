"""
Algorithm recommendation for Step 2.

Recommends one of six algorithms based on simple data-driven heuristics,
and — importantly — on what the uploaded data actually *is*:

- If only a DEG summary table (gene, p-value, log2FC) was uploaded, there
  is no per-sample data for a model to learn from, so only
  "Traditional DEG (Volcano)" is offered. The other five are genuinely
  unusable in this mode, not just "less recommended" — recommending them
  here used to be misleading, since they were silently reduced to
  rank-by-formula tricks on the same two numbers.
- If a real expression matrix (samples x genes) + group labels was
  uploaded, all six are usable, and the heuristics below pick a sensible
  default among them.

The output is designed to be shown in a modal dialog with per-algorithm
explanations in plain language.

Design notes:
- No jargon in the output text. Users may be biologists without ML
  background.
- Recommendations are heuristics, not guarantees. The rationale is
  always explained so the user can override if their intuition differs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------- data classes -----------------------------------------------------

@dataclass
class DatasetSummary:
    """Numeric summary of the loaded dataset."""
    n_total: int
    n_significant: int          # p < 0.05
    n_strong_up: int            # log2fc > 1 AND p < 0.05
    n_strong_down: int          # log2fc < -1 AND p < 0.05
    pct_significant: float
    balance: str                # "up-heavy", "down-heavy", "balanced"
    signal_strength: str        # "very strong", "strong", "moderate", "weak"
    size_class: str             # "very small", "small", "medium", "large", "very large"


@dataclass
class AlgorithmInfo:
    """Human-readable info about one algorithm."""
    key: str                    # matches backend/algorithms.py ALGORITHMS
    label: str                  # display name
    description: str            # 1-2 sentences, plain language
    use_when: list[str]         # bullet points
    avoid_when: list[str]       # bullet points
    requires_expression_matrix: bool = False  # True for all but Traditional DEG
    is_recommended: bool = False
    recommendation_rank: int = 0    # 0 = not ranked, lower = better fit


# ---------- algorithm catalog (plain-language) ------------------------------

ALGORITHM_CATALOG = [
    AlgorithmInfo(
        key="Traditional DEG (Volcano)",
        label="Traditional DEG",
        description=(
            "Picks genes that pass a p-value and fold-change cut-off, then "
            "ranks them by how strong the p-value is. The simplest and most "
            "transparent option."
        ),
        use_when=[
            "You want a clear, transparent selection rule",
            "The differential expression signal is already strong",
            "You want results that any reviewer can reproduce by hand",
        ],
        avoid_when=[
            "The dataset has very few significant genes",
            "You suspect the ranking should consider gene co-behavior",
        ],
    ),
    AlgorithmInfo(
        key="Random Forest (Ensemble)",
        label="Random Forest",
        description=(
            "Trains many small decision trees to tell your two sample groups "
            "apart using every gene's actual expression values, then ranks "
            "genes by how useful each one was for that split. Requires an "
            "expression matrix (samples x genes) with group labels — it "
            "cannot run on a p-value/log2FC table alone, since there's "
            "nothing per-sample to learn from."
        ),
        use_when=[
            "You uploaded an expression matrix with group labels, not just a DEG table",
            "The signal is not overwhelmingly strong",
            "You want a robust, general-purpose ranking",
            "The dataset has many genes (thousands or more)",
        ],
        avoid_when=[
            "You only have a DEG summary table (no per-sample data) — use Traditional DEG instead",
            "You need a very transparent, rule-based selection",
            "The dataset is very small (fewer than 100 genes)",
        ],
        requires_expression_matrix=True,
    ),
    AlgorithmInfo(
        key="XGBoost-style Boosting",
        label="Gradient Boosting",
        description=(
            "Trains a sequence of decision trees on your samples' expression "
            "values, where each new tree focuses on the samples the previous "
            "ones misclassified. Genes are ranked by how much they "
            "contributed to telling the two groups apart. Requires an "
            "expression matrix with group labels."
        ),
        use_when=[
            "You uploaded an expression matrix with group labels, not just a DEG table",
            "The dataset is medium to large",
            "You expect the important genes to interact with each other",
            "You want a strong ranking even when the signal is subtle",
        ],
        avoid_when=[
            "You only have a DEG summary table (no per-sample data) — use Traditional DEG instead",
            "You want the fastest possible run",
            "You need the ranking to be easy to explain in one sentence",
        ],
        requires_expression_matrix=True,
    ),
    AlgorithmInfo(
        key="LASSO (L1) Regression",
        label="LASSO",
        description=(
            "Fits a model across your samples that is forced to zero out "
            "the coefficient of any gene it doesn't need to separate your "
            "two groups — the genes left with a non-zero coefficient are "
            "the selected panel, not just a re-ranking. Requires an "
            "expression matrix with group labels; on a p-value/log2FC table "
            "there are only two numbers total to 'select' from, so it "
            "cannot run."
        ),
        use_when=[
            "You uploaded an expression matrix with group labels, not just a DEG table",
            "You want a small, focused panel",
            "The dataset has many genes with correlated behavior",
            "You are looking for a minimal set that still captures the signal",
        ],
        avoid_when=[
            "You only have a DEG summary table (no per-sample data) — use Traditional DEG instead",
            "You need many candidate genes rather than a short list",
            "The dataset has very few samples relative to genes (LASSO may be too aggressive)",
        ],
        requires_expression_matrix=True,
    ),
    AlgorithmInfo(
        key="Elastic Net (L1+L2)",
        label="Elastic Net",
        description=(
            "A middle ground between LASSO and a standard linear model, fit "
            "across your samples' expression values. Keeps a compact set of "
            "genes with non-zero coefficients but is more forgiving than "
            "LASSO when many genes rise and fall together. Requires an "
            "expression matrix with group labels."
        ),
        use_when=[
            "You uploaded an expression matrix with group labels, not just a DEG table",
            "You want a compact panel but LASSO seems too aggressive",
            "Many genes in the dataset behave similarly",
            "You want stable results across repeated runs",
        ],
        avoid_when=[
            "You only have a DEG summary table (no per-sample data) — use Traditional DEG instead",
            "You want the shortest possible list (LASSO is stricter)",
            "You need a fully rule-based, transparent selection",
        ],
        requires_expression_matrix=True,
    ),
    AlgorithmInfo(
        key="SVR (Support Vector)",
        label="SVM (linear)",
        description=(
            "Fits a linear support-vector classifier to separate your two "
            "sample groups, then repeatedly drops the least useful gene and "
            "refits (recursive feature elimination) until a compact, ranked "
            "panel remains. Requires an expression matrix with group labels."
        ),
        use_when=[
            "You uploaded an expression matrix with group labels, not just a DEG table",
            "You want a panel narrowed down by repeatedly testing which genes matter least",
            "The dataset is medium sized",
        ],
        avoid_when=[
            "You only have a DEG summary table (no per-sample data) — use Traditional DEG instead",
            "The dataset is very large (RFE re-fits many times and can be slow)",
            "You want quick, transparent output",
        ],
        requires_expression_matrix=True,
    ),
]


# ---------- dataset summary --------------------------------------------------

def summarize_dataset(
    df: pd.DataFrame,
    g_col: str,
    p_col: str,
    fc_col: str,
) -> DatasetSummary:
    """
    Compute the summary statistics that drive the recommendation.
    Uses only the columns already mapped by the user in Step 1.
    """
    # Clean coercion — the pipeline does this too, but this endpoint may
    # be called before the pipeline is run, so we re-apply.
    p = pd.to_numeric(df[p_col], errors="coerce")
    fc = pd.to_numeric(df[fc_col], errors="coerce")
    mask = p.notna() & fc.notna()
    p = p[mask]
    fc = fc[mask]

    n_total = int(len(p))
    if n_total == 0:
        return DatasetSummary(
            n_total=0, n_significant=0, n_strong_up=0, n_strong_down=0,
            pct_significant=0.0, balance="balanced",
            signal_strength="weak", size_class="very small",
        )

    sig_mask = p < 0.05
    n_significant = int(sig_mask.sum())
    n_strong_up = int(((fc > 1) & sig_mask).sum())
    n_strong_down = int(((fc < -1) & sig_mask).sum())
    pct_significant = 100.0 * n_significant / n_total

    # Balance
    n_strong = n_strong_up + n_strong_down
    if n_strong == 0:
        balance = "balanced"
    else:
        up_ratio = n_strong_up / n_strong
        if up_ratio > 0.75:
            balance = "up-heavy"
        elif up_ratio < 0.25:
            balance = "down-heavy"
        else:
            balance = "balanced"

    # Signal strength — based on percentage of strong DEGs relative to total
    pct_strong = 100.0 * n_strong / n_total
    if pct_strong > 30:
        signal_strength = "very strong"
    elif pct_strong > 10:
        signal_strength = "strong"
    elif pct_strong > 2:
        signal_strength = "moderate"
    else:
        signal_strength = "weak"

    # Size class
    if n_total < 200:
        size_class = "very small"
    elif n_total < 2000:
        size_class = "small"
    elif n_total < 10000:
        size_class = "medium"
    elif n_total < 30000:
        size_class = "large"
    else:
        size_class = "very large"

    return DatasetSummary(
        n_total=n_total,
        n_significant=n_significant,
        n_strong_up=n_strong_up,
        n_strong_down=n_strong_down,
        pct_significant=round(pct_significant, 1),
        balance=balance,
        signal_strength=signal_strength,
        size_class=size_class,
    )


# ---------- recommendation logic --------------------------------------------

def recommend_algorithm(
    summary: DatasetSummary,
    has_expression_matrix: bool = False,
) -> tuple[str, str]:
    """
    Decide which algorithm to recommend. Returns (algorithm_key, rationale).
    Rationale is plain language, meant to be shown to the user.

    If `has_expression_matrix` is False (only a DEG summary table was
    uploaded), Traditional DEG is the ONLY option that can genuinely run —
    the other five need per-sample expression values to select genes as
    features from. Recommending them here without that data would just
    reproduce the original bug, so this is a hard gate, not a heuristic.

    Heuristics (only apply once an expression matrix is available):
    - Very strong signal + small/medium dataset -> Traditional DEG
      (transparent, adequate)
    - Very small dataset -> Traditional DEG
      (ensembles need more data to shine)
    - Very large dataset with weak/moderate signal -> Random Forest
      (robust, handles noise)
    - Very large dataset with strong signal -> Random Forest OR Traditional DEG
      (RF for more nuanced ranking)
    - Medium dataset, moderate signal -> Random Forest
      (safe default)
    - Anywhere with heavily unbalanced direction -> Random Forest
      (not fooled by direction bias)
    - Otherwise -> Random Forest (safe default)
    """
    if not has_expression_matrix:
        return (
            "Traditional DEG (Volcano)",
            "You've uploaded a DEG summary table (gene, p-value, log2FC) "
            "without per-sample expression data. The other five methods "
            "need real per-sample values to genuinely select which genes "
            "matter — on a summary table there's nothing for them to learn "
            "from, so only the transparent threshold rule is offered here. "
            "Upload an expression matrix (samples x genes) with group "
            "labels in Step 1 to unlock model-based feature selection."
        )

    if summary.n_total < 100:
        return (
            "Traditional DEG (Volcano)",
            "Your dataset is small, so a simple threshold-based approach "
            "gives a clear, transparent selection. More complex methods "
            "need more data to be useful."
        )

    if summary.size_class == "very small":
        return (
            "Traditional DEG (Volcano)",
            "Your dataset has relatively few genes. A simple threshold-based "
            "approach is transparent and adequate at this scale."
        )

    if summary.signal_strength == "weak":
        return (
            "Random Forest (Ensemble)",
            "The differential expression signal is subtle in your data. "
            "Random Forest is robust to weak signals and can pick up genes "
            "that a simple cut-off would miss."
        )

    if summary.signal_strength == "very strong" and summary.size_class in ("small", "medium"):
        return (
            "Traditional DEG (Volcano)",
            "Your data shows a very strong differential signal. A simple "
            "threshold-based selection is transparent and will capture the "
            "biology clearly."
        )

    if summary.balance != "balanced":
        return (
            "Random Forest (Ensemble)",
            f"Your dataset is {summary.balance} (most strong changes go in "
            "one direction). Random Forest is not fooled by this asymmetry "
            "and gives a balanced ranking."
        )

    if summary.size_class in ("large", "very large"):
        return (
            "Random Forest (Ensemble)",
            "Your dataset is large. Random Forest handles this size well "
            "and gives a robust, general-purpose ranking."
        )

    # Default
    return (
        "Random Forest (Ensemble)",
        "Random Forest is a strong general-purpose choice for datasets like "
        "yours: it handles noise, does not overfit to a few genes, and "
        "produces a stable ranking."
    )


# ---------- main entrypoint --------------------------------------------------

def make_recommendation(
    df: pd.DataFrame,
    g_col: str,
    p_col: str,
    fc_col: str,
    has_expression_matrix: bool = False,
) -> dict:
    """
    Full pipeline: summarize the dataset, pick a recommendation, and
    prepare the algorithm catalog with the recommended one flagged.

    `has_expression_matrix` should reflect the session's actual data_mode
    (state.py: session.data_mode == "expression_matrix"). When False, only
    "Traditional DEG (Volcano)" is marked available — the frontend should
    disable/grey out the other five cards rather than let the user pick
    one that will just error out at /api/run.

    Returns a dict ready to be JSON-serialized for the frontend.
    """
    summary = summarize_dataset(df, g_col, p_col, fc_col)
    rec_key, rec_rationale = recommend_algorithm(summary, has_expression_matrix)

    # Mark the recommended algorithm in the catalog, and whether each one
    # is actually usable given the current session's data.
    catalog_out = []
    for a in ALGORITHM_CATALOG:
        available = has_expression_matrix or not a.requires_expression_matrix
        catalog_out.append({
            "key": a.key,
            "label": a.label,
            "description": a.description,
            "use_when": a.use_when,
            "avoid_when": a.avoid_when,
            "requires_expression_matrix": a.requires_expression_matrix,
            "available": available,
            "is_recommended": (a.key == rec_key),
        })

    # Human-readable summary lines for the "About your data" section
    # of the modal. Kept plain-language.
    summary_lines = [
        f"{summary.n_total:,} genes in total",
        f"{summary.n_significant:,} of them are statistically significant "
        f"(p < 0.05) — that's {summary.pct_significant}% of the dataset",
    ]
    if summary.n_strong_up + summary.n_strong_down > 0:
        summary_lines.append(
            f"{summary.n_strong_up:,} genes are strongly upregulated and "
            f"{summary.n_strong_down:,} are strongly downregulated"
        )
    summary_lines.append(
        f"Overall signal strength: {summary.signal_strength}"
    )
    if summary.balance != "balanced":
        summary_lines.append(
            f"The direction of change is {summary.balance}"
        )

    return {
        "ok": True,
        "has_expression_matrix": has_expression_matrix,
        "summary": {
            "n_total": summary.n_total,
            "n_significant": summary.n_significant,
            "n_strong_up": summary.n_strong_up,
            "n_strong_down": summary.n_strong_down,
            "pct_significant": summary.pct_significant,
            "balance": summary.balance,
            "signal_strength": summary.signal_strength,
            "size_class": summary.size_class,
        },
        "summary_lines": summary_lines,
        "recommendation": {
            "key": rec_key,
            "rationale": rec_rationale,
        },
        "algorithms": catalog_out,
    }
