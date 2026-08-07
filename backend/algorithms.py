"""
The 6 algorithm cards shown in the UI.

IMPORTANT — this module was rewritten to fix a correctness bug in the
original implementation. The original code ran LASSO / Elastic Net /
Random Forest / Gradient Boosting / SVR on a (n_genes, 2) array whose only
two "features" were p_value and log2FC — i.e. genes were rows, not
features, so there was nothing for these algorithms to actually *select*.
They were fitting a regressor to reproduce -log10(p)*|log2FC| (a value
computable in closed form from the same two inputs) and calling the
resulting per-gene prediction "feature selection". That is a ranking
trick, not feature selection.

Genuine feature selection requires genes to be the *features* (columns)
and samples to be the rows, with a group label to predict. That data
only exists if the user uploaded a raw expression matrix + sample
metadata (see data.py: parse_expression_matrix / parse_sample_metadata /
align_matrix_and_groups). This module now operates on that shape:

    X : (n_samples, n_genes)   — expression values, samples x genes
    y : (n_samples,)           — binary group label (0/1)

and returns, per algorithm, an actual per-gene importance/weight derived
from a model that was fit using genes as columns — so "selection" means
what it says: some genes get zero weight / get dropped, others don't.

"Traditional DEG (Volcano)" needs none of this — it's a transparent
threshold rule on p-value/log2FC and is handled directly in pipeline.py.
It remains available even when only a DEG summary table (no per-sample
data) was uploaded. The other five algorithms below now REQUIRE an
expression matrix; pipeline.py enforces that and raises a clear error
otherwise, instead of silently faking a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


ALGORITHMS = [
    {"key": "Traditional DEG (Volcano)",  "title": "Traditional DEG",   "icon": "ti-chart-dots",  "desc": "Volcano-plot thresholds on p-value and log2FC. Works on a DEG table alone.", "requires_expression_matrix": False},
    {"key": "Random Forest (Ensemble)",   "title": "Random Forest",     "icon": "ti-binary-tree", "desc": "Tree ensemble trained to classify samples by group; genes ranked by feature importance.", "requires_expression_matrix": True},
    {"key": "XGBoost-style Boosting",     "title": "Gradient Boosting", "icon": "ti-trending-up", "desc": "Boosted trees trained to classify samples by group; genes ranked by feature importance.", "requires_expression_matrix": True},
    {"key": "LASSO (L1) Regression",      "title": "LASSO",             "icon": "ti-line",        "desc": "L1-regularized logistic regression — genes with non-zero coefficients are the selected panel.", "requires_expression_matrix": True},
    {"key": "Elastic Net (L1+L2)",        "title": "Elastic Net",       "icon": "ti-grid-dots",   "desc": "Combined L1+L2 regularized logistic regression, more stable than LASSO when genes are correlated.", "requires_expression_matrix": True},
    {"key": "SVR (Support Vector)",       "title": "SVM (linear)",      "icon": "ti-vector",      "desc": "Linear support-vector classifier with recursive feature elimination — genes ranked by how much removing them hurts separation.", "requires_expression_matrix": True},
]

# Algorithms that only make sense with a real expression matrix (samples x
# genes). "Traditional DEG (Volcano)" is deliberately excluded — it is a
# threshold rule, not a model, and works fine on a DEG summary table alone.
REQUIRES_EXPRESSION_MATRIX = {
    a["key"] for a in ALGORITHMS if a["requires_expression_matrix"]
}


@dataclass
class SelectionResult:
    gene_order: List[str]          # genes ranked best -> worst
    scores: Dict[str, float]       # per-gene importance/weight (comparable within algo, not across)
    n_nonzero: int                 # how many genes actually got non-zero weight (algo-dependent)
    method_detail: str             # one-line human-readable note on what happened


def _encode_labels(y_labels: np.ndarray, group_a: str, group_b: str) -> np.ndarray:
    return (y_labels == group_a).astype(int)


def select_features(
    X: np.ndarray,
    y_labels: np.ndarray,
    gene_names: List[str],
    group_a: str,
    group_b: str,
    strategy: str,
    random_state: int = 42,
) -> SelectionResult:
    """Run genuine feature selection: genes are columns of X, samples are
    rows, y_labels is the group each sample belongs to.

    X must be (n_samples, n_genes). Raises ValueError for
    "Traditional DEG (Volcano)" — that method is threshold-based and is
    handled in pipeline.py directly, not here.
    """
    if strategy not in REQUIRES_EXPRESSION_MATRIX:
        raise ValueError(
            f"{strategy!r} is not a model-based selection method handled "
            "by select_features(); Traditional DEG uses thresholds instead."
        )

    n_samples, n_genes = X.shape
    if n_samples != len(y_labels):
        raise ValueError("X and y_labels have mismatched sample counts.")
    if len(gene_names) != n_genes:
        raise ValueError("gene_names length does not match X's gene columns.")

    y = _encode_labels(np.asarray(y_labels), group_a, group_b)
    # Standardize per gene (column) across samples — required for the
    # regularized linear models and helps the tree models converge faster.
    X_sc = StandardScaler().fit_transform(X)

    if strategy == "Random Forest (Ensemble)":
        model = RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ).fit(X, y)
        importances = model.feature_importances_
        n_nonzero = int((importances > 0).sum())
        detail = (
            f"Random Forest trained to classify {group_a} vs {group_b} "
            f"across {n_samples} samples; ranked by Gini importance."
        )

    elif strategy == "XGBoost-style Boosting":
        model = GradientBoostingClassifier(
            n_estimators=300, random_state=random_state
        ).fit(X, y)
        importances = model.feature_importances_
        n_nonzero = int((importances > 0).sum())
        detail = (
            f"Gradient Boosting trained to classify {group_a} vs {group_b} "
            f"across {n_samples} samples; ranked by boosted feature importance."
        )

    elif strategy == "LASSO (L1) Regression":
        model = LogisticRegression(
            penalty="l1", solver="liblinear", C=0.5, random_state=random_state,
            max_iter=2000,
        ).fit(X_sc, y)
        coefs = model.coef_.ravel()
        importances = np.abs(coefs)
        n_nonzero = int((coefs != 0).sum())
        detail = (
            f"L1-regularized logistic regression selected {n_nonzero}/{n_genes} "
            f"genes with non-zero coefficients (the rest were dropped entirely)."
        )

    elif strategy == "Elastic Net (L1+L2)":
        model = LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=0.5,
            random_state=random_state, max_iter=5000,
        ).fit(X_sc, y)
        coefs = model.coef_.ravel()
        importances = np.abs(coefs)
        n_nonzero = int((coefs != 0).sum())
        detail = (
            f"Elastic Net logistic regression selected {n_nonzero}/{n_genes} "
            f"genes with non-zero coefficients."
        )

    elif strategy == "SVR (Support Vector)":
        # A non-linear (RBF) SVM has no per-feature coefficients, so it
        # cannot do feature selection at all. We use a linear-kernel SVC,
        # which does, wrapped in Recursive Feature Elimination (RFE) to
        # get a genuine model-driven ranking: RFE repeatedly refits and
        # drops the currently-least-useful gene(s) until only one remains,
        # so every gene's elimination order becomes its rank. (A fixed
        # "keep the top K" cutoff would do nothing when K >= n_genes, so
        # we always eliminate down to 1 rather than capping at a constant.)
        base = SVC(kernel="linear", C=1.0, random_state=random_state)
        step = max(1, n_genes // 50)
        rfe = RFE(base, n_features_to_select=1, step=step).fit(X_sc, y)
        ranking = rfe.ranking_  # 1 = last gene standing, higher = eliminated earlier
        # Convert ranking (lower is better, 1 = best) into a descending
        # importance score for consistent sorting below.
        importances = (ranking.max() + 1 - ranking).astype(float)
        n_nonzero = n_genes  # RFE here ranks everything rather than zeroing genes out
        detail = (
            f"Linear-kernel SVM + recursive feature elimination ranked all "
            f"{n_genes} genes by how much removing each one hurt "
            f"separation of {group_a} vs {group_b} (least damaging removed first)."
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    scores = {gene_names[i]: float(importances[i]) for i in range(n_genes)}
    gene_order = [g for g, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]

    return SelectionResult(
        gene_order=gene_order,
        scores=scores,
        n_nonzero=n_nonzero,
        method_detail=detail,
    )
