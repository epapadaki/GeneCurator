"""
Generate a realistic TCGA-BRCA tumor-vs-normal DEG table for the GeneCurator
use case. The table mirrors well-documented patterns from the published
literature — dominant proliferation signal (Ki67, cell cycle), CDCA family,
extracellular matrix remodeling, loss of tissue-specific markers.

Critically, this scaffold is designed so that Random Forest and other
supervised feature-selection algorithms will preferentially pick the
strongest-effect proliferation and immune-adjacent genes, while missing
the canonical breast cancer biomarkers (ESR1, PGR, ERBB2, FOXA1, GATA3)
that are moderately significant but not top-ranked by pure statistics.

This creates the exact "algorithm picks statistically strong but
biologically incomplete panel" scenario that the case study is meant to
demonstrate.

Once the user runs the R script (prepare_brca_deg.R), this scaffold is
replaced by the real TCGA-BRCA DESeq2 output.

NEW: this script now ALSO generates a synthetic samples x genes expression
matrix + sample metadata twin of the DEG table (same log2FC/p-value story,
reverse-engineered into per-sample values — see `generate_expression_matrix`
below), so the case study can demonstrate genuine feature selection
(Random Forest / Gradient Boosting / LASSO / Elastic Net / linear SVM+RFE)
offline, without running the ~10 minute real GDC download. It mirrors what
prepare_brca_deg.R now does with real per-sample data.

Output:
  TCGA_BRCA_tumor_vs_normal_DEG.tsv    — DEG summary table (~15,000 genes)
  TCGA_BRCA_expression_matrix.tsv      — synthetic samples x genes matrix
                                          (subset of genes — see N_SAMPLES,
                                          MATRIX_GENE_CAP below)
  TCGA_BRCA_sample_metadata.tsv        — sample_id, condition
Cols (DEG table): Symbol, Ensembl, baseMean, log2FoldChange, lfcSE, stat,
                  pvalue, padj
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

rng = np.random.default_rng(42)

# Samples for the synthetic expression-matrix twin, and how many genes to
# include in it (curated genes always included; filled out with a sample
# of the random background so RF/GB/LASSO/SVM-RFE have realistic noise to
# sift through, without ballooning to the full ~15,000-gene DEG table size).
N_SAMPLES_PER_GROUP = 30       # -> 60 samples total, tumor vs normal
MATRIX_GENE_CAP = 400          # curated genes + this many background genes

# =============================================================================
# 1. Canonical BRCA biomarker set (moderate significance — algorithm will
#    likely miss these because they are not top-ranked by pure statistics)
# =============================================================================
# These are what an expert MUST add to the panel. Values reflect typical
# TCGA-BRCA tumor-vs-normal contrasts.
canonical_brca = {
    # Hormone receptor and canonical breast markers
    "ESR1":   ( 1.2, 1e-8),
    "PGR":    ( 1.5, 1e-9),
    "ERBB2":  ( 1.8, 1e-11),
    "FOXA1":  ( 1.6, 1e-10),
    "GATA3":  ( 1.4, 1e-9),
    # Tumor suppressors and cancer drivers
    "TP53":   (-0.9, 1e-6),
    "PIK3CA": ( 0.6, 1e-4),
    "BRCA1":  ( 1.1, 1e-7),
    "BRCA2":  ( 0.9, 1e-5),
    "PTEN":   (-1.0, 1e-6),
    # Cell death / apoptosis markers
    "BCL2":   (-1.3, 1e-8),
    # Proliferation
    "MKI67":  ( 3.2, 1e-25),  # Ki67 — this one IS strong
}

# =============================================================================
# 2. Strong-effect proliferation and cell-cycle genes (algorithm WILL pick
#    these — they dominate the ranking but tell you "this is cancer",
#    not "this is breast cancer")
# =============================================================================
proliferation_dominant = {
    # Cell cycle: CDC/CDK/CCN families
    "CDK1":    ( 4.1, 1e-42),
    "CDC20":   ( 4.5, 1e-45),
    "CCNB1":   ( 3.9, 1e-38),
    "CCNB2":   ( 3.8, 1e-36),
    "CCNE1":   ( 2.8, 1e-25),
    "CCNA2":   ( 3.6, 1e-33),
    "CDCA5":   ( 3.7, 1e-34),
    "CDCA8":   ( 3.9, 1e-37),
    "CDK4":    ( 2.1, 1e-15),
    "CDK6":    ( 1.9, 1e-13),
    # Mitotic machinery (extremely common in cancer DEG top hits)
    "BIRC5":   ( 4.8, 1e-50),  # survivin
    "PLK1":    ( 4.3, 1e-43),
    "AURKA":   ( 4.2, 1e-41),
    "AURKB":   ( 4.4, 1e-44),
    "TOP2A":   ( 4.6, 1e-47),
    "KIF20A":  ( 4.5, 1e-46),
    "KIF23":   ( 4.0, 1e-40),
    "KIF11":   ( 3.7, 1e-35),
    "KIF14":   ( 3.9, 1e-37),
    "KIF2C":   ( 4.1, 1e-40),
    "KIF4A":   ( 3.8, 1e-36),
    # E2F targets
    "E2F1":    ( 3.4, 1e-30),
    "E2F2":    ( 3.6, 1e-32),
    # Centromere/kinetochore
    "CENPA":   ( 4.0, 1e-39),
    "CENPE":   ( 3.7, 1e-34),
    "CENPF":   ( 4.2, 1e-42),
    "NDC80":   ( 3.9, 1e-37),
    "SPC25":   ( 3.6, 1e-32),
    "MAD2L1":  ( 3.5, 1e-31),
    "BUB1":    ( 3.7, 1e-34),
    "BUB1B":   ( 3.8, 1e-35),
    # DNA replication
    "MCM2":    ( 3.3, 1e-28),
    "MCM3":    ( 3.2, 1e-27),
    "MCM4":    ( 3.6, 1e-32),
    "MCM10":   ( 3.4, 1e-29),
    "RRM2":    ( 4.5, 1e-45),  # very common top hit
    "TYMS":    ( 3.9, 1e-37),
    # Ubiquitin cell cycle
    "UBE2C":   ( 4.7, 1e-48),
    "UBE2T":   ( 4.0, 1e-39),
    # Chromatin
    "TPX2":    ( 4.3, 1e-42),
    "PBK":     ( 4.5, 1e-46),
    "MELK":    ( 4.2, 1e-41),
    # Nucleic acid / helicases
    "PARP1":   ( 1.8, 1e-12),
}

# =============================================================================
# 3. ECM/collagen down-regulation (tumor stroma) — algorithm may pick some
# =============================================================================
ecm_down = {
    "COL1A1":  (-3.2, 1e-27),
    "COL1A2":  (-3.1, 1e-26),
    "COL3A1":  (-2.9, 1e-24),
    "COL14A1": (-2.7, 1e-22),
    "COL5A1":  (-2.5, 1e-20),
    "COL6A3":  (-2.4, 1e-19),
    "LUM":     (-2.6, 1e-21),
    "DCN":     (-2.8, 1e-23),
    "MMP2":    (-2.5, 1e-20),
    "MMP11":   ( 4.0, 1e-40),  # up in tumors, epithelial
    "SPARC":   (-1.8, 1e-12),
}

# =============================================================================
# 4. Loss of normal breast tissue markers (down in tumor) — algorithm might
#    pick because signal is strong, but expert knows why
# =============================================================================
breast_normal_markers = {
    "LEP":     (-4.5, 1e-45),  # adipose signature
    "ADIPOQ":  (-5.2, 1e-52),
    "PLIN1":   (-4.8, 1e-48),
    "CIDEA":   (-4.6, 1e-46),
    "FABP4":   (-4.9, 1e-50),
    "LPL":     (-3.8, 1e-36),
    "GPD1":    (-3.6, 1e-32),
    "PPARG":   (-2.4, 1e-18),
    "AQP7":    (-4.1, 1e-40),
    "PLIN4":   (-3.9, 1e-37),
    "PLIN5":   (-3.4, 1e-29),
    # Milk/lactation
    "CSN2":    (-6.5, 1e-60),
    "LALBA":   (-6.2, 1e-58),
    "CSN1S1":  (-6.0, 1e-56),
    "WAP":     (-5.8, 1e-54),
    # Myoepithelial
    "KRT5":    (-2.3, 1e-16),
    "KRT14":   (-2.5, 1e-19),
    "KRT17":   (-2.1, 1e-14),
}

# =============================================================================
# 5. Immune-related genes (partial — realistic mix)
# =============================================================================
immune = {
    "CD3D":    ( 0.8, 1e-4),
    "CD8A":    ( 0.9, 1e-4),
    "PDCD1":   ( 0.6, 1e-3),
    "CD274":   ( 1.1, 1e-6),
    "CTLA4":   ( 0.7, 1e-4),
    "IFNG":    ( 0.5, 1e-3),
    "IL10":    ( 0.3, 1e-2),
    # Immunoglobulins — algorithm may pick these too (like COVID case!)
    "IGHG1":   ( 2.1, 1e-15),
    "IGHG3":   ( 1.9, 1e-13),
    "IGHM":    ( 1.7, 1e-11),
    "IGKC":    ( 2.3, 1e-17),
    "IGLC1":   ( 2.0, 1e-14),
}

# =============================================================================
# Merge all curated genes
# =============================================================================
curated = {}
for group in [canonical_brca, proliferation_dominant, ecm_down,
              breast_normal_markers, immune]:
    curated.update(group)


# =============================================================================
# 5b. Synthetic expression-matrix twin (samples x genes)
# =============================================================================
# Reverse-engineers per-sample values from a target (log2FC, p_value) pair:
# for a two-sample Welch/pooled t-test with n samples per group, the target
# p-value implies a critical t-statistic (via the t-distribution's inverse
# CDF), and t = mean_diff / (pooled_sd * sqrt(2/n)) lets us solve for the
# pooled_sd that makes the *expected* t-test on freshly-sampled data land
# close to the target. Realized values will vary a bit sample-to-sample
# (as real data would) rather than reproducing the target exactly — that's
# fine for a scaffold; it only needs to be qualitatively realistic.
def _pooled_sd_for_target(log2fc: float, pvalue: float, n: int) -> float:
    pvalue = min(max(pvalue, 1e-300), 0.999)
    df = 2 * n - 2
    t_crit = stats.t.ppf(1 - pvalue / 2, df)
    t_crit = max(t_crit, 1e-6)
    return abs(log2fc) / (t_crit * np.sqrt(2.0 / n))


def generate_expression_matrix(
    curated_genes: dict,
    n_per_group: int = N_SAMPLES_PER_GROUP,
    gene_cap: int = MATRIX_GENE_CAP,
    seed: int = 43,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Synthetic samples x genes matrix + sample metadata reproducing the
    curated genes' log2FC/p-value story, plus a filler of background genes
    with weak/no signal (so the ML algorithms have realistic noise to sift
    through, not just the "obvious" curated set)."""
    local_rng = np.random.default_rng(seed)
    n_samples = n_per_group * 2
    sample_ids = [f"TCGA_scaffold_{i:03d}" for i in range(n_samples)]
    conditions = ["tumor"] * n_per_group + ["normal"] * n_per_group

    curated_names = list(curated_genes.keys())
    n_background = max(0, gene_cap - len(curated_names))
    background_names = [f"BGGENE{i}" for i in range(n_background)]
    all_genes = curated_names + background_names

    X = np.empty((n_samples, len(all_genes)))
    baseline = local_rng.uniform(4.0, 9.0, size=len(all_genes))

    for j, gene in enumerate(curated_names):
        target_fc, target_p = curated_genes[gene]
        sd = _pooled_sd_for_target(target_fc, target_p, n_per_group)
        sd = float(np.clip(sd, 0.15, 3.0))  # keep it in a plausible range
        mean_tumor = baseline[j] + target_fc / 2
        mean_normal = baseline[j] - target_fc / 2
        X[:n_per_group, j] = local_rng.normal(mean_tumor, sd, size=n_per_group)
        X[n_per_group:, j] = local_rng.normal(mean_normal, sd, size=n_per_group)

    # Background genes: mostly flat/no signal (mirrors the DEG table's own
    # 70/25/5 mixture at a smaller scale), so the panel isn't unrealistically
    # "clean".
    for k, gene in enumerate(background_names):
        j = len(curated_names) + k
        sd = local_rng.uniform(0.5, 1.2)
        roll = local_rng.random()
        if roll < 0.85:
            fc = local_rng.normal(0, 0.3)  # essentially noise
        else:
            fc = local_rng.choice([-1, 1]) * local_rng.uniform(0.8, 2.0)  # occasional real signal
        mean_tumor = baseline[j] + fc / 2
        mean_normal = baseline[j] - fc / 2
        X[:n_per_group, j] = local_rng.normal(mean_tumor, sd, size=n_per_group)
        X[n_per_group:, j] = local_rng.normal(mean_normal, sd, size=n_per_group)

    expr_df = pd.DataFrame(X, columns=all_genes, index=sample_ids)
    expr_df.index.name = "sample_id"
    meta_df = pd.DataFrame({"sample_id": sample_ids, "condition": conditions})
    return expr_df, meta_df


# =============================================================================
# 6. Generate a full background of ~15,000 genes with realistic statistics
# =============================================================================
# We create a realistic distribution:
#   - most genes have small log2FC (~0.1-0.5) and non-significant p-values
#   - a long tail of moderately-affected genes
#   - the curated set overrides random values

# Pool of gene symbols (mix of curated + realistic-looking random symbols)
n_total = 15000
n_random = n_total - len(curated)

# Random gene names for filler
random_prefixes = ["ZNF", "OR", "KRT", "CDH", "TRIM", "SLC", "SLCO", "TMEM",
                   "MRPL", "MRPS", "RPL", "RPS", "PPP1R", "TBC1D", "USP",
                   "LOC", "LINC", "MIR", "SNORD", "ANKRD", "FAM", "C1orf",
                   "C2orf", "TMEM", "CCDC", "WDR", "DDX", "DHX", "EIF"]

random_symbols = set()
attempt = 0
while len(random_symbols) < n_random and attempt < n_random * 5:
    attempt += 1
    prefix = random_prefixes[rng.integers(0, len(random_prefixes))]
    number = rng.integers(1, 5000)
    sym = f"{prefix}{number}"
    if sym not in curated:
        random_symbols.add(sym)
random_symbols = list(random_symbols)
# Fill any shortage with sequential filler names
while len(random_symbols) < n_random:
    random_symbols.append(f"UNKN{len(random_symbols)}")

all_symbols = list(curated.keys()) + random_symbols

# Log2FC and p-value distributions
log2fc = np.zeros(n_total)
pvalue = np.ones(n_total)

# Curated
for i, sym in enumerate(curated.keys()):
    log2fc[i] = curated[sym][0]
    pvalue[i] = curated[sym][1]

# Random: mostly non-significant with a long tail
# Use a mixture: 70% small effect, 25% moderate, 5% larger
start = len(curated)
sizes = rng.choice([0, 1, 2], size=n_random, p=[0.70, 0.25, 0.05])
for i, size in enumerate(sizes):
    idx = start + i
    if size == 0:  # non-significant noise
        log2fc[idx] = rng.normal(0, 0.4)
        pvalue[idx] = rng.uniform(0.05, 1.0)
    elif size == 1:  # moderate
        direction = rng.choice([-1, 1])
        log2fc[idx] = direction * rng.uniform(0.5, 1.5)
        pvalue[idx] = 10 ** rng.uniform(-5, -2)
    else:  # larger but not extreme
        direction = rng.choice([-1, 1])
        log2fc[idx] = direction * rng.uniform(1.5, 3.0)
        pvalue[idx] = 10 ** rng.uniform(-15, -5)

# Derive baseMean, stat, lfcSE (rough approximations)
base_mean = np.exp(rng.normal(5, 2, n_total))
lfc_se = np.abs(log2fc) / np.maximum(-np.log10(pvalue), 1) + 0.1
stat = log2fc / lfc_se

# BH-adjusted padj
sorted_idx = np.argsort(pvalue)
sorted_p = pvalue[sorted_idx]
ranks = np.arange(1, n_total + 1)
padj_sorted = sorted_p * n_total / ranks
padj_sorted = np.minimum.accumulate(padj_sorted[::-1])[::-1]
padj_sorted = np.clip(padj_sorted, 0, 1)
padj = np.empty_like(pvalue)
padj[sorted_idx] = padj_sorted

# Fake Ensembl IDs
ensembl_ids = [f"ENSG{i:011d}" for i in range(n_total)]

# Build the dataframe
df = pd.DataFrame({
    "Symbol":         all_symbols,
    "Ensembl":        ensembl_ids,
    "baseMean":       np.round(base_mean, 2),
    "log2FoldChange": np.round(log2fc, 4),
    "lfcSE":          np.round(lfc_se, 4),
    "stat":           np.round(stat, 4),
    "pvalue":         pvalue,
    "padj":           padj,
})

# Sort by p-value (like DESeq2 default)
df = df.sort_values("pvalue").reset_index(drop=True)

# Write
out_dir = Path(__file__).resolve().parent / "data"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "TCGA_BRCA_tumor_vs_normal_DEG.tsv"
df.to_csv(out_path, sep="\t", index=False,
          float_format="%.4g")

# --- NEW: synthetic expression matrix + sample metadata twin ---------------
expr_df, meta_df = generate_expression_matrix(curated)
matrix_path = out_dir / "TCGA_BRCA_expression_matrix.tsv"
metadata_path = out_dir / "TCGA_BRCA_sample_metadata.tsv"
expr_df.reset_index().to_csv(matrix_path, sep="\t", index=False, float_format="%.4f")
meta_df.to_csv(metadata_path, sep="\t", index=False)

# Summary
n_sig = (df["padj"] < 0.05).sum()
n_up  = ((df["padj"] < 0.05) & (df["log2FoldChange"] >  1)).sum()
n_dn  = ((df["padj"] < 0.05) & (df["log2FoldChange"] < -1)).sum()
print(f"Wrote {len(df)} genes → {out_path}")
print(f"  Significant (padj < 0.05): {n_sig}")
print(f"  Up-regulated   (log2FC >  1): {n_up}")
print(f"  Down-regulated (log2FC < -1): {n_dn}")
print(f"\nWrote {expr_df.shape[0]} samples x {expr_df.shape[1]} genes → {matrix_path}")
print(f"Wrote {len(meta_df)} sample labels → {metadata_path}")

# Sanity check — show what a Random Forest might pick (top 20 by p-value)
print("\nTop 20 by p-value (algorithm's likely picks):")
print(df.head(20)[["Symbol", "log2FoldChange", "pvalue"]].to_string(index=False))

print("\nCanonical BRCA markers — algorithm may MISS these:")
for sym in ["ESR1", "PGR", "ERBB2", "FOXA1", "GATA3", "BCL2", "PIK3CA"]:
    row = df[df["Symbol"] == sym].iloc[0]
    rank = df[df["Symbol"] == sym].index[0]
    print(f"  {sym:8s} rank {rank+1:5d}  log2FC={row['log2FoldChange']:6.2f}  p={row['pvalue']:.2e}")
