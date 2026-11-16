# =============================================================================
# TCGA-BRCA differential expression analysis for GeneCurator use case
# =============================================================================
#
# Downloads TCGA-BRCA RNA-seq raw counts (tumor vs. normal), runs DESeq2, and
# writes a single differential expression table compatible with GeneCurator's
# input format (Symbol, log2FoldChange, pvalue, padj).
#
# Runtime: ~5-10 minutes depending on network speed. Requires ~2 GB free disk.
#
# Dependencies (install once):
#   BiocManager::install(c("TCGAbiolinks", "DESeq2", "SummarizedExperiment",
#                          "org.Hs.eg.db"))
#
# Output: TCGA_BRCA_tumor_vs_normal_DEG.tsv (in the working directory)
# =============================================================================

suppressPackageStartupMessages({
  library(TCGAbiolinks)
  library(DESeq2)
  library(SummarizedExperiment)
  library(org.Hs.eg.db)
})

message("[1/6] Querying GDC for TCGA-BRCA STAR-Counts...")
query <- GDCquery(
  project        = "TCGA-BRCA",
  data.category  = "Transcriptome Profiling",
  data.type      = "Gene Expression Quantification",
  workflow.type  = "STAR - Counts",
  sample.type    = c("Primary Tumor", "Solid Tissue Normal")
)

# Subsample to keep the demo tractable — matches the "300-500 samples"
# recommended balance of statistical power vs. runtime.
sample_info <- getResults(query)
tumors      <- sample_info$cases[sample_info$sample_type == "Primary Tumor"]
normals     <- sample_info$cases[sample_info$sample_type == "Solid Tissue Normal"]

set.seed(42)
tumors_use  <- sample(tumors, min(400, length(tumors)))
normals_use <- normals  # keep all normals (~113)
barcodes    <- c(tumors_use, normals_use)

query <- GDCquery(
  project        = "TCGA-BRCA",
  data.category  = "Transcriptome Profiling",
  data.type      = "Gene Expression Quantification",
  workflow.type  = "STAR - Counts",
  sample.type    = c("Primary Tumor", "Solid Tissue Normal"),
  barcode        = barcodes
)

message("[2/6] Downloading count matrices (this is the slow step)...")
GDCdownload(query, method = "api", files.per.chunk = 20)

message("[3/6] Assembling SummarizedExperiment...")
se <- GDCprepare(query, summarizedExperiment = TRUE)

# Sample types
sample_types <- ifelse(se$sample_type == "Primary Tumor", "tumor", "normal")
sample_types <- factor(sample_types, levels = c("normal", "tumor"))
colData(se)$condition <- sample_types

# Filter low-expression genes
message("[4/6] Filtering low-expression genes...")
counts_mat <- assay(se, "unstranded")
keep       <- rowSums(counts_mat >= 10) >= 10
se_filt    <- se[keep, ]
message(sprintf("  Kept %d of %d genes.", nrow(se_filt), nrow(se)))

# DESeq2
message("[5/6] Running DESeq2 (tumor vs normal, this takes 2-3 minutes)...")
dds <- DESeqDataSet(se_filt, design = ~ condition)
dds <- DESeq(dds, parallel = FALSE)
res <- results(dds, contrast = c("condition", "tumor", "normal"))

# Map Ensembl IDs → HGNC symbols
message("[6/6] Mapping to HGNC symbols and writing output...")
ensembl_ids <- sub("\\..*$", "", rownames(res))  # strip version suffix
symbols     <- mapIds(org.Hs.eg.db,
                       keys      = ensembl_ids,
                       column    = "SYMBOL",
                       keytype   = "ENSEMBL",
                       multiVals = "first")

out <- data.frame(
  Symbol         = symbols,
  Ensembl        = ensembl_ids,
  baseMean       = res$baseMean,
  log2FoldChange = res$log2FoldChange,
  lfcSE          = res$lfcSE,
  stat           = res$stat,
  pvalue         = res$pvalue,
  padj           = res$padj,
  stringsAsFactors = FALSE
)

# Drop rows without a symbol or without a valid p-value
out <- out[!is.na(out$Symbol) & !is.na(out$pvalue), ]

# Collapse duplicate symbols by keeping the most significant
out <- out[order(out$pvalue), ]
out <- out[!duplicated(out$Symbol), ]

# Order by significance for readability
out <- out[order(out$pvalue), ]

# Write
out_path <- "TCGA_BRCA_tumor_vs_normal_DEG.tsv"
write.table(out, file = out_path, sep = "\t", row.names = FALSE,
            quote = FALSE)

message(sprintf("Done. Wrote %d genes to %s", nrow(out), out_path))
message(sprintf("  Significant (padj < 0.05): %d",
                sum(out$padj < 0.05, na.rm = TRUE)))
message(sprintf("  Up-regulated (log2FC > 1):   %d",
                sum(out$log2FoldChange > 1  & out$padj < 0.05, na.rm = TRUE)))
message(sprintf("  Down-regulated (log2FC < -1): %d",
                sum(out$log2FoldChange < -1 & out$padj < 0.05, na.rm = TRUE)))
