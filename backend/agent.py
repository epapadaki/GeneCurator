"""
Curation suggestion agent for GeneCurator.

Uses Google Gemini API to analyze the current gene panel and suggest
removals (redundant/non-specific genes) and additions (missing canonical
markers based on the biological context).

The agent output is a structured JSON with rationales — never a free-form
text — so the frontend can render it as an auditable list of suggestions
that the user accepts or rejects individually.
"""
from __future__ import annotations
import os
import json
import re
from dataclasses import dataclass
from typing import Optional

# python-dotenv loads .env into environment. Silent-fail if not installed
# (we'll enforce it via requirements.txt but keep the import defensive).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ---------- data classes -----------------------------------------------------

@dataclass
class Suggestion:
    """A single curation suggestion, add or remove."""
    symbol: str
    action: str  # "add" or "remove"
    reason: str
    category: str  # e.g. "hormone_receptor", "immunoglobulin", "canonical_marker"


@dataclass
class AgentResult:
    """The full result of an agent call."""
    ok: bool
    remove_suggestions: list[Suggestion]
    add_suggestions: list[Suggestion]
    context_summary: str
    error: Optional[str] = None


# ---------- prompt construction ---------------------------------------------

# System instruction — kept sharp and role-focused so the model does not
# drift into general chit-chat.
SYSTEM_PROMPT = """You are a bioinformatics domain expert reviewing an automated gene panel selected by a feature-selection algorithm from a differential expression experiment.

Your job is to identify:
1. Genes in the current panel that are likely REDUNDANT or NON-SPECIFIC to the tissue/disease context (candidates for removal)
2. Well-established canonical markers for the tissue/disease context that are MISSING from the current panel (candidates for addition, only if they appear in the provided dataset)

RULES:
- Base your reasoning ONLY on well-established biology (canonical markers, textbook pathways, clinical guidelines). Do not speculate.
- For every suggestion, give a one-sentence rationale grounded in specific biological function.
- Never suggest genes that are not in the provided candidate lists.
- Output valid JSON only — no preamble, no explanation, no code fences.
- Be conservative: prefer to suggest fewer, high-confidence changes rather than many weak ones.
- If a user-declared context is provided, TREAT IT AS GROUND TRUTH. Do not second-guess or override it based on the gene expression pattern.
- If NO user-declared context is provided, you must attempt inference from the top genes but explicitly flag the uncertainty in your context_summary.
- Maximum 15 removal suggestions and 12 addition suggestions."""


USER_PROMPT_TEMPLATE_WITH_CONTEXT = """USER-DECLARED CONTEXT (TREAT AS GROUND TRUTH):
{user_context}

DATASET TOP 20 GENES (by significance):
{top_genes}

CURRENT ALGORITHM-SELECTED PANEL ({n_panel} genes):
{panel_genes}

CANDIDATE ADDITIONS (well-known markers present in dataset but NOT in panel):
{candidate_additions}

The user has explicitly stated the biological context above. Your job is to suggest curation appropriate for THAT context. Do not attempt to independently infer or override the context based on the gene list.

Return JSON with this exact structure:
{{
  "context_summary": "One sentence confirming the user-declared context and briefly noting how the gene panel aligns with it, e.g., 'User-declared: breast cancer TCGA-BRCA. The panel shows expected proliferation and matrix remodeling signatures.'",
  "remove_suggestions": [
    {{"symbol": "GENE_NAME", "reason": "One-sentence biological rationale grounded in the user-declared context", "category": "one of: immunoglobulin, adipocyte_marker, milk_protein, redundant_cell_cycle, non_coding, pseudogene, cancer_testis_antigen, unclear_function"}}
  ],
  "add_suggestions": [
    {{"symbol": "GENE_NAME", "reason": "One-sentence biological rationale grounded in the user-declared context", "category": "one of: hormone_receptor, transcription_factor, tumor_suppressor, oncogene, clinical_biomarker, pathway_hub"}}
  ]
}}"""


USER_PROMPT_TEMPLATE_NO_CONTEXT = """NO USER-DECLARED CONTEXT. You must infer the tissue/disease context from the gene expression pattern below, and flag the uncertainty in your context_summary.

DATASET TOP 20 GENES (by significance):
{top_genes}

CURRENT ALGORITHM-SELECTED PANEL ({n_panel} genes):
{panel_genes}

CANDIDATE ADDITIONS (well-known markers present in dataset but NOT in panel):
{candidate_additions}

Return JSON with this exact structure:
{{
  "context_summary": "One sentence that BEGINS WITH 'Inferred (no user context): ' and describes your best guess plus the reason, e.g., 'Inferred (no user context): likely epithelial malignancy based on MMP11, S100A7, and matrix remodeling signature. Could be breast, prostate, or another epithelial tumor.'",
  "remove_suggestions": [
    {{"symbol": "GENE_NAME", "reason": "One-sentence biological rationale", "category": "one of: immunoglobulin, adipocyte_marker, milk_protein, redundant_cell_cycle, non_coding, pseudogene, cancer_testis_antigen, unclear_function"}}
  ],
  "add_suggestions": [
    {{"symbol": "GENE_NAME", "reason": "One-sentence biological rationale — be extra conservative here since context is unknown", "category": "one of: hormone_receptor, transcription_factor, tumor_suppressor, oncogene, clinical_biomarker, pathway_hub"}}
  ]
}}"""


# ---------- candidate curation for additions --------------------------------

# Canonical marker library. Grouped by tissue/context so we can filter the
# "candidate additions" list intelligently. The agent will only suggest
# additions from this pool that are ALSO present in the user's dataset.
#
# This is a hand-curated, textbook-grade list — the kind of knowledge a
# domain expert brings to the table.
CANONICAL_MARKERS = {
    # Breast cancer
    "breast": ["ESR1", "PGR", "ERBB2", "FOXA1", "GATA3", "BCL2", "MKI67",
               "PTEN", "TP53", "PIK3CA", "BRCA1", "BRCA2", "CDH1", "MYC",
               "CCND1", "AR", "EGFR"],
    # Prostate cancer
    "prostate": ["AR", "KLK3", "KLK2", "NKX3-1", "FOXA1", "PTEN", "TP53",
                 "MYC", "ERG", "TMPRSS2", "PIK3CA", "RB1"],
    # Lung cancer
    "lung": ["EGFR", "KRAS", "ALK", "ROS1", "MET", "BRAF", "TP53", "STK11",
             "KEAP1", "NF1", "RB1", "MYC"],
    # Colorectal
    "colorectal": ["APC", "TP53", "KRAS", "BRAF", "PIK3CA", "SMAD4", "MLH1",
                   "MSH2", "MSH6", "CDX2", "MYC"],
    # Ovarian
    "ovarian": ["BRCA1", "BRCA2", "TP53", "PTEN", "PIK3CA", "PAX8", "WT1",
                "CA125", "MUC16"],
    # Universal cancer drivers
    "universal": ["TP53", "MYC", "PTEN", "PIK3CA", "RB1", "CDKN2A"],
}


def _flatten_canonical() -> set[str]:
    """All canonical markers, deduplicated."""
    out = set()
    for markers in CANONICAL_MARKERS.values():
        out.update(markers)
    return out


def _candidate_additions(panel: list[str], all_symbols: list[str]) -> list[str]:
    """
    Return canonical markers that (a) exist in the user's dataset and
    (b) are not already in the panel. The agent picks from this pool.
    """
    canonical = _flatten_canonical()
    dataset_set = set(all_symbols)
    panel_set = set(panel)
    return sorted(canonical & dataset_set - panel_set)


# ---------- Gemini API call --------------------------------------------------

def _extract_json(text: str) -> Optional[dict]:
    """
    Extract the first JSON object from a text blob. LLMs occasionally wrap
    JSON in code fences or add trailing commentary despite instructions.
    """
    # Strip common code fence wrappers
    text = re.sub(r"^\s*```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    # Find the first { and its matching }
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None


def suggest_curation(
    top_20_genes: list[dict],
    current_panel: list[str],
    all_dataset_symbols: list[str],
    user_context: str = "",
) -> AgentResult:
    """
    Call the Gemini API to get curation suggestions.

    Parameters
    ----------
    top_20_genes : list of {symbol, log2fc, pvalue}
        Top-ranked genes by significance — used by the agent for context
        when user_context is empty.
    current_panel : list of gene symbols
        The current gene panel (algorithm picks + user modifications).
    all_dataset_symbols : list of gene symbols
        All gene symbols present in the user's uploaded dataset. Used to
        constrain add suggestions to genes the user could actually add.
    user_context : str, optional
        User-declared biological context. When non-empty, this is treated
        as ground truth and the agent will not attempt tissue inference.
        Example: "TCGA-BRCA tumor vs normal (breast cancer)".

    Returns
    -------
    AgentResult
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return AgentResult(
            ok=False,
            remove_suggestions=[],
            add_suggestions=[],
            context_summary="",
            error="GEMINI_API_KEY not set. Add it to a .env file in the project root.",
        )

    # Lazy import so the app runs without google-genai installed if the
    # user never enables agent features.
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return AgentResult(
            ok=False,
            remove_suggestions=[],
            add_suggestions=[],
            context_summary="",
            error="google-genai package not installed. Run: pip install google-genai python-dotenv",
        )

    # Build the prompt content
    top_genes_str = "\n".join(
        f"  {g['symbol']}: log2FC={g['log2fc']:+.2f}, p={g['pvalue']:.2e}"
        for g in top_20_genes
    )
    panel_str = ", ".join(current_panel) if current_panel else "(empty)"
    candidates = _candidate_additions(current_panel, all_dataset_symbols)
    candidates_str = ", ".join(candidates) if candidates else "(none — panel already contains most canonical markers)"

    # Branch on whether the user provided context. When they do, we use a
    # prompt that treats it as ground truth. When they don't, we use one
    # that forces the model to flag its inference as uncertain.
    user_context_clean = (user_context or "").strip()
    if user_context_clean:
        user_prompt = USER_PROMPT_TEMPLATE_WITH_CONTEXT.format(
            user_context=user_context_clean[:500],  # cap length for prompt safety
            top_genes=top_genes_str,
            n_panel=len(current_panel),
            panel_genes=panel_str,
            candidate_additions=candidates_str,
        )
    else:
        user_prompt = USER_PROMPT_TEMPLATE_NO_CONTEXT.format(
            top_genes=top_genes_str,
            n_panel=len(current_panel),
            panel_genes=panel_str,
            candidate_additions=candidates_str,
        )

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,  # low temp → more consistent, less creative
                response_mime_type="application/json",
            ),
        )
    except Exception as e:
        return AgentResult(
            ok=False,
            remove_suggestions=[],
            add_suggestions=[],
            context_summary="",
            error=f"Gemini API error: {type(e).__name__}: {str(e)[:200]}",
        )

    raw = response.text or ""
    parsed = _extract_json(raw)
    if not parsed:
        return AgentResult(
            ok=False,
            remove_suggestions=[],
            add_suggestions=[],
            context_summary="",
            error=f"Could not parse agent output as JSON. Raw output: {raw[:300]}",
        )

    # Validate and constrain the output
    panel_set = set(current_panel)
    dataset_set = set(all_dataset_symbols)
    canonical_pool = set(_candidate_additions(current_panel, all_dataset_symbols))

    remove_out = []
    for s in parsed.get("remove_suggestions", [])[:15]:
        sym = str(s.get("symbol", "")).strip().upper()
        if sym in panel_set:
            remove_out.append(Suggestion(
                symbol=sym,
                action="remove",
                reason=str(s.get("reason", ""))[:300],
                category=str(s.get("category", "unclear_function"))[:60],
            ))

    add_out = []
    for s in parsed.get("add_suggestions", [])[:12]:
        sym = str(s.get("symbol", "")).strip().upper()
        # Only allow additions that (a) are in the dataset and (b) not
        # already in panel. This constrains hallucinations.
        if sym in dataset_set and sym not in panel_set:
            add_out.append(Suggestion(
                symbol=sym,
                action="add",
                reason=str(s.get("reason", ""))[:300],
                category=str(s.get("category", "clinical_biomarker"))[:60],
            ))

    return AgentResult(
        ok=True,
        remove_suggestions=remove_out,
        add_suggestions=add_out,
        context_summary=str(parsed.get("context_summary", ""))[:400],
    )
