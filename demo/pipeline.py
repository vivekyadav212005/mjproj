"""
demo/pipeline.py — Baseline RAG, SecureRAG Defence, and SecureRAG+ pipelines.

Two completely independent query modes:

  baseline_query()   — retrieve + send everything to llama3 (no security)
  securerag_query()  — query check → retrieve → per-doc scan → quarantine
                        → send only clean context to llama3
  securerag_plus_query() — query check → retrieve → document risk + sentence
                            windows → sanitize/quarantine → llama3

Both modes use:
  - DemoRetriever    (demo/retriever.py — 6 fictional documents)
  - RuleClassifier   (security/rule_classifier.py — read-only, stateless)
  - GeminiClient     (rag/llm_client.py — Ollama llama3:latest wrapper)

This module does NOT instantiate BaselineRAG, HybridDefendedRAG, or any
research pipeline class. It does NOT touch results/final/, the 89,500-chunk
ChromaDB, or the S2 trained model.

Note: The demo labels the defended mode as "SecureRAG Defence" (not "S3")
because it uses only RuleClassifier, not the full S3 hybrid classifier.
"""

import sys
import os
import time

# Allow imports from the project root (one level up from demo/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo.retriever import DemoRetriever
from security.rule_classifier import RuleClassifier
from security.s4_retrieval_risk import RetrievalRiskScorer, S4RiskConfig
from security.s4_sentence_scanner import SentenceWindowScanner
from rag.llm_client import GeminiClient

# ── Prompt template (same structure as the research S0 template) ─────────
DEMO_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based on the "
    "provided context documents. Use only the information in the context "
    "to answer the user's question. If the context does not contain "
    "sufficient information to answer, state that clearly."
)

DEMO_PROMPT_TEMPLATE = (
    "Context information:\n"
    "{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

BLOCKED_RESPONSE = (
    "⚠️ This query was blocked by the SecureRAG query security layer. "
    "It appears to contain instructions that may attempt to override "
    "the assistant's operational guidelines."
)

NO_CLEAN_CONTEXT = (
    "All retrieved documents were flagged as potentially malicious and "
    "quarantined. No clean context is available to answer this question."
)


def _build_prompt(question: str, context_docs: list[str]) -> str:
    context = "\n\n".join(context_docs) if context_docs else "No context available."
    return DEMO_PROMPT_TEMPLATE.format(context=context, question=question)


def _call_llm(llm: GeminiClient, prompt: str) -> dict:
    """Call Ollama llama3 and return (response, error, latency_ms)."""
    t0 = time.perf_counter()
    try:
        response = llm.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt=DEMO_SYSTEM_PROMPT,
        )
        error = llm.last_error
        if not response and error:
            response = f"[LLM returned empty — error: {error}]"
    except Exception as e:
        response = ""
        error = str(e)
    latency = round((time.perf_counter() - t0) * 1000, 2)
    return {"response": response, "error": error, "latency_ms": latency}


# ══════════════════════════════════════════════════════════════════════════
# Baseline RAG — no security
# ══════════════════════════════════════════════════════════════════════════

def baseline_query(query: str, retriever: DemoRetriever, llm: GeminiClient,
                   top_k: int = 5) -> dict:
    """
    Baseline RAG: retrieve and pass all documents to the LLM without any
    security filtering. Poisoned documents will reach the model.
    """
    t_start = time.perf_counter()

    # ── Retrieve ─────────────────────────────────────────────────────────
    retrieved = retriever.retrieve(query, top_k=top_k)

    # ── Build prompt from ALL retrieved docs ─────────────────────────────
    context_docs = [doc["text"] for doc in retrieved]
    prompt = _build_prompt(query, context_docs)

    # ── Call LLM ─────────────────────────────────────────────────────────
    llm_result = _call_llm(llm, prompt)

    total_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "mode": "Baseline RAG",
        "query": query,
        "query_security": None,  # no security check performed
        "retrieved_docs": retrieved,
        "doc_scan_results": None,  # no document scanning
        "quarantined_docs": [],
        "clean_docs": retrieved,
        "context_sent_to_llm": context_docs,
        "llm_called": True,
        "llm_response": llm_result["response"],
        "llm_error": llm_result["error"],
        "prompt": prompt,
        "total_latency_ms": total_ms,
    }


# ══════════════════════════════════════════════════════════════════════════
# SecureRAG Defence — RuleClassifier-based document security
# ══════════════════════════════════════════════════════════════════════════

def securerag_query(query: str, retriever: DemoRetriever, llm: GeminiClient,
                    classifier: RuleClassifier, top_k: int = 5) -> dict:
    """
    SecureRAG Defence pipeline:
      1. Query security check
      2. Retrieve top-k documents
      3. Per-document security scan
      4. Quarantine malicious documents
      5. Send only clean context to llama3
    """
    t_start = time.perf_counter()

    # ── Step 1: Query Security Check ─────────────────────────────────────
    query_check = classifier.classify(query)
    query_security = {
        "score": query_check["score"],
        "blocked": query_check["blocked"],
        "triggered_groups": query_check["triggered_groups"],
        "decision": "BLOCKED" if query_check["blocked"] else "ALLOWED",
    }

    if query_check["blocked"]:
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return {
            "mode": "SecureRAG Defence",
            "query": query,
            "query_security": query_security,
            "retrieved_docs": [],
            "doc_scan_results": [],
            "quarantined_docs": [],
            "clean_docs": [],
            "context_sent_to_llm": [],
            "llm_called": False,
            "llm_response": BLOCKED_RESPONSE,
            "llm_error": None,
            "prompt": None,
            "total_latency_ms": total_ms,
        }

    # ── Step 2: Retrieve ─────────────────────────────────────────────────
    retrieved = retriever.retrieve(query, top_k=top_k)

    # ── Step 3: Per-document security scan ───────────────────────────────
    doc_scan_results = []
    quarantined = []
    clean = []

    for doc in retrieved:
        check = classifier.classify(doc["text"])
        scan_entry = {
            "doc_name": doc["doc_name"],
            "similarity": doc["similarity"],
            "rank": doc["rank"],
            "score": check["score"],
            "blocked": check["blocked"],
            "triggered_groups": check["triggered_groups"],
            "decision": "MALICIOUS — QUARANTINED" if check["blocked"] else "SAFE",
        }
        doc_scan_results.append(scan_entry)

        if check["blocked"]:
            quarantined.append(doc)
        else:
            clean.append(doc)

    # ── Step 4: Generate from clean context only ─────────────────────────
    if not clean:
        total_ms = round((time.perf_counter() - t_start) * 1000, 2)
        return {
            "mode": "SecureRAG Defence",
            "query": query,
            "query_security": query_security,
            "retrieved_docs": retrieved,
            "doc_scan_results": doc_scan_results,
            "quarantined_docs": quarantined,
            "clean_docs": clean,
            "context_sent_to_llm": [],
            "llm_called": False,
            "llm_response": NO_CLEAN_CONTEXT,
            "llm_error": None,
            "prompt": None,
            "total_latency_ms": total_ms,
        }

    context_docs = [doc["text"] for doc in clean]
    prompt = _build_prompt(query, context_docs)
    llm_result = _call_llm(llm, prompt)

    total_ms = round((time.perf_counter() - t_start) * 1000, 2)

    return {
        "mode": "SecureRAG Defence",
        "query": query,
        "query_security": query_security,
        "retrieved_docs": retrieved,
        "doc_scan_results": doc_scan_results,
        "quarantined_docs": quarantined,
        "clean_docs": clean,
        "context_sent_to_llm": context_docs,
        "llm_called": True,
        "llm_response": llm_result["response"],
        "llm_error": llm_result["error"],
        "prompt": prompt,
        "total_latency_ms": total_ms,
    }


# ══════════════════════════════════════════════════════════════════════════
# SecureRAG+ (S4) — document risk plus sentence-window sanitisation
# ══════════════════════════════════════════════════════════════════════════

def securerag_plus_query(query: str, retriever: DemoRetriever, llm: GeminiClient,
                         classifier: RuleClassifier, top_k: int = 5) -> dict:
    """Run the S4-style, auditable sanitisation path for the demo corpus.

    This remains a self-contained demo pipeline: it reuses the existing
    deterministic RuleClassifier and S4 scanner/risk primitives, but does not
    instantiate or mutate the research benchmark system or its artifacts.
    """
    t_start = time.perf_counter()
    query_check = classifier.classify(query)
    query_security = {
        "score": query_check["score"],
        "blocked": query_check["blocked"],
        "triggered_groups": query_check["triggered_groups"],
        "decision": "BLOCKED" if query_check["blocked"] else "ALLOWED",
    }
    if query_check["blocked"]:
        return {
            "mode": "SecureRAG+ (S4)", "query": query,
            "query_security": query_security, "retrieved_docs": [],
            "doc_scan_results": [], "quarantined_docs": [], "clean_docs": [],
            "sanitized_docs": [], "context_sent_to_llm": [], "llm_called": False,
            "llm_response": BLOCKED_RESPONSE, "llm_error": None, "prompt": None,
            "total_latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
        }

    retrieved = retriever.retrieve(query, top_k=top_k)
    scanner = SentenceWindowScanner(classifier, max_window_chars=420)
    risk_scorer = RetrievalRiskScorer(S4RiskConfig())
    doc_scan_results, quarantined, clean, sanitized = [], [], [], []

    for doc in retrieved:
        document_check = classifier.classify(doc["text"])
        window_scan = scanner.scan(doc["text"])
        # DemoRetriever reports cosine similarity; map it to the distance form
        # used by the S4 risk scorer's alignment calculation.
        alignment = risk_scorer.alignment(
            query, doc["text"], retrieval_distance=1.0 - doc["similarity"]
        )
        risk = risk_scorer.score(
            document_score=document_check["score"],
            max_sentence_score=window_scan["max_sentence_score"],
            suspicious_window_proportion=window_scan["suspicious_window_proportion"],
            query_document_alignment=alignment,
        )
        has_flagged_window = bool(window_scan["suspicious_window_count"])
        if risk["risk_level"] == "MALICIOUS":
            policy = "MALICIOUS — QUARANTINED"
            quarantined.append(doc)
        elif has_flagged_window:
            policy = "SUSPICIOUS — SANITIZED"
            sanitized_doc = dict(doc)
            sanitized_doc["original_text"] = doc["text"]
            sanitized_doc["text"] = window_scan["clean_text"]
            sanitized_doc["removed_windows"] = window_scan["flagged_text"]
            sanitized.append(sanitized_doc)
            clean.append(sanitized_doc)
        else:
            policy = "SAFE"
            clean.append(doc)

        windows = []
        for window in window_scan["windows"]:
            window_level = (
                "MALICIOUS" if window["blocked"] else
                "SUSPICIOUS" if window["score"] > 0 else "SAFE"
            )
            windows.append({**window, "risk_level": window_level})
        doc_scan_results.append({
            "doc_name": doc["doc_name"], "similarity": doc["similarity"],
            "rank": doc["rank"], "score": document_check["score"],
            "blocked": risk["risk_level"] == "MALICIOUS",
            "triggered_groups": document_check["triggered_groups"],
            "decision": policy, "risk_score": risk["risk_score"],
            "risk_level": risk["risk_level"], "alignment": alignment,
            "windows": windows, "removed_windows": window_scan["flagged_text"],
        })

    context_docs = [doc["text"] for doc in clean]
    if not clean:
        return {
            "mode": "SecureRAG+ (S4)", "query": query,
            "query_security": query_security, "retrieved_docs": retrieved,
            "doc_scan_results": doc_scan_results, "quarantined_docs": quarantined,
            "clean_docs": [], "sanitized_docs": sanitized,
            "context_sent_to_llm": [], "llm_called": False,
            "llm_response": NO_CLEAN_CONTEXT, "llm_error": None, "prompt": None,
            "total_latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
        }

    prompt = _build_prompt(query, context_docs)
    llm_result = _call_llm(llm, prompt)
    return {
        "mode": "SecureRAG+ (S4)", "query": query,
        "query_security": query_security, "retrieved_docs": retrieved,
        "doc_scan_results": doc_scan_results, "quarantined_docs": quarantined,
        "clean_docs": clean, "sanitized_docs": sanitized,
        "context_sent_to_llm": context_docs, "llm_called": True,
        "llm_response": llm_result["response"], "llm_error": llm_result["error"],
        "prompt": prompt,
        "total_latency_ms": round((time.perf_counter() - t_start) * 1000, 2),
    }
