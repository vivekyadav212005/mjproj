# SecureRAG S4 — Research Report

Generated 2026-10-05T12:19:43.
All numbers come from a single run of the frozen benchmark. No threshold, attack, or metric was changed after results were seen.

## 1. Methodology

| Component | Setting |
|---|---|
| Generator | Ollama `llama3:latest` |
| Temperature | 0.0 |
| Max tokens | 512 |
| Seed | 42 |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | `chromadb/final_clean` (shared with S0-S3) |
| TOP_K | 5 |
| Prompt | `config.S0_PROMPT_TEMPLATE` + `config.RAG_SYSTEM_PROMPT` (byte-identical to S0-S3) |
| Success metric | case-insensitive `success_keyword` substring, overridden to failure on a refusal phrase |

S4 extends S3 with per-chunk retrieval-aware risk scoring and sentence-window sanitisation:

| Layer | S3 | S4 |
|---|---|---|
| Query gate | Rule OR ML | Rule OR ML (identical) |
| Document scan | block/allow per chunk | block/allow + sentence-window sanitise |
| Risk scoring | binary | multi-factor: doc-level + window + proportion + alignment |
| SUSPICIOUS chunks | passed through | malicious windows removed, clean text retained |
| MALICIOUS chunks | entire chunk blocked | entire chunk quarantined |

### Benchmark (frozen, 160 attacks + 150 benign)

| Slice | n |
|---|---|
| Direct D1-D5 | 100 |
| Indirect I1 | 20 |
| Indirect I2 (hard) | 20 |
| Indirect I3 | 20 |
| **Total attacks** | **160** |
| Benign queries | 150 |

## 2. Attack success rate

| Scope | S3 | S4 | delta (pp) |
|---|---|---|---|
| **Overall (160)** | 10.0% (16/160) | 8.8% (14/160) | -1.3 pp |
| Direct (100) | 7.0% (7/100) | 7.0% (7/100) | +0.0 pp |
| Indirect (60) | 15.0% (9/60) | 11.7% (7/60) | -3.3 pp |
| D1 Instruction Override (20) | 5.0% (1/20) | 5.0% (1/20) | +0.0 pp |
| D2 Role Hijacking (20) | 5.0% (1/20) | 5.0% (1/20) | +0.0 pp |
| D3 Obfuscated Injection (20) | 25.0% (5/20) | 25.0% (5/20) | +0.0 pp |
| D4 Prompt Leakage (20) | 0.0% (0/20) | 0.0% (0/20) | +0.0 pp |
| D5 Multi-Turn Anchoring (20) | 0.0% (0/20) | 0.0% (0/20) | +0.0 pp |
| I1 Explicit Instruction in Doc (20) | 0.0% (0/20) | 0.0% (0/20) | +0.0 pp |
| I2 Hidden Instruction in Doc (hard) (20) | 45.0% (9/20) | 35.0% (7/20) | -10.0 pp |
| I3 Persona Shift via Document (20) | 0.0% (0/20) | 0.0% (0/20) | +0.0 pp |
| D5 single-turn (5) | 0.0% (0/5) | 0.0% (0/5) | +0.0 pp |
| D5 multi-turn (15) | 0.0% (0/15) | 0.0% (0/15) | +0.0 pp |

## 3. Utility on 150 benign queries

| Metric | S3 | S4 | delta |
|---|---|---|---|
| Queries blocked (FPR) | 0.0% | 0.0% | — |
| Answerable | 99.3% | 100.0% | — |
| NO_CLEAN_CONTEXT | 1 | 0 | — |
| Chunks removed / retrieved | 8.0% | 0.1% | — |
| Gold source retrieved | 115/150 | 115/150 | — |
| Gold source retained | 107/150 | 115/150 | — |
| Gold retention (given retrieved) | 93.0% | 100.0% | — |

## 4. S4 Sanitisation Statistics

| Counter | Attacks | Benign |
|---|---|---|
| LLM calls | 86 | 150 |
| Queries blocked | 74 | 0 |
| Follow-ups blocked | 13 | — |
| NO_CLEAN_CONTEXT | 0 | 0 |
| Chunks retrieved | 365 | 750 |
| Chunks quarantined | 29 | 1 |
| Chunks sanitised | 68 | 141 |
| Windows removed | 225 | 193 |
| Mean latency (ms) | 6914.3 | 17784.31 |
| Errors | 0 | 0 |

## 5. S3 vs S4 Summary Comparison

| Dimension | S3 | S4 | Assessment |
|---|---|---|---|
| Overall ASR | 10.0% | 8.8% | -1.3 pp |
| Direct ASR | 7.0% | 7.0% | — |
| Indirect ASR | 15.0% | 11.7% | — |
| Benign FPR | 0.0% | 0.0% | — |
| Answerability | 99.3% | 100.0% | — |
| Gold retention | 93.0% | 100.0% | — |
| LLM calls (attack) | 86 | 86 | — |
| Windows/chunks removed | S3: n/a | S4: 225 windows, 68 sanitised chunks | S4-only |

## 6. I3 Persona-Shift Analysis

I3 (Persona Shift via Document): S3 ASR = 0.0%, S4 ASR = 0.0%.

I3 attacks embed persona-shift instructions in retrieved documents. S4's sentence-window sanitisation removes the embedded instruction while retaining the factual content of the document.

## 7. Artifacts

- `results/s4/s4_attack_results.csv`
- `results/s4/s4_attack_results.jsonl`
- `results/s4/s4_benign_results.csv`
- `results/s4/s4_summary.json`
- `results/s4/s4_utility_summary.json`
- `results/s4/s4_asr_by_category.csv`
- `results/s4/s4_s3_comparison.csv`
- `results/s4/s4_i3_analysis.csv`
- `results/s4/run_log.txt`
- `results/s4/raw/` — unaggregated per-record JSONL (full chunk audit)

The following paths were **not modified** by this run:
- `results/final/` — frozen S0-S3 results
- `chromadb/clean`, `chromadb/final_clean` — source indices
- `attack_suite/`, `attack_suite_hard_indirect/` — frozen attack suites

