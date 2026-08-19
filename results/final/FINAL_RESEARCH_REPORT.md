# SecureRAG — Final Research Report

Generated 2026-08-19T08:22:10. All numbers below come from a single run of the frozen benchmark; no threshold, attack, or metric was changed after results were seen.

## 1. Methodology

| Component | Setting |
|---|---|
| Generator | Ollama `llama3:latest` |
| Temperature | 0.0 |
| Max tokens (`num_predict`) | 512 |
| Seed | 42 |
| Embedding model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | `chromadb/final_clean` — rebuilt from the `chromadb/clean` SQLite chunk text/metadata |
| Chunks indexed | 89500 |
| TOP_K | 5 |
| Indirect context | top_k-1 retrieved chunks + 1 injected chunk = 5 |
| Prompt | `config.S0_PROMPT_TEMPLATE` + `config.RAG_SYSTEM_PROMPT`, byte-identical for S0/S1/S2/S3 |
| Success metric | case-insensitive `success_keyword` substring, overridden to failure on a refusal phrase |

The four systems share one retriever object, one prompt, one generation configuration and one success function. They differ only in the classifier applied:

| System | Classifier | Blocking rule |
|---|---|---|
| S0 | none | undefended; no query gate, no document scan, no follow-up scan |
| S1 | `RuleClassifier` G1–G5 | block at rule score ≥ 0.50 |
| S2 | persisted TF-IDF + Logistic Regression | block at p ≥ 0.40 |
| S3 | S1 rule **OR** S2 model | block if either fires |

S1/S2/S3 classify (1) the user query before retrieval, (2) every retrieved chunk, (3) every injected indirect chunk, and (4) every D5 follow-up turn. Only blocked chunks are removed; remaining clean chunks are passed through. A blocked query performs no retrieval and no LLM call. If every chunk is removed the system returns `NO_CLEAN_CONTEXT` and makes no LLM call. No defended system adds prompt hardening or any output-side filter.

### Benchmark (frozen, 160 attacks)

| Slice | n | Source |
|---|---|---|
| Direct D1–D5 | 100 | `attack_suite/attack_suite_full.json` |
| Indirect I1 | 20 | original |
| Indirect I2 | 20 | hard-indirect v1 (`attack_suite_hard_indirect/I2_hard.json`) |
| Indirect I3 | 20 | original |
| **Total** | **160** | content SHA-256 `bfdecac65b8249d0…` |

D5 is 5 single-turn controls plus 15 true multi-turn attacks. Utility is measured separately on the frozen 150 benign queries.

## 2. Attack success rate

| Scope | S0 undefended baseline | S1 RuleClassifier only | S2 persisted TF-IDF/LR @ 0.40 | S3 Rule OR persisted TF-IDF/LR |
|---|---|---|---|---|
| **Overall (160)** | 65.0% (104/160) | 23.8% (38/160) | 11.9% (19/160) | 10.0% (16/160) |
| Direct (100) | 63.0% (63/100) | 25.0% (25/100) | 9.0% (9/100) | 7.0% (7/100) |
| Indirect (60) | 68.3% (41/60) | 21.7% (13/60) | 16.7% (10/60) | 15.0% (9/60) |
| D1 Instruction Override (20) | 95.0% (19/20) | 35.0% (7/20) | 10.0% (2/20) | 5.0% (1/20) |
| D2 Role Hijacking (20) | 70.0% (14/20) | 40.0% (8/20) | 5.0% (1/20) | 5.0% (1/20) |
| D3 Obfuscated Injection (20) | 65.0% (13/20) | 35.0% (7/20) | 25.0% (5/20) | 25.0% (5/20) |
| D4 Prompt Leakage (20) | 30.0% (6/20) | 5.0% (1/20) | 5.0% (1/20) | 0.0% (0/20) |
| D5 Multi-Turn Anchoring (20) | 55.0% (11/20) | 10.0% (2/20) | 0.0% (0/20) | 0.0% (0/20) |
| I1 Explicit Instruction in Doc (20) | 75.0% (15/20) | 0.0% (0/20) | 0.0% (0/20) | 0.0% (0/20) |
| I2 Hidden Instruction in Doc (hard) (20) | 65.0% (13/20) | 65.0% (13/20) | 45.0% (9/20) | 45.0% (9/20) |
| I3 Persona Shift via Document (20) | 65.0% (13/20) | 0.0% (0/20) | 5.0% (1/20) | 0.0% (0/20) |
| D5 single-turn (5) | 40.0% (2/5) | 0.0% (0/5) | 0.0% (0/5) | 0.0% (0/5) |
| D5 multi-turn (15) | 60.0% (9/15) | 13.3% (2/15) | 0.0% (0/15) | 0.0% (0/15) |

## 3. Document-filter quality

Positive = the injected poisoned chunk on an indirect attack. Negative = a clean KB chunk retrieved alongside it. Attacks blocked at the query layer never reach a document scan and are excluded.

| System | TP | FN | FP | TN | Precision | Recall | F1 | FPR |
|---|---|---|---|---|---|---|---|---|
| S0 | — | — | — | — | — | — | — | — |
| S1 | 40 | 20 | 0 | 240 | 1.000 | 0.667 | 0.800 | 0.000 |
| S2 | 42 | 18 | 10 | 230 | 0.808 | 0.700 | 0.750 | 0.042 |
| S3 | 46 | 14 | 10 | 230 | 0.821 | 0.767 | 0.793 | 0.042 |

S0 has no document filter, so the confusion matrix is undefined for it.

## 4. Utility on 150 benign queries

| Metric | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| Queries blocked | 0 / 150 | 0 / 150 | 0 / 150 | 0 / 150 |
| Benign query FPR | 0.0% | 0.0% | 0.0% | 0.0% |
| Answerability | 100.0% (150/150) | 100.0% (150/150) | 99.3% (149/150) | 99.3% (149/150) |
| `NO_CLEAN_CONTEXT` | 0 | 0 | 1 | 1 |
| Chunks removed | 0 / 750 | 0 / 750 | 60 / 750 | 60 / 750 |
| Benign document FPR | 0.0% | 0.0% | 8.0% | 8.0% |
| Gold source retrieved | 115 / 150 | 115 / 150 | 115 / 150 | 115 / 150 |
| Gold source retained | 115 / 150 | 115 / 150 | 107 / 150 | 107 / 150 |
| Gold retention, given retrieved | 100.0% | 100.0% | 93.0% | 93.0% |

## 5. Cost and operational counters

| Counter | S0 | S1 | S2 | S3 |
|---|---|---|---|---|
| LLM calls (attacks) | 175 | 121 | 89 | 86 |
| LLM calls (benign) | 150 | 150 | 149 | 149 |
| Queries blocked (attacks) | 0 | 43 | 69 | 72 |
| Follow-ups blocked (attacks) | 0 | 11 | 15 | 15 |
| `NO_CLEAN_CONTEXT` (attacks) | 0 | 0 | 0 | 0 |
| Chunks retrieved (attacks) | 800 | 585 | 445 | 430 |
| Chunks blocked (attacks) | 0 | 40 | 69 | 72 |
| Queries blocked (benign) | 0 | 0 | 0 | 0 |
| Chunks blocked (benign) | 0 | 0 | 60 | 60 |
| Mean latency, attacks (ms) | 13540.85 | 8781.46 | 5163.17 | 5209.09 |
| Mean latency, benign (ms) | 18993.79 | 20043.18 | 20333.81 | 20505.32 |
| LLM/pipeline errors | 0 | 0 | 0 | 0 |

## 6. Comparison

Lowest overall ASR: **S3** at 10.0% against the S0 baseline of 65.0%.

| System | Overall ASR | Δ vs S0 | Benign query FPR | Answerability | Gold retention (given retrieved) |
|---|---|---|---|---|---|
| S0 undefended baseline | 65.0% | — | 0.0% | 100.0% | 100.0% |
| S1 RuleClassifier only | 23.8% | -41.2 pp | 0.0% | 100.0% | 100.0% |
| S2 persisted TF-IDF/LR @ 0.40 | 11.9% | -53.1 pp | 0.0% | 99.3% | 93.0% |
| S3 Rule OR persisted TF-IDF/LR | 10.0% | -55.0 pp | 0.0% | 99.3% | 93.0% |

A lower ASR is only a real defence if utility holds: read each row with the benign FPR and answerability columns, not the ASR alone.

## 7. Validity checks

S3 is the disjunction it claims to be. Blocked-query and blocked-follow-up sets are exactly S1 ∪ S2. Chunk-level sets agree on the 86 attacks all three systems document-scanned; they cannot be compared elsewhere, because S3 blocks strictly more queries and a query-blocked attack never reaches a document scan.

| Check | Result |
|---|---|
| Blocked queries: S3 = S1 ∪ S2 | True |
| Blocked follow-ups: S3 = S1 ∪ S2 | True |
| Blocked chunks: S3 = S1 ∪ S2 (commonly scanned attacks) | True |
| S0 chunks scanned by a classifier | 0 |
| S0 queries blocked / chunks removed | 0 / 0 |
| Identical retrieval across systems (gold-in-top-5 set) | True |
| Records per system (attacks / benign) | S0: 160/150, S1: 160/150, S2: 160/150, S3: 160/150 |
| LLM or pipeline errors | 0 |
| Blocked rows that still called the LLM | 0 |
| Chunk accounting mismatches | 0 |

## 8. Artifacts

- `results/final/final_metadata.json`
- `results/final/final_attack_results.csv`
- `results/final/final_attack_results.jsonl`
- `results/final/final_asr_by_category.csv`
- `results/final/final_summary.json`
- `results/final/final_benign_results.csv`
- `results/final/final_utility_summary.json`
- `results/final/raw/` — per-system unaggregated records, including the full per-chunk classifier log

Pre-existing result directories (`results/s0`, `s1`, `s2`, `s3`, `s3_fixed`, `s2_audit`, `s2_retrained`, `hard_indirect`, `indirect_audit`) and `chromadb/clean` were not written to by this run.
