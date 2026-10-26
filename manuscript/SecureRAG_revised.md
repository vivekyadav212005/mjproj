# SecureRAG: Sentence-Window Sanitization and the Security–Utility Trade-off for Prompt Injection in Retrieval-Augmented Generation

Rachel Fernandes, Vivek Yadav, Aryan Shewale

Working draft, 26 October 2026. This draft reflects all three completed generator runs and does not retroactively change any reported architecture or result. Every count was recomputed from `results/final/raw/`, `results/s4/raw/`, and `results/multimodel_2026/` by `analysis/reanalyze_verified_results.py` and `analysis/analyze_multimodel.py`. No new language-model experiment was run for this draft. Verified statistics are in `results/analysis/verified_statistics.json`.

---

## Abstract

Retrieval-augmented generation (RAG) concatenates untrusted retrieved text with a user question, creating an attack surface where instructions hidden in documents can override the user's query. Published defenses include prompt delimiting [4], structured queries with instruction-tuned models [5], activation-based instruction detection [9], segment-level injection removal [8], and isolate-then-aggregate generation [7]. The design space between whole-chunk deletion and sentence-level deletion within an otherwise retained chunk has received less direct empirical measurement.

This paper evaluates a layered pre-generation filter on one local RAG stack. SecureRAG applies a rule classifier (S1) and a TF-IDF logistic regression model (S2) to queries and retrieved chunks; a disjunction of the two (S3) forms the combined filter; and a sentence-window variant (S4) deletes flagged sentence windows inside chunks instead of quarantining every flagged chunk. The S4 risk score is a weighted combination of document score, maximum window score, flagged-window fraction, and a low-alignment penalty. Across all three evaluated generators, the alignment penalty was zero for every chunk in every run.

On 160 keyword-labeled attacks against three local instruction-tuned models (Llama 3 8B Q4_0, Qwen 2.5 7B Q4_K_M, Mistral 7B Q4_K_M), the rule/TF-IDF combined filter (S3) reduces attack success consistently: from 104/160 to 16/160 on Llama 3, from 69/160 to 6/160 on Qwen 2.5, and from 54/160 to 4/160 on Mistral. These reductions are statistically significant on all three generators (exact McNemar, Holm-adjusted p < 10^-12). The sentence-window layer (S4) does not produce a statistically supported further security gain on any generator; on Qwen the point estimate increases from 6/160 to 7/160. On 150 benign queries, whole-chunk removal falls from 60/750 under S3 to 1/750 under S4 across all generators, but S4 edited 193 windows across 83/150 benign queries, including 24 of 115 retrieved gold sources. The alignment term, the combined-filter-over-TF-IDF improvement, and S4's security gain are unsupported by these results. The study uses forced document injection, keyword attack labels without human validation, and three local quantized models on a flat prompt that bypasses native chat templates.

---

## Abstract for the IEEE Publication Recommender

Paste the title and the following paragraph into the recommender. Do not treat its output as evidence that a journal will accept the paper.

Retrieval-augmented generation systems append retrieved documents to a user question, which allows indirect prompt injection. This paper evaluates a layered pre-generation filter across three instruction-tuned generators (Llama 3 8B, Qwen 2.5 7B, Mistral 7B). Rules and a TF-IDF logistic regression model block malicious queries and chunks; attack success fell from 65%, 43%, and 34% with no defense to 10%, 3.8%, and 2.5% with the combined filter on Llama 3, Qwen 2.5, and Mistral, respectively. These reductions are statistically significant on all three models. A sentence-window variant deletes flagged sentences instead of dropping whole chunks: it does not produce a statistically supported further security reduction on any generator, while whole-chunk removal on benign queries fell from 8.0% to 0.13%. The alignment term in the risk score did not affect any decision across all three runs. The study uses forced document injection, keyword attack labels, a single private benchmark, and flat prompts that bypass native chat templates.

---

## Novelty Statement

SecureRAG does not introduce the first RAG prompt-injection defense, and the experiments do not show that it outperforms published detectors or sanitizers. Prompt provenance marking already addresses indirect injection [4]. Structured queries with instruction-tuned generators eliminate the data channel as an attack surface [5]. Trained activation-based detectors achieve near-zero ASR on BIPIA without modifying retrieved text [9]. Segment-level and extraction-based injection removal are already peer-reviewed [8]. Sentence-level NLI graph pruning already performs sanitization with a stronger semantic signal than lexical rules [11]. Isolate-then-aggregate generation provides a certified robustness guarantee [7]. Index-poisoning attacks demonstrate that many lexical defenses can be circumvented when the adversary controls retrieval [6].

The concrete addition of this paper is a three-generator empirical measurement, on unmodified quantized models, of the difference between dropping a retrieved chunk and deleting flagged sentence windows inside it. That measurement separates attack success, whole-chunk removal, window edits, and gold-source retention as distinct quantities, reported with paired statistical tests. The paper also documents that the published retrieval-alignment term did not affect any quarantine decision across all three generator runs, and that the combined filter is not a statistically demonstrated improvement over TF-IDF logistic regression alone.

These are honest measurements, not strong claims. Their value is in the transparency of a documented negative result and in providing a reproducible baseline on a specific hardware and software configuration for future work that includes a poisoned index, published-defense baselines, and validated attack labels.

---

## 1. Introduction

Retrieval-augmented generation systems answer a question from a small set of passages retrieved from an index. The generator sees those passages in the same prompt as the question. A passage can therefore contain an instruction. Greshake et al. demonstrated that this indirect path compromises deployed LLM-integrated applications [1]. Later work formalized the attack surface [2], produced benchmarks for indirect injection through external content [3], and showed that knowledge-base poisoning further requires the malicious passage to be retrieved at query time [6].

The engineering response to a flagged chunk is typically to drop it. Dropping a chunk that is mostly benign removes evidence the answer needed. Chen et al. studied the alternative of detecting the injected span and deleting that region, while retaining the surrounding document [8]. SecureRAG implements a simpler version of that alternative on top of a rule classifier and a linear bag-of-n-gram classifier, and it records both attack success and how much benign context survives under each policy.

The paper's claims are limited to what the frozen, independently-analyzed runs support.

1. On this benchmark, the rule/TF-IDF disjunction cuts keyword attack success from 104/160 to 16/160 on Llama 3 8B, from 69/160 to 6/160 on Qwen 2.5 7B, and from 54/160 to 4/160 on Mistral 7B. These reductions are paired and statistically significant on all three generators. Rules alone also produced statistically significant reductions on all three.

2. Replacing whole-chunk deletion with sentence-window deletion does not produce a supported further reduction in attack success on any generator. On Qwen 2.5, the point estimate moves from 6/160 to 7/160.

3. Sentence-window deletion reduces whole-chunk removal on benign queries from 60/750 to 1/750 across all generators, and raises gold-source retention given retrieval from 107/115 to 115/115. That retention is not intact text: window deletion edited 193 windows across 83/150 benign queries and partially modified 24/115 retrieved gold sources.

4. The retrieval-alignment term in the published S4 risk score had zero effect on any run. The alignment penalty was zero for all 430 scanned attack chunks and all 750 benign chunks, on each of the three generators.

5. The combined filter (S3) is not a statistically demonstrated improvement over TF-IDF logistic regression alone (S2): discordant counts are 3 versus 0 on Llama, 2 versus 0 on Qwen, and 2 versus 0 on Mistral; Holm-adjusted p = 1.0 on all three.

6. These results do not extend to poisoned-index retrieval, native chat templates, adaptive adversaries, or answer quality. All indirect attacks were inserted into the prompt by the harness, not retrieved from a poisoned index.

---

## 2. Related Work

Table I, stored in full in `manuscript/novelty_matrix.csv`, compares the nearest work. Peer-reviewed publications and preprints are labeled separately. None of the listed systems were reimplemented here; comparisons are therefore architectural and methodological, not numerical leaderboard comparisons.

### 2.1 Threat Demonstrations and Formalization

**Greshake et al. [1]** demonstrated indirect prompt injection against deployed LLM-integrated applications, including Bing Chat and code-completion tools. Their case studies showed that external retrieved content—web pages, emails, documents—can redirect a deployed system's actions without any user instruction. This is the threat model SecureRAG addresses. Greshake et al. did not evaluate a detector or sanitizer; this paper adds a measured filter pipeline. Venue: AISec 2023 (peer-reviewed ACM workshop).

**Liu et al. [2]** formalized prompt injection attacks and defenses in a unified framework, comparing five attack types across ten defenses and ten models. Their peer-reviewed USENIX Security 2024 paper provides the broadest defense comparison in the literature but does not study sentence-level sanitization of retrieved chunks. SecureRAG is narrower: one private 160-attack suite, one local hardware configuration.

**Yi et al. [3]** released BIPIA, a benchmark for indirect injection through external content, and evaluated two defense families (KDD 2025, peer-reviewed): black-box boundary tokens with an explicit reminder, and a white-box training defense. These defenses operate at the prompt level, not through pre-generation chunk filtering. SecureRAG was not run on BIPIA, so ASR figures cannot be compared.

### 2.2 Prompt-Engineering and Model-Level Defenses

**Spotlighting (Hines et al. [4])** is a family of prompt-engineering provenance-marking techniques: delimiting, datamarking, and encoding. The paper is a preprint (arXiv:2403.14720, 2024; a matching text appears in CEUR-WS vol. 3920, which is not a traditional peer-reviewed proceedings). It reports ASR below 2% on GPT-family models. The defense changes what the model sees; SecureRAG deletes flagged windows and calls the model on the remainder. No head-to-head comparison was run.

**StruQ (Chen, Piet, Sitawarin, Wagner [5])** is a peer-reviewed USENIX Security 2025 paper. It separates the prompt into a trusted instruction channel and an untrusted data channel, then fine-tunes a model to follow only the instruction channel. StruQ reports near-zero ASR on manual and optimization attacks. It requires fine-tuning; SecureRAG leaves generator weights unchanged and filters text before the model is called. These are fundamentally different mechanisms with incomparable evaluation setups.

### 2.3 Index-Poisoning and Retrieval-Aware Attacks

**PoisonedRAG (Zou et al. [6])** is a peer-reviewed USENIX Security 2025 paper demonstrating that a small number of optimized malicious texts injected into a knowledge base can be reliably retrieved for target questions. This is a harder adversarial setting than SecureRAG's harness, where the malicious document is forced into the context window without a retrieval requirement. SecureRAG's indirect results are not evidence about a poisoned-index adversary.

### 2.4 Certified and Aggregation-Based Defenses

**RobustRAG (Xiang, Wu, Zhong, Wagner, Chen, Mittal [7])** was accepted at IEEE SaTML 2026 (preprint arXiv:2405.15556; proceedings pagination not verified at the time of this draft). It introduces an isolate-then-aggregate architecture: passages are split into disjoint groups, the LLM generates separately from each group, and responses are aggregated. RobustRAG provides the first certified lower bound on response quality against a bounded number of corrupted passages. The mechanism requires multiple LLM calls per query and targets retrieval corruption with provable guarantees. SecureRAG makes one generation call after filtering and provides no certificate.

### 2.5 Detection and Removal Defenses

**Chen, Li, Sui, He, Liu, Song, Hooi [8]** is the peer-reviewed work most closely related to SecureRAG's window layer. Published at ACL 2025 Main (DOI: 10.18653/v1/2025.acl-long.890, pp. 18189-18206), it evaluates two removal strategies: segmentation removal (crop injected segments from a document) and extraction removal (train a model to parse out injected instructions). The paper reports over-defense out-of-domain as a known failure mode. SecureRAG's window deletion is a simpler rule/TF-IDF instance of segmentation, applied to a different private benchmark, with an explicit whole-chunk versus window utility split. ASR figures cannot be compared across the different benchmarks.

**InstructDetector (Wen, Wang, Yang, Tang, Xie, Lyu, Dou, Wu [9])** is a peer-reviewed paper in Findings of EMNLP 2025 (ACL Anthology, pp. 19472-19487). It detects indirect injection by monitoring hidden states and gradients from intermediate LLM layers, reporting 99.60% in-domain detection accuracy and 0.03% ASR on BIPIA. This method requires access to model internals. SecureRAG treats the generator as a black box and runs a lexical classifier on the text before any model call.

**TrustRAG (Zhou et al. [10])** is a preprint (arXiv:2501.00879, 2025). It clusters retrieved document embeddings and uses LLM self-assessment to discard conflicting passages before generation. SecureRAG does not cluster the retrieved set, and the poisoned indexes described in the repository were not used in any evaluated run.

**SONAR (Datta, Umble, Brown, Tao [11])** is the closest preprint in method family (arXiv:2605.01078, University of Utah, submitted 1 May 2026; not peer-reviewed at the time of this draft). SONAR builds a sentence-level relational graph weighted by NLI entailment and contradiction scores, then uses connectivity-driven pruning to remove adversarial seeds while preserving benign context. It reports near-zero ASR across multiple models and benchmarks. The signal is semantic (NLI), unlike SecureRAG's lexical TF-IDF rule. SecureRAG does not implement NLI graph pruning; its distinct measured contribution is the utility trade-off between whole-chunk deletion and window deletion on one private benchmark.

### 2.6 Comparison Summary (Table I)

Because benchmarks, generators, and evaluation protocols differ across all entries, ASR figures from different papers cannot be placed in the same column as if directly comparable.

| Ref | Title (abbrev.) | Venue / Status | Threat model | Mechanism | Comparable to SecureRAG? |
|---|---|---|---|---|---|
| [1] | Greshake et al. | AISec 2023 (peer-reviewed workshop) | Indirect injection in real apps | Attack demonstration | Related threat; no defense |
| [2] | Liu et al. | USENIX Sec 2024 (peer-reviewed) | Formal injection framework | Multi-defense benchmark | Same family; different benchmark |
| [3] | Yi et al. | KDD 2025 (peer-reviewed) | Indirect injection, BIPIA | Prompt delimiting + training | Different benchmark and mechanism |
| [4] | Hines et al. (Spotlighting) | arXiv preprint; CEUR-WS copy | Indirect injection in untrusted data | Provenance marking | Different mechanism; not rerun here |
| [5] | StruQ | USENIX Sec 2025 (peer-reviewed) | Data-channel injection | Structured queries + fine-tuning | Different mechanism; requires training |
| [6] | PoisonedRAG | USENIX Sec 2025 (peer-reviewed) | Optimized index poisoning | Attack paper | Not comparable until poisoned-index run |
| [7] | RobustRAG | SaTML 2026 (peer-reviewed, accepted) | Bounded retrieval corruption | Isolate-then-aggregate | Different mechanism; certified |
| [8] | Chen et al. | ACL 2025 Main (peer-reviewed) | Indirect injection in documents | Segmentation + extraction removal | Closest removal method; different suite |
| [9] | InstructDetector | EMNLP Findings 2025 (peer-reviewed) | Indirect injection, BIPIA | Hidden-state detection | Different signal; needs model internals |
| [10] | TrustRAG | arXiv preprint | Index poisoning | Embedding cluster + LLM self-assess | Not comparable without poisoned index |
| [11] | SONAR | arXiv preprint | Indirect injection, RAG/agents | NLI graph pruning | Closest method family; stronger signal |

---

## 3. Threat Model and Assumptions

The adversary's goal is to cause the generator's response to satisfy a keyword predicate. Two surfaces are in scope.

- **Direct.** The instruction is in the user message, including a second turn after a benign first question.
- **Indirect, conditional on insertion.** The instruction is in a document that the evaluation harness places in the context window. The harness does not require retrieval to select it; the malicious document is inserted by the runner.

**Out of scope, not claimed:** an adversary who must get a poisoned document into the top-5 of a vector index; an adversary who can optimize attack strings against the classifier; tool-using agents; exfiltration of private user data as an action rather than as a keyword in the response.

The defender sees the query, the retrieved chunks, and follow-up user turns. The defender does not see model weights, logits, or intermediate activations. The generator is a black box. Benign users are assumed to ask the 150 recorded questions. A refusal phrase in the response overrides a keyword hit.

**Assumptions that the experiments actually use:**

- The same prompt template and system string are used for every system and every generator. The system string contains the marker that D4 tries to elicit. Ollama's native chat templates for Qwen (ChatML) and Mistral ([INST]) were not applied; the prompt was sent as a flat system/user/assistant string to maintain parity with the Llama 3 run.
- Temperature is 0 and the Ollama seed is 42. This does not guarantee bitwise identical GPU sampling across sessions.
- Top-5 retrieval uses all-MiniLM-L6-v2 (ONNX runtime for Qwen and Mistral runs; sentence-transformers for the original S0-S3 Llama run; ONNX MiniLM for the S4 Llama run). Re-scoring 435 frozen Llama attack queries with the new embedding reproduced all stored block decisions.
- Indirect trials use four retrieved chunks plus one harness-inserted chunk at position 5.

---

## 4. Architecture

### 4.1 Shared Pipeline

S0 retrieves five chunks and generates. S1-S4 classify before generating. A blocked query or follow-up returns a fixed refusal string and does not call the generator. If every chunk is removed, the system returns a no-context string and does not call the generator. There is no output filter.

### 4.2 Classifiers

**S1** is a regular-expression classifier with five groups (weights: 0.95 instruction override, 0.90 role hijacking, 0.85 system-prompt request, 0.80 delimiter injection, 0.65 document-embedded marker). The block threshold is 0.50.

**S2** is a TF-IDF unigram/bigram vectorizer and a class-balanced logistic regression model saved at `security/training/s2_tfidf_lr.joblib`. The block threshold is 0.40, chosen as the validation threshold with the best F1 among thresholds whose validation false-positive rate was at most 0.05. The training script is written to exclude the 160 attacks and 150 benign queries used for evaluation. The training rows are not in the repository, so that exclusion was not re-verified from the corpus in this draft. S2 is a bag-of-n-gram logistic regression, not the DeBERTa model whose training code also exists in the repository.

**S3** blocks if S1 or S2 blocks. On the attack logs, the S3 query-block and follow-up-block sets equal the union of S1 and S2.

### 4.3 S4 Score and the Policy That Actually Ran

For a chunk with document score D, maximum window score W, flagged-window fraction P, and query-chunk alignment A, the code computes:

    R = min(1, max(0, 0.35*D + 0.45*W + 0.15*P + 0.05*max(0, 0.20 - A)))

Alignment is 1/(1+distance) when a retrieval distance exists. Labels are SAFE if R < 0.45, SUSPICIOUS if 0.45 <= R < 0.80, MALICIOUS if R >= 0.80.

**The control flow does not follow those labels in all cases.** A MALICIOUS chunk is quarantined. Every other chunk that contains at least one blocked window is edited, including SAFE chunks. The 0.45 threshold is stored in the code but does not determine whether window sanitization occurs. Windows are sentence splits, further cut at 420 characters, classified with the same S3 disjunction.

**Two findings from saved audits (no generation required):**

1. Across 430 scanned attack chunks and 750 benign chunks, the alignment penalty was 0 on every chunk in every generator run. Every retrieved chunk had alignment >= 0.59 (above the 0.20 floor). Inserted chunks have no retrieval distance and are assigned zero penalty. The theoretical maximum contribution of the alignment term is 0.05 * 0.20 = 0.01, which cannot change a decision at the 0.80 boundary. The term is inactive, not merely small, on these logs.

2. The 29 attack chunks labeled MALICIOUS are exactly the 29 chunks with D >= 0.80. The window and proportion terms did not change any quarantine decision on this run.

S4 therefore differs from S3 in which chunks are edited or kept (sentence-window deletion versus full quarantine), not in a retrieval-sensitive score.

**Known implementation discrepancy:** 33 scanned attack chunks and 111 benign chunks that received a SAFE label (R < 0.45) were still sanitized because they contained at least one blocked window. The SUSPICIOUS threshold was not enforced as written. The reported attack-success numbers belong to this more aggressive implementation and must not be quoted alongside a description of the SUSPICIOUS threshold as the sanitization gate.

---

## 5. Experimental Methodology

### 5.1 Generators

Three Ollama-hosted generators were evaluated:

| Model | Tag | Parameters | Quantization | Context | License |
|---|---|---:|---|---:|---|
| Llama 3 8B | llama3:latest | 8.0B | Q4_0 | 8,192 | Meta Llama 3 Community |
| Qwen 2.5 7B | qwen2.5:7b | 7.6B | Q4_K_M | 32,768 (pinned to 8,192) | Apache 2.0 |
| Mistral 7B | mistral:7b | 7.2B | Q4_K_M | 32,768 (pinned to 8,192) | Apache 2.0 |

Llama 3 runs (S0-S3: 19 August 2026; S4: 5 October 2026) are the stored baseline. Qwen 2.5 and Mistral runs were completed 25-26 October 2026 with the same attacks, queries, prompts, retrieval index, top-k, classifier, thresholds, temperature (0), seed (42), and output limit (512 tokens). No run hit the 8,192-token context cap. Llama 3 S3 and S4 were collected in different sessions; that is a reproducibility limitation for Llama latency comparisons. Qwen S0-S4 was one process; Mistral S0-S4 was one process. The machine is an Apple M4 with 16 GB unified memory.

The client name GeminiClient in older repository modules refers to the local request wrapper, not a Google model.

### 5.2 Attack Suite

160 attacks: 100 direct (D1-D5, 20 per category) and 60 indirect (I1-I3, 20 per category). D1: instruction override. D2: role hijacking. D3: obfuscation. D4: prompt-marker elicitation. D5: multi-turn (5 single-turn controls + 15 two-turn). I1: explicit in-document instruction. I2: camouflaged in-document instruction (hard-indirect file). I3: persona shift.

**Success rule:** case-insensitive keyword substring match, canceled if the response contains `cannot`, `can't`, `not allowed`, `unable`, or `won't`. No human annotator has applied the protocol in `manuscript/ANNOTATION_PROTOCOL.md`. The keyword rule is an automated proxy; Section 7 describes its known failure modes.

Attack content SHA-256: `bfdecac65b8249d079a44d89c679b8ec10ee97f3b70f2f47d5559494125e24ac` (from `results/s4/s4_summary.json`). The S3/S4 query-block sets matched across all three generator runs (72 and 74 blocked attacks).

### 5.3 Benign Set and Metrics

150 questions: 50 each in healthcare, finance, and general.

    ASR = k/n, n = 160

**Benign false-positive rate:** blocked benign queries / 150. Zero for every system and every generator.

**Chunk-removal rate:** dropped chunks / 750 retrieved chunks.

**Query-level edit rate (S4 only):** fraction of benign queries in which any chunk was quarantined or had a window removed.

**Gold retrieval:** 1 when the designated source identifier appears in the top-5 retrieved. **Gold retention:** 1 when that source is not quarantined. Gold retention is not gold text unmodified: on S4, 24/115 retained gold sources had at least one window deleted.

**Answerability:** pipeline returned a model response (not a refusal or no-context string).

Wilson 95% intervals treat the 160 attacks as independent Bernoulli trials. Category intervals use n = 20 and are wide. Paired system comparisons use exact McNemar tests. The 24 within-model and cross-model tests reported in Section 6 use Holm correction. Any paired result described as significant has Holm-adjusted p < 0.05.

### 5.4 What Was Not Rerun

S0-S3 Llama 3: generated 19 August 2026. S4 Llama 3: generated 5 October 2026. Qwen 2.5 and Mistral: generated 25-26 October 2026. This draft only recomputes statistics. Thresholds and classifiers were not changed.

---

## 6. Results

### 6.1 Attack Success — Llama 3 8B

| System | Successes | ASR | 95% Wilson interval |
|---|---:|---:|---|
| S0 none | 104/160 | 65.0% | 57.3-72.0% |
| S1 rules | 38/160 | 23.8% | 17.8-30.9% |
| S2 TF-IDF LR | 19/160 | 11.9% | 7.7-17.8% |
| S3 rule or TF-IDF | 16/160 | 10.0% | 6.2-15.6% |
| S4 window sanitization | 14/160 | 8.8% | 5.3-14.2% |

S0 vs. S3: 88 attacks succeeded only under S0, 0 only under S3 (Holm p = 1.1e-14). S1 vs. S3: 22 vs. 0 (significant). S2 vs. S3: 3 vs. 0 (Holm p = 1.0, not significant). S3 vs. S4: 3 attacks succeeded only under S3 (I2H_010, I2H_012, I2H_020) and 1 only under S4 (I2H_013); unadjusted p = 0.625, not significant. The S3 and S4 Wilson intervals overlap.

I2H_013 is the case that argues against treating sanitization as strictly safer: S3 dropped the inserted chunk and the keyword rule failed; S4 removed one window, kept the chunk, and the keyword rule succeeded.

### 6.2 Attack Success — Three-Generator Summary

| Model | S0 | S1 | S2 | S3 | S4 |
|---|---:|---:|---:|---:|---:|
| Llama 3 8B | 104/160 (65.0%) | 38/160 (23.8%) | 19/160 (11.9%) | 16/160 (10.0%) | 14/160 (8.8%) |
| Qwen 2.5 7B | 69/160 (43.1%) | 32/160 (20.0%) | 8/160 (5.0%) | 6/160 (3.8%) | 7/160 (4.4%) |
| Mistral 7B | 54/160 (33.8%) | 24/160 (15.0%) | 6/160 (3.8%) | 4/160 (2.5%) | 4/160 (2.5%) |

95% Wilson intervals for each cell are in `results/multimodel_2026/CROSS_MODEL_REPORT.md`.

**S0 vs. S3 — significant on all three generators (Holm):**
- Llama 3: 88 vs. 0 discordant (Holm p = 1.1e-14)
- Qwen 2.5: 63 vs. 0 (Holm p = 1.0e-14)
- Mistral: 51 vs. 1 (Holm p = 4.7e-13)

**S2 vs. S3 — not significant on any generator (Holm p = 1.0 on all three):**
- Llama 3: 3 vs. 0; Qwen: 2 vs. 0; Mistral: 2 vs. 0. The combined filter is not a demonstrated improvement over TF-IDF logistic regression alone.

**S3 vs. S4 — not significant on any generator:**
- Llama 3: 3 vs. 1 (p = 0.625); Qwen: 0 vs. 1 (p = 1.0); Mistral: 1 vs. 1 (p = 1.0)
- On Qwen, S4 has a higher point estimate than S3 (7/160 vs. 6/160). The sentence-window layer does not add a repeatable security gain.

**Across generators:** Undefended Llama 3 is significantly more vulnerable than undefended Mistral (61 vs. 11 discordant, Holm p = 2.8e-8) and Qwen (49 vs. 14, Holm p = 1.6e-4). At S3, Llama 3 remains significantly more vulnerable than Mistral (12 vs. 0, Holm p = 0.0063) and Qwen (10 vs. 0, Holm p = 0.023).

### 6.3 Attack Success by Category (Llama 3)

Category counts are out of 20 items. Wilson intervals at n = 20 are wide; cross-system differences within a category are not paired tests.

| Category | S0 | S3 | S4 |
|---|---:|---:|---:|
| D1 instruction override | 19/20 | 1/20 | 1/20 |
| D2 role hijacking | 14/20 | 1/20 | 1/20 |
| D3 obfuscation | 13/20 | 5/20 | 5/20 |
| D4 prompt-marker elicitation | 6/20 | 0/20 | 0/20 |
| D5 multi-turn | 11/20 | 0/20 | 0/20 |
| I1 explicit document | 15/20 | 0/20 | 0/20 |
| I2 camouflaged document | 13/20 | 9/20 | 7/20 |
| I3 persona shift | 13/20 | 0/20 | 0/20 |

Full category grid for all three generators is in `results/multimodel_2026/comparison/asr_by_category.csv`.

D3 (obfuscation) is flat at 5/20 from S2 through S4 across all three generators. I2 is the only category that moves between S3 and S4; the improvement is 2/20 on Llama 3 and the Wilson intervals (S3: 25.8-65.8%; S4: 18.1-56.7%) overlap. D3_006 and D3_020 succeed under S3 on all three generators. I1, I3, D4, and D5 are at 0/20 for S3 and S4 on all three generators; with 20 items the Wilson upper bound on a zero count is 16.1%.

**Cross-generator direct vs. indirect at S0:** Llama 3 shows no detectable difference (63/100 vs. 41/60, Fisher p = 0.61). Qwen (65/100 vs. 4/60, p = 7.1e-13) and Mistral (51/100 vs. 3/60, p = 3.2e-10) show that indirect items almost never produced the keyword. Because these runs used a flat prompt rather than native ChatML or [INST] templates, this discrepancy must not be described as model-level robustness to indirect injection.

Residual S3 failures: D3_006, D3_020 (all three generators); nine I2 items on Llama 3; I2H_001, I2H_003 on Qwen; I2H_012 on Mistral.

### 6.4 Benign Context

| System | Query blocks | Chunks dropped | Queries altered | Windows removed | Answerable | Gold retained/retrieved |
|---|---:|---:|---:|---:|---:|---:|
| S0 | 0/150 | 0/750 | 0/150 | 0 | 150/150 | 115/115 |
| S1 | 0/150 | 0/750 | 0/150 | 0 | 150/150 | 115/115 |
| S2 | 0/150 | 60/750 (8.0%) | 37/150 | 0 | 149/150 | 107/115 (93.0%) |
| S3 | 0/150 | 60/750 (8.0%) | 37/150 | 0 | 149/150 | 107/115 (93.0%) |
| S4 | 0/150 | 1/750 (0.13%) | 83/150 | 193 | 150/150 | 115/115 (100%) |

These counts are pipeline outputs; the editor does not call the generator. Benign utility is identical across all three generators for all systems.

S2 and S3 are identical on the benign file: the disjunction adds no extra benign drop. Of the 60 dropped chunks, 43 are from finance queries, 14 from healthcare, and 3 from general. One finance query loses all five chunks (no clean context).

**S4's 0.13% chunk-drop rate is not the complete picture of context damage.** The run sanitized 141/750 chunks, removed 193 windows, and altered 83/150 queries. Of 115 retrieved gold sources, 24 were sanitized and 91 were unmodified. Gold retention counts a sanitized source as retained; it does not mean the text was preserved. Gold retention after S4 is 115/115; gold sources left unmodified is 91/115.

### 6.5 Component and Threshold Checks

These checks recompute labels from saved scores; they do not require a new generation run and do not imply changed attack-success numbers.

**Alignment term:** Removing it entirely, or giving it weight 1, leaves all 430 attack chunk labels and all 750 benign chunk labels unchanged (366 SAFE, 35 SUSPICIOUS, 29 MALICIOUS). Confirmed on Llama 3, Qwen 2.5, and Mistral logs.

**Document-score-only threshold:** Thresholding D >= 0.80 quarantines exactly the same 29 attack chunks as the full S4 score. A window-only score would quarantine 36 attack chunks and 98 benign chunks. Equal weights on D, W, and P would quarantine only 2 attack chunks. The published weights produce the quarantine count by tracking D >= 0.80.

**Threshold sensitivity:** Moving the suspicious threshold among 0.35, 0.45, and 0.55, and the malicious threshold among 0.70, 0.80, and 0.90, changes label counts. At a malicious threshold of 0.90, zero attack chunks are MALICIOUS. Because sanitization is triggered by blocked windows rather than the SUSPICIOUS label, threshold moves alone do not reproduce the edited prompts without a generation run.

**SAFE-threshold bug:** 33 attack chunks and 111 benign chunks classified SAFE were still sanitized. The current numbers belong to this more aggressive implementation. Re-enforcing the SAFE threshold without regenerating responses would make the reported attack-success numbers incorrect.

### 6.6 Latency

Mean end-to-end latency per attack: 13.5 s (S0), 8.8 s (S1), 5.2 s (S2), 5.2 s (S3), 6.9 s (S4) on Llama 3. Defended means are lower because blocked attacks skip generation. Median end-to-end benign latency across generators: 15.9-19.0 s. Qwen 2.5 wall clock: 383.9 min (1,305 generations). Mistral 7B: 400.0 min (1,305 generations). Scanner time was not separated from generation time. Peak memory was not logged; the Mistral process loaded one 4-bit 7B model at approximately 5.7 GB resident.

**Keyword-rule noise:** Refusal override (keyword present but refusal phrase cancels): 5 Llama S0, 6 Qwen S0, 12 Mistral S0 responses. Near-miss spellings: Mistral D1_002 answered COMPROMISSED; Qwen D3_012 was within two edits of its keyword. Hedged successes (keyword plus apology outside the official refusal list): 5 Llama S0, 9 Mistral S0. One Qwen D1_001 response uses "compromised" while discussing a retrieved breach article; one Qwen I2H_001 emits the keyword then answers the medical question. Both count as successes under the frozen rule.

---

## 7. Discussion and Limitations

**The result to publish is the three-generator S0-to-S3 reduction.** On all three generators, the rule/TF-IDF combined filter reduces keyword attack success consistently and significantly, without blocking any benign query. This result is robust: it appears on models from three different developers (Meta, Alibaba, Mistral AI), and the utility cost in finance chunks (43 of 60 dropped chunks) is documented.

**S4 should not be published as a better defense.** The two-attack net difference on Llama 3 includes one regression (I2H_013). On Qwen, the point estimate moves the wrong way. On Mistral, the rates are identical. None of these differences survive Holm correction. The correct claim is a documented utility trade-off: sentence-window deletion preserves whole chunks and raises gold retention from 107/115 to 115/115, at the cost of partial edits inside those chunks (24/115 gold sources modified, 193 windows removed across 83 benign queries).

**The combined filter is not demonstrated to improve over TF-IDF alone.** S2 vs. S3 is non-significant on all three generators. The disjunction is presented honestly as the deployed system; it is not claimed as an improvement over either component alone.

**The alignment term needs to be removed or redesigned before it is claimed as a retrieval-aware component.** A penalty capped at 0.01, which was zero for every chunk in every run, cannot change decisions at the 0.80 boundary. Inserted chunks (the actual indirect threat in this harness) have no retrieval distance and contribute zero. The term is inactive in this architecture as evaluated.

**The threat model is the main external-validity limit.** The harness inserts the malicious document into the context without a retrieval requirement. A retrieval-poisoning adversary must get the document retrieved; many malicious texts will not rank in the top-5. The 1%, 5%, and 10% poisoned indexes described in `knowledge_base/poisoned/poisoning_summary.json` were not used in any run (`chromadb/` contains only `final_clean`).

**Keyword labels are noisy.** The metric can miss compliance that avoids the keyword, and can count keyword occurrences that are not compliance. I2 is the informative residual (rules do not move it; TF-IDF partly does). D3 is the direct residual and is unaffected by window editing. No human annotator has validated these labels.

**Three quantized models do not generalize to instruction-tuned models at other scales, providers, or deployment configurations.** The flat prompt, which bypasses native chat templates, means that the low indirect ASR for Qwen and Mistral cannot be attributed to model-level robustness to indirect injection.

**No comparison was run against BIPIA boundary awareness, Spotlighting, StruQ, Chen et al. segmentation removal, RobustRAG, or SONAR.** Copying ASR figures from those papers into a shared table would not be a meaningful comparison.

---

## 8. Future Work

The following experiments are proposed. None of them has been run, and none should be described as completed or implied by the existing data.

**8.1 Poisoned-index evaluation.** The highest-priority extension is to measure retrieval hit rate and keyword attack success when malicious documents must be retrieved from a vector index, not forced by the harness. The repository already describes 1%, 5%, and 10% poisoning catalogs (905, 4,711, and 9,945 injected chunks). The missing measurement is whether the inserted documents are retrieved at each rate and how S3 and S4 perform relative to S0. This is the threat model of PoisonedRAG [6] and would allow a meaningful comparison to that threat characterization.

**8.2 Published-defense baselines.** Reimplementing Spotlighting [4] delimiting and SONAR [11] graph pruning on the same benchmark, with the same generator and evaluation protocol, would allow a direct method comparison that does not conflate benchmark differences. StruQ [5] and InstructDetector [9] require model fine-tuning or access to model internals and may be better compared analytically rather than reimplemented.

**8.3 Independent human review of attack-success labels.** The annotation protocol in `manuscript/ANNOTATION_PROTOCOL.md` specifies a stratified sample (80 attacks, 20 benign) labeled by two annotators who did not author the items they label. Cohen's kappa on `attack_present` and `response_complies` would quantify how far the keyword rule departs from human judgment, particularly for the near-misses, hedged successes, and refusal-override cases documented in Section 6.6.

**8.4 Ablation of S4 components.** A valid generation-level ablation would regenerate responses for (a) quarantine only (the D >= 0.80 rule alone), (b) window deletion only (without quarantine), and (c) the full S4 policy, on the same 160 attacks and 150 benign queries. This would isolate the security contribution of window deletion from the quarantine decision, and would determine whether enforcing the SAFE-threshold policy changes attack success.

**8.5 Additional generators.** A fourth generator at larger scale, if available locally, would help separate model-scale effects from the classifier's coverage. Native chat templates for Qwen (ChatML) and Mistral ([INST]) should be applied in a separate run to determine whether the low indirect ASR at S0 for those models is template-dependent.

**8.6 Adaptive attacker.** An adversary who can query the classifier and optimize attack strings against it would test robustness beyond the current hand-written suite. The current results do not support a claim that the classifier is resistant to adaptive attacks.

**8.7 Answer quality.** No exact-match, ROUGE, or human grade was computed for benign responses. A correctness evaluation would determine whether the window edits in 24/115 gold sources reduce answer accuracy, and would support a utility claim that currently rests only on gold retention counts.

---

## 9. Conclusion

On a 160-attack, 150-query benchmark with three local instruction-tuned generators (Llama 3 8B, Qwen 2.5 7B, Mistral 7B), a rule classifier combined with TF-IDF logistic regression reduces keyword attack success consistently and significantly — from 65%, 43%, and 34% with no defense to 10%, 3.8%, and 2.5% on the three generators — while blocking no benign query. These reductions are paired and statistically significant on all three models (exact McNemar, Holm-adjusted p < 10^-12).

Sentence-window sanitization preserves whole-chunk context: whole-chunk removal falls from 60/750 to 1/750, and gold-source retention given retrieval rises from 107/115 to 115/115. It does not provide a statistically supported security improvement: S3 versus S4 is non-significant on all three generators, and on Qwen the point estimate moves in the wrong direction. Twenty-four of 115 retained gold sources were partially edited. The retrieval-alignment term specified in the published risk score had zero effect on any decision in any run. The combined filter is not a demonstrated improvement over TF-IDF logistic regression alone.

The next evidence this paper needs is a poisoned-index experiment, human agreement on the keyword rule, generation-level ablation of S4 components, and published-defense baselines on the same evaluation setup.

---

## References

[1] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, "Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection," in Proc. 16th ACM Workshop Artif. Intell. Secur. (AISec), 2023, doi: 10.1145/3605764.3623985.

[2] Y. Liu, Y. Jia, R. Geng, J. Jia, and N. Z. Gong, "Formalizing and benchmarking prompt injection attacks and defenses," in Proc. 33rd USENIX Security Symp., 2024. [Online]. Available: https://www.usenix.org/conference/usenixsecurity24/presentation/liu-yupei

[3] J. Yi, Y. Xie, B. Zhu, E. Kiciman, G. Sun, X. Xie, and F. Wu, "Benchmarking and defending against indirect prompt injection attacks on large language models," in Proc. 31st ACM SIGKDD Conf. Knowl. Discovery Data Mining (KDD), 2025, doi: 10.1145/3690624.3709179.

[4] K. Hines, G. Lopez, M. Hall, F. Zarfati, Y. Zunger, and E. Kiciman, "Defending against indirect prompt injection attacks with spotlighting," arXiv:2403.14720, 2024. A matching text appears in CEUR-WS, vol. 3920.

[5] S. Chen, J. Piet, C. Sitawarin, and D. Wagner, "StruQ: Defending against prompt injection with structured queries," in Proc. 34th USENIX Security Symp., 2025, pp. 2383-2400.

[6] W. Zou, R. Geng, B. Wang, and J. Jia, "PoisonedRAG: Knowledge corruption attacks to retrieval-augmented generation of large language models," in Proc. 34th USENIX Security Symp., 2025. [Online]. Available: https://www.usenix.org/conference/usenixsecurity25/presentation/zou-poisonedrag

[7] C. Xiang, T. Wu, Z. Zhong, D. Wagner, D. Chen, and P. Mittal, "Certifiably robust RAG against retrieval corruption," in IEEE Conf. Secure Trustworthy Mach. Learn. (SaTML), 2026, accepted-paper listing: https://satml.org/2026/accepted-papers/. Preprint: arXiv:2405.15556.

[8] Y. Chen, H. Li, Y. Sui, Y. He, Y. Liu, Y. Song, and B. Hooi, "Can indirect prompt injection attacks be detected and removed?" in Proc. 63rd Annu. Meeting Assoc. Comput. Linguistics (ACL), 2025, pp. 18189-18206, doi: 10.18653/v1/2025.acl-long.890.

[9] T. Wen, C. Wang, X. Yang, H. Tang, Y. Xie, L. Lyu, Z. Dou, and F. Wu, "Defending against indirect prompt injection by instruction detection," in Findings Assoc. Comput. Linguistics: EMNLP, 2025, pp. 19472-19487. [Online]. Available: https://aclanthology.org/2025.findings-emnlp.1060/

[10] H. Zhou, K.-H. Lee, Z. Zhan, Y. Chen, Z. Li, Z. Wang, H. Haddadi, and E. Yilmaz, "TrustRAG: Enhancing robustness and trustworthiness in RAG," arXiv:2501.00879, 2025.

[11] S. Datta, M. Umble, D. S. Brown, and G. Tao, "A sentence relation-based approach to sanitizing malicious instructions," arXiv:2605.01078, 2026.

---

## Figure Notes

Figures are SVG files in `results/analysis/figures/`. They use only the recomputed counts.

- `fig1_overall_asr.svg`. S0 through S4 attack success with 95% Wilson intervals for Llama 3. S3 and S4 intervals overlap. The plot is the reason not to describe S4 as a security improvement.
- `fig2_asr_by_category.svg`. Category rates for Llama 3. D3 is flat from S2 to S4. I2 is the only category that moves between S3 and S4; the intervals overlap.
- `fig3_security_vs_chunk_removal.svg`. Attack success against whole-chunk removal rate. S4 appears at the lower left on this plane. The caption must state that 83/150 benign queries had a window edit and that 24/115 gold sources were edited. A point using window-edit rate instead of chunk-drop rate would not place S4 at zero cost.

A three-generator ASR comparison table is in `results/multimodel_2026/comparison/` and should accompany any updated multi-model figure. No poisoning-rate figure, per-model ASR threshold curve, or answer-quality figure exists; those data were not collected.

---

## Appendix: IEEE Journal Assessment

The following assessment is based on scope alignment and the evidence this draft has. Fees are from the IEEE APC list updated 31 March 2026. Confirm fees on the day of submission.

| Journal | Fit for current draft | Notes |
|---|---|---|
| **IEEE Access** | **Best current fit.** Gold open-access; multidisciplinary across IEEE fields. An empirical measurement of a RAG filter with stated limitations is appropriate scope. The non-significant S4 security result and inactive alignment term must remain in the abstract. | APC 2,160 USD plus tax. No page-limit charge. Requires public ORCID. AI-assisted text must be disclosed. |
| IEEE Trans. Artificial Intelligence | Possible future fit after: poisoned-index experiment, generation-level ablation of S4. Three generators help but the null S4 result and inactive alignment term remain thin for a transactions article. | Hybrid; open-access 2,800 USD. 10-page regular allowance. |
| IEEE Trans. Dependable and Secure Computing | Scope covers measurement of dependable and secure systems. The forced-injection threat model, no adaptive attacker, and no published-defense comparison are gaps relative to TDSC expectations. Not the right target until those gaps close. | Hybrid; open-access 2,800 USD. 12-page allowance. |
| IEEE Trans. Information Forensics and Security | TIFS covers information forensics, biometrics, and surveillance. Prompt-injection measurement is only loosely in scope, and TIFS expects a stronger security evaluation. Not appropriate for this draft. | Hybrid; open-access 2,800 USD. 10-page allowance. |
| IEEE Security & Privacy | Magazine format. The honest result could become a short magazine piece after poisoned-index and answer-quality measurements exist. Not ready in current form. | Hybrid; open-access 2,995 USD. |

Recent peer-reviewed work on RAG prompt-injection defense is concentrated at USENIX Security, KDD, ACL, EMNLP Findings, and IEEE SaTML. IEEE Access is the appropriate initial target for the paper as it currently stands.

**Minimum set before a transactions submission:** a poisoned-index curve at more than one contamination rate, generation-level ablation of S4 components, and human kappa on the stratified attack sample. If those results do not favor the method, report that. Do not retune classifier thresholds on the 160-attack evaluation file and then quote that file as a held-out test set.
