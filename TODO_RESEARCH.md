# SecureRAG — implemented vs missing

Audit date: 2026-08-18. **Do not treat this as a rewrite list.** Existing S0–S3 classes, classifiers, attack JSON, and saved JSONL should stay until a change is explicitly chosen.

---

## Already implemented

### Data and knowledge base

- [x] Phase 1A loaders for seven prompt-injection datasets, balancing, exact + near dedup, 70/15/15 split scripts
- [x] Phase 1B corpus download / chunk / Chroma ingest scripts (PubMedQA, finance phrasebank + alpaca, SQuAD)
- [x] Clean ChromaDB on disk (`chromadb/clean`, collection `securerag_clean`, ~89.5k chunks)
- [x] Poisoned DB copies at 1% / 5% / 10% contamination
- [x] 63-doc poisoned catalog + 150 benign queries + retrieval-baseline JSON
- [x] 160-attack suite (100 direct + 60 indirect), validator, and paper-style suite report

### Models and defences (code)

- [x] S0 undefended RAG (`rag/baseline_rag.py`)
- [x] `RuleClassifier` G1–G5 with documented weights and threshold 0.5
- [x] S1 `RuleDefendedRAG.query()` (query + retrieved-doc rules + `_generate_response`)
- [x] S2 `MLDefendedRAG.query()` (rules + `SimpleMLDetector`)
- [x] S3 `HybridDefendedRAG.query()` (weighted hybrid + extra doc patterns)
- [x] DeBERTa **training and inference scripts** (`prepare_training_data.py`, `train_deberta.py`, `security/ml_classifier.py`)
- [x] Local LLM client (Ollama llama3, Gemini-compatible names)
- [x] Shared attack runner, ASR reporter, JSONL resume, `recompute_success_metrics.py`

### Results on disk

- [x] S0: 160/160 attacks, overall ASR 48.1%
- [x] S3: 160/160 attacks, overall ASR 23.8%
- [x] Retrieval baseline top-5 recall 0.78 (clean KB)

---

## Missing or not actually wired

### Critical for a valid S0–S3 comparison

- [ ] **S1 evaluation artifacts** — `results/s1/` does not exist
- [ ] **S2 evaluation artifacts** — `results/s2/` does not exist
- [ ] **DeBERTa weights** — `security/training/fine_tuned_model/` absent
- [ ] **Phase 1A frozen CSVs** — `data/processed/` absent; `hf_dataset/` empty
- [ ] **S2/S3 do not call DeBERTa** — they train TF-IDF LR on S0 responses at startup
- [ ] **Indirect-document defence never runs** on S1/S2/S3 (`BaselineRAG.query_with_injected_doc` wins MRO)
- [ ] **D5 multi-turn defence never runs** (same MRO; 8/15 S3 multi-turn attacks still succeed)
- [ ] **Mistral is not in the stack** — generation is `llama3:latest`; comments still say Gemini
- [ ] **Utility evaluation** — 150 benign queries exist; no ROUGE / EM / refusal-on-benign script
- [ ] **Poisoned-DB retrieval evaluation** — DBs built; attacks inject the doc instead of measuring retrieval-under-poisoning

### Bugs / evaluation holes to resolve before extending

- [ ] S3 query substitution (`[POTENTIALLY MALICIOUS QUERY FILTERED]`) retrieves unrelated chunks and answers them (SQL-injection traces in `results/s3`)
- [ ] `SimpleMLDetector` trains on **attack responses**, then is applied to **queries/docs**
- [ ] I2 ASR is 0% even on S0 — category cannot show defence gain
- [ ] S3 I1 ASR rose vs S0 (85% vs 65%) on the undefended injection path
- [ ] D4 can “succeed” by echoing the `SECURERAG_SYSTEM` marker from the S3 prompt
- [ ] `rag/s2_rag.py` calls missing `predict()`; unused but misleading
- [ ] `LLM_TEMPERATURE` / `LLM_MAX_TOKENS` never sent to Ollama
- [ ] Runners still require `GEMINI_API_KEY` and import `google.genai`
- [ ] Duplicate attack-suite and knowledge-base trees; retrieval reports disagree (0.78 vs 0.767)
- [ ] 63 poisoned catalog docs vs 60 I-attacks; two different payload template sets
- [ ] `S3_HARD_BLOCK_THRESHOLD = 0.85` unused (runtime uses 0.75)
- [ ] No README

### Artifacts gitignored / not checked in

- [ ] `data/raw`, `data/processed`
- [ ] `knowledge_base/corpora`, `knowledge_base/chunks`
- [ ] `security/training/checkpoints`, populated `hf_dataset`
- [ ] Chroma binaries (present locally, not in git)

---

## Recommended research extensions (preserve current code)

Order is “make the existing experiment interpretable” first, then add new work. Each item should be a small additive change, not a rewrite of `BaselineRAG` / `RuleClassifier` / the attack JSON.

1. **Document the actual generator** in config and reports (Ollama llama3). Decide explicitly whether to keep llama3, restore Gemini, or add Mistral as a *new* client behind `get_llm_client()` without deleting the Ollama path.
2. **Run S1 and S2** with the current classes so the paper has four comparable JSONLs. Do not overwrite `results/s0` or `results/s3`.
3. **Fix MRO without deleting S0 methods** — e.g. override `query_with_injected_doc` / `query_multi_turn` on S1–S3 (or reverse mixin order only on defended classes) so `BaseHelpers` defences actually execute. Then re-run S1–S3.
4. **Stop S3 query substitution** or retrieve with the original query after a block. Keep the current substitution behind a flag if you need to reproduce `results/s3`.
5. **Wire `security.ml_classifier.MLClassifier`** once DeBERTa is trained; keep `SimpleMLDetector` as a named baseline (S2-lite) rather than deleting it.
6. **Retrain SimpleML (if kept)** on queries/documents with labels that are not the evaluation suite, never on S0 responses.
7. **Utility pass** on the 150 benign queries for S0 vs each defence (answer quality + false-block rate).
8. **Retrieval-poisoning experiment** using `chromadb/poisoned_*` and the 63-doc catalog, separate from conditional-success-given-injection.
9. **I2 redesign or model swap** — llama3 currently ignores hidden-instruction payloads, so the category is uninformative.
10. **D4 metric** should not count the system-prompt marker when the defended prompt itself contains `SECURERAG_SYSTEM`.
11. **Reconcile attack-suite I1–I3 text** with `poisoned_docs_catalog.json` (or cite them as two different payload families).
12. **Threshold / ablation study** for G1–G5 weights, S3 0.75 vs unused 0.85, and doc threshold 0.40 — currently many numbers are hard-coded in more than one place.
13. **Clean entry-point naming** (`s1_rag.py` / `s2_rag.py` vs `defended_rag.py`) without removing either until you know which JSONLs were produced by which file. Current S1/S3 runners import `defended_rag.py`.

---

## What “done” looks like for the next paper-facing run

- Four result folders `results/s0`–`results/s3`, each 160 attacks, produced by a **named** LLM and a **named** classifier.
- A short note in each report stating whether indirect/D5 defences were active.
- DeBERTa test metrics (`training_report.txt`) **or** an explicit statement that S2 is TF-IDF LR.
- Utility table on the 150-query set.
- One attack-suite source of truth (no drifting duplicate JSON).
