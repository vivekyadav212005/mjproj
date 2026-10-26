# IEEE journal fit and unfinished experiments

Date: 25 October 2026. The IEEE Publication Recommender at https://publication-recommender.ieee.org/home was not queried from this session. Paste the title and abstract below yourself. A recommender score is not an acceptance decision.

## Title to paste

SecureRAG: Sentence-Window Sanitization and the Security–Utility Trade-off for Prompt Injection in Retrieval-Augmented Generation

## Abstract to paste

Retrieval-augmented generation systems append retrieved documents to a user question, which allows indirect prompt injection. This paper evaluates a layered filter for that setting. Rules and a TF-IDF logistic regression model block malicious queries and chunks. A sentence-window variant deletes flagged sentences inside a chunk instead of dropping the chunk. On 160 attacks against Llama 3 8B, attack success fell from 104/160 with no defense to 16/160 with the combined filter. Sentence-window editing changed this to 14/160, which is not a statistically supported further reduction, while whole-chunk removal on 150 benign queries fell from 60/750 to 1/750. Retrieved gold sources were retained in 115/115 cases after window editing, but 24 of those sources were still partially deleted. The alignment term in the risk score did not affect any decision. The study uses forced document injection, keyword attack labels, and a single model.

## How the journals compare

Impact factors were not taken from a Journal Citation Reports extract in this session. The choice below follows scope, article type, and the evidence this draft actually has. Fees are from IEEE's article-processing-charge list updated 31 March 2026 (https://magazines.ieeeauthorcenter.ieee.org/wp-content/uploads/sites/10/IEEE-Article-Processing-Charges-List.pdf) and, for Access, from https://ieeeaccess.ieee.org/about/article-processing-charges/. Fees change. Confirm them on the day of submission. Open-access fees below are author-paid; hybrid journals also accept subscription publication.

| Journal | Why it is or is not a fit | Article type and fees, as verified |
|---|---|---|
| IEEE Access | Best current fit if the claims stay as narrow as the revised draft. Access is a multidisciplinary gold open-access journal covering IEEE fields. An empirical measurement of a RAG filter, with limitations stated, can sit there. It is still peer reviewed. The inactive alignment term and the non-significant S4 security gap must stay in the abstract, or the paper will read as over-claimed. | Full open access. APC 2,160 USD plus tax, no page-limit charge. IEEE recommends staying under about 20 pages. Word or LaTeX plus a matching PDF. Public ORCID. AI-assisted text must be disclosed in the acknowledgements and cited. Source: https://ieeeaccess.ieee.org/authors/submission-guidelines/ |
| IEEE Transactions on Artificial Intelligence | A possible later fit if a second model, a poisoned-index experiment, and a real component ablation are added, and if the contribution is the measurement method rather than an unused alignment formula. Not a fit for the draft as it stands: one model and a null security difference for the proposed layer are thin for a transactions article. | Hybrid. Open-access fee 2,800 USD. Overlength 200 USD per page. Regular-paper allowance 10 pages on the March 2026 APC list. Aims page was not re-read in this session; read it before deciding. |
| IEEE Transactions on Dependable and Secure Computing | Scope, from the Computer Society call, is foundations, design, modeling, and evaluation of dependable and secure systems, including measurement. A RAG defense can fit that sentence. The current threat model does not: documents are forced into the prompt, there is no adaptive attacker, and there is no comparison to published defenses. Submit here only after those gaps close. | Hybrid. Open-access fee 2,800 USD. Overlength 220 USD per page. Regular-paper allowance 12 pages. Source: https://www.computer.org/digital-library/journals/tq/cfp-dependable-secure-computing |
| IEEE Transactions on Information Forensics and Security | The Signal Processing Society describes TIFS as covering information forensics, information security, biometrics, surveillance, and systems that use those technologies. A prompt-injection measurement is only loosely in that scope. TIFS also expects a stronger security evaluation than keyword success on 160 hand-written items. Not the journal for this draft. | Hybrid. Open-access fee 2,800 USD. Overlength 220 USD per page. Regular-paper allowance 10 pages. Source: https://signalprocessingsociety.org/publications-resources/ieee-transactions-information-forensics-and-security |
| IEEE Security & Privacy | Magazine articles need a broader audience and a result that changes practice. The honest result here is a negative finding about an alignment score plus a utility trade-off on a private suite. That can become a short magazine piece after the poisoned-index and second-model measurements exist. It is not ready. | Hybrid. Open-access fee 2,995 USD on the March 2026 list. Page allowance was not clearly populated in the extracted table, so it is not stated here. |

Recent peer-reviewed work on this topic is concentrated at USENIX Security, KDD, ACL, EMNLP Findings, and IEEE SaTML, not in the transactions above. That is a reason to be careful with TIFS and TDSC, not a reason to avoid IEEE Access.

## Unfinished experiments

Do not describe any of the following as done.

1. **Attack-success ablation of S4.** Offline label counts are in `results/analysis/weight_variant_labels.csv` and `threshold_sensitivity_labels.csv`. They show that the alignment term changes nothing and that quarantine matches `document_score >= 0.80`. They do not say what the generator would have answered under a different policy. A valid ablation regenerates responses for S3, for window deletion alone, and for quarantine alone, on the same 160 attacks and 150 queries.

2. **Poisoned-index evaluation.** The runners insert the malicious document. Catalog files describe 905, 4,711, and 9,945 injected chunks at 1%, 5%, and 10%, but `chromadb/poisoned_1pct`, `poisoned_5pct`, and `poisoned_10pct` are not in the workspace. Only `chromadb/final_clean` is present. Retrieval hit rate and attack success at those rates were not measured.

3. **Additional attack types.** Obfuscation, role shift, and multi-turn items already exist (D2, D3, D5). What is missing is an attack that must be retrieved, and an attack optimized against the classifier. New attack strings were not added in this session. Duplicating the current 160 items would not create that evidence.

4. **Human annotation.** The protocol is `manuscript/ANNOTATION_PROTOCOL.md`. Zero items were double-annotated. Cohen's kappa was not computed.

5. **Second and third generators.** Installed and unused: `qwen2.5:7b` (7.6B, Q4_K_M, context 32768, Apache 2.0). Also installed: `qwen2.5-coder:7b`, which is a weak choice for general questions. The evaluated model is `llama3:latest` (8.0B, Q4_0, context 8192, Meta Llama 3 license). A full extra model is a multi-hour run at the observed latencies (about 5–20 seconds per item). It was not started.

6. **Answer correctness.** No exact match, ROUGE, or human grade against the benign ground truth.

7. **Memory and isolated defense latency.** End-to-end times are in the result files. Scanner time was not separated from generation time. Peak memory was not recorded.

8. **Head-to-head baselines.** BIPIA boundary awareness, Spotlighting, StruQ, Chen et al. segmentation removal, RobustRAG, and SONAR were not reimplemented.

9. **Training-corpus audit.** The S2 script excludes the evaluation texts. The training rows are not in the tree, so overlap was not re-checked.

10. **Spec-faithful S4.** The code sanitizes SAFE chunks that contain a blocked window. A unit test now records that behavior. Switching the policy without regenerating answers would make the old attack-success numbers false.

## Minimum set before a transactions submission

Keep Access as the working target until three results exist: a poisoned-index curve at more than one contamination rate, Llama 3 plus Qwen 2.5 under the same prompt and seed, and a generation ablation in which window deletion is the only change. Human kappa on the stratified sample should accompany the keyword metric. If those results do not favor the method, report that. Do not retune thresholds on the 160-attack file and then quote that file as a test set.
