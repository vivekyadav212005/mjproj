# SecureRAG Revision Audit

**Internal document — not submitted with the manuscript.**  
Initial revision: 26 October 2026.  
Independent audit appended: 26 October 2026.  
Phase 1–5 submission-preparation work completed: 26 October 2026.

---

## Phase 0 — Baseline

| File | Present | Notes |
|---|---|---|
| `manuscript/SecureRAG_revised.md` | Yes | 421 lines; starting draft |
| `manuscript/ANNOTATION_PROTOCOL.md` | Yes | Status: not executed |
| `results/analysis/verified_statistics.json` | Yes | 943 lines; primary data source |
| `results/multimodel_2026/CROSS_MODEL_REPORT.md` | Yes | Cross-model McNemar results |
| `results/s4/s4_summary.json` | Yes | |
| `results/s4/s4_utility_summary.json` | Yes | |
| `results/s4/s4_benign_results.csv` | Yes | 150 rows; source for 83/150 count |
| `results/analysis/figures/fig1_overall_asr.svg` | Yes | |
| `results/analysis/figures/fig2_asr_by_category.svg` | Yes | |
| `results/analysis/figures/fig3_security_vs_chunk_removal.svg` | Yes | |
| `security/s4_retrieval_risk.py` | Yes | |
| `security/s4_sentence_scanner.py` | Yes | |
| IEEE LaTeX class | No | IEEEtran [journal] used; appropriate |

---

## Corrections applied

### 1. Compile-breaking LaTeX command (FIXED — initial audit)

`\IEEEcompsocitemizethanks` / `\IEEEcompsocthanksitem` are defined only in
compsoc mode. Replaced with plain `\thanks{}` inside `\author{}`.

### 2. Unverifiable "435 frozen Llama attack queries" (FIXED — initial audit)

The number 435 appears nowhere in any data file. The verified count from
`verified_statistics.json` is 430 attack chunks scanned by the risk scorer.
Corrected to "Re-scoring 430 Llama attack chunks."

### 3. Window-only 98 benign description (FIXED — initial audit)

The text said "quarantine 36 attack and 98 benign chunks" for the window-only
score. Data: window_only benign MALICIOUS=2, SUSPICIOUS=98. The 98 benign
chunks would be flagged for sanitization, not quarantined. Fixed to:
"quarantine 36 attack chunks (MALICIOUS) and flag 98 benign chunks for
sanitization (SUSPICIOUS); only 2 benign chunks would be quarantined."

### 4. `chen2025detecting` — all author names wrong (FIXED — Phase 3)

Old: Chen, Yifan; Li, Hangfan; Sui, Yulun; He, Yue; Liu, Yang; Song, Dawn; Hooi, Bryan  
Corrected from ACL Anthology 2025.acl-long.890:  
**Chen, Yulin; Li, Haoran; Sui, Yuan; He, Yufei; Liu, Yue; Song, Yangqiu; Hooi, Bryan**

### 5. `wen2025instructdetector` — four wrong author names (FIXED — Phase 3)

Old: Wang, Chujie; Yang, Xingyao; Tang, Hao; Wu, Fei  
Corrected from ACL Anthology 2025.findings-emnlp.1060:  
**Wang, Chenglong; Yang, Xiyuan; Tang, Haoyu; Wu, Fangzhao**  
DOI 10.18653/v1/2025.findings-emnlp.1060 also added.

### 6. `hines2024spotlighting` — author name and DOI (FIXED — Phase 3)

arXiv:2403.14720 records the author as **Federico Zarfati**, not Farida Zarfati.
Added arXiv DOI 10.48550/arXiv.2403.14720.
Note updated: "CEUR-WS volume number not independently re-verified."

### 7. Figure environments (FIXED — Phase 1)

Three SVG files converted to PDF using cairosvg:
- `manuscript/fig1_overall_asr.pdf` (23 466 bytes)
- `manuscript/fig2_asr_by_category.pdf` (20 331 bytes)
- `manuscript/fig3_security_vs_chunk_removal.pdf` (25 420 bytes)

Three `\begin{figure}...\end{figure}` environments activated in the body with
`\includegraphics[width=\columnwidth]{figN}` and `\label{fig:...}` defined.
In-text `\ref{}` calls added at: end of §V-A (Fig. 1), end of category
paragraph in §V-C (Fig. 2), S4 context-damage paragraph in §V-D (Fig. 3).

### 8. Author block (FIXED — Phase 2)

Removed fabricated "Manuscript received" date (publisher-assigned metadata).
Added explicit comment instructing authors to supply ORCID iDs before submission.

---

## Numerical verification summary (independent audit, all checks PASS)

| Claim | Source | Result |
|---|---|---|
| S0–S4 ASR Llama 3 (all 5 rows) | verified_statistics.json | PASS |
| Wilson CIs (all 5 rows) | verified_statistics.json | PASS |
| McNemar b/c counts and p-values | verified_statistics.json | PASS |
| Holm-adjusted p S0 vs S3 = 1.1e-14 | 4.60e-16 × 24 = 1.10e-14 | PASS |
| 83/150 queries altered (S4) | s4_benign_results.csv direct count | PASS |
| 193 windows removed | s4_utility_summary + CSV sum | PASS |
| 141/750 chunks sanitized | s4_utility_summary | PASS |
| 24/115 gold sanitized, 91 unmodified | verified_statistics.json | PASS |
| 430 attack chunks scanned | verified_statistics.json | PASS |
| Alignment penalty = 0 on all chunks | penalty_nonzero=0 | PASS |
| All category counts (D1–I3, Llama 3) | verified_statistics.json | PASS |
| Latency S0–S4 (13.5, 8.8, 5.2, 5.2, 6.9 s) | verified_statistics mean_latency_ms | PASS |
| Qwen/Mistral wall clock (383.9, 400.0 min) | CROSS_MODEL_REPORT.md | PASS |
| Three-generator table (all 15 cells) | CROSS_MODEL_REPORT.md | PASS |

---

## LaTeX structural checks

| Check | Result |
|---|---|
| All `\ref{}` labels resolve | PASS (0 undefined) |
| Brace balance (non-comment lines) | PASS (282 open = 282 close) |
| Active `\begin{figure}` environments | 3 (PASS) |
| `\includegraphics` calls | 3 (PASS) |
| PDF figure files in `manuscript/` | 3 × present (PASS) |
| Markdown remnants | PASS (none) |
| Unsupported claims | PASS (none) |
| Future work as completed | PASS (none) |
| 11 cite keys = 11 bib entries | PASS |
| `\IEEEcompsocitemizethanks` absent | PASS |
| BibTeX entry types | PASS (@inproceedings, @misc) |
| `\documentclass[journal]{IEEEtran}` | Appropriate for IEEE Access |

---

## Reference verification

| Ref | Source checked | Status |
|---|---|---|
| Greshake 2023 AISec | ACM DL | Verified |
| Liu 2024 USENIX Sec | usenix.org | Verified |
| Yi 2025 KDD DOI | Publisher page | Verified |
| Hines 2024 arXiv:2403.14720 | arxiv.org | Verified; Federico Zarfati corrected; DOI added |
| Hines CEUR-WS vol. 3920 | Not re-verified | **Unresolved** |
| StruQ USENIX Sec 2025 (authors, abstract) | usenix.org presentation page | Verified |
| StruQ pp. 2383–2400 | Not in USENIX proceedings page | **Unresolved** |
| PoisonedRAG USENIX Sec 2025 | usenix.org | Verified |
| RobustRAG SaTML 2026 | Preprint arXiv:2405.15556; accepted | Pagination unverified |
| Chen et al. ACL 2025 (authors, pages, DOI) | ACL Anthology 2025.acl-long.890 | Verified; all 7 author names corrected |
| Wen et al. EMNLP 2025 (authors, pages, DOI) | ACL Anthology 2025.findings-emnlp.1060 | Verified; 4 author names corrected; DOI added |
| TrustRAG arXiv:2501.00879 | arXiv (preprint status) | Accurate |
| SONAR arXiv:2605.01078 | arXiv (preprint status) | Accurate |

---

## Compilation status

**No LaTeX compiler installed** (pdflatex, xelatex, lualatex all absent).
Compilation was NOT performed. Syntax validated by:
- Brace balance count: 282 open = 282 close
- All `\ref{}` labels resolve (0 undefined)
- No `\IEEEcompsocitemizethanks` (compsoc-only command removed)
- 3 active figure environments; 3 PDF files present in `manuscript/`
- 11 cite keys = 11 bib entries; 0 unused or missing

**Compile commands** (run from `manuscript/` with TeX Live installed):
```
pdflatex SecureRAG_IEEE_Access.tex
bibtex   SecureRAG_IEEE_Access
pdflatex SecureRAG_IEEE_Access.tex
pdflatex SecureRAG_IEEE_Access.tex
```

---

## Remaining issues after all completed phases

### Must fix before submission

1. **Supply ORCID iDs.** IEEE Access requires a public ORCID for each
   corresponding author. A placeholder comment is in the `\thanks{}` block.
   Authors must insert their actual ORCID identifiers.

2. **Compile and inspect the PDF.** TeX Live is not installed in this
   environment; compilation has not been performed. A complete compile
   (pdflatex × 2 + bibtex) must be run to confirm no layout errors,
   missing glyphs, or unresolved references.

3. **Verify StruQ page range** (pp. 2383–2400, USENIX Security 2025 proceedings).
   The USENIX presentation page does not list page numbers. Verify against the
   published proceedings PDF or the USENIX open-access download.

4. **Verify RobustRAG page range** (SaTML 2026). Proceedings pagination was not
   available at time of this draft; the bib note records this.

5. **Verify Spotlighting CEUR-WS vol. 3920** attribution. Volume number taken
   from original manuscript draft; not independently confirmed.

### Important but not blocking

6. `\markboth` is redundant in IEEEtran `[journal]` mode; safe to remove.

7. S1 vs S3 discordant counts (22/0 Llama, 26/0 Qwen, 21/1 Mistral) are
   sourced from `CROSS_MODEL_REPORT.md` only, not from `verified_statistics.json`.
   Source is authoritative but the provenance chain is one step longer.

8. Fig. 3 caption notes "S4 occupies the same low chunk-removal position as S1."
   This is accurate: both have near-zero whole-chunk removal. The caption is
   deliberately precise about what the x-axis does not capture.

### Requires new experiments

9. Human annotation of attack-success labels (protocol exists, not executed).
10. Answer-quality metric (no ROUGE/EM computed for benign responses).
11. Poisoned-index retrieval evaluation.
12. Generation-level S4 ablation (quarantine-only vs. window-only vs. full S4).
13. S2 training-corpus exclusion re-verification (corpus not in repository).
