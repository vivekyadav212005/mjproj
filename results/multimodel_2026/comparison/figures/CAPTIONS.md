# Figure captions

All rates use the frozen keyword rule. Wilson bars are 95% score intervals for a single proportion. They are not paired tests. Paired tests are in `comparison/confirmatory_tests.csv`.

`fig1_overall_asr.svg`. Attack success on 160 attacks for S0–S4. Llama 3 8B is the historical run. Qwen 2.5 7B and Mistral 7B are the new runs. Observed rates are 104, 38, 19, 16, and 14 out of 160 for Llama 3; 69, 32, 8, 6, and 7 out of 160 for Qwen; and 54, 24, 6, 4, and 4 out of 160 for Mistral.

`fig2_category_asr_S0.svg`, `fig2_category_asr_S3.svg`, and `fig2_category_asr_S4.svg`. The same rate within each attack category. Every category has 20 items. I1 and I3 are 0/20 for undefended Qwen and undefended Mistral. Under S3 and S4, the remaining successes are concentrated in D3 and I2.

`fig3_security_vs_chunk_removal.svg`. Attack success against the fraction of benign retrieved chunks that were dropped, out of 750 chunks from 150 queries. S4 sits at 1/750 chunks removed. That axis does not show the 193 windows deleted from benign text, the 83/150 queries with an edit, or the 24/115 gold sources that were sanitized.

`fig4_benign_query_fpr.svg`. Share of the 150 benign queries blocked before retrieval. The rate is 0/150 for every model and every defense. Chunk removal and window edits are different quantities and are not plotted here.

`fig5_median_latency.svg`. Median end-to-end latency of the 150 benign queries, in milliseconds, including retrieval, filtering, and generation. Llama 3 S3 and S4 were timed in different sessions. No run was limited by the 8,192-token context cap.

`fig6_generalization.svg`. S0, S3, and S4 only. The drop from no defense to the combined filter appears for all three generators. The S3-to-S4 change does not. Exact McNemar tests of that change are non-significant for Llama 3 (p = 0.625), Qwen (p = 1), and Mistral (p = 1) before multiplicity correction.
