# S4 SecureRAG+ Research Extension

This document describes the architectural design and experimental motivation for the **S4 SecureRAG+** defense system. S4 extends the S0–S3 defense framework in this repository to provide a highly interpretable, retrieval-aware, and fine-grained threat containment strategy.

## 1. Motivation: Beyond Binary Document Gating (S3)

In SecureRAG (S3), documents retrieved from the vector database are treated as monolithic blocks. If a document triggers any classifier threshold (e.g. RuleClassifier or SimpleMLDetector), it is completely quarantined (removed from LLM context). This binary gater presents two significant limitations:
1. **Utility Degradation (Over-blocking)**: If a legitimate document contains instruction-like language (such as customer support templates, legal disclaimers, or historical quotes) or a minor indirect injection embedded at the end, the *entire* document is dropped. The LLM is deprived of the benign factual content.
2. **Context Blindness**: Document classifiers treat the document's relevance (retrieval rank/distance) and threat density (window scanning) independently. A highly relevant document with a minor threat profile should be sanitized rather than blindly dropped.

## 2. S4 SecureRAG+ Pipeline Architecture

S4 addresses these limitations through a layered, retrieval-aware risk-scoring model combined with sentence-level sanitization:

```
                  [ USER QUERY ]
                        │
                        ▼
             ┌─────────────────────┐
             │ S3 Query Gate Check │ ──(Blocked)──► [ BLOCKED_RESPONSE ]
             └─────────────────────┘
                        │
                  (Not Blocked)
                        │
                        ▼
             ┌─────────────────────┐
             │  Document Retrieval │
             └─────────────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │   Window Scanner    │ ◄─── (Sentence splitting +
             │ & S3 Doc Classifier │       S3 window classification)
             └─────────────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │   Retrieval-Aware   │ ◄─── (Computes risk using alignment
             │    Risk Scorer      │       and weights)
             └─────────────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │ Adaptive Policy     │
             │ quarantine/sanitize │
             └─────────────────────┘
                  /          \
                 /            \
        (MALICIOUS)       (SUSPICIOUS or SAFE)
           /                    \
          ▼                      ▼
  [ QUARANTINE ]            [ SANITIZE ]
  (Entire document          (Remove blocked windows,
   removed)                  send clean text to LLM)
                                 │
                                 ▼
                             [ LLM ]
```

## 3. Mathematical Risk Formulation & Thresholds

For each retrieved document, S4 calculates an interpretable risk score $R \in [0, 1]$:

$$R = \text{min}\left(\text{max}\left(w_d \cdot D + w_w \cdot W_{max} + w_p \cdot P + w_a \cdot A_p, 0.0\right), 1.0\right)$$

Where:
- $D \in [0, 1]$ is the **Document Score** from the whole-document classifier.
- $W_{max} \in [0, 1]$ is the **Max Window Score** across all sentence windows in the document.
- $P \in [0, 1]$ is the **Suspicious Window Proportion** (blocked windows / total windows).
- $A_p \in [0, 1]$ is the **Low-Alignment Penalty** computed from query-document alignment $A_{align}$:
  $$A_p = \text{max}(0.0, 0.20 - A_{align})$$
  *(The alignment $A_{align}$ is calculated via cosine similarity or mapped from retrieval distance: $1 / (1 + \text{distance})$)*.

### Configured Weights and Thresholds:
S4 preserves the following pre-declared weights and thresholds:
* **Document Score Weight ($w_d$)**: `0.35`
* **Max Window Score Weight ($w_w$)**: `0.45`
* **Suspicious Window Proportion Weight ($w_p$)**: `0.15`
* **Alignment Penalty Weight ($w_a$)**: `0.05`
* **Alignment Floor**: `0.20`
* **SUSPICIOUS Threshold**: `0.45` (Risk $\ge 0.45$ trigger sanitization)
* **MALICIOUS Threshold**: `0.80` (Risk $\ge 0.80$ triggers whole-document quarantine)

## 4. Mitigation Policies

* **SAFE (Risk < 0.45)**: The document has a low threat profile. It is fully allowed into the context.
* **SUSPICIOUS (0.45 $\le$ Risk < 0.80)**: The threat can be isolated. Blocked sentence windows (flagged text) are surgically removed. The remaining clean text is joined and included in the LLM prompt.
* **MALICIOUS (Risk $\ge$ 0.80)**: The threat is too dense or cannot be safely isolated. The whole document is quarantined and removed from the prompt.
