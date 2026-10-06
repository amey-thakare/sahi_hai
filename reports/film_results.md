# Phase 8 Structural-Semantic FiLM Fusion Results

## 1. Research Motivation
Semantic information alone drops in performance on URLs selected by ICP as uncertain (`NEEDS_STAGE2`). Conditioning semantic representations on structural URL evidence may improve discrimination.

## 2. Hypothesis
H1: Conditioning the MiniLM semantic representation using lightweight structural URL features through FiLM will improve classification performance on ICP-ambiguous URLs compared with semantic-only representations and naive structural-semantic concatenation.

## 3. Architecture & Data Split
- **MiniLM Baseline:** MLP (Linear) on 384-D
- **Naive Concatenation:** MLP (Linear) on 396-D
- **FiLM Architecture:** Gamma (12->64->384), Beta (12->64->384), Stable Formulation: $E_{FiLM} = (1+\gamma) \odot E + eta$
- **Data:** Trained on 40k subset, Validated on 10k, Evaluated on 126k Test set and its NEEDS_STAGE2 subset.
- **Structural Preprocessing:** StandardScaler fit on 40k train set only.

## 4. Parameter Counts
- MiniLM Classifier: 385
- Concatenation Classifier: 397
- FiLM Classifier + Generator: 51969

## 5. Required Results Table

| Model | Population | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| MiniLM | Full | 0.8613 | 0.8295 | 0.7342 | 0.7789 | 0.9274 | 0.0753 | 0.2658 |
| Concatenation | Full | 0.8922 | 0.8773 | 0.7859 | 0.8291 | 0.9547 | 0.0548 | 0.2141 |
| FiLM | Full | 0.9488 | 0.9513 | 0.8917 | 0.9206 | 0.9869 | 0.0227 | 0.1083 |
| MiniLM | NEEDS_STAGE2 | 0.6523 | 0.6481 | 0.3871 | 0.4847 | 0.7176 | 0.1537 | 0.6129 |
| Concatenation | NEEDS_STAGE2 | 0.6627 | 0.6539 | 0.4282 | 0.5175 | 0.7349 | 0.1658 | 0.5718 |
| FiLM | NEEDS_STAGE2 | 0.7544 | 0.7662 | 0.6025 | 0.6746 | 0.8329 | 0.1345 | 0.3975 |

## 6. Required Latency Table
Measurements for classifier overhead (batch size 1000). MiniLM encoding latency is excluded here to isolate fusion overhead.

| Model | Mean ms | Median ms | P95 ms | P99 ms | Throughput |
|---|---:|---:|---:|---:|---:|
| MiniLM (Classifier Only) | 0.03 | 0.03 | 0.04 | 0.04 | 35703325.93 |
| Concatenation | 0.03 | 0.03 | 0.03 | 0.03 | 36568287.40 |
| FiLM | 0.15 | 0.15 | 0.17 | 0.18 | 6839168.90 |

## 7. Gamma and Beta Findings
- Gamma Mean: 0.7757, Std: 11.0029, Min: -132.8281, Max: 138.3354
- Beta Mean: 0.0090, Std: 0.0754, Min: -1.3413, Max: 1.0997

## 8. Final Conclusion
H1 is SUPPORTED. FiLM demonstrated improvement over naive concatenation for the ambiguous `NEEDS_STAGE2` population.
