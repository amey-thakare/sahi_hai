# Phase 11 — Statistical Analysis

## Comparison: P1: α_b=0.025, α_m=0.05 vs P0: Baseline (α=0.05 sym)

### McNemar's Test
- Test statistic: 114.5242
- p-value: 1.0004e-26
- Interpretation: Statistically significant (p < 0.05)

### Bootstrap 95% Confidence Intervals
- F1 difference (best − baseline): [0.0025, 0.0042]
- Recall difference (best − baseline): [-0.0118, -0.0099]

### Conclusion
- F1 improvement is statistically supported (CI excludes 0).
- Recall improvement is not statistically supported (CI includes 0).
