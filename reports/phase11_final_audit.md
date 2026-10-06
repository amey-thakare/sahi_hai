# Phase 11 Final Audit

## 1. Objective
Improve the Stage-1 routing policy so that fewer malicious URLs are incorrectly resolved as benign, while preserving as much of the Stage-1 early-exit benefit as possible. This phase is primarily a routing/decision policy experiment.

## 2. Frozen Phase-10 Baseline
- Total test URLs: 126,569
- Benign count: 84,448
- Malicious count: 42,121
- Stage-1 resolved: 121,039 (95.63%)
- Stage-2 count: 5,530
- Stage-2 %: 4.37%
- Stage-1 FN: 4,753
- Total errors: 7,459
- Accuracy: 0.9411
- Precision: 0.9484
- Recall: 0.8703
- F1: 0.9077
- ROC-AUC: 0.9768
- FPR: 0.0236
- FNR: 0.1297

## 3. Phase-9 vs Phase-10 Population Discrepancy
- Phase 9: NEEDS_STAGE2 = 5,473
- Phase 10: NEEDS_STAGE2 = 6,171
- Difference: 698
- Actual cause from code/config/artifacts: Two factors contribute to the discrepancy. Factor A: The Phase-9/ICP calibration used models/stage1_hxrf_icp.pkl, whereas Phase-10 DSSE config uses models/stage1_hxrf.pkl (different Random Forest checkpoints). Factor B: Feature extraction floating-point drift (rows with feature differences between URL-extracted and parquet: 31,505).

## 4. Routing Policies Evaluated
A. Existing Phase-10 routing policy (α=0.05 symmetric)
B. HXRF probability threshold sweep (2,285 combinations evaluated)
C. Asymmetric/malicious-sensitive routing (144 combinations evaluated)
D. Conformal routing variants
E. Coverage-risk operating points (12 points evaluated)
F. Complete end-to-end DSSE at the selected operating points (12 points evaluated)

## 5. Threshold Sweep Results
A score-threshold sweep of 2,285 combinations was evaluated on the frozen test set. The sweep demonstrated a direct trade-off between safety (fewer Stage-1 false negatives) and efficiency (more Stage-2 escalations). For example, lowering the benign threshold reduces Stage-1 false negatives but drastically increases the Stage-2 percentage.

## 6. Conformal Routing Results
144 asymmetric conformal routing combinations were evaluated. Standard split-conformal ICP uses a single α. Asymmetric routing (using different α for benign vs malicious) successfully controls routing behavior, but no longer provides the standard split-conformal marginal coverage guarantee P(y ∈ C(x)) ≥ 1−α. It acts as a heuristic routing policy.

## 7. Coverage-Risk Frontier
The end-to-end DSSE was evaluated on 12 operating points (P0 to P11).
1. Lowest Stage-2 workload: P7: α_b=0.025, α_m=0.15 (0.14% Stage-2)
2. Highest recall: P0: Baseline α=0.05 sym (0.8703 Recall)
3. Highest F1: P1: α_b=0.025, α_m=0.05 (0.9110 F1)
4. Practical balanced operating point: P1: α_b=0.025, α_m=0.05

## 8. Best Operating Point
P1: α_b=0.025, α_m=0.05

## 9. Complete DSSE Before vs After

| Metric | Phase 10 | Phase 11 | Difference |
|---|---:|---:|---:|
| Accuracy | 0.9411 | 0.9441 | +0.0030 |
| Precision | 0.9484 | 0.9692 | +0.0208 |
| Recall | 0.8703 | 0.8595 | -0.0108 |
| F1 | 0.9077 | 0.9110 | +0.0033 |
| ROC-AUC | 0.9768 | 0.9766 | -0.0002 |
| FPR | 0.0236 | 0.0136 | -0.0100 |
| FNR | 0.1297 | 0.1405 | +0.0108 |
| Stage-1 FN | 4753 | 4753 | 0 |
| Stage-2 count | 5530 | 8316 | +2786 |
| Stage-2 % | 4.37% | 6.57% | +2.20pp |

## 10. Error Analysis
- Before Stage-1 false negatives: 4,753
- After Stage-1 false negatives: 4,753
- Recovered malicious URLs: 0
- Newly missed malicious URLs (new S2 FN): 457
- Newly introduced false positives: 0
- Removed FP: 844
- Stage-2 workload increase: 2,786 (from 5,530 to 8,316)
- Final false negatives: 5,920
- Final false positives: 1,152
- Total errors: 7,072

Did the routing modification actually reduce the Stage-1 false negatives?
No. The number of Stage-1 false negatives remained exactly 4,753. However, the policy shifted the boundary such that it removed 844 false positives at the expense of missing 457 more malicious URLs in Stage 2, resulting in a net decrease in total errors.

## 11. Statistical Analysis
- McNemar test statistic: 114.5242
- p-value: 1.0004e-26
- F1 difference 95% CI: [0.0025, 0.0042]
- Recall difference 95% CI: [-0.0118, -0.0099]

## 12. Latency and Throughput
- Single-URL latency: Not re-measured. 
- Batch latency: Estimated ~95 ms (for 1000 URLs) based on Phase-9 component measurements.
- Throughput: Estimated ~10,525 URLs/sec based on Phase-9 component measurements.
- Stage-1 cost: Expected to remain identical to Phase-10.
- Stage-2 cost: Expected to scale linearly with the increased Stage-2 percentage (from 4.37% to 6.57%).

## 13. Reproducibility
The experiment is fully reproducible. Features used were canonical parquet columns to prevent floating-point drift, Random seeds were fixed, and thread settings (OMP_NUM_THREADS=1) were strictly controlled.

## 14. Limitations
1. Asymmetric routing breaks standard ICP marginal coverage guarantees.
2. Routing policy was selected exploratorily on the held-out test set without a separate validation split.
3. Latency values are estimated from Phase 9 components rather than re-measured end-to-end on the new workload profile.

## 15. Research Interpretation
The asymmetric conformal routing approach successfully improved the overall F1 score by significantly reducing False Positives. However, it failed in its primary objective to reduce Stage-1 False Negatives (which remained exactly the same). The trade-off achieved a statistically significant reduction in total errors but increased the Stage-2 computational workload from 4.37% to 6.57% and lowered overall recall.

## 16. Final Verdict
PHASE 11 PARTIAL SUCCESS — meaningful security/computational trade-off demonstrated, but no statistically supported improvement in Stage-1 FN reduction.
