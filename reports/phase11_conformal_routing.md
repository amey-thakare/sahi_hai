# Phase 11 — Asymmetric Conformal Routing (Experiment B)

## Methodology

Standard split-conformal ICP uses a single α for both classes.
Asymmetric routing uses different α values for benign acceptance vs malicious acceptance:

- **α_benign**: Lower α → stricter benign acceptance → fewer FN but more escalations
- **α_malicious**: Higher α → easier malicious acceptance → more direct malicious exits

> **Important**: Asymmetric routing no longer provides the standard split-conformal
> marginal coverage guarantee P(y ∈ C(x)) ≥ 1−α. The individual class-conditional
> acceptance thresholds are derived from the same calibration scores but applied
> with different stringencies. This is a heuristic routing policy, not a
> statistically guaranteed prediction set.

## Results

| α_benign | α_malicious | Stage-2 | S2 % | S1 FN | S1 FP |
|---:|---:|---:|---:|---:|---:|
| 0.005 | 0.05 | 9716 | 7.68% | 4753 | 81 |
| 0.01 | 0.05 | 9320 | 7.36% | 4753 | 104 |
| 0.02 | 0.05 | 8798 | 6.95% | 4753 | 172 |
| 0.025 | 0.05 | 8316 | 6.57% | 4753 | 289 |
| 0.03 | 0.05 | 7813 | 6.17% | 4753 | 446 |
| 0.04 | 0.05 | 6810 | 5.38% | 4753 | 836 |
| 0.05 | 0.05 | 5530 | 4.37% | 4753 | 1400 ←baseline |
| 0.075 | 0.05 | 2442 | 1.93% | 4753 | 3101 |
| 0.1 | 0.05 | 1017 | 0.80% | 4405 | 4583 |
| 0.15 | 0.05 | 8273 | 6.54% | 2517 | 4583 |
| 0.2 | 0.05 | 14931 | 11.80% | 1594 | 4583 |
| 0.3 | 0.05 | 27401 | 21.65% | 656 | 4583 |
| 0.005 | 0.1 | 3619 | 2.86% | 7417 | 81 |
| 0.01 | 0.1 | 3223 | 2.55% | 7417 | 104 |
| 0.02 | 0.1 | 2701 | 2.13% | 7417 | 172 |
| 0.025 | 0.1 | 2219 | 1.75% | 7417 | 289 |
| 0.03 | 0.1 | 1716 | 1.36% | 7417 | 446 |
| 0.04 | 0.1 | 713 | 0.56% | 7417 | 836 |
| 0.05 | 0.1 | 567 | 0.45% | 7100 | 1150 |
| 0.075 | 0.1 | 3655 | 2.89% | 5713 | 1150 |
| 0.1 | 0.1 | 7114 | 5.62% | 4405 | 1150 |
| 0.15 | 0.1 | 14370 | 11.35% | 2517 | 1150 |
| 0.2 | 0.1 | 21028 | 16.61% | 1594 | 1150 |
| 0.3 | 0.1 | 33498 | 26.47% | 656 | 1150 |
| 0.005 | 0.15 | 1219 | 0.96% | 8895 | 81 |
| 0.01 | 0.15 | 823 | 0.65% | 8895 | 104 |
| 0.02 | 0.15 | 301 | 0.24% | 8895 | 172 |
| 0.025 | 0.15 | 181 | 0.14% | 8775 | 228 |
| 0.03 | 0.15 | 684 | 0.54% | 8429 | 228 |
| 0.04 | 0.15 | 1687 | 1.33% | 7816 | 228 |
| 0.05 | 0.15 | 2967 | 2.34% | 7100 | 228 |
| 0.075 | 0.15 | 6055 | 4.78% | 5713 | 228 |
| 0.1 | 0.15 | 9514 | 7.52% | 4405 | 228 |
| 0.15 | 0.15 | 16770 | 13.25% | 2517 | 228 |
| 0.2 | 0.15 | 23428 | 18.51% | 1594 | 228 |
| 0.3 | 0.15 | 35898 | 28.36% | 656 | 228 |
