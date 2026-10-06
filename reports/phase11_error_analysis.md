# Phase 11 — Error Analysis

## Before vs After

| Metric | Phase-10 Baseline | Best Phase-11 Policy |
|---|---:|---:|
| Policy | P0: Baseline (α=0.05 sym) | P1: α_b=0.025, α_m=0.05 |
| Stage-1 FN | 4753 | 4753 |
| Stage-1 FP | 1400 | 289 |
| Stage-2 FN | 710 | 1167 |
| Stage-2 FP | 596 | 863 |
| Total FN | 5463 | 5920 |
| Total FP | 1996 | 1152 |
| Total Errors | 7459 | 7072 |
| Stage-2 Escalations | 5530 | 8316 |
| Stage-2 Rate | 4.37% | 6.57% |

## Recovery Analysis

- Malicious URLs recovered (FN→TP/TN): **0**
- Malicious URLs newly missed: **457**
- Net FN reduction: **-457**

## False Positive Changes

- Newly introduced FP: **0**
- Removed FP: **844**
- Net FP change: **-844**
