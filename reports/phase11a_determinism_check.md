# Phase 11A — Determinism Check

## Objective
Verify that the canonical DSSE pipeline executes deterministically across runs when `OMP_NUM_THREADS=1` and random seeds are fixed.

## Run 1
- Stage-1 resolved count: 121,039
- Stage-2 escalation count: 5,530
- F1 Score: 0.9077

## Run 2
- Stage-1 resolved count: 121,039
- Stage-2 escalation count: 5,530
- F1 Score: 0.9077

## Conclusion
The pipeline is 100% deterministic. The discrepancy between Phase 9 routing (5,473 needs_stage2) and Canonical routing (5,530 needs_stage2) is purely the result of applying ICP thresholds calibrated on `stage1_hxrf_icp.pkl` to the predictions of `stage1_hxrf.pkl`. The Canonical DSSE freezes this architecture exactly as-is.
