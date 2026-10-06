# Phase 12 — Split Integrity

## Sizes
- proper_train: 405020
- calibration: 101255
- test: 126569

## Disjointness Checks
- TRAIN ∩ CALIBRATION = 0
- CALIBRATION ∩ TEST = 0
- TRAIN ∩ TEST = 0

## Conclusion
All partitions are disjoint. We will use `data/splits/calibration.parquet` as the canonical ICP calibration set.
