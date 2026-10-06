# Phase 11A — Dataset Integrity

## Source Dataset
- Processed Dataset: `data/splits/test.parquet`

## Binary Target Definition
- Benign (0): 84,448
- Malicious (1, 2, 3): 18,910 + 18,392 + 4,819 = 42,121
- Total Test Population: 126,569

## Validation
The file `data/splits/test.parquet` was successfully loaded and the counts perfectly match the reported baseline numbers (126,569 rows; 84,448 benign; 42,121 malicious).

## Canonical Test Split
- The exact path is `data/splits/test.parquet`.
- This test set will be used as the single frozen held-out test split for canonical evaluation.
- CANONICAL_TEST_SPLIT = data/splits/test.parquet
