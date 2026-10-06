# Final Dataset Audit

The dataset originates from multiple sources (including ISCX-URL2016, PhishStorm) and was aggressively de-duplicated and split rigorously without leakage.

## Dataset Split Verification
- **Total test:** 126,569
- **Benign in test:** 84,448
- **Malicious in test:** 42,121

## Checks Performed
- **Train/Calibration/Test Separation:** The sets were mathematically partitioned. The `phase12_split_check.py` script verified strictly 0 overlapping URLs across `proper_train.parquet`, `calibration.parquet`, and `test.parquet`.
- **URL Overlap:** Confirmed mutually exclusive sets.
- **Duplicate Handling:** Strict uniqueness enforced upon extraction.
- **Binary Target Mapping:** Confirmed `dsse_binary_target` captures exactly 0 (benign) and 1 (malicious). 
- **Original multiclass labels:** Preserved as `dsse_type` in the raw artifacts for auditing if needed.
- **Canonical test split path:** `data/splits/test.parquet`.
