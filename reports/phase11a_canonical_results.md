# Phase 11A — Canonical Results

## 1. Architecture
Canonical DSSE (HXRF -> ICP -> FiLM/XGBoost)

## 2. Dataset
Test set only (`data/splits/test.parquet`). Using Parquet Columns directly, no dynamic extraction float drift.

## 3. Routing Statistics
- Total test URLs: 126569
- Stage-1 resolved count: 121039
- Stage-2 escalation count: 5530
- Stage-1 resolution rate: 95.63%
- Stage-2 escalation rate: 4.37%

## 4. Classification Results
| System | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Canonical DSSE | 0.9411 | 0.9484 | 0.8703 | 0.9077 | 0.9768 | 0.0236 | 0.1297 |
