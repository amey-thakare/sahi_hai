# Phase 11A — HXRF Reconciliation

## models/stage1_hxrf.pkl
- Type: RandomForestClassifier
- n_estimators: 200
- max_depth: 15
- random_state: 42
- classes_: [0 1]
- total_tree_nodes: 1005388

## models/stage1_hxrf_icp.pkl
- Type: RandomForestClassifier
- n_estimators: 200
- max_depth: 15
- random_state: 42
- classes_: [0 1]
- total_tree_nodes: 937732

## Conclusion
Comparing the two RF models, we determine if they are identical or different checkpoints. If `total_tree_nodes` differs, they are distinct models.
Canonical Checkpoint Decision: We select `models/stage1_hxrf.pkl` as it represents the production Phase-10 DSSE model, whereas the ICP checkpoint was meant for calibration. The ICP calibrator will have a slight mismatch, but this is the frozen Phase 10 baseline.
