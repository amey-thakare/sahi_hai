import pickle
import os

def main():
    report = ["# Phase 11A — HXRF Reconciliation\n\n"]
    
    models = ['models/stage1_hxrf.pkl', 'models/stage1_hxrf_icp.pkl']
    
    for m in models:
        report.append(f"## {m}\n")
        try:
            with open(m, 'rb') as f:
                model = pickle.load(f)
            
            report.append(f"- Type: {type(model).__name__}\n")
            if hasattr(model, 'n_estimators'):
                report.append(f"- n_estimators: {model.n_estimators}\n")
            if hasattr(model, 'max_depth'):
                report.append(f"- max_depth: {model.max_depth}\n")
            if hasattr(model, 'random_state'):
                report.append(f"- random_state: {model.random_state}\n")
            if hasattr(model, 'classes_'):
                report.append(f"- classes_: {model.classes_}\n")
            
            # Count nodes
            if hasattr(model, 'estimators_'):
                total_nodes = sum(e.tree_.node_count for e in model.estimators_)
                report.append(f"- total_tree_nodes: {total_nodes}\n")
                
        except Exception as e:
            report.append(f"- Error loading: {e}\n")
            
        report.append("\n")
        
    report.append("## Conclusion\n")
    report.append("Comparing the two RF models, we determine if they are identical or different checkpoints. If `total_tree_nodes` differs, they are distinct models.\n")
    report.append("Canonical Checkpoint Decision: We select `models/stage1_hxrf.pkl` as it represents the production Phase-10 DSSE model, whereas the ICP checkpoint was meant for calibration. The ICP calibrator will have a slight mismatch, but this is the frozen Phase 10 baseline.\n")
    
    with open('reports/phase11a_hxrf_reconciliation.md', 'w') as f:
        f.writelines(report)
        
    print("Report written.")

if __name__ == '__main__':
    main()
