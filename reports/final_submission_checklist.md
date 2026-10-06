# FINAL SUBMISSION CHECKLIST

## 1. Document Assets
- [x] Final Markdown Manuscript (`paper/final/DSSE_Malicious_URL_Detection_IEEE.md`)
- [x] Final PDF Manuscript (`paper/final/DSSE_Malicious_URL_Detection_IEEE.pdf`)
- [x] Final DOCX Manuscript (`paper/final/DSSE_Malicious_URL_Detection_IEEE.docx`)
- [x] All 6 figures correctly referenced and rendering without errors

## 2. Research Integrity Audit
- [x] **No Re-training**: Confirmed no model files in `models/` have been modified during the publication phases.
- [x] **No Checkpoint Swaps**: Confirmed canonical Stage-1 and ICP use the exact same canonical checkpoint (`models/stage1_hxrf.pkl`).
- [x] **No Metric Inflation**: All metrics perfectly match the Phase-12 benchmark standard:
  - F1 Score: 0.9043
  - ROC-AUC: 0.9767
  - Stage-1 Routing: 96.69%
  - FNR: 13.27% (5,591 FNs total)
- [x] **Reproducibility**: `OMP_NUM_THREADS=1` constraint explicitly documented for thread-safe exact reproducibility.

## 3. Formatting & Style
- [x] **Author Placeholders**: Present (`[Author Name]`, `[Affiliation]`, `[Email]`).
- [x] **Title**: "Dual-Stage Semantic Ensemble with Conformal Selective Routing for Malicious URL Detection".
- [x] **Language**: Objective, professional, avoiding hyperbolic claims ("state-of-the-art", "perfect").
- [x] **Citations**: All 6 citation gaps resolved with authoritative, peer-reviewed academic sources.

## 4. Submission Readiness
- [x] The manuscript is visually consistent and strictly adheres to academic publishing standards.
- [x] All paths are correctly resolved relative to the output artifacts.
- [x] Ready for submission to peer-reviewed cybersecurity venues.
