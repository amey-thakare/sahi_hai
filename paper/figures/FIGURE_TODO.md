# DSSE Paper Figures TODO

This document contains the exact instructions for generating the figures referenced in `paper/DSSE_research_paper.md`.

## Figure 1: Complete DSSE architecture
**Type:** Flowchart / Architecture Diagram
**Instructions:** 
Draw a vertical or horizontal flowchart showing the pipeline:
1. RAW URL input
2. Feature extraction (12 Structural Features)
3. HXRF Gatekeeper
4. Canonical ICP Routing Layer
5. Decision split based on prediction set:
   - Confident (Set = {BENIGN} or {MALICIOUS}) -> Final Decision
   - Ambiguous (Set = {BENIGN, MALICIOUS} or Empty) -> Stage 2 Escalation
6. MiniLM Semantic processing
7. FiLM Modulation (conditioned on the 12 features)
8. XGBoost Classification -> Final Decision

## Figure 2: ICP routing flow
**Type:** Diagram
**Instructions:** 
Illustrate the split-conformal calibration and inference.
Show the nonconformity score function $s(x,y) = 1 - p(y|x)$ calculated over the calibration set.
Show the threshold $q_\alpha$ dividing the probability space.
Show how the prediction set size dictates the routing (0/2 -> Stage 2, 1 -> Stage 1).

## Figure 3: FiLM structural-semantic fusion
**Type:** Neural Network Diagram
**Instructions:**
Show the two streams:
- Semantic Stream: MiniLM outputs $E_{MiniLM}$ (384-dimensional).
- Structural Stream: $X_{12}$ passes through MLPs to output $\gamma$ and $\beta$.
- Fusion Node: Display the element-wise multiplication ($\odot$) and addition ($+$) operations resulting in $E_{FiLM}$ (128-dimensional).

## Figure 4: Model performance comparison
**Type:** Bar Chart
**Instructions:**
Plot the F1 and ROC-AUC scores from Table III comparing:
- HXRF only
- Phase-11A Canonical DSSE
- Phase-12 Canonical ICP DSSE
*(Exclude the isolated subset historical baselines to prevent visual misrepresentation).*

## Figure 5: Stage-1 vs Stage-2 routing distribution
**Type:** Pie Chart or Stacked Bar Chart
**Instructions:**
Visually represent the computational trade-off:
- 96.69% Stage-1 Resolution (Lightweight)
- 3.31% Stage-2 Escalation (Heavyweight)

## Figure 6: Error analysis
**Type:** Bar Chart / Matrix
**Instructions:**
Plot the progression of errors:
- Stage-1 FN (5,043) vs Final DSSE FN (5,591)
- Stage-1 FP (1,670) vs Final DSSE FP (2,137)
Highlight the base level of errors originating in Stage-1 that bypass Stage-2.
