# Experimental Fairness Audit

## Audit of Model Comparisons
The paper presents results for multiple models in Table III (Model Comparison).

### End-to-End Evaluations
- **HXRF only (Stage 1):** Directly comparable to end-to-end performance on the entire test set.
- **Phase-11A canonical DSSE:** Directly comparable end-to-end performance.
- **Phase-12 canonical ICP DSSE:** Directly comparable end-to-end performance.

### Historical Isolated Subset Evaluations
- **MiniLM + XGB:** Not directly comparable end-to-end. Evaluated historically on the `NEEDS_STAGE2` subset.
- **Concat + XGB:** Not directly comparable end-to-end. Evaluated historically on the `NEEDS_STAGE2` subset.
- **FiLM + XGB:** Not directly comparable end-to-end. Evaluated historically on the `NEEDS_STAGE2` subset.

## Conclusion
The paper correctly includes a note under Table III stating: "*(Note: MiniLM, Concat, and FiLM baselines represent historical isolated subset metrics and are NOT DIRECTLY COMPARABLE to the end-to-end evaluations)*". 

The experimental comparisons are presented fairly and accurately.
