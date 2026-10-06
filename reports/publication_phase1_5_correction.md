# Publication Phase 1.5 Correction Summary

## Corrections Made
1. Computational Claim: Replaced 96% computational cost bypass with precise wording on avoiding Stage-2 processing.
2. Exchangeability Wording: Corrected to specify reliance on the standard exchangeability assumption between calibration and test examples.
3. Stage-1 False Negative Interpretation: Rewrote the "absolute lower bound" interpretation to correctly describe propagation of confident errors.
4. McNemar Interpretation: Reworded the statistical significance result to emphasize asymmetry in paired disagreements rather than superior predictive performance.
5. Dataset Description: Updated provenance to explicitly mention ISCX-URL2016 and PhishStorm.
6. MiniLM Configuration: Fixed the truncation token limit from 512 to 256.
7. FiLM Dimension Verification: Fixed the dimension from 128 to 384.
8. Unverified hype terms: Removed instances of "substantial computational trade-off" and "drastically minimizing".

## Verification Checklist
- **MiniLM configuration verified:** YES (max_seq_length is 256, normalization is false).
- **FiLM dimension verified:** YES (output dim is 384, not 128).
- **Dataset provenance verified:** YES (from ISCX-URL2016 and PhishStorm).

## Unsupported Claims Removed
- Removed 2 unsupported emphasis phrases ("substantial", "drastically").
- Corrected 2 scientifically imprecise claims (McNemar superiority, identical distribution exchangeability).
- Corrected 1 absolute computational claim (96% cost reduction).

## Status
READY FOR REVIEWER AUDIT
