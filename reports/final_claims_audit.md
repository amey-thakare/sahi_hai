# Final Claims Audit

### SUPPORTED
- DSSE uses a lightweight Stage-1 structural classifier based entirely on 12 character/structural features without external network calls.
- Only a subset of URLs are dynamically escalated to the semantic Stage 2.
- The canonical ICP algorithm rigorously uses the exact same HXRF checkpoint for both calibration and inference.
- Under the standard exchangeability assumption, the split-conformal procedure provides the corresponding marginal coverage guarantee.
- The measured Stage-2 escalation rate is 3.31% (for $\alpha=0.05$).
- The measured end-to-end F1 is 0.9043 (for $\alpha=0.05$).
- The architecture correctly short-circuits structurally obvious URLs, avoiding heavy Transformer execution overhead.

### NOT SUPPORTED
- Universal adversarial robustness (not explicitly evaluated against adversarial string mutations in tests).
- Distribution-shift robustness (assumes data comes from the same distribution).
- Zero false negatives (the system still incurs FN in both Stage-1 and Stage-2).
- Conditional coverage / Class-conditional coverage (only marginal coverage is formally guaranteed by the base setup).
- Guaranteed superior F1 (it provides an operational computational tradeoff, occasionally slightly reducing absolute F1 for massive computational savings).
- Universal sub-millisecond latency (network and IO overhead per batch will vary wildly).
- Statistical superiority without appropriate testing (where tested via McNemar's, superiority or differences were observed).
