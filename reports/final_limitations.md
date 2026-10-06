# Final Limitations

- **Dataset Provenance & Generalization:** The data derives from stationary historical datasets (ISCX-URL2016, PhishStorm). Modern evolving phishing or short-lived domains may possess unrepresented feature distributions.
- **Exchangeability Assumption:** The conformal guarantees intrinsically depend on the test queries being exchangeable with the calibration set. Temporal drift actively violates this in production deployments without recalibration.
- **Marginal vs Conditional Guarantees:** ICP provides standard marginal coverage over the average distribution. It does *not* provide formal conditional coverage for specific URLs or even strict class-conditional coverage.
- **Adversarial & Evasion Attacks:** The Stage-1 structural features (`domain_len`, `path_depth`, `special_count_v2`) are highly susceptible to targeted length manipulations and obfuscation by adversaries.
- **Stage-1 False Negatives:** Stage-1 continues to be a bottleneck. Confident misclassifications at Stage-1 completely bypass Stage-2 evaluation, injecting an absolute floor on the final False Negative Rate.
- **Hardware-specific latency:** Throughput metrics rely on a specific batch size configuration and Apple Silicon (MPS/CPU) hardware states; exact execution profiles will vary under different network topologies or GPU architectures.
