# Phase 12 — Conformal Calibration Method

## Mathematical Construction
We implemented a clean, standard split-conformal classifier for the binary problem using Inductive Conformal Prediction (ICP).

### Score Definition
The non-conformity score for a sample $x$ and class $y \in \{0, 1\}$ is defined as:
$s(x, y) = 1 - p_\theta(y | x)$
where $p_\theta(y | x)$ is the probability emitted by the frozen Canonical Stage-1 model (`models/stage1_hxrf.pkl`).

### Quantile Calculation
Let $D_{cal} = \{(x_i, y_i)\}_{i=1}^n$ be the disjoint calibration set. We compute the true-class nonconformity scores $s_i = s(x_i, y_i)$ for all $n$ calibration samples. We sort these scores in ascending order.
For a target significance level $\alpha$, the conformal threshold $q_\alpha$ is the finite-sample corrected quantile:
$q_\alpha$ = the element at index $\lceil(n+1)(1-\alpha)\rceil - 1$, clipped to $[0, n-1]$.

### Prediction Set Construction
For a new test point $x_{test}$, we compute the nonconformity score for all candidate classes:
$s(x_{test}, 0) = 1 - p_\theta(0 | x_{test})$
$s(x_{test}, 1) = 1 - p_\theta(1 | x_{test})$

The prediction set $C_\alpha(x_{test})$ includes class $y$ if and only if $s(x_{test}, y) \leq q_\alpha$.

### Conformal P-values
For any class $y$, the conformal p-value is the fraction of calibration true-class scores greater than or equal to the candidate score $s(x_{test}, y)$, applying a finite-sample correction:
$p(y) = \frac{|\{ i : s_i \geq s(x_{test}, y) \}| + 1}{n + 1}$
The prediction set can equivalently be defined as $C_\alpha(x_{test}) = \{y : p(y) > \alpha\}$.

## Guarantee Conditions
Under the standard exchangeability assumptions (where the calibration set and test set are drawn exchangeably from the same underlying distribution), the split-conformal construction provides the corresponding **marginal coverage guarantee**:
$P(Y_{test} \in C_\alpha(X_{test})) \geq 1 - \alpha$

- **Model Consistency:** The same predictive model (`models/stage1_hxrf.pkl`) is used for generating calibration scores and test scores.
- **Split Separation:** The calibration set (`data/splits/calibration.parquet`) was not used to fit the predictive model.
- **Data:** This method provides a marginal coverage guarantee. It does not automatically guarantee conditional coverage, class-conditional coverage, or robustness to arbitrary distribution shift.
