"""
Inductive Conformal Prediction (ICP) Calibrator for binary classification.

Implements the split-conformal methodology for the HXRF Stage-1 gatekeeper.

Mathematical formulation
------------------------
Let D_cal = {(x_i, y_i)}_{i=1}^n be the calibration set.

Non-conformity score:
    s(x, y) = 1 - p_θ(y | x)

where p_θ(y | x) is the HXRF predicted probability for class y.

For calibration sample i:
    s_i = 1 - p_θ(y_i | x_i)

Conformal quantile (finite-sample corrected):
    q_α = the ⌈(n+1)(1-α)⌉ / n -th quantile of {s_1, ..., s_n}

    Equivalently, sort {s_1, ..., s_n} in ascending order and take the
    element at index ⌈(n+1)(1-α)⌉ - 1  (0-indexed), clipped to [0, n-1].

Prediction set for new input x:
    C_α(x) = { y ∈ {0,1} : s(x,y) ≤ q_α }

Coverage guarantee (under exchangeability):
    P(y_{n+1} ∈ C_α(x_{n+1})) ≥ 1 - α
"""
import math
import pickle
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ConformalCalibrator:
    """Split-conformal calibrator for binary classification.

    Attributes
    ----------
    alpha : float
        Significance level (e.g. 0.05 for 95% nominal coverage).
    method : str
        Description of the conformal method.
    nonconformity_score : str
        Human-readable definition of the score function.
    calibration_scores : np.ndarray
        Sorted non-conformity scores from the calibration set.
    quantile : float
        The conformal threshold q_α.
    n_calibration : int
        Number of calibration samples used.
    class_distribution : dict
        {0: count_benign, 1: count_malicious} in calibration set.
    random_seed : int
        Seed used for reproducibility.
    """

    alpha: float = 0.05
    method: str = "inductive_conformal_prediction"
    nonconformity_score: str = "1 - predicted_probability_true_class"
    calibration_scores: Optional[np.ndarray] = None
    quantile: float = 0.0
    n_calibration: int = 0
    class_distribution: dict = field(default_factory=dict)
    random_seed: int = 42

    def fit(self, y_cal: np.ndarray, probas_cal: np.ndarray) -> "ConformalCalibrator":
        """Fit the conformal calibrator using calibration predictions.

        Parameters
        ----------
        y_cal : np.ndarray, shape (n,)
            True labels for calibration samples (0 or 1).
        probas_cal : np.ndarray, shape (n, 2)
            HXRF predicted probabilities [P(y=0|x), P(y=1|x)] for each
            calibration sample.  These MUST come from a model that was
            NOT trained on these samples.

        Returns
        -------
        self
        """
        n = len(y_cal)
        assert n > 0, "Calibration set is empty"
        assert probas_cal.shape == (n, 2), (
            f"Expected shape ({n}, 2), got {probas_cal.shape}"
        )

        # Non-conformity scores: s_i = 1 - p(y_i | x_i)
        scores = np.array([
            1.0 - probas_cal[i, int(y_cal[i])]
            for i in range(n)
        ])

        # Sort ascending for quantile computation
        self.calibration_scores = np.sort(scores)
        self.n_calibration = n
        self.class_distribution = {
            0: int((y_cal == 0).sum()),
            1: int((y_cal == 1).sum()),
        }

        # Finite-sample conformal quantile
        # Index = ceil((n+1)(1-α)) - 1, clipped to [0, n-1]
        self.quantile = self._compute_quantile(self.alpha)

        return self

    def _compute_quantile(self, alpha: float) -> float:
        """Compute the conformal quantile q_α with finite-sample correction.

        Uses: index = ceil((n+1)(1-α)) - 1, clipped to [0, n-1].

        Parameters
        ----------
        alpha : float
            Significance level.

        Returns
        -------
        float
            The conformal threshold q_α.
        """
        n = self.n_calibration
        raw_index = math.ceil((n + 1) * (1 - alpha)) - 1
        index = max(0, min(raw_index, n - 1))
        return float(self.calibration_scores[index])

    def get_quantile(self, alpha: float) -> float:
        """Get the conformal quantile for an arbitrary alpha.

        Useful for sensitivity analysis without refitting.
        """
        assert self.calibration_scores is not None, "Calibrator not fitted"
        return self._compute_quantile(alpha)

    def predict(
        self,
        probas: np.ndarray,
        alpha: Optional[float] = None,
    ) -> dict:
        """Generate prediction sets and conformal p-values for new samples.

        Parameters
        ----------
        probas : np.ndarray, shape (m, 2)
            HXRF predicted probabilities [P(y=0|x), P(y=1|x)].
        alpha : float, optional
            Override the fitted alpha for sensitivity analysis.

        Returns
        -------
        dict with keys:
            prediction_sets : list of lists
                Each element is a list of classes in the prediction set.
            prediction_set_sizes : np.ndarray
                Size of each prediction set (1 or 2).
            pvalues : np.ndarray, shape (m, 2)
                Conformal p-values for [class 0, class 1].
            quantile_used : float
                The conformal threshold used.
            alpha_used : float
                The alpha value used.
        """
        assert self.calibration_scores is not None, "Calibrator not fitted"

        if alpha is None:
            alpha = self.alpha
            q = self.quantile
        else:
            q = self._compute_quantile(alpha)

        m = probas.shape[0]

        # Non-conformity scores for each candidate class
        # s(x, y) = 1 - p(y|x)
        scores_class0 = 1.0 - probas[:, 0]  # score for y=0 (benign)
        scores_class1 = 1.0 - probas[:, 1]  # score for y=1 (malicious)

        # Prediction set: include class y if s(x,y) <= q
        include_0 = scores_class0 <= q  # include benign
        include_1 = scores_class1 <= q  # include malicious

        # Build prediction sets
        prediction_sets = []
        for i in range(m):
            pset = []
            if include_0[i]:
                pset.append(0)
            if include_1[i]:
                pset.append(1)
            prediction_sets.append(pset)

        sizes = np.array([len(ps) for ps in prediction_sets])

        # Conformal p-values
        # p(y) = (#{i : s_i >= s(x,y)} + 1) / (n + 1)
        # = (n - rank_of_score + 1) / (n + 1)  where rank is 0-indexed
        n = self.n_calibration
        pvalues = np.zeros((m, 2))
        for cls_idx, scores_cls in enumerate([scores_class0, scores_class1]):
            for i in range(m):
                s = scores_cls[i]
                # Number of calibration scores >= s
                count_geq = np.sum(self.calibration_scores >= s)
                pvalues[i, cls_idx] = (count_geq + 1) / (n + 1)

        return {
            "prediction_sets": prediction_sets,
            "prediction_set_sizes": sizes,
            "pvalues": pvalues,
            "quantile_used": q,
            "alpha_used": alpha,
        }

    def save(self, path: str):
        """Serialize the calibrator to disk."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "ConformalCalibrator":
        """Load a serialized calibrator."""
        with open(path, "rb") as f:
            return pickle.load(f)
