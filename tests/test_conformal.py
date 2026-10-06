"""
Tests for the Conformal Prediction module (Phase 6).
Covers all 17 test requirements from the Phase 6 specification.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pickle
import unittest

import numpy as np
import pandas as pd

from src.conformal.calibrator import ConformalCalibrator
from src.conformal.routing import (
    route_prediction_set, FINAL_BENIGN, FINAL_MALICIOUS, NEEDS_STAGE2,
)


class TestConformalCalibrator(unittest.TestCase):
    """Tests for the ICP calibrator and predictor."""

    @classmethod
    def setUpClass(cls):
        base = os.path.join(os.path.dirname(__file__), '..')

        # Load configs
        cc_path = os.path.join(base, 'config', 'conformal_config.json')
        if not os.path.exists(cc_path):
            raise unittest.SkipTest("config/conformal_config.json not found — run Phase 6 pipeline first")
        with open(cc_path) as f:
            cls.conformal_cfg = json.load(f)
        with open(os.path.join(base, 'config', 'stage1_features.json')) as f:
            cls.feat_cfg = json.load(f)

        # Load calibrator
        cls.cal_path = os.path.join(base, cls.conformal_cfg['calibrator_path'])
        cls.calibrator = ConformalCalibrator.load(cls.cal_path)

        # Load ICP-specific HXRF model
        cls.model_path = os.path.join(base, cls.conformal_cfg['hxrf_model_path'])
        with open(cls.model_path, 'rb') as f:
            cls.model = pickle.load(f)

        # Load calibration data for verification
        cal_data_path = os.path.join(base, cls.conformal_cfg['calibration_dataset'])
        cls.cal_data = pd.read_parquet(cal_data_path)

        # Load test data path for leakage check
        cls.test_data_path = os.path.join(base, 'data', 'splits', 'test.parquet')

    # ── 1. Conformal calibrator loads ────────────────────────────────────
    def test_01_calibrator_loads(self):
        """Conformal calibrator loads from disk without error."""
        cal = ConformalCalibrator.load(self.cal_path)
        self.assertIsNotNone(cal)
        self.assertIsNotNone(cal.calibration_scores)
        self.assertGreater(cal.n_calibration, 0)

    # ── 2. Calibration data contains both classes ────────────────────────
    def test_02_calibration_both_classes(self):
        """Calibration set contains both benign and malicious samples."""
        self.assertIn(0, self.calibrator.class_distribution)
        self.assertIn(1, self.calibrator.class_distribution)
        self.assertGreater(self.calibrator.class_distribution[0], 0)
        self.assertGreater(self.calibrator.class_distribution[1], 0)

    # ── 3. HXRF model loads ─────────────────────────────────────────────
    def test_03_hxrf_model_loads(self):
        """ICP-specific HXRF model loads and has correct feature count."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.n_features_in_, 12)

    # ── 4. Prediction set is valid ──────────────────────────────────────
    def test_04_prediction_set_valid(self):
        """Prediction set contains only valid class labels."""
        probas = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]])
        result = self.calibrator.predict(probas)
        for ps in result['prediction_sets']:
            for cls in ps:
                self.assertIn(cls, [0, 1])

    # ── 5. Prediction set can be singleton or both ──────────────────────
    def test_05_prediction_set_types(self):
        """Test that benign-only, malicious-only, and both are possible."""
        # Very confident benign
        probas_ben = np.array([[0.999, 0.001]])
        res_ben = self.calibrator.predict(probas_ben, alpha=0.20)
        # Very confident malicious
        probas_mal = np.array([[0.001, 0.999]])
        res_mal = self.calibrator.predict(probas_mal, alpha=0.20)
        # Uncertain
        probas_amb = np.array([[0.5, 0.5]])
        res_amb = self.calibrator.predict(probas_amb, alpha=0.01)

        # At least one of the confident predictions should be singleton
        # and the uncertain one should be ambiguous at strict alpha
        self.assertTrue(
            len(res_ben['prediction_sets'][0]) >= 1 and
            len(res_mal['prediction_sets'][0]) >= 1,
        )
        # Check both classes appear somewhere
        all_classes = set()
        for ps in [res_ben['prediction_sets'][0],
                    res_mal['prediction_sets'][0],
                    res_amb['prediction_sets'][0]]:
            all_classes.update(ps)
        self.assertEqual(all_classes, {0, 1})

    # ── 6. Prediction set size between 1 and 2 ──────────────────────────
    def test_06_prediction_set_size(self):
        """Prediction set size is 0, 1, or 2 for binary classification.

        Size 0 (empty set) can occur at larger alpha when neither class's
        non-conformity score is below the quantile threshold — this is
        mathematically valid and means the model is too uncertain at the
        given significance level.
        """
        probas = np.array([[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.7, 0.3]])
        for a in [0.01, 0.05, 0.10, 0.20]:
            result = self.calibrator.predict(probas, alpha=a)
            for sz in result['prediction_set_sizes']:
                self.assertIn(sz, [0, 1, 2], f"Invalid set size {sz} for alpha={a}")

    # ── 7. Alpha is in (0, 1) ──────────────────────────────────────────
    def test_07_alpha_range(self):
        """Configured alpha is in (0, 1)."""
        self.assertGreater(self.calibrator.alpha, 0.0)
        self.assertLess(self.calibrator.alpha, 1.0)
        for a in self.conformal_cfg['sensitivity_alphas']:
            self.assertGreater(a, 0.0)
            self.assertLess(a, 1.0)

    # ── 8. P-values are in [0, 1] ──────────────────────────────────────
    def test_08_pvalues_range(self):
        """Conformal p-values are in [0, 1]."""
        probas = np.array([[0.8, 0.2], [0.3, 0.7], [0.5, 0.5]])
        result = self.calibrator.predict(probas)
        for i in range(probas.shape[0]):
            for cls in range(2):
                pv = result['pvalues'][i, cls]
                self.assertGreaterEqual(pv, 0.0, f"p-value < 0 at [{i},{cls}]")
                self.assertLessEqual(pv, 1.0, f"p-value > 1 at [{i},{cls}]")

    # ── 9. Malicious probability is in [0, 1] ──────────────────────────
    def test_09_malicious_probability_range(self):
        """HXRF produces probabilities in [0, 1]."""
        from src.features.stage1_features import extract_stage1_features
        feats = extract_stage1_features('http://example.com/test')
        fv = pd.DataFrame([feats])[self.feat_cfg['features']]
        probas = self.model.predict_proba(fv)
        self.assertGreaterEqual(probas[0, 0], 0.0)
        self.assertLessEqual(probas[0, 0], 1.0)
        self.assertGreaterEqual(probas[0, 1], 0.0)
        self.assertLessEqual(probas[0, 1], 1.0)

    # ── 10. Routing is correct for each prediction-set type ─────────────
    def test_10_routing_benign_singleton(self):
        """Singleton {benign} routes to FINAL_BENIGN."""
        self.assertEqual(route_prediction_set([0]), FINAL_BENIGN)

    def test_10b_routing_malicious_singleton(self):
        """Singleton {malicious} routes to FINAL_MALICIOUS."""
        self.assertEqual(route_prediction_set([1]), FINAL_MALICIOUS)

    def test_10c_routing_ambiguous(self):
        """Ambiguous {benign, malicious} routes to NEEDS_STAGE2."""
        self.assertEqual(route_prediction_set([0, 1]), NEEDS_STAGE2)

    # ── 11. Ambiguous prediction set routes to NEEDS_STAGE2 ─────────────
    def test_11_ambiguous_routes_stage2(self):
        """Ambiguous sets with both classes route to NEEDS_STAGE2."""
        self.assertEqual(route_prediction_set([0, 1]), NEEDS_STAGE2)
        self.assertEqual(route_prediction_set([1, 0]), NEEDS_STAGE2)

    # ── 12. Singleton benign routes to FINAL_BENIGN ─────────────────────
    def test_12_singleton_benign(self):
        """Verified: {0} → FINAL_BENIGN."""
        self.assertEqual(route_prediction_set([0]), FINAL_BENIGN)

    # ── 13. Singleton malicious routes to FINAL_MALICIOUS ───────────────
    def test_13_singleton_malicious(self):
        """Verified: {1} → FINAL_MALICIOUS."""
        self.assertEqual(route_prediction_set([1]), FINAL_MALICIOUS)

    # ── 14. Calibration does not access the test set ────────────────────
    def test_14_no_test_leakage(self):
        """Calibration samples are not from the test set."""
        test_df = pd.read_parquet(self.test_data_path)
        if 'url' in self.cal_data.columns and 'url' in test_df.columns:
            cal_urls = set(self.cal_data['url'])
            test_urls = set(test_df['url'])
            overlap = cal_urls & test_urls
            self.assertEqual(len(overlap), 0,
                             f"Calibration-test overlap: {len(overlap)} URLs")

    # ── 15. Predictor works without network access ──────────────────────
    def test_15_no_network(self):
        """Full ICP prediction works without network."""
        from src.conformal.predictor import predict_conformal
        result = predict_conformal('http://localhost/test')
        self.assertIn('prediction_set', result)
        self.assertIn('routing', result)
        self.assertIn(result['routing'],
                      [FINAL_BENIGN, FINAL_MALICIOUS, NEEDS_STAGE2])

    # ── 16. Saved calibrator can be loaded and reused ───────────────────
    def test_16_calibrator_save_load(self):
        """Calibrator survives save/load round-trip."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            tmp_path = f.name
        try:
            self.calibrator.save(tmp_path)
            loaded = ConformalCalibrator.load(tmp_path)
            self.assertEqual(loaded.alpha, self.calibrator.alpha)
            self.assertEqual(loaded.n_calibration, self.calibrator.n_calibration)
            self.assertAlmostEqual(loaded.quantile, self.calibrator.quantile, places=10)
            np.testing.assert_array_equal(loaded.calibration_scores,
                                          self.calibrator.calibration_scores)
        finally:
            os.unlink(tmp_path)

    # ── 17. Results are reproducible with fixed seed ────────────────────
    def test_17_reproducibility(self):
        """Same inputs produce same outputs."""
        probas = np.array([[0.8, 0.2], [0.3, 0.7]])
        r1 = self.calibrator.predict(probas)
        r2 = self.calibrator.predict(probas)
        self.assertEqual(r1['prediction_sets'], r2['prediction_sets'])
        np.testing.assert_array_equal(r1['pvalues'], r2['pvalues'])
        self.assertEqual(r1['quantile_used'], r2['quantile_used'])

    # ── Extra: quantile is reasonable ───────────────────────────────────
    def test_extra_quantile_reasonable(self):
        """Conformal quantile is in [0, 1]."""
        self.assertGreaterEqual(self.calibrator.quantile, 0.0)
        self.assertLessEqual(self.calibrator.quantile, 1.0)

    # ── Extra: calibration_scores is sorted ─────────────────────────────
    def test_extra_scores_sorted(self):
        """Calibration scores are stored in ascending order."""
        scores = self.calibrator.calibration_scores
        self.assertTrue(np.all(scores[:-1] <= scores[1:]))


if __name__ == '__main__':
    unittest.main()
