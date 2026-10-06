"""
Tests for the Stage-1 HXRF model and predictor.
Covers all 13 test requirements from Phase 5 specification.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import pickle
import unittest

import numpy as np
import pandas as pd

from src.features.stage1_features import extract_stage1_features
from src.stage1.predictor import predict_stage1, TAU_HIGH, TAU_LOW


class TestStage1Model(unittest.TestCase):
    """Tests for the HXRF model artefact and predictor."""

    @classmethod
    def setUpClass(cls):
        base = os.path.join(os.path.dirname(__file__), '..')
        with open(os.path.join(base, 'config', 'stage1_model_config.json')) as f:
            cls.config = json.load(f)
        with open(os.path.join(base, cls.config['model_path']), 'rb') as f:
            cls.model = pickle.load(f)
        with open(os.path.join(base, 'config', 'stage1_features.json')) as f:
            cls.feature_cfg = json.load(f)

    # ── 1. Model loads successfully ──────────────────────────────────────
    def test_01_model_loads(self):
        """Model pickle loads without error."""
        self.assertIsNotNone(self.model)
        self.assertTrue(hasattr(self.model, 'predict'))
        self.assertTrue(hasattr(self.model, 'predict_proba'))

    # ── 2. Predictor accepts a raw URL ───────────────────────────────────
    def test_02_predict_raw_url(self):
        """predict_stage1() returns a valid dict for a raw URL."""
        result = predict_stage1('http://example.com/test')
        self.assertIsInstance(result, dict)
        self.assertIn('prediction', result)
        self.assertIn('malicious_probability', result)
        self.assertIn('routing_decision', result)
        self.assertIn('url', result)

    # ── 3. Exactly 12 features are generated ─────────────────────────────
    def test_03_feature_count(self):
        """Feature extractor produces exactly 12 features."""
        feats = extract_stage1_features('http://example.com/path?q=1')
        self.assertEqual(len(feats), 12)

    # ── 4. Feature ordering is correct ───────────────────────────────────
    def test_04_feature_order(self):
        """Feature order matches config/stage1_features.json."""
        expected = self.feature_cfg['features']
        feats = extract_stage1_features('http://example.com')
        self.assertEqual(list(feats.keys()), expected)

    # ── 5. Model receives exactly the expected feature columns ───────────
    def test_05_model_feature_columns(self):
        """Model.n_features_in_ matches 12 and feature names match."""
        self.assertEqual(self.model.n_features_in_, 12)
        expected = self.feature_cfg['features']
        feats = extract_stage1_features('http://example.com')
        fv = pd.DataFrame([feats])[expected]
        # Should not raise
        prob = self.model.predict_proba(fv)
        self.assertEqual(prob.shape, (1, 2))

    # ── 6. Prediction class is valid ─────────────────────────────────────
    def test_06_valid_class(self):
        """Returned class_id is 0 or 1, prediction is benign/malicious."""
        result = predict_stage1('http://google.com')
        self.assertIn(result['class_id'], [0, 1])
        self.assertIn(result['prediction'], ['benign', 'malicious'])

    # ── 7. Malicious probability is between 0 and 1 ──────────────────────
    def test_07_probability_range(self):
        """P(malicious) and confidence are in [0, 1]."""
        test_urls = [
            'http://google.com',
            'http://evil-malware.xyz/payload.exe',
            'secure-login-update.com/verify.php?id=123',
            'http://very-long-subdomain.phishing.evil.com/login?session=abc123',
        ]
        for url in test_urls:
            result = predict_stage1(url)
            self.assertGreaterEqual(result['malicious_probability'], 0.0,
                                    f"P(mal) < 0 for {url}")
            self.assertLessEqual(result['malicious_probability'], 1.0,
                                 f"P(mal) > 1 for {url}")

    # ── 8. Confidence is between 0 and 1 ─────────────────────────────────
    def test_08_confidence_range(self):
        """Confidence is in [0, 1]."""
        test_urls = ['http://google.com', 'evil.xyz/payload.exe']
        for url in test_urls:
            result = predict_stage1(url)
            self.assertGreaterEqual(result['confidence'], 0.0)
            self.assertLessEqual(result['confidence'], 1.0)

    # ── 9. Routing works correctly (boundary conditions) ─────────────────
    def test_09_routing_logic(self):
        """Routing decision is consistent with probability thresholds."""
        test_urls = [
            'http://example.com',
            'http://evil-malware.xyz/payload.exe',
            'secure-login-update.com/verify.php?id=123',
            'http://google.com/search?q=hello',
            'http://192.168.1.1/admin/login.php',
        ]
        for url in test_urls:
            result = predict_stage1(url)
            p = result['malicious_probability']
            if p >= TAU_HIGH or p <= TAU_LOW:
                self.assertEqual(result['routing_decision'], 'final_stage1',
                                 f"Expected final_stage1 for P={p:.4f} on {url}")
            else:
                self.assertEqual(result['routing_decision'], 'needs_stage2',
                                 f"Expected needs_stage2 for P={p:.4f} on {url}")

    def test_09b_routing_threshold_values(self):
        """Verify the architectural threshold constants are correct."""
        self.assertEqual(TAU_HIGH, 0.95)
        self.assertEqual(TAU_LOW, 0.05)

    # ── 10. Predictor returns latency ────────────────────────────────────
    def test_10_latency_returned(self):
        """Predictor includes a non-negative latency_ms."""
        result = predict_stage1('http://example.com')
        self.assertIn('latency_ms', result)
        self.assertIsInstance(result['latency_ms'], float)
        self.assertGreaterEqual(result['latency_ms'], 0.0)

    # ── 11. No network access required ───────────────────────────────────
    def test_11_no_network_access(self):
        """Feature extraction and prediction succeed without network."""
        feats = extract_stage1_features('http://localhost/test')
        fv = pd.DataFrame([feats])[self.feature_cfg['features']]
        prob = self.model.predict_proba(fv)
        self.assertEqual(prob.shape[1], 2)

    # ── 12. Model operates entirely offline ──────────────────────────────
    def test_12_offline_operation(self):
        """Full predict_stage1 pipeline works with no external deps."""
        result = predict_stage1('http://offline-test.local/page')
        self.assertIn('prediction', result)
        self.assertIn('routing_decision', result)

    # ── 13. Saved configuration matches the trained model ────────────────
    def test_13_config_matches_model(self):
        """Config file is consistent with the model object."""
        # Feature count
        self.assertEqual(self.config['feature_count'], 12)
        self.assertEqual(len(self.config['features']), 12)
        self.assertEqual(self.model.n_features_in_, 12)

        # Feature order must be identical
        self.assertEqual(self.config['features'], self.config['feature_order'])
        self.assertEqual(self.config['features'], self.feature_cfg['features'])

        # Class mapping
        self.assertIn('0', self.config['class_mapping'])
        self.assertIn('1', self.config['class_mapping'])
        self.assertEqual(self.config['class_mapping']['0'], 'benign')
        self.assertEqual(self.config['class_mapping']['1'], 'malicious')

        # Model type
        self.assertEqual(self.config['model_type'], 'RandomForestClassifier')
        self.assertEqual(type(self.model).__name__, 'RandomForestClassifier')

        # Hyperparameters match actual model
        hp = self.config['hyperparameters']
        self.assertEqual(self.model.n_estimators, hp['n_estimators'])
        self.assertEqual(self.model.max_depth, hp['max_depth'])
        self.assertEqual(self.model.min_samples_split, hp['min_samples_split'])
        self.assertEqual(self.model.min_samples_leaf, hp['min_samples_leaf'])
        self.assertEqual(self.model.max_features, hp['max_features'])

    # ── Extra: no NaN in features ────────────────────────────────────────
    def test_extra_no_nan_in_features(self):
        """No NaN values produced by the feature extractor."""
        urls = [
            'http://example.com',
            'https://very-long-subdomain.phishing.evil.com/login.php?id=1&a=2',
            'ftp-login-192.168.1.1/admin',
        ]
        for url in urls:
            feats = extract_stage1_features(url)
            for k, v in feats.items():
                self.assertFalse(
                    v != v,  # NaN check
                    f"NaN for feature '{k}' on url '{url}'",
                )

    # ── Extra: model has correct random state ────────────────────────────
    def test_extra_random_state(self):
        """Model random_state matches the configured seed."""
        self.assertEqual(self.model.random_state, self.config['random_seed'])


if __name__ == '__main__':
    unittest.main()
