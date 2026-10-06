"""
Tests for the Stage-1 feature extractor.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import unittest

from src.features.stage1_features import extract_stage1_features


class TestStage1Features(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        base = os.path.join(os.path.dirname(__file__), '..')
        with open(os.path.join(base, 'config', 'stage1_features.json')) as f:
            cls.config = json.load(f)
        cls.expected_features = cls.config['features']

    def test_feature_count(self):
        feats = extract_stage1_features('http://secure-update-login.com/login.php?session=123')
        self.assertEqual(len(feats), 12)

    def test_feature_names_match_config(self):
        feats = extract_stage1_features('http://example.com')
        for f in self.expected_features:
            self.assertIn(f, feats, f"Missing feature: {f}")

    def test_feature_order(self):
        feats = extract_stage1_features('http://example.com')
        self.assertEqual(list(feats.keys()), self.expected_features)

    def test_types_numeric(self):
        feats = extract_stage1_features('http://example.com/path?q=1')
        for k, v in feats.items():
            self.assertIsInstance(v, (int, float), f"{k} is {type(v)}")

    def test_suspicious_extension_positive(self):
        feats = extract_stage1_features('http://example.com/download.php')
        self.assertEqual(feats['suspicious_extension'], 1)

    def test_suspicious_extension_negative(self):
        feats = extract_stage1_features('http://example.com/page.aspx')
        self.assertEqual(feats['suspicious_extension'], 0)

    def test_abnormal_url_no_scheme(self):
        feats = extract_stage1_features('example.com/path')
        self.assertEqual(feats['abnormal_url'], 1)

    def test_abnormal_url_with_scheme(self):
        feats = extract_stage1_features('http://example.com/path')
        self.assertEqual(feats['abnormal_url'], 0)

    def test_url_len_original(self):
        url = 'example.com/path'
        feats = extract_stage1_features(url)
        self.assertEqual(feats['url_len'], len(url))

    def test_character_counts(self):
        url = 'http://a.b.com/x?y=1&z=2'
        feats = extract_stage1_features(url)
        self.assertEqual(feats['='], url.count('='))
        self.assertEqual(feats['.'], url.count('.'))
        self.assertEqual(feats['//'], url.count('//'))


if __name__ == '__main__':
    unittest.main()
