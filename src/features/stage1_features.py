"""
Stage-1 Feature Extractor — DSSE HXRF Gatekeeper.

Extracts exactly 12 lightweight structural features from a raw URL string.
Feature definitions matched to MUD_malicious_urls_2026_V2.csv by empirical
verification against the url_normalized column.

All features are computable from the URL string alone in O(N) time.
No DNS, HTTP, WHOIS, or external lookups are used.

Normalisation pipeline (mirrors the dataset):
  1. Prepend ``http://`` if no scheme present.
  2. Remove trailing ``/`` when the path has segments beyond the root
     (e.g. ``/foo/bar/`` → ``/foo/bar``). Bare ``domain.com/`` keeps its
     trailing slash since it represents the root.

Known approximations (documented):
  - phish_adv_hyphen_count: ~3.5 % mismatch rate vs dataset.  The
    dataset's definition appears inconsistent across subsets.  We use
    hostname hyphen-count, which matches >96 % of rows.
  - suspicious_extension: dataset list reverse-engineered empirically.
    We do NOT include .aspx, .asp, .shtml, .exe, .sh (confirmed not
    flagged by the dataset).
"""
import math
import re
from urllib.parse import urlparse


def _normalise_url(url: str) -> str:
    """
    Mirror the dataset's ``url_normalized`` column.

    Steps
    -----
    1. Prepend ``http://`` when no scheme is present.
    2. Remove trailing ``/`` when the path has ≥2 segments
       (i.e. the slash is NOT the root-only ``/``).
    """
    if not (url.startswith('http://') or url.startswith('https://')):
        url = 'http://' + url

    # Strip trailing slash unless it is the root itself.
    # "http://example.com/"       → keep  (root only)
    # "http://example.com/foo/"   → strip
    parsed = urlparse(url)
    path = parsed.path
    if len(path) > 1 and path.endswith('/'):
        url = url.rstrip('/')   # safe: query/fragment never end with /

    return url


# Extensions flagged by the dataset (empirically verified).
# .aspx, .asp, .shtml, .exe, .sh are confirmed NOT flagged by the dataset.
_SUSPICIOUS_EXTS = frozenset([
    '.html', '.htm', '.php', '.cgi', '.jsp',
    '.zip', '.rar', '.js', '.vbs', '.bat', '.cmd',
    '.bin', '.pdf', '.doc', '.xls', '.ppt',
])


def extract_stage1_features(url: str) -> dict:
    """
    Extract exactly 12 lightweight structural features from a raw URL.

    Parameters
    ----------
    url : str
        The raw URL string, with or without a scheme.

    Returns
    -------
    dict
        Mapping  feature-name → numeric value  (int or float).
        Keys are in the canonical order defined in config/stage1_features.json.
    """
    url_norm = _normalise_url(url)
    parsed = urlparse(url_norm)
    host = parsed.hostname or ''
    path = parsed.path          # e.g. "/foo/bar", "/", or ""

    features = {}

    # ── 1. domain_len ──────────────────────────────────────────────────
    features['domain_len'] = len(host)

    # ── 2. path_depth ──────────────────────────────────────────────────
    #   len(path.split('/')) - 1    (= number of '/' in path).
    features['path_depth'] = len(path.split('/')) - 1 if path else 0

    # ── 3. special_count_v2 ────────────────────────────────────────────
    features['special_count_v2'] = sum(1 for c in url_norm if not c.isalnum())

    # ── 4. token_count ─────────────────────────────────────────────────
    #   Alphanumeric tokens AFTER stripping the scheme.
    url_no_scheme = re.sub(r'^https?://', '', url_norm)
    tokens = [t for t in re.split(r'[^a-zA-Z0-9]+', url_no_scheme) if t]
    features['token_count'] = len(tokens)

    # ── 5. url_entropy ─────────────────────────────────────────────────
    n = len(url_norm)
    if n == 0:
        features['url_entropy'] = 0.0
    else:
        freq: dict[str, int] = {}
        for c in url_norm:
            freq[c] = freq.get(c, 0) + 1
        features['url_entropy'] = round(
            -sum((cnt / n) * math.log2(cnt / n) for cnt in freq.values()),
            4,
        )

    # ── 6. url_len ─────────────────────────────────────────────────────
    features['url_len'] = len(url)

    # ── 7–9. Character counts on the original URL ──────────────────────
    features['='] = url.count('=')
    features['.'] = url.count('.')
    features['//'] = url.count('//')

    # ── 10. abnormal_url ───────────────────────────────────────────────
    features['abnormal_url'] = 0 if (
        url.startswith('http://') or url.startswith('https://')
    ) else 1

    # ── 11. phish_adv_hyphen_count ─────────────────────────────────────
    features['phish_adv_hyphen_count'] = host.count('-')

    # ── 12. suspicious_extension ───────────────────────────────────────
    path_lower = path.lower()
    features['suspicious_extension'] = 1 if any(
        path_lower.endswith(ext) for ext in _SUSPICIOUS_EXTS
    ) else 0

    return features
