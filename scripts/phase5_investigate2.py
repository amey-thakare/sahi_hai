"""
Deep investigation round 2 — remaining mismatches.
"""
import sys, os
sys.path.insert(0, os.getcwd())
import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
import math

df = pd.read_parquet('data/splits/train.parquet')

# ============================================================================
# 1. path_depth — off-by-1 with trailing slashes
# ============================================================================
print("=" * 70)
print("path_depth: checking slash-counting logic")
print("=" * 70)
sample = df.sample(30, random_state=99)
for _, row in sample.iterrows():
    url = row['url']
    url_norm = url if url.startswith('http') else 'http://' + url
    parsed = urlparse(url_norm)
    path = parsed.path
    # Various counting methods
    slash_count = path.count('/')
    segments = [s for s in path.split('/') if s]
    n_segments = len(segments)
    # Does the dataset use n_segments?
    print(f"  ds={row['path_depth']:2d}  slashes={slash_count:2d}  segments={n_segments:2d}  path='{path}'  url={url[:50]}")

# ============================================================================
# 2. token_count — try using original URL (not normalised)
# ============================================================================
print()
print("=" * 70)
print("token_count: trying original URL")
print("=" * 70)
for _, row in sample.iterrows():
    url = row['url']
    url_norm = url if url.startswith('http') else 'http://' + url
    tokens_orig = [t for t in re.split(r'[^a-zA-Z0-9]+', url) if t]
    tokens_norm = [t for t in re.split(r'[^a-zA-Z0-9]+', url_norm) if t]
    # Try: split only by / . - _ ? & = @ : etc. (not all non-alphanum)
    tokens_word = re.findall(r'[a-zA-Z0-9]+', url)
    print(f"  ds={row['token_count']:3d}  orig={len(tokens_orig):3d}  norm={len(tokens_norm):3d}  findall={len(tokens_word):3d}  url={url[:50]}")

# ============================================================================
# 3. phish_adv_hyphen_count — check registered domain vs full hostname
# ============================================================================
print()
print("=" * 70)
print("phish_adv_hyphen_count: hostname subparts")
print("=" * 70)
# Check: maybe it counts hyphens in the registered domain only (not subdomains)?
mismatch_urls = df[df['url'].str.contains('-', na=False)].sample(20, random_state=55)
for _, row in mismatch_urls.iterrows():
    url = row['url']
    url_norm = url if url.startswith('http') else 'http://' + url
    parsed = urlparse(url_norm)
    host = parsed.hostname or ''
    # Split host into parts
    parts = host.split('.')
    # Registered domain = last 2 parts for .com, last 3 for .co.uk etc
    # Simple: try just the SLD (second-level domain)
    sld = parts[-2] if len(parts) >= 2 else host
    hyphens_host = host.count('-')
    hyphens_sld = sld.count('-')
    hyphens_full = url.count('-')
    print(f"  ds={row['phish_adv_hyphen_count']:2d}  host={hyphens_host:2d}  sld={hyphens_sld:2d}  full={hyphens_full:2d}  host={host[:40]:40s}  url={url[:50]}")

# ============================================================================
# 4. suspicious_extension — check what .aspx means
# ============================================================================
print()
print("=" * 70)
print("suspicious_extension: .aspx")
print("=" * 70)
aspx_urls = df[df['url'].str.contains(r'\.aspx', na=False, case=False)].head(10)
for _, row in aspx_urls.iterrows():
    url = row['url']
    print(f"  ds={row['suspicious_extension']}  url={url[:70]}")
# Also check .asp
asp_urls = df[df['url'].str.contains(r'\.asp\b', na=False, case=False)].head(5)
for _, row in asp_urls.iterrows():
    print(f"  .asp: ds={row['suspicious_extension']}  url={url[:70]}")
