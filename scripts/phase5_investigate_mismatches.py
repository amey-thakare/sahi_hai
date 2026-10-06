"""
Deep investigation of mismatched features to understand the dataset's
exact definitions for each of the 7 mismatched features.
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
# 1. path_depth — dataset has 1 for trailing-slash URLs, we get 0
# ============================================================================
print("=" * 70)
print("INVESTIGATING: path_depth")
print("=" * 70)
# Trailing slash URLs like christophebeck.com/
urls_trailing = df[df['url'].str.match(r'^[^/]+/$', na=False)].head(10)
for _, row in urls_trailing.iterrows():
    url = row['url']
    path_parsed = urlparse('http://' + url if not url.startswith('http') else url).path
    segments = [s for s in path_parsed.split('/') if s]
    print(f"  URL: {url[:60]:60s}  dataset={row['path_depth']}  parsed_segments={len(segments)}  path='{path_parsed}'")

# Check: does dataset count trailing slash as +1?
# Let's check non-trailing slash
print()
urls_no_trail = df[~df['url'].str.endswith('/') & (df['path_depth'] == 0)].head(5)
for _, row in urls_no_trail.iterrows():
    url = row['url']
    path_parsed = urlparse('http://' + url if not url.startswith('http') else url).path
    print(f"  URL: {url[:60]:60s}  dataset={row['path_depth']}  path='{path_parsed}'")

# ============================================================================
# 2. special_count_v2 — systematic offset of ~3
# ============================================================================
print()
print("=" * 70)
print("INVESTIGATING: special_count_v2")
print("=" * 70)
# The delta is consistently ~3 for many URLs. Maybe the dataset includes
# the scheme "://" in the count but URLs without scheme get 0 extra?
# Or maybe the dataset strips the scheme before counting?
sample = df.sample(20, random_state=42)
for _, row in sample.iterrows():
    url = row['url']
    has_scheme = url.startswith('http://') or url.startswith('https://')
    count_full = sum(1 for c in url if not c.isalnum())
    # Try without scheme
    url_no_scheme = re.sub(r'^https?://', '', url)
    count_no_scheme = sum(1 for c in url_no_scheme if not c.isalnum())
    print(f"  ds={row['special_count_v2']:3d}  full={count_full:3d}  no_scheme={count_no_scheme:3d}  scheme={'Y' if has_scheme else 'N'}  url={url[:60]}")

# ============================================================================
# 3. token_count
# ============================================================================
print()
print("=" * 70)
print("INVESTIGATING: token_count")
print("=" * 70)
for _, row in sample.head(10).iterrows():
    url = row['url']
    has_scheme = url.startswith('http://') or url.startswith('https://')
    url_no_scheme = re.sub(r'^https?://', '', url)
    tokens_full = [t for t in re.split(r'[^a-zA-Z0-9]+', url) if t]
    tokens_no_scheme = [t for t in re.split(r'[^a-zA-Z0-9]+', url_no_scheme) if t]
    print(f"  ds={row['token_count']:3d}  full={len(tokens_full):3d}  no_scheme={len(tokens_no_scheme):3d}  url={url[:60]}")

# ============================================================================
# 4. url_entropy
# ============================================================================
print()
print("=" * 70)
print("INVESTIGATING: url_entropy")
print("=" * 70)
for _, row in sample.head(10).iterrows():
    url = row['url']
    url_no_scheme = re.sub(r'^https?://', '', url)
    
    def entropy(s):
        prob = [s.count(c) / len(s) for c in set(s)]
        return -sum(p * math.log(p) / math.log(2.0) for p in prob)
    
    ent_full = entropy(url) if len(url) > 0 else 0
    ent_no_scheme = entropy(url_no_scheme) if len(url_no_scheme) > 0 else 0
    print(f"  ds={row['url_entropy']:.4f}  full={ent_full:.4f}  no_scheme={ent_no_scheme:.4f}  url={url[:60]}")

# ============================================================================
# 5. abnormal_url — check what the dataset actually means
# ============================================================================
print()
print("=" * 70)
print("INVESTIGATING: abnormal_url")
print("=" * 70)
# Check distribution
print(f"  Value counts: {df['abnormal_url'].value_counts().to_dict()}")
# Check if it's related to having a scheme
for _, row in sample.head(20).iterrows():
    url = row['url']
    has_scheme = url.startswith('http://') or url.startswith('https://')
    parsed = urlparse(url if has_scheme else 'http://' + url)
    host = parsed.hostname or ''
    host_in_url = host in url if host else False
    print(f"  ds={row['abnormal_url']}  scheme={'Y' if has_scheme else 'N'}  host='{host[:30]}'  host_in_url={host_in_url}  url={url[:50]}")

# ============================================================================
# 6. phish_adv_hyphen_count
# ============================================================================
print()
print("=" * 70)
print("INVESTIGATING: phish_adv_hyphen_count")
print("=" * 70)
# Maybe it only counts hyphens in the hostname?
for _, row in sample.head(20).iterrows():
    url = row['url']
    has_scheme = url.startswith('http://') or url.startswith('https://')
    parsed = urlparse(url if has_scheme else 'http://' + url)
    host = parsed.hostname or ''
    hyphens_full = url.count('-')
    hyphens_host = host.count('-')
    print(f"  ds={row['phish_adv_hyphen_count']:2d}  full={hyphens_full:2d}  host_only={hyphens_host:2d}  host={host[:30]:30s}  url={url[:50]}")

# ============================================================================
# 7. suspicious_extension
# ============================================================================
print()
print("=" * 70)
print("INVESTIGATING: suspicious_extension")
print("=" * 70)
# Check URLs that dataset says are suspicious but we say are not
mismatched = df[(df['suspicious_extension'] == 1)].sample(20, random_state=42)
for _, row in mismatched.iterrows():
    url = row['url']
    has_scheme = url.startswith('http://') or url.startswith('https://')
    parsed = urlparse(url if has_scheme else 'http://' + url)
    path = parsed.path.lower()
    # Check with broader extension list
    exts = path.split('.')[-1] if '.' in path else ''
    print(f"  ds=1  path_ext='{exts}'  path={path[:50]}  url={url[:60]}")
