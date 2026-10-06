# Feature Eligibility and Leakage Audit

This document classifies every column in `dataset/MUD_malicious_urls_2026_V2.csv` for use in the DSSE (Dual-Stage Semantic Ensemble) architecture.

## 1. RAW URL
These columns contain the raw or minimally processed URL strings. They are the primary inputs for Stage 2 of DSSE (Semantic feature extraction via text embeddings).
- `url`
- `url_normalized`
- `url_hash`

## 2. TARGET
These columns contain the ground-truth labels for the data. They must NOT be used as input features. The primary target for DSSE will be derived from these.
- `label`: Multi-class integer (0=benign, 1=defacement, 2=phishing, 3=malware)
- `class_label`: String representation (benign, defacement, phishing, malware)
- `type`: Redundant string representation (same as class_label)

*Note: For the initial DSSE implementation, we will map `label == 0` to `0` (benign) and `label > 0` to `1` (malicious) to create a binary target.*

## 3. METADATA
These columns contain dataset curation context, parsed components, or temporal information. They are useful for analysis but generally should not be used as direct input features to the model to avoid overfitting to specific data collection timelines.
- `source_dataset`
- `dataset_version`
- `ts_first_seen`
- `ts_aggregated`
- `reachability_class`
- `is_syntax_valid`
- `norm_actions`
- `host`
- `path`
- `query`
- `tld`

## 4. ENGINEERED FEATURES
These columns are lexical and heuristic features that can be extracted instantaneously from the URL string itself. These will form the input for Stage 1 of DSSE.
- `domain_len`, `path_len`, `query_len`, `path_depth`, `subdomain_count`, `query_param_count`
- `has_port`, `is_ip_host`, `has_punycode`, `is_https`, `is_shortener`
- `digit_count_v2`, `letter_count_v2`, `special_count_v2`, `digit_ratio`, `letter_ratio`, `special_ratio`
- `token_count`, `max_token_len`, `avg_token_len`, `url_entropy`, `host_entropy`, `url_len_v2`, `url_len`
- `@`, `?`, `-`, `=`, `.`, `#`, `%`, `+`, `$`, `!`, `*`, `,`, `//`, `digits`, `letters`
- `abnormal_url`, `https`, `Shortining_Service`, `having_ip_address`
- `phish_urgency_words`, `phish_security_words`, `phish_brand_mentions`, `phish_brand_hijack`, `phish_multiple_subdomains`, `phish_long_path`, `phish_many_params`, `phish_suspicious_tld`
- `phish_adv_exact_brand_match`, `phish_adv_brand_in_subdomain`, `phish_adv_brand_in_path`, `phish_adv_hyphen_count`, `phish_adv_number_count`, `phish_adv_suspicious_tld`, `phish_adv_long_domain`, `phish_adv_many_subdomains`, `phish_adv_encoded_chars`, `phish_adv_path_keywords`, `phish_adv_has_redirect`, `phish_adv_many_params`
- `path_has_hacked_terms`, `suspicious_extension`, `path_underscore_count`, `is_gov_edu`

## 5. LEAKAGE (MUST BE DROPPED)
The following columns are either direct leaks of the target or impossible to obtain at inference time without breaking the latency requirements of a real-time URL detector.

- **Web Scraping/Crawler Features**: 
  - `web_http_status`, `web_is_live`, `web_ext_ratio`, `web_unique_domains`, `web_favicon`, `web_csp`, `web_xframe`, `web_hsts`, `web_xcontent`, `web_security_score`, `web_forms_count`, `web_password_fields`, `web_hidden_inputs`, `web_has_login`, `web_ssl_valid`
  - **Reasoning**: A real-time malicious URL detector (like a browser extension) must block the URL *before* the user visits it. Requesting the page to extract features like `web_password_fields` or `web_ssl_valid` is impossible at inference time without a background crawler. Using them during training would give the model information it will not have in production, leading to severe data leakage and inflated offline accuracy. They must be dropped.
