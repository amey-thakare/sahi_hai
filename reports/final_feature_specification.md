# Final 12 Features

Stage-1 structural gatekeeper uses EXACTLY the following 12 predefined URL-only features, ordered deterministically:

1. `domain_len`
2. `path_depth`
3. `special_count_v2`
4. `token_count`
5. `url_entropy`
6. `url_len`
7. `=`
8. `.`
9. `//`
10. `abnormal_url`
11. `phish_adv_hyphen_count`
12. `suspicious_extension`

- **Exact Feature Ordering:** Confirmed strictly locked by `config/stage1_features.json`.
- **Exact Feature Definitions:** Derived completely from URL character-level and structural extraction.
- **No hidden features:** The model's `.n_features_in_` is exactly 12.
- **No DNS / External State:** No DNS resolutions, WHOIS checks, or external reputation APIs exist in the extraction pipeline.
- **No future information:** Only available string context from the incoming URL is utilized.
