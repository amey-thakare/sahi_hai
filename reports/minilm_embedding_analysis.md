# Exploratory Semantic Similarity Analysis

Cosine similarities between sample URLs:

| URL 1 | URL 2 | Cosine Similarity |
|-------|-------|-------------------|
| `https://example.com/login` | `http://example.com/login.php` | 0.8586 |
| `https://example.com/login` | `https://secure-login.example.com` | 0.8145 |
| `https://example.com/login` | `http://phish.com/verify` | 0.3529 |
| `https://example.com/login` | `https://google.com` | 0.4535 |
| `https://example.com/login` | `https://googIe.com` | 0.3927 |
| `http://example.com/login.php` | `https://secure-login.example.com` | 0.7457 |
| `http://example.com/login.php` | `http://phish.com/verify` | 0.2734 |
| `http://example.com/login.php` | `https://google.com` | 0.2688 |
| `http://example.com/login.php` | `https://googIe.com` | 0.2536 |
| `https://secure-login.example.com` | `http://phish.com/verify` | 0.4251 |
| `https://secure-login.example.com` | `https://google.com` | 0.3421 |
| `https://secure-login.example.com` | `https://googIe.com` | 0.3489 |
| `http://phish.com/verify` | `https://google.com` | 0.2871 |
| `http://phish.com/verify` | `https://googIe.com` | 0.3786 |
| `https://google.com` | `https://googIe.com` | 0.5853 |
