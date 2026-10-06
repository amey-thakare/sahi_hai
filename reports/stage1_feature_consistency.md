# Stage-1 Feature Consistency Report

Sample size: 200 URLs (100 benign + 100 malicious)

| Feature | Match Rate | Status |
|---------|-----------|--------|
| `domain_len` | 100.0% | ✅ OK |
| `path_depth` | 98.5% | ⚠️ MISMATCH (3/200) |
| `special_count_v2` | 98.5% | ⚠️ MISMATCH (3/200) |
| `token_count` | 81.5% | ⚠️ MISMATCH (37/200) |
| `url_entropy` | 99.5% | ⚠️ MISMATCH (1/200) |
| `url_len` | 100.0% | ✅ OK |
| `=` | 100.0% | ✅ OK |
| `.` | 100.0% | ✅ OK |
| `//` | 100.0% | ✅ OK |
| `abnormal_url` | 100.0% | ✅ OK |
| `phish_adv_hyphen_count` | 95.5% | ⚠️ MISMATCH (9/200) |
| `suspicious_extension` | 99.0% | ⚠️ MISMATCH (2/200) |


## Mismatches for `path_depth`

- URL: `http://olx.co.id/riau/q-%7Bq%7D/?utm_source=google&utm_medium=search&utm_campaig`
  - Dataset: 2, Extractor: 3, Delta: 1
- URL: `bd4bc.ufh7.com/de/?Ympfc2NodXN0ZXJAeWFob28uZGUMessage-ID:`
  - Dataset: 1, Extractor: 2, Delta: 1
- URL: `twistedfactory.com/models/?name=Hollie%20Stevens`
  - Dataset: 1, Extractor: 2, Delta: 1

## Mismatches for `special_count_v2`

- URL: `http://olx.co.id/riau/q-%7Bq%7D/?utm_source=google&utm_medium=search&utm_campaig`
  - Dataset: 20, Extractor: 21, Delta: 1
- URL: `bd4bc.ufh7.com/de/?Ympfc2NodXN0ZXJAeWFob28uZGUMessage-ID:`
  - Dataset: 9, Extractor: 10, Delta: 1
- URL: `twistedfactory.com/models/?name=Hollie%20Stevens`
  - Dataset: 8, Extractor: 9, Delta: 1

## Mismatches for `token_count`

- URL: `'services.runescape.co.cm/m=forums/forums.vs?1534,94,544,63330546,94,544,6333054`
  - Dataset: 9, Extractor: 15, Delta: 6
- URL: `christophebeck.com/`
  - Dataset: 3, Extractor: 2, Delta: 1
- URL: `http://olx.co.id/riau/q-%7Bq%7D/?utm_source=google&utm_medium=search&utm_campaig`
  - Dataset: 16, Extractor: 17, Delta: 1
- URL: `habbogeradordemoedas2011.rel7.com/`
  - Dataset: 4, Extractor: 3, Delta: 1
- URL: `lordoftherings.kevnet.com:9010/1.%20The%20Fellowship%20of%20the%20Ring.html`
  - Dataset: 6, Extractor: 11, Delta: 5
- URL: `norteltechnj.com/`
  - Dataset: 3, Extractor: 2, Delta: 1
- URL: `utulocal239.org/`
  - Dataset: 3, Extractor: 2, Delta: 1
- URL: `http://torrentdn.com/bbs/s.php?bo_table=torrent_ani&wr_id=41350&k=%EB%82%98%EB%A`
  - Dataset: 16, Extractor: 24, Delta: 8
- URL: `ntrieb.home.comcast.net/`
  - Dataset: 5, Extractor: 4, Delta: 1
- URL: `completeathletedevelopment.com/`
  - Dataset: 3, Extractor: 2, Delta: 1

## Mismatches for `url_entropy`

- URL: `bd4bc.ufh7.com/de/?Ympfc2NodXN0ZXJAeWFob28uZGUMessage-ID:`
  - Dataset: 5.0602, Extractor: 5.0466, Delta: 0.013600000000000279

## Mismatches for `phish_adv_hyphen_count`

- URL: `secure.runescape.com.rs.yf-login.com/m=weblogin/loginform.html`
  - Dataset: 0, Extractor: 1, Delta: 1
- URL: `the-artists.org/artist/Jean-Pierre-Khazem`
  - Dataset: 0, Extractor: 1, Delta: 1
- URL: `paseo-ftmyers.com/news.aspx`
  - Dataset: 0, Extractor: 1, Delta: 1
- URL: `nikki-nya.livejournal.com/`
  - Dataset: 0, Extractor: 1, Delta: 1
- URL: `instagram-ris-risa.com`
  - Dataset: 0, Extractor: 2, Delta: 2
- URL: `digimedia-mls.com/dmxplus/`
  - Dataset: 0, Extractor: 1, Delta: 1
- URL: `arry-nano.com/index.html`
  - Dataset: 0, Extractor: 1, Delta: 1
- URL: `talented112-writer.xyz`
  - Dataset: 0, Extractor: 1, Delta: 1
- URL: `it-70-pro.com.br`
  - Dataset: 0, Extractor: 2, Delta: 2

## Mismatches for `suspicious_extension`

- URL: `mi.uib.no/~hansfn/tex/advocacydft.pdf`
  - Dataset: 0, Extractor: 1, Delta: 1
- URL: `http://michael-kupfer.de/x.txt`
  - Dataset: 1, Extractor: 0, Delta: 1
