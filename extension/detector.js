/**
 * In-browser Feature Extraction & Neural Network Inference Engine
 * Sahi Hai (Manifest V3)
 */

const SHORTENING_REGEX = /(bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|lnkd\.in|db\.tt|qr\.ae|adf\.ly|bitly\.com|cur\.lv|ity\.im|q\.gs|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|link\.zip\.net)/i;

const IP_V4_REGEX = /^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$/;

const SUSPICIOUS_DOMAINS = /(amtso\.org\/.*phishing|wicar\.org|at\.ua|usa\.cc|baltazarpresentes\.com\.br|pe\.hu|esy\.es|hol\.es|sweddy\.com|myjino\.ru|96\.lt|ow\.ly|\.tk|\.ml|\.ga|\.cf|\.gq)/i;

/**
 * Extract 22 features matching the trained model pipeline
 */
function extractFeatures(rawUrl, domStats = {}) {
  let urlObj;
  try {
    urlObj = new URL(rawUrl);
  } catch (e) {
    urlObj = { hostname: rawUrl, href: rawUrl, protocol: 'https:' };
  }

  const hostname = urlObj.hostname || '';
  const url = rawUrl;

  const features = {};

  // 1. IP Address (Feature 0)
  const isIP = IP_V4_REGEX.test(hostname);
  features.ip_address = isIP ? -1 : 1;

  // 2. URL Length (Feature 1)
  if (url.length < 54) {
    features.url_length = 1;
  } else if (url.length <= 75) {
    features.url_length = 0;
  } else {
    features.url_length = -1;
  }

  // 3. URL Shortening (Feature 2)
  features.shortening = SHORTENING_REGEX.test(url) ? -1 : 1;

  // 4. Having @ Symbol (Feature 3)
  features.at_symbol = url.includes('@') ? -1 : 1;

  // 5. Double Slash Redirect (Feature 4)
  const lastDoubleSlash = url.lastIndexOf('//');
  features.double_slash = lastDoubleSlash > 7 ? -1 : 1;

  // 6. Prefix / Suffix with '-' in Domain (Feature 5)
  features.prefix_suffix = hostname.includes('-') ? -1 : 1;

  // 7. Subdomains (Feature 6)
  const dotCount = (hostname.match(/\./g) || []).length;
  if (dotCount <= 2) {
    features.subdomains = 1;
  } else if (dotCount === 3) {
    features.subdomains = 0;
  } else {
    features.subdomains = -1;
  }

  // 8. SSL / Protocol (Feature 8 in original dataset)
  const isHttps = urlObj.protocol === 'https:';
  features.ssl = isHttps ? 1 : -1;

  // 9. Favicon (Feature 9)
  const faviconExternal = domStats.faviconExternal || false;
  features.favicon = faviconExternal ? -1 : 1;

  // 10. HTTP/HTTPS token in domain part (Feature 11)
  const cleanDomain = hostname.replace(/^(https?:\/\/)?(www\.)?/, '');
  features.https_token = cleanDomain.includes('http') || cleanDomain.includes('https') ? -1 : 1;

  // 11. Request URL / external media ratio (Feature 12)
  const reqExternalPct = domStats.requestUrlExternalPct ?? 10;
  if (reqExternalPct < 22.0) {
    features.request_url = 1;
  } else if (reqExternalPct < 61.0) {
    features.request_url = 0;
  } else {
    features.request_url = -1;
  }

  // 12. URL of Anchor (Feature 13)
  const anchorUnsafePct = domStats.anchorUnsafePct ?? 15;
  if (anchorUnsafePct < 31.0) {
    features.anchor_url = 1;
  } else if (anchorUnsafePct < 67.0) {
    features.anchor_url = 0;
  } else {
    features.anchor_url = -1;
  }

  // 13. Links in <script> and <link> tags (Feature 14)
  const linksExternalPct = domStats.linksInTagsExternalPct ?? 10;
  if (linksExternalPct < 17.0) {
    features.links_in_tags = 1;
  } else if (linksExternalPct < 81.0) {
    features.links_in_tags = 0;
  } else {
    features.links_in_tags = -1;
  }

  // 14. Server Form Handler (SFH) (Feature 15)
  features.sfh = domStats.sfhStatus ?? 1;

  // 15. Submitting to Email (mailto: form action) (Feature 16)
  features.submitting_to_email = domStats.hasMailtoForm ? -1 : 1;

  // 16. Abnormal URL (Feature 17)
  features.abnormal_url = hostname.length > 0 && url.includes(hostname) ? 1 : -1;

  // 17. Hidden / Suspicious iFrame (Feature 22)
  features.iframe = domStats.hasHiddenIframe ? -1 : 1;

  // 18. Age of Domain / WHOIS heuristic (Feature 23)
  const isTopDomain = /(google|youtube|github|wikipedia|microsoft|apple|amazon|cloudflare|mozilla|facebook|twitter|instagram|linkedin|yahoo|reddit|netflix)\./i.test(hostname);
  features.domain_age = isTopDomain ? 1 : (isIP ? -1 : 1);

  // 19. DNS Record / Valid domain format (Feature 24)
  features.dns_record = hostname.includes('.') && hostname.split('.').pop().length >= 2 ? 1 : -1;

  // 20. Web Traffic heuristic (Feature 25)
  features.web_traffic = isTopDomain ? 1 : 0;

  // 21. Google Search Index (Feature 27)
  features.google_index = isTopDomain ? 1 : 1;

  // 22. Statistical Reports / Blacklists (Feature 29)
  const isKnownBadDomain = SUSPICIOUS_DOMAINS.test(url) || isIP;
  features.statistical_report = isKnownBadDomain ? -1 : 1;

  // Ordered array of 22 features matching model indices
  const featureVector = [
    features.ip_address,
    features.url_length,
    features.shortening,
    features.at_symbol,
    features.double_slash,
    features.prefix_suffix,
    features.subdomains,
    features.ssl,
    features.favicon,
    features.https_token,
    features.request_url,
    features.anchor_url,
    features.links_in_tags,
    features.sfh,
    features.submitting_to_email,
    features.abnormal_url,
    features.iframe,
    features.domain_age,
    features.dns_record,
    features.web_traffic,
    features.google_index,
    features.statistical_report
  ];

  return { featureVector, features };
}

/**
 * Neural Network Inference in JavaScript
 */
function runNeuralNetwork(featureVector, model) {
  let activations = featureVector;

  // Layer by layer forward pass
  for (let l = 0; l < model.weights.length - 1; l++) {
    const W = model.weights[l];
    const b = model.biases[l];
    const nextActivations = new Array(b.length).fill(0);

    for (let j = 0; j < b.length; j++) {
      let sum = b[j];
      for (let i = 0; i < activations.length; i++) {
        sum += activations[i] * W[i][j];
      }
      // ReLU activation
      nextActivations[j] = Math.max(0, sum);
    }
    activations = nextActivations;
  }

  // Output layer (Single neuron with Sigmoid)
  const lastW = model.weights[model.weights.length - 1];
  const lastB = model.biases[model.biases.length - 1];
  let outSum = lastB[0];
  for (let i = 0; i < activations.length; i++) {
    outSum += activations[i] * lastW[i][0];
  }

  const safeProbability = 1 / (1 + Math.exp(-outSum));
  return safeProbability;
}

// Export functions to global scope for popup.js
window.SahiHaiDetector = {
  extractFeatures,
  runNeuralNetwork
};
