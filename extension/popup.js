/**
 * Modern Manifest V3 Popup Controller with Threat Alert Visuals
 */

let cachedModel = null;
let currentTab = null;

// Load Neural Network Weights
async function loadModel() {
  if (cachedModel) return cachedModel;
  try {
    const res = await fetch(chrome.runtime.getURL('model_weights.json'));
    cachedModel = await res.json();
    return cachedModel;
  } catch (err) {
    console.error('Failed to load neural model weights:', err);
    return null;
  }
}

// Inspect DOM features from active tab safely
async function collectDOMStats(tabId) {
  try {
    const [result] = await chrome.scripting.executeScript({
      target: { tabId },
      func: () => {
        const hostname = window.location.hostname;
        
        // 1. External Media / Request URL
        const media = Array.from(document.querySelectorAll('img, audio, embed, iframe'));
        let externalMedia = 0;
        media.forEach(el => {
          const src = el.src || '';
          if (src && !src.includes(hostname) && src.startsWith('http')) {
            externalMedia++;
          }
        });
        const requestUrlExternalPct = media.length > 0 ? (externalMedia / media.length) * 100 : 0;

        // 2. Anchor tags
        const anchors = Array.from(document.querySelectorAll('a'));
        let unsafeAnchors = 0;
        anchors.forEach(a => {
          const href = a.getAttribute('href') || '';
          if (
            href.startsWith('#') || 
            href.toLowerCase().includes('javascript:') || 
            href.toLowerCase().includes('mailto:') || 
            (href.startsWith('http') && !href.includes(hostname))
          ) {
            unsafeAnchors++;
          }
        });
        const anchorUnsafePct = anchors.length > 0 ? (unsafeAnchors / anchors.length) * 100 : 0;

        // 3. Links in script and link tags
        const tagLinks = Array.from(document.querySelectorAll('script[src], link[href]'));
        let extTagLinks = 0;
        tagLinks.forEach(el => {
          const ref = el.src || el.href || '';
          if (ref && !ref.includes(hostname) && ref.startsWith('http')) {
            extTagLinks++;
          }
        });
        const linksInTagsExternalPct = tagLinks.length > 0 ? (extTagLinks / tagLinks.length) * 100 : 0;

        // 4. Server Form Handler (SFH)
        const forms = Array.from(document.querySelectorAll('form'));
        let sfhStatus = 1;
        let hasMailtoForm = false;
        forms.forEach(f => {
          const action = f.getAttribute('action') || '';
          if (action === '' || action === 'about:blank') {
            sfhStatus = -1;
          } else if (action.toLowerCase().startsWith('mailto:')) {
            hasMailtoForm = true;
          } else if (action.startsWith('http') && !action.includes(hostname)) {
            sfhStatus = 0;
          }
        });

        // 5. Hidden / Suspicious iFrames
        let hasHiddenIframe = false;
        const iframes = Array.from(document.querySelectorAll('iframe'));
        iframes.forEach(f => {
          const rect = f.getBoundingClientRect();
          if (rect.width === 0 || rect.height === 0 || f.style.display === 'none' || f.style.visibility === 'hidden') {
            hasHiddenIframe = true;
          }
        });

        return {
          requestUrlExternalPct,
          anchorUnsafePct,
          linksInTagsExternalPct,
          sfhStatus,
          hasMailtoForm,
          hasHiddenIframe
        };
      }
    });
    return result?.result || {};
  } catch (err) {
    // If running on a chrome:// or restricted page
    return {};
  }
}

// Display target URL in UI
function updateUrlUI(urlStr) {
  const chip = document.getElementById('url-security-chip');
  try {
    const url = new URL(urlStr);
    document.getElementById('url-protocol').textContent = url.protocol + '//';
    document.getElementById('url-domain').textContent = url.hostname;
    document.getElementById('url-path').textContent = url.pathname.length > 20 ? url.pathname.slice(0, 20) + '...' : url.pathname;
    
    if (url.protocol === 'https:') {
      chip.textContent = 'HTTPS Encrypted';
      chip.className = 'url-security-chip secure';
    } else {
      chip.textContent = 'HTTP Insecure';
      chip.className = 'url-security-chip insecure';
    }
  } catch {
    document.getElementById('url-protocol').textContent = '';
    document.getElementById('url-domain').textContent = urlStr.slice(0, 30);
    document.getElementById('url-path').textContent = '';
    chip.textContent = 'Custom URL';
    chip.className = 'url-security-chip';
  }
}

// Render security signals
function renderSignals(features) {
  const grid = document.getElementById('signals-grid');
  const tagsContainer = document.getElementById('threat-tags-container');
  grid.innerHTML = '';
  tagsContainer.innerHTML = '';

  const list = [
    { label: 'IP Address as Host', val: features.ip_address, ok: 'Domain Name', warn: 'Raw IP Used', tag: 'Raw IP' },
    { label: 'SSL / HTTPS Security', val: features.ssl, ok: 'Secure (HTTPS)', warn: 'Insecure (HTTP)', tag: 'No HTTPS' },
    { label: 'URL Length', val: features.url_length, ok: 'Normal', warn: 'Abnormally Long', tag: 'Long URL' },
    { label: 'URL Shortener', val: features.shortening, ok: 'No', warn: 'Obfuscated Shortlink', tag: 'Shortener' },
    { label: 'Prefix/Suffix Hyphen (-)', val: features.prefix_suffix, ok: 'Clean', warn: 'Hyphen Domain Spoof', tag: 'Hyphen Spoof' },
    { label: 'Subdomain Depth', val: features.subdomains, ok: 'Normal', warn: 'Excessive Subdomains', tag: 'Multi-Subdomain' },
    { label: 'Hidden iFrames', val: features.iframe, ok: 'None', warn: '0-Pixel Hidden Frame', tag: 'Hidden iFrame' },
    { label: 'Server Form Handler', val: features.sfh, ok: 'Standard', warn: 'External Action', tag: 'External SFH' },
    { label: 'Mailto in Forms', val: features.submitting_to_email, ok: 'None', warn: 'Direct Mail Action', tag: 'Mailto Form' },
    { label: 'Domain Age & Reputation', val: features.domain_age, ok: 'Established', warn: 'Unverified / New', tag: 'Unverified Domain' },
    { label: 'Threat Intel Blacklist', val: features.statistical_report, ok: 'Clean', warn: 'Flagged Pattern', tag: 'Blacklisted Pattern' }
  ];

  list.forEach(item => {
    const row = document.createElement('div');
    row.className = 'signal-row' + (item.val === -1 ? ' threat-row' : '');

    const title = document.createElement('span');
    title.className = 'signal-title';
    title.textContent = item.label;

    const badge = document.createElement('span');
    if (item.val === 1) {
      badge.className = 'signal-badge pass';
      badge.textContent = item.ok;
    } else if (item.val === 0) {
      badge.className = 'signal-badge warn';
      badge.textContent = 'Moderate';
    } else {
      badge.className = 'signal-badge fail';
      badge.textContent = item.warn;

      // Add pill to quick summary tags
      const pill = document.createElement('span');
      pill.className = 'threat-pill';
      pill.innerHTML = `⚠️ ${item.tag}`;
      tagsContainer.appendChild(pill);
    }

    row.appendChild(title);
    row.appendChild(badge);
    grid.appendChild(row);
  });
}

// Perform complete inspection
async function performScan() {
  const body = document.getElementById('app-body');
  const btn = document.getElementById('inspect-btn');
  const btnText = document.getElementById('btn-text');
  const container = document.getElementById('result-container');
  const verdictEl = document.getElementById('result-verdict');
  const subtextEl = document.getElementById('result-subtext');
  const confContainer = document.getElementById('confidence-container');
  const confTitleEl = document.getElementById('confidence-title');
  const confPctEl = document.getElementById('confidence-pct');
  const meterFill = document.getElementById('meter-fill');
  const iconWrapper = document.getElementById('result-icon');
  const badgeStatus = document.getElementById('badge-status');

  btn.classList.add('loading');
  btnText.textContent = 'Analyzing Site...';

  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab || !tab.url) {
      throw new Error('No active web page detected.');
    }

    currentTab = tab;
    updateUrlUI(tab.url);

    // Browser internal pages check
    if (tab.url.startsWith('chrome://') || tab.url.startsWith('edge://') || tab.url.startsWith('about:')) {
      body.classList.remove('threat-active');
      container.className = 'result-container state-safe';
      badgeStatus.textContent = 'System Page';
      badgeStatus.style.background = 'rgba(6, 182, 212, 0.12)';
      badgeStatus.style.color = 'var(--accent-cyan)';
      verdictEl.textContent = 'Browser System Page';
      subtextEl.textContent = 'This is a secure internal browser system page.';
      confContainer.style.display = 'none';
      btn.classList.remove('loading');
      btnText.textContent = 'Re-analyze Page';
      return;
    }

    const [domStats] = await Promise.all([
      collectDOMStats(tab.id)
    ]);

    // Keep extracting local DOM features for the UI signals grid
    const { featureVector, features } = window.SahiHaiDetector.extractFeatures(tab.url, domStats);
    renderSignals(features);

    let isSafe = true;
    let confidence = 95;
    let needsStage2 = false;

    try {
      const formData = new URLSearchParams();
      formData.append('url', tab.url);
      
      const response = await fetch('http://127.0.0.1:8000/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: formData.toString()
      });
      
      const result = await response.json();
      
      if (result.result === 'FINAL_BENIGN') {
        isSafe = true;
      } else if (result.result === 'FINAL_MALICIOUS') {
        isSafe = false;
      } else if (result.result === 'NEEDS_STAGE2') {
        isSafe = false;
        needsStage2 = true;
      } else {
        // Fallback for errors or older backend
        isSafe = result.result === 'SAFE';
      }
      
      confidence = Math.round(result.confidence || 0);
      
    } catch (err) {
      console.error("Backend error:", err);
      // Fallback heuristic scoring if server is down
      const score = featureVector.reduce((a, b) => a + b, 0);
      isSafe = score >= 0;
      confidence = Math.min(99, Math.max(60, Math.round(50 + Math.abs(score) * 2.5)));
    }

    confContainer.style.display = 'block';
    confPctEl.textContent = `${confidence}%`;
    meterFill.style.width = `${confidence}%`;

    if (needsStage2) {
      // AMBIGUOUS / NEEDS STAGE 2 STATE (Orange/Yellow Alert Interface)
      body.classList.remove('threat-active'); // maybe add a warning class later if desired
      container.className = 'result-container state-warning';
      badgeStatus.textContent = 'DEEP SCAN REQUIRED';
      badgeStatus.style.background = 'rgba(245, 158, 11, 0.25)'; // amber
      badgeStatus.style.color = '#d97706'; // darker amber
      badgeStatus.style.borderColor = 'rgba(245, 158, 11, 0.5)';

      confTitleEl.textContent = 'Analysis Pending';
      verdictEl.textContent = 'STAGE 2 REQUIRED';
      subtextEl.textContent = 'Ambiguous signals detected. This URL requires deeper semantic analysis (Stage 2) which is currently pending.';
      meterFill.style.backgroundColor = '#f59e0b'; // amber
      iconWrapper.innerHTML = `
        <svg viewBox="0 0 24 24" width="38" height="38" fill="none" stroke="currentColor" stroke-width="2.5">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
      `;
    } else if (isSafe) {
      // SAFE STATE (Green)
      body.classList.remove('threat-active');
      container.className = 'result-container state-safe';
      badgeStatus.textContent = 'SAFE & VERIFIED';
      badgeStatus.style.background = 'rgba(16, 185, 129, 0.15)';
      badgeStatus.style.color = 'var(--safe-green)';
      badgeStatus.style.borderColor = 'rgba(16, 185, 129, 0.35)';

      confTitleEl.textContent = 'Safety Confidence';
      verdictEl.textContent = 'SAHI HAI (SAFE)';
      subtextEl.textContent = 'This website exhibits legitimate domain credentials, SSL encryption, and safe structural markers.';
      meterFill.style.backgroundColor = 'var(--safe-green)';
      iconWrapper.innerHTML = `
        <svg viewBox="0 0 24 24" width="38" height="38" fill="none" stroke="currentColor" stroke-width="2.5">
          <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
          <polyline points="9 12 11 14 15 10"/>
        </svg>
      `;
    } else {
      // DANGEROUS / PHISHING STATE (Red Alert Interface)
      body.classList.add('threat-active');
      container.className = 'result-container state-danger';
      badgeStatus.textContent = 'THREAT DETECTED';
      badgeStatus.style.background = 'rgba(239, 68, 68, 0.25)';
      badgeStatus.style.color = '#ff6b6b';
      badgeStatus.style.borderColor = 'rgba(239, 68, 68, 0.5)';

      confTitleEl.textContent = 'Threat Probability';
      verdictEl.textContent = 'DANGER: PHISHING / MALICIOUS';
      subtextEl.textContent = 'High risk detected! This website exhibits deceptive structural indicators typical of credential theft.';
      meterFill.style.backgroundColor = 'var(--danger-red)';
      iconWrapper.innerHTML = `
        <svg viewBox="0 0 24 24" width="38" height="38" fill="none" stroke="currentColor" stroke-width="2.5">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
          <line x1="12" y1="9" x2="12" y2="13"/>
          <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
      `;

      // Auto-open diagnostics so user immediately sees what went wrong
      const diagAccordion = document.getElementById('diagnostics-accordion');
      if (diagAccordion) diagAccordion.open = true;
    }
  } catch (err) {
    body.classList.remove('threat-active');
    container.className = 'result-container state-warning';
    verdictEl.textContent = 'Scan Incomplete';
    subtextEl.textContent = err.message || 'Unable to scan this page.';
    confContainer.style.display = 'none';
  } finally {
    btn.classList.remove('loading');
    btnText.textContent = 'Re-analyze Page';
  }
}

// Setup event listeners
document.addEventListener('DOMContentLoaded', async () => {
  document.getElementById('inspect-btn').addEventListener('click', performScan);

  // Auto-fetch active tab & auto-scan on open
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab && tab.url) {
    updateUrlUI(tab.url);
    await performScan();
  }
});
