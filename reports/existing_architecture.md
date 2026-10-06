# Audit of Existing Architecture

Based on a thorough inspection of the project files, here is the breakdown of the existing system before upgrading to the Dual-Stage Semantic Ensemble (DSSE).

## A. Current Architecture
The current project comprises three disjointed components:
1. **Model Training & Testing Scripts (Python)**: Scripts for training MLP and Random Forest classifiers on a legacy `.arff` dataset. Test scripts can perform live extraction of features from a given URL and infer using a `.pkl` model.
2. **Local REST Backend (`server.py`)**: A simple Python backend that handles POST requests to evaluate URLs. However, it completely bypasses the ML models and uses basic regex matching to classify URLs.
3. **Browser Extension (Manifest V3)**: The primary frontend. It extracts features client-side via content scripts and runs a custom, hardcoded neural network forward pass in JavaScript (using `model_weights.json`) instead of querying the backend API.

## B. Current ML Models
- **MLP Classifier**: Uses Scikit-learn's `MLPClassifier` with hidden layer sizes `(100, 50, 30, 10, 5)`, using `relu` activation and `adam` solver. Saved as `models/mlp_model.pkl`.
- **Random Forest**: Uses Scikit-learn's `RandomForestClassifier`. Saved as `models/random_forest.pkl`.
- **JS Model**: Extracted weights (`extension/model_weights.json`) are used in a custom JS implementation (`extension/detector.js`) to mimic the MLP.

## C. Current Features
The system uses 22 manually engineered features:
- **URL-based**: Length, IP address presence, Shortening service, `@` symbol, Double slash redirect, Prefix/Suffix hyphen, Subdomain count, HTTP/HTTPS token in domain, Abnormal URL.
- **DOM/Page-based**: SSL final state, Favicon source, Request URL (external media ratio), URL of Anchor (unsafe anchor ratio), Links in `<script>` and `<link>` tags, Server Form Handler (SFH), Submitting to email (`mailto:`), IFrame presence.
- **External/Third-party**: Domain registration length, Age of domain (WHOIS), DNS record validity, Web traffic rank (Alexa), Google Index status, Statistical reports (Blacklist/regex check).

## D. Current Target/Labels
- Labels distinguish between `SAFE` (1) and `PHISHING` (-1 / 0 depending on the specific script).
- Training scripts label-encode the target variable using `LabelEncoder`.
- The JS neural network uses a sigmoid output, where a probability >= 0.5 classifies the site as SAFE.

## E. Current Dataset
- **Active Dataset**: Training currently relies on a legacy file `train/data/web_data.arff`.
- **Dormant Datasets**: The `dataset/` directory contains much larger and modern CSV files (e.g., `MUD_malicious_urls_2026_V2.csv`, `dataset_with_all_features v2.csv`) which appear to be intended for the upgrade but are not currently utilized by the training scripts.

## F. Training Pipeline
1. Reads `web_data.arff`.
2. Hardcodes the selection of 22 specific feature columns by index: `[0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17, 22, 23, 24, 25, 27, 29]`.
3. Converts data to `float64` and applies `LabelEncoder` to the target.
4. Performs a random train/test split.
5. Fits the model and dumps the artifact via `joblib`.

## G. Inference Pipeline (Python)
- The script `test/test.py` takes a URL argument and passes it to `features_extraction.py`.
- Features are extracted dynamically (involving live WHOIS queries, BS4 scraping, and DNS checks).
- The array of 22 features is fed to the loaded `.pkl` model to generate a prediction.

## H. Backend/API Flow
- `server.py` listens on port 8000.
- It accepts POST requests with a URL parameter.
- **Critical Flaw**: It completely ignores the trained ML models and feature extractors. Instead, it classifies a URL as `PHISHING` if it matches a simplistic regex (`\.tk|\.ml|\.ga|\.cf|\.gq|login.*\.php|verify.*account`).

## I. Frontend/Extension Flow
- The user clicks the extension popup (`popup.html`).
- `popup.js` executes a script in the active tab to extract DOM-based features (e.g., external media ratio, hidden iframes).
- It passes these to `detector.js`, which calculates URL-based features and creates the 22-feature vector.
- Inference is performed directly in the browser via `runNeuralNetwork()`, bypassing `server.py`.
- Displays the confidence score and safety status to the user.

## J. Files that will need modification for DSSE
- **Training Scripts**: `train/train_mlp.py` (and potentially `train_rf.py`) must be rewritten to load the new CSV datasets and utilize the new DSSE architecture instead of the legacy ARFF pipeline.
- **Feature Extractors**: `test/features_extraction.py` and `extension/detector.js` will need major updates to support Dual-Stage Semantic features (which likely involve semantic embeddings or LLM-based text analysis, not just structural heuristics).
- **Inference Server**: `server.py` must be completely rewritten to properly load the DSSE models and serve predictions, rather than relying on regex.
- **Extension Logic**: `extension/popup.js` might need updates if the feature extraction or inference logic is pushed back to the backend due to the complexity of DSSE.

## K. Files that should remain unchanged
- **Extension UI**: `extension/popup.html`, `extension/style.css`, and `extension/icons/` can remain intact as long as the data structure passed to the UI is maintained.
- **Raw Datasets**: The CSV files in `dataset/` should be used as-is for training.

## L. Potential Data Leakage Issues
- Randomly splitting the dataset using `train_test_split` can cause leakage if multiple URLs from the same domain share near-identical DOM features, artificially inflating test accuracy.
- Training on static ARFF data but inferring with live HTTP/WHOIS requests (which change over time) can lead to a significant train-serve skew.

## M. Potential Technical Problems
- **Fragmented Truth**: There are three disparate logic paths: the Python test scripts (using `.pkl`), the API (using regex), and the extension (using custom JS NN). None of them are strictly aligned.
- **Hardcoded Column Indices**: Extracting features by hardcoded index numbers (`[0, 1, 2...]`) in `train_mlp.py` is brittle and will break immediately when the dataset schema changes.
- **JS Model Limitations**: A complex DSSE architecture likely cannot be easily exported to simple JSON weights for client-side JavaScript execution. The extension will almost certainly need to pivot to using the backend API.
