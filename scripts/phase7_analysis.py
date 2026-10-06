import os
import time
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from src.semantic.minilm_encoder import MiniLMEncoder
from src.conformal.calibrator import ConformalCalibrator
from src.conformal.routing import route_prediction_set

# Suppress HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

ensure_dir('reports/figures')

def run_phase7():
    print("Loading data...")
    train_df = pd.read_parquet('data/splits/train.parquet')
    test_df = pd.read_parquet('data/splits/test.parquet')

    # Configs
    with open('config/conformal_config.json') as f:
        conformal_config = json.load(f)
    with open('config/stage1_model_config.json') as f:
        stage1_config = json.load(f)

    print("Identifying NEEDS_STAGE2 subset using ICP...")
    # Load conformal model and calibrator
    with open(conformal_config['hxrf_model_path'], 'rb') as f:
        hxrf_model = pickle.load(f)
    calibrator = ConformalCalibrator.load(conformal_config['calibrator_path'])
    
    features = stage1_config['features']
    probas = hxrf_model.predict_proba(test_df[features])
    result = calibrator.predict(probas)
    
    routes = [route_prediction_set(ps) for ps in result['prediction_sets']]
    test_df['routing'] = routes
    
    needs_stage2_mask = test_df['routing'] == 'NEEDS_STAGE2'
    test_ns2_df = test_df[needs_stage2_mask].copy()
    print(f"Total test samples: {len(test_df)}")
    print(f"NEEDS_STAGE2 test samples: {len(test_ns2_df)}")
    
    # Semantic Encoder Init
    encoder = MiniLMEncoder(device='auto')
    dim = encoder.get_embedding_dimension()
    max_seq_len = encoder.max_seq_length
    device = encoder.get_device()
    print(f"Encoder initialized: dim={dim}, max_seq={max_seq_len}, device={device}")

    # ==========================
    # 7 & 8: Tokenization Analysis
    # ==========================
    print("Tokenization Analysis...")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    # Sample 100,000 for tokenization analysis to be fast
    sample_urls = test_df['url'].sample(min(100000, len(test_df)), random_state=42).tolist()
    token_lengths = [len(tokenizer.encode(u, add_special_tokens=True)) for u in sample_urls]
    
    max_tok = max(token_lengths)
    avg_tok = np.mean(token_lengths)
    med_tok = np.median(token_lengths)
    p95_tok = np.percentile(token_lengths, 95)
    pct_trunc = np.mean(np.array(token_lengths) > max_seq_len) * 100

    plt.figure(figsize=(10, 6))
    plt.hist(token_lengths, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(max_seq_len, color='red', linestyle='--', label=f'Max Seq Len ({max_seq_len})')
    plt.title("MiniLM Token Length Distribution for URLs")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig('reports/figures/minilm_token_distribution.png')
    plt.close()

    with open('reports/minilm_tokenization.md', 'w') as f:
        f.write("# MiniLM Tokenization Analysis\n\n")
        f.write("## Token Length Statistics\n")
        f.write(f"- **Max Configured Token Length:** {max_seq_len}\n")
        f.write(f"- **Average Token Count:** {avg_tok:.2f}\n")
        f.write(f"- **Median Token Count:** {med_tok:.2f}\n")
        f.write(f"- **P95 Token Count:** {p95_tok:.2f}\n")
        f.write(f"- **Maximum Observed Token Count:** {max_tok}\n")
        f.write(f"- **Percentage Truncated:** {pct_trunc:.2f}%\n\n")
        f.write("![Token Distribution](figures/minilm_token_distribution.png)\n")

    # ==========================
    # 14 & 15: Latency & Throughput Benchmark
    # ==========================
    print("Benchmarking latency and throughput...")
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    latency_stats = []
    
    # Warmup
    _ = encoder.encode_urls(sample_urls[:128], batch_size=32)
    
    for bs in batch_sizes:
        n_samples = min(2000, len(sample_urls)) if bs > 1 else 100
        subset = sample_urls[:n_samples]
        
        times = []
        # Batch inference measurement
        if bs == 1:
            for url in subset:
                t0 = time.perf_counter()
                _ = encoder.encode_url(url)
                times.append((time.perf_counter() - t0) * 1000) # ms
            total_time = sum(times) / 1000
        else:
            t0 = time.perf_counter()
            for i in range(0, len(subset), bs):
                batch = subset[i:i+bs]
                _ = encoder.encode_urls(batch, batch_size=bs)
            total_time = time.perf_counter() - t0
            # For latency per batch:
            # We want to measure the time taken to process ONE batch
            times = []
            for i in range(0, min(200, len(subset)), bs):
                batch = subset[i:i+bs]
                b_t0 = time.perf_counter()
                _ = encoder.encode_urls(batch, batch_size=bs)
                times.append((time.perf_counter() - b_t0) * 1000) # ms
            
        throughput = n_samples / total_time if bs == 1 else len(subset) / total_time
        
        latency_stats.append({
            'batch_size': bs,
            'mean_ms': np.mean(times),
            'median_ms': np.median(times),
            'p95_ms': np.percentile(times, 95),
            'p99_ms': np.percentile(times, 99),
            'throughput_urls_per_sec': throughput
        })

    # Plot Latency vs Batch
    plt.figure(figsize=(10, 6))
    plt.plot([s['batch_size'] for s in latency_stats], [s['mean_ms'] for s in latency_stats], marker='o')
    plt.title("MiniLM Mean Latency per Batch vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Mean Latency (ms)")
    plt.grid(True)
    plt.savefig('reports/figures/minilm_latency_vs_batch.png')
    plt.close()

    # Plot Throughput vs Batch
    plt.figure(figsize=(10, 6))
    plt.plot([s['batch_size'] for s in latency_stats], [s['throughput_urls_per_sec'] for s in latency_stats], marker='s', color='green')
    plt.title("MiniLM Throughput vs Batch Size")
    plt.xlabel("Batch Size")
    plt.ylabel("Throughput (URLs/sec)")
    plt.grid(True)
    plt.savefig('reports/figures/minilm_throughput_vs_batch.png')
    plt.close()

    # ==========================
    # 11 & 12: Baseline Evaluation
    # ==========================
    print("Training Semantic Baseline (Logistic Regression)...")
    
    # Storage analysis
    # For 632k dataset: 632844 * 384 * 4 bytes = ~972 MB
    # We will sample 50,000 for training the LR baseline to save time
    train_sample = train_df.sample(min(50000, len(train_df)), random_state=42)
    train_urls = train_sample['url'].tolist()
    train_labels = train_sample['dsse_binary_target'].values
    
    train_embs = encoder.encode_urls(train_urls, batch_size=32)
    
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(train_embs, train_labels)

    print("Evaluating on Full Test Set...")
    test_urls = test_df['url'].tolist()
    test_labels = test_df['dsse_binary_target'].values
    test_embs = encoder.encode_urls(test_urls, batch_size=128) # Larger batch size for bulk inference
    
    test_preds = lr.predict(test_embs)
    test_probs = lr.predict_proba(test_embs)[:, 1]
    
    def get_metrics(y_true, y_pred, y_prob):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'specificity': tn / (tn + fp),
            'fpr': fp / (fp + tn),
            'fnr': fn / (fn + tp),
            'cm': cm
        }
        
    full_metrics = get_metrics(test_labels, test_preds, test_probs)
    
    print("Evaluating on NEEDS_STAGE2 Subset...")
    ns2_urls = test_ns2_df['url'].tolist()
    ns2_labels = test_ns2_df['dsse_binary_target'].values
    ns2_embs = encoder.encode_urls(ns2_urls, batch_size=128)
    
    ns2_preds = lr.predict(ns2_embs)
    ns2_probs = lr.predict_proba(ns2_embs)[:, 1]
    
    ns2_metrics = get_metrics(ns2_labels, ns2_preds, ns2_probs)

    # Plot Confusion Matrix for Baseline (Full Test)
    import seaborn as sns
    plt.figure(figsize=(6, 5))
    sns.heatmap(full_metrics['cm'], annot=True, fmt='d', cmap='Blues')
    plt.title("MiniLM Baseline Confusion Matrix (Full Test)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('reports/figures/minilm_confusion_matrix.png')
    plt.close()

    # ==========================
    # Write Results Report
    # ==========================
    with open('reports/minilm_results.md', 'w') as f:
        f.write("# Stage-2 Semantic Representation — MiniLM Baseline\n\n")
        f.write("## 1. Motivation\n")
        f.write("Introduce a deep semantic encoder to process raw URLs for the NEEDS_STAGE2 subset to catch sophisticated malicious patterns missed by structural analysis.\n\n")
        f.write("## 2. Model & Settings\n")
        f.write(f"- Model: `sentence-transformers/all-MiniLM-L6-v2`\n")
        f.write(f"- Embedding Dimension: {dim}\n")
        f.write(f"- Max Sequence Length: {max_seq_len}\n")
        f.write(f"- Device: {device}\n")
        f.write(f"- Normalization: {encoder.normalize_embeddings}\n\n")
        
        f.write("## 3. Dataset Storage Analysis\n")
        f.write("- Full 632k dataset ~972 MB (uncompressed float32)\n")
        f.write(f"- Test set (126k) ~194 MB\n")
        f.write(f"- NEEDS_STAGE2 subset ({len(ns2_urls)}) ~{len(ns2_urls)*384*4/1024/1024:.2f} MB\n\n")
        
        f.write("## 4. Latency & Throughput Benchmark\n")
        f.write("| Batch Size | Mean (ms/batch) | Median (ms/batch) | P95 (ms/batch) | P99 (ms/batch) | Throughput (URLs/sec) |\n")
        f.write("|------------|-----------------|-------------------|----------------|----------------|-----------------------|\n")
        for s in latency_stats:
            f.write(f"| {s['batch_size']} | {s['mean_ms']:.2f} | {s['median_ms']:.2f} | {s['p95_ms']:.2f} | {s['p99_ms']:.2f} | {s['throughput_urls_per_sec']:.2f} |\n")
        f.write("\n![Latency vs Batch](figures/minilm_latency_vs_batch.png)\n")
        f.write("![Throughput vs Batch](figures/minilm_throughput_vs_batch.png)\n\n")

        f.write("## 5. Classification Metrics\n")
        f.write("Baseline model: Logistic Regression trained on 50,000 URLs.\n\n")
        f.write("| Metric | Full Test Set | NEEDS_STAGE2 Subset |\n")
        f.write("|--------|---------------|---------------------|\n")
        f.write(f"| Accuracy | {full_metrics['accuracy']:.4f} | {ns2_metrics['accuracy']:.4f} |\n")
        f.write(f"| ROC-AUC | {full_metrics['roc_auc']:.4f} | {ns2_metrics['roc_auc']:.4f} |\n")
        f.write(f"| Precision | {full_metrics['precision']:.4f} | {ns2_metrics['precision']:.4f} |\n")
        f.write(f"| Recall | {full_metrics['recall']:.4f} | {ns2_metrics['recall']:.4f} |\n")
        f.write(f"| F1 Score | {full_metrics['f1']:.4f} | {ns2_metrics['f1']:.4f} |\n")
        f.write(f"| Specificity | {full_metrics['specificity']:.4f} | {ns2_metrics['specificity']:.4f} |\n")
        f.write(f"| FPR | {full_metrics['fpr']:.4f} | {ns2_metrics['fpr']:.4f} |\n")
        f.write(f"| FNR | {full_metrics['fnr']:.4f} | {ns2_metrics['fnr']:.4f} |\n\n")
        
        f.write("![Confusion Matrix](figures/minilm_confusion_matrix.png)\n\n")
        
        f.write("## 6. Tokenization Analysis\n")
        f.write(f"Avg tokens: {avg_tok:.2f}, Truncated: {pct_trunc:.2f}%\n\n")

    # ==========================
    # 19: Exploratory Similarity
    # ==========================
    from sklearn.metrics.pairwise import cosine_similarity
    print("Semantic Similarity Analysis...")
    urls_to_test = [
        "https://example.com/login",
        "http://example.com/login.php",
        "https://secure-login.example.com",
        "http://phish.com/verify",
        "https://google.com",
        "https://googIe.com"
    ]
    embs_test = encoder.encode_urls(urls_to_test, batch_size=8)
    sim_matrix = cosine_similarity(embs_test)
    
    with open('reports/minilm_embedding_analysis.md', 'w') as f:
        f.write("# Exploratory Semantic Similarity Analysis\n\n")
        f.write("Cosine similarities between sample URLs:\n\n")
        f.write("| URL 1 | URL 2 | Cosine Similarity |\n")
        f.write("|-------|-------|-------------------|\n")
        for i in range(len(urls_to_test)):
            for j in range(i+1, len(urls_to_test)):
                f.write(f"| `{urls_to_test[i]}` | `{urls_to_test[j]}` | {sim_matrix[i,j]:.4f} |\n")
                
    print("Phase 7 Analysis Script Completed!")

if __name__ == "__main__":
    run_phase7()
