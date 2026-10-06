import matplotlib.pyplot as plt
import networkx as nx
import os

def create_fig1():
    G = nx.DiGraph()
    G.add_edges_from([
        ("Raw URL", "12 Structural Features"),
        ("12 Structural Features", "HXRF"),
        ("HXRF", "Conformal Router"),
        ("Conformal Router", "Final Benign\n(BENIGN)"),
        ("Conformal Router", "Final Malicious\n(MALICIOUS)"),
        ("Conformal Router", "Stage 2\n(Uncertain / EMPTY_SET)"),
        ("Stage 2\n(Uncertain / EMPTY_SET)", "MiniLM"),
        ("MiniLM", "FiLM"),
        ("FiLM", "XGBoost"),
        ("XGBoost", "Final Prediction")
    ])
    
    pos = {
        "Raw URL": (0, 8),
        "12 Structural Features": (0, 7),
        "HXRF": (0, 6),
        "Conformal Router": (0, 5),
        "Final Benign\n(BENIGN)": (-1.5, 4),
        "Final Malicious\n(MALICIOUS)": (0, 4),
        "Stage 2\n(Uncertain / EMPTY_SET)": (1.5, 4),
        "MiniLM": (1.5, 3),
        "FiLM": (1.5, 2),
        "XGBoost": (1.5, 1),
        "Final Prediction": (1.5, 0)
    }
    
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=10, font_size=10, 
            font_weight='bold', edge_color='gray', arrows=True, arrowsize=20, node_shape='s', 
            bbox=dict(facecolor='lightblue', edgecolor='black', boxstyle='round,pad=0.3'))
    
    plt.title("Figure 1: Overall DSSE Architecture", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('paper/figures/fig1_dsse_architecture.png', dpi=300)
    plt.savefig('paper/figures/fig1_dsse_architecture.pdf')
    plt.close()

def create_fig2():
    features = [
        "domain_len", "path_depth", "special_count_v2", "token_count", 
        "url_entropy", "url_len", "=", ".", "//", "abnormal_url", 
        "phish_adv_hyphen_count", "suspicious_extension"
    ]
    plt.figure(figsize=(8, 5))
    plt.axis('off')
    
    plt.text(0.5, 0.95, "Stage-1 12-Feature Gatekeeper", fontsize=14, fontweight='bold', ha='center')
    
    for i, f in enumerate(features):
        x = 0.2 if i < 6 else 0.8
        y = 0.8 - (i % 6) * 0.12
        plt.text(x, y, f"{i+1}. {f}", fontsize=12, bbox=dict(facecolor='lightcyan', edgecolor='black', boxstyle='round,pad=0.4'), ha='center')
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig2_stage1_features.png', dpi=300)
    plt.savefig('paper/figures/fig2_stage1_features.pdf')
    plt.close()

def create_fig3():
    G = nx.DiGraph()
    G.add_edges_from([
        ("HXRF probabilities", "Conformal calibration"),
        ("Conformal calibration", "Prediction set"),
        ("Prediction set", "Routing decision"),
        ("Routing decision", "{BENIGN}"),
        ("Routing decision", "{MALICIOUS}"),
        ("Routing decision", "{BENIGN, MALICIOUS}"),
        ("Routing decision", "EMPTY_SET"),
        ("{BENIGN}", "final benign"),
        ("{MALICIOUS}", "final malicious"),
        ("{BENIGN, MALICIOUS}", "Stage 2 (Escalation)"),
        ("EMPTY_SET", "Stage 2 (Escalation)")
    ])
    
    pos = {
        "HXRF probabilities": (0, 7),
        "Conformal calibration": (0, 6),
        "Prediction set": (0, 5),
        "Routing decision": (0, 4),
        "{BENIGN}": (-1.5, 3),
        "{MALICIOUS}": (-0.5, 3),
        "{BENIGN, MALICIOUS}": (0.5, 3),
        "EMPTY_SET": (1.5, 3),
        "final benign": (-1.5, 2),
        "final malicious": (-0.5, 2),
        "Stage 2 (Escalation)": (1.0, 2)
    }
    
    plt.figure(figsize=(10, 7))
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=10, font_size=9, 
            font_weight='bold', edge_color='gray', arrows=True, arrowsize=20, node_shape='s', 
            bbox=dict(facecolor='lightgreen', edgecolor='black', boxstyle='round,pad=0.3'))
    
    plt.title("Figure 3: Conformal Selective Routing", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('paper/figures/fig3_conformal_routing.png', dpi=300)
    plt.savefig('paper/figures/fig3_conformal_routing.pdf')
    plt.close()

def create_fig4():
    G = nx.DiGraph()
    G.add_edges_from([
        ("Raw URL", "MiniLM"),
        ("MiniLM", "384-D semantic representation\n(E_MiniLM)"),
        ("12 structural features", "FiLM conditioning network"),
        ("FiLM conditioning network", "gamma and beta"),
        ("384-D semantic representation\n(E_MiniLM)", "E_FiLM = gamma(X12) ⊙ E_MiniLM + beta(X12)"),
        ("gamma and beta", "E_FiLM = gamma(X12) ⊙ E_MiniLM + beta(X12)"),
        ("E_FiLM = gamma(X12) ⊙ E_MiniLM + beta(X12)", "XGBoost"),
        ("XGBoost", "Prediction")
    ])
    
    pos = {
        "Raw URL": (-1, 5),
        "MiniLM": (-1, 4),
        "384-D semantic representation\n(E_MiniLM)": (-1, 3),
        "12 structural features": (1, 5),
        "FiLM conditioning network": (1, 4),
        "gamma and beta": (1, 3),
        "E_FiLM = gamma(X12) ⊙ E_MiniLM + beta(X12)": (0, 2),
        "XGBoost": (0, 1),
        "Prediction": (0, 0)
    }
    
    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightcoral', node_size=10, font_size=10, 
            font_weight='bold', edge_color='gray', arrows=True, arrowsize=20, node_shape='s', 
            bbox=dict(facecolor='lightcoral', edgecolor='black', boxstyle='round,pad=0.3'))
    
    plt.title("Figure 4: Stage-2 Semantic Fusion Architecture\n(structural-conditioned modulation of the semantic representation)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('paper/figures/fig4_stage2_film.png', dpi=300)
    plt.savefig('paper/figures/fig4_stage2_film.pdf')
    plt.close()

def create_fig5():
    labels = ['Stage 1 Resolution', 'Stage 2 Escalation']
    sizes = [96.69, 3.31]
    colors = ['#ff9999','#66b3ff']
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%', startangle=90, textprops={'fontsize': 14, 'weight': 'bold'})
    ax.axis('equal')  
    plt.title("Figure 5: Input Routing Distribution", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('paper/figures/fig5_routing_distribution.png', dpi=300)
    plt.savefig('paper/figures/fig5_routing_distribution.pdf')
    plt.close()

def create_fig6():
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
    values = [93.89, 94.47, 86.73, 90.43, 97.67]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(metrics, values, color=['#c2c2f0','#ffb3e6', '#c2f0c2', '#ffcc99', '#99ff99'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Figure 6: Final Performance Summary (Phase-12 DSSE)', fontsize=14, fontweight='bold')
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('paper/figures/fig6_final_performance.png', dpi=300)
    plt.savefig('paper/figures/fig6_final_performance.pdf')
    plt.close()

if __name__ == '__main__':
    create_fig1()
    create_fig2()
    create_fig3()
    create_fig4()
    create_fig5()
    create_fig6()
