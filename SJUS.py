# -*- coding: utf-8 -*-
"""
Budgeted Union (C, U) Key-Sample Selection
-------------------------------------------
This script performs key-sample selection by:
1. Building a Minimum Spanning Tree (MST).
2. Performing Spectral Clustering on the MST.
3. Calculating Centrality (phi) and Boundary Uncertainty (psi).
4. Budget-Aware Selection (Priority: Intersection -> Union -> Alternating Supplementary).
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import cdist

import networkx as nx
import matplotlib.pyplot as plt

# =================================================================
# [MODIFIABLE] USER CONFIGURATION
# =================================================================
INPUT_DATA_PATH = "data/your_dataset.csv" 
COLUMN_LABEL = "Class"
COLUMN_ID = "Sample"
OUTPUT_DIR = "results"

# Algorithm Hyperparameters
PERCENTILE = 95           # Threshold for candidates (Eq. 1)
N_CLUSTERS = 2            # K
MIN_CLUSTER_SIZE = 20     # For small-cluster relaxation
RANDOM_STATE = 42

# Budget Options
USE_BUDGET = True         # Enable Budget-Aware Selection
BUDGET_RATE = 0.10        # b: 10% labeling budget
# =================================================================

def setup_environment():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    np.random.seed(RANDOM_STATE)

def load_data(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at {path}.")
    
    df = pd.read_csv(path)
    sample_ids = df[COLUMN_ID] if COLUMN_ID in df.columns else pd.Series(np.arange(len(df)))
    features = [c for c in df.columns if c not in [COLUMN_ID, COLUMN_LABEL]]
    X = df[features].values
    y = df[COLUMN_LABEL].values
    X_scaled = StandardScaler().fit_transform(X)
    return df, X_scaled, y, sample_ids

def analyze_mst_selection(dist_matrix: np.ndarray, dist_name: str, y, sample_ids):
    n = dist_matrix.shape[0]
    # Calculate Budget B = ceil(b * n)
    B = int(np.ceil(BUDGET_RATE * n)) if USE_BUDGET else n

    # --- Step 1: Structural Learning ---
    mst = minimum_spanning_tree(dist_matrix)
    adj = ((mst.toarray() + mst.toarray().T) > 0).astype(float)
    sc = SpectralClustering(n_clusters=N_CLUSTERS, affinity='precomputed', random_state=RANDOM_STATE)
    labels = sc.fit_predict(adj)

    # --- Step 2: Scoring (SJUS) ---
    G = nx.Graph(adj)
    degrees = np.array([d for _, d in G.degree()], dtype=float)
    # phi: structural score (Eq. phi)
    phi = (degrees - degrees.min()) / (degrees.max() - degrees.min() + 1e-12)

    Z = sc.embedding_ if hasattr(sc, 'embedding_') else adj
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10).fit(Z)
    dists = cdist(Z, km.cluster_centers_)
    d_sorted = np.sort(dists, axis=1)
    # psi: uncertainty score (Eq. psi)
    psi = np.clip(1.0 - (d_sorted[:, 1] - d_sorted[:, 0]) / (d_sorted[:, 0] + d_sorted[:, 1] + 1e-12), 0, 1)

    # --- Step 3: Budget-Aware Selection ---
    omega_cap_all = [] # Priority 1: Joint set
    omega_cup_rest = [] # Priority 2: Union set (excluding joint)

    # Thresholding per cluster
    for k in range(N_CLUSTERS):
        idx = np.where(labels == k)[0]
        if len(idx) == 0: continue
        
        pct = PERCENTILE if len(idx) >= MIN_CLUSTER_SIZE else 85
        theta_phi = np.percentile(phi[idx], pct)
        theta_psi = np.percentile(psi[idx], pct)
        
        omega_phi = idx[phi[idx] >= theta_phi]
        omega_psi = idx[psi[idx] >= theta_psi]
        
        cap_k = np.intersect1d(omega_phi, omega_psi)
        cup_k = np.union1d(omega_phi, omega_psi)
        rest_k = np.setdiff1d(cup_k, cap_k)
        
        omega_cap_all.extend(cap_k.tolist())
        omega_cup_rest.extend(rest_k.tolist())

    # Final selection pool Omega
    omega = []

    # Phase 1: Core Selection (Joint set)
    omega.extend(omega_cap_all)

    # Phase 2: Union Completion (If budget remains)
    if len(omega) < B:
        needed = B - len(omega)
        # Sort rest of union by combined score to prioritize
        omega_cup_rest = sorted(omega_cup_rest, key=lambda i: phi[i] + psi[i], reverse=True)
        omega.extend(omega_cup_rest[:needed])

    # Phase 3: Supplementary Selection (If budget still remains)
    if len(omega) < B:
        exclude_set = set(omega)
        remaining_indices = [i for i in range(n) if i not in exclude_set]
        
        # Sort remaining by phi and psi
        phi_rank = sorted(remaining_indices, key=lambda i: phi[i], reverse=True)
        psi_rank = sorted(remaining_indices, key=lambda i: psi[i], reverse=True)
        
        p_ptr, s_ptr = 0, 0
        while len(omega) < B and (p_ptr < len(phi_rank) or s_ptr < len(psi_rank)):
            # Pick from phi
            while p_ptr < len(phi_rank) and phi_rank[p_ptr] in omega:
                p_ptr += 1
            if p_ptr < len(phi_rank) and len(omega) < B:
                omega.append(phi_rank[p_ptr])
            
            # Pick from psi
            while s_ptr < len(psi_rank) and psi_rank[s_ptr] in omega:
                s_ptr += 1
            if s_ptr < len(psi_rank) and len(omega) < B:
                omega.append(psi_rank[s_ptr])

    # Phase 4: Truncation (If Priority 1 exceeds budget)
    if len(omega) > B:
        # Prioritize samples within the joint set by their average score
        omega = sorted(omega[:len(omega)], key=lambda i: phi[i] + psi[i], reverse=True)[:B]

    # Convert to flags
    final_selection = np.zeros(n, dtype=int)
    final_selection[omega] = 1

    # --- Save Results ---
    res_df = pd.DataFrame({
        'ID': sample_ids, 'Cluster': labels, 
        'phi_score': phi, 'psi_score': psi,
        'IsSelected': final_selection, 'Label': y
    })
    res_df.to_csv(os.path.join(OUTPUT_DIR, f"selection_{dist_name}.csv"), index=False)
    
    # Visualization (only if PLOT_MST is True)
    visualize_mst(G, final_selection, labels, dist_name)

    return final_selection.sum()

def visualize_mst(G, selected, clusters, name):
    plt.figure(figsize=(10, 8))
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx_edges(G, pos, alpha=0.2, edge_color='gray')
    
    # Background nodes
    nx.draw_networkx_nodes(G, pos, node_size=40, node_color=clusters, 
                           cmap=plt.cm.Paired, alpha=0.6)
    
    # Selected nodes
    key_nodes = [i for i, val in enumerate(selected) if val == 1]
    nx.draw_networkx_nodes(G, pos, nodelist=key_nodes, node_size=100, 
                           node_color='red', node_shape='*', label="Key Samples (Omega)")
    
    plt.title(f"Budget-Aware Selection - {name} (B={selected.sum()})")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"plot_{name}.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    print("Starting Key Sample Selection Process (Budget-Aware)...")
    setup_environment()
    
    try:
        raw_df, X, y, ids = load_data(INPUT_DATA_PATH)
        
        distances = {
            'Euclidean': euclidean_distances(X),
            'Cosine': cosine_distances(X)
        }

        for name, mat in distances.items():
            count = analyze_mst_selection(mat, name, y, ids)
            print(f"Finished {name}: Selected {count} samples.")

        print(f"Done! Check the '{OUTPUT_DIR}' folder.")
    except Exception as e:
        print(f"Error: {e}")