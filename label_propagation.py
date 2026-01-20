# -*- coding: utf-8 -*-
"""
Measure Propagation for Semi-Supervised Learning
------------------------------------------------
This script implements a Label Propagation algorithm on KNN graphs.
User can place their datasets in the 'datasets' folder and modify settings.
"""

import os
import time
import numpy as np
import pandas as pd
from queue import PriorityQueue
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm

# =================================================================
# [MODIFIABLE] USER CONFIGURATION
# =================================================================
# 1. Folder Setup
DATA_FOLDER = "datasets"       # Place your .xlsx and .csv files here
OUTPUT_FOLDER = "results"      # Results will be saved here

# 2. File Selection (Can be set to None for auto-detection)
# If None, the script picks the first .csv file found in DATA_FOLDER
GROUND_TRUTH_FILENAME = None   

# 3. Algorithm Parameters
K_VALUES = [4, 6]              # Number of neighbors for KNN
PARAM_GRID = {
    'mu': [1e-4, 0.01, 1, 100],
    'nu': [1e-6, 1e-4, 0.01],
    'max_iter': [1000],
    'tol': [1e-1]
}
# =================================================================

class KNNGraph(dict):
    """Universal KNN Graph builder supporting Multiple Distance Metrics."""
    def __init__(self, K, similarity_type="Euclidean"):
        super().__init__()
        self.K = K
        self.similarity_type = similarity_type

    def _get_sim(self, x, y):
        if self.similarity_type == "Euclidean":
            return 1.0 / (1.0 + np.linalg.norm(x - y))
        elif self.similarity_type == "Manhattan":
            return 1.0 / (1.0 + np.sum(np.abs(x - y)))
        elif self.similarity_type == "Cosine":
            return 1.0 / (1.0 + cosine_distances([x], [y])[0][0])
        return 1.0 / (1.0 + np.linalg.norm(x - y))

    def build(self, X):
        n = len(X)
        for i in range(n):
            pq = PriorityQueue(maxsize=self.K)
            for j in range(n):
                if i == j: continue
                sim = self._get_sim(X[i], X[j])
                if not pq.full():
                    pq.put((sim, j))
                else:
                    lowest_sim, _ = pq.get()
                    pq.put((max(sim, lowest_sim), j if sim > lowest_sim else _))
            
            self[i] = {'neighbours': [], 'weights': []}
            while not pq.empty():
                s, target = pq.get()
                self[i]['neighbours'].append(target)
                self[i]['weights'].append([s])
        return self

class MeasurePropagation:
    """Label Propagation Core Logic."""
    def __init__(self, mu=0.1, nu=0.01, tol=1e-1, max_iter=100):
        self.mu, self.nu, self.tol, self.max_iter = mu, nu, tol, max_iter
        self.SMALL = 1e-10

    def fit_predict(self, graph, y_mask):
        n_nodes = len(y_mask)
        labeled_idx = np.where(y_mask != -1)[0]
        n_classes = len(np.unique(y_mask[labeled_idx]))
        
        r = {i: np.eye(n_classes)[int(y_mask[i])] for i in labeled_idx}
        p = np.full((n_nodes, n_classes), 1.0 / n_classes)
        q = np.full((n_nodes, n_classes), 1.0 / n_classes)

        for _ in range(self.max_iter):
            q_old = q.copy()
            for i in range(n_nodes):
                nb, wt = graph[i]['neighbours'], np.array(graph[i]['weights'])
                gamma = self.nu + self.mu * wt.sum()
                p[i] = np.exp((np.log(q[nb] + self.SMALL) * wt).sum(axis=0) * (self.mu / gamma))
                p[i] /= (p[i].sum() + self.SMALL)
            
            for i in range(n_nodes):
                nb, wt = graph[i]['neighbours'], np.array(graph[i]['weights'])
                div_r = self.mu * (wt * p[nb]).sum(axis=0) + (r[i] if i in r else 0)
                div_l = self.mu * wt.sum() + (1 if i in r else 0)
                q[i] = div_r / (div_l + self.SMALL)
            
            if np.linalg.norm(q - q_old) < self.tol: break
        return np.argmax(q, axis=1)

def compute_ccer(graph, labels):
    """Calculates Cluster-Contrastive Edge Ratio."""
    cross, total, seen = 0, 0, set()
    for u in graph.keys():
        for v in graph[u]['neighbours']:
            pair = tuple(sorted((u, v)))
            if pair not in seen:
                seen.add(pair); total += 1
                if labels[u] != labels[v]: cross += 1
    return cross / total if total > 0 else 0

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # --- Universal Ground Truth Detection ---
    if GROUND_TRUTH_FILENAME:
        gt_path = os.path.join(DATA_FOLDER, GROUND_TRUTH_FILENAME)
    else:
        # Automatically find the first .csv file in the folder
        csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')]
        if not csv_files:
            print(f"Error: No Ground Truth (.csv) found in '{DATA_FOLDER}' folder.")
            return
        gt_path = os.path.join(DATA_FOLDER, csv_files[0])
    
    print(f"[*] Using Ground Truth: {os.path.basename(gt_path)}")
    y_true = pd.read_csv(gt_path)['Class'].values
    
    # Process all Excel files (Semi-labeled data)
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.xlsx')]
    
    for filename in files:
        print(f"\n>>> EXPERIMENT: {filename}")
        df = pd.read_excel(os.path.join(DATA_FOLDER, filename))
        X = StandardScaler().fit_transform(df.iloc[:, 1:-1].values)
        y_mask = df['Class'].values
        
        # Detection of Similarity type from filename
        sim_type = "Euclidean"
        for name in ["Manhattan", "Cosine"]:
            if name in filename: sim_type = name

        for K in K_VALUES:
            print(f"[*] Building {sim_type} Graph (K={K})...")
            graph = KNNGraph(K, sim_type).build(X)
            results = []

            for params in tqdm(ParameterGrid(PARAM_GRID), desc="Grid Search"):
                preds = MeasurePropagation(**params).fit_predict(graph, y_mask)
                results.append({
                    **params, 'K': K,
                    'ACC': accuracy_score(y_true, preds),
                    'F1': f1_score(y_true, preds, average='macro'),
                    'CCER': compute_ccer(graph, preds)
                })
            
            out_name = f"metrics_{filename.replace('.xlsx', '')}_K{K}.csv"
            pd.DataFrame(results).to_csv(os.path.join(OUTPUT_FOLDER, out_name), index=False)

if __name__ == "__main__":
    main()