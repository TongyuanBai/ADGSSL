# -*- coding: utf-8 -*-
"""
Measure Propagation for Semi-Supervised Learning
Optimized for GitHub distribution.
"""

import os
import time
import logging
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from queue import PriorityQueue
from tqdm import tqdm
from adjustText import adjust_text

from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (accuracy_score, recall_score, precision_score, 
                             f1_score, roc_auc_score)
from sklearn.model_selection import ParameterGrid
from sklearn.base import BaseEstimator, ClassifierMixin

# ========= Configuration =========
RUNS_PER_PARAM = 1
PRINT_TOP_N     = 5

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========= Graph Classes =========
class Graph(dict):
    NEIGHBOURS_KEY = 'neighbours'
    WEIGHTS_KEY = 'weights'

    @property
    def vertices(self):
        return self.keys()

    @property
    def nb_vertices(self):
        return len(self)

    def add_edge(self, vertex, neighbour, weight=1):
        if vertex not in self:
            self[vertex] = {self.NEIGHBOURS_KEY: [neighbour],
                            self.WEIGHTS_KEY: [weight]}
        elif neighbour not in self[vertex][self.NEIGHBOURS_KEY]:
            self[vertex][self.NEIGHBOURS_KEY].append(neighbour)
            self[vertex][self.WEIGHTS_KEY].append(weight)

    def build(self, *args):
        self._build(*args)
        self.post_build_hook()
        return self

    def _build(self, edges):
        for source_id, dest_id in edges:
            if source_id != dest_id:
                self.add_edge(source_id, dest_id)
                self.add_edge(dest_id, source_id)

    def post_build_hook(self):
        for vertex, edge_info in self.items():
            edge_info[self.WEIGHTS_KEY] = np.expand_dims(np.array(edge_info[self.WEIGHTS_KEY]), 1)

# ========= Similarity Functions =========
def euc_similarity(x, y):
    return 1.0 / (1.0 + np.linalg.norm(x - y))

def manhattan_similarity(x, y):
    return 1.0 / (1.0 + np.sum(np.abs(x - y)))

def cosine_similarity(x, y):
    return 1.0 / (1.0 + cosine_distances([x], [y])[0][0])

# ========= KNN Graph =========
class KNNGraph(Graph):
    def __init__(self, K, similarity_func, *args, **kwargs):
        super(KNNGraph, self).__init__(*args, **kwargs)
        self.K = K
        self.similarity_func = similarity_func

    class Neighbour:
        def __init__(self, id_, similarity):
            self.id = id_
            self.similarity = similarity

        def __lt__(self, other):
            return self.similarity < other.similarity

    def _build(self, X):
        n_samples = len(X)
        for i in range(n_samples):
            neighbours = PriorityQueue(maxsize=self.K)
            for j in range(n_samples):
                if i != j:
                    neighbour = KNNGraph.Neighbour(j, self.similarity_func(X[i], X[j]))
                    if not neighbours.full():
                        neighbours.put(neighbour)
                    else:
                        lowest_entry = neighbours.get()
                        if neighbour.similarity > lowest_entry.similarity:
                            neighbours.put(neighbour)
                        else:
                            neighbours.put(lowest_entry)
            while not neighbours.empty():
                neighbour = neighbours.get()
                self.add_edge(i, neighbour.id, weight=neighbour.similarity)

# ========= Measure Propagation Core =========
class MeasurePropagation:
    def __init__(self, mu=0.1, nu=0.01, tol=2e-2, max_iter=100):
        self.graph = None
        self.r, self.nb_classes = None, None
        self.p, self.q = None, None
        self.mu = mu
        self.nu = nu
        self.tol = tol
        self.max_iter = max_iter
        self.SMALL = 1e-10

    def _labels_to_probabilities(self, vertices_labels_dct):
        unique_labels = np.unique(list(vertices_labels_dct.values()))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        nb_classes = len(unique_labels)
        probs = {}
        for vertex, label in vertices_labels_dct.items():
            probs[vertex] = np.zeros(nb_classes)
            probs[vertex][label_to_index[label]] = 1
        return probs, nb_classes

    def _init_probability_distributions(self):
        p = np.full((self.graph.nb_vertices, self.nb_classes), 1 / self.nb_classes)
        q = np.full((self.graph.nb_vertices, self.nb_classes), 1 / self.nb_classes)
        return p, q

    def compute_p_update(self, vertex):
        neighbours = self.graph[vertex][self.graph.NEIGHBOURS_KEY]
        w_neighbours = self.graph[vertex][self.graph.WEIGHTS_KEY]
        gamma = self.nu + self.mu * w_neighbours.sum()
        p_new = np.exp((np.log(self.q[neighbours] + self.SMALL) * w_neighbours).sum(axis=0) * (self.mu / gamma))
        return p_new / p_new.sum()

    def compute_p_updates(self):
        for vertex in self.graph.vertices:
            self.p[vertex] = self.compute_p_update(vertex)

    def compute_q_update(self, vertex):
        neighbours = self.graph[vertex][self.graph.NEIGHBOURS_KEY]
        w_neighbours = self.graph[vertex][self.graph.WEIGHTS_KEY]
        divident_right_sum = self.mu * (w_neighbours * self.p[neighbours]).sum(axis=0)
        divident_left_sum = self.r[vertex] if vertex in self.r else 0
        divisor_right_sum = self.mu * w_neighbours.sum()
        divisor_left_sum = 1 if vertex in self.r else 0
        return (divident_right_sum + divident_left_sum) / (divisor_right_sum + divisor_left_sum)

    def compute_q_updates(self):
        q_new = np.zeros(self.q.shape)
        for vertex in self.graph.vertices:
            q_new[vertex] = self.compute_q_update(vertex)
        return q_new

    def alternate_minimization_step(self):
        self.compute_p_updates()
        q_new = self.compute_q_updates()
        test_convergence = self.compute_test_convergence(q_new)
        self.q = q_new
        return test_convergence

    def compute_test_convergence(self, q_new):
        div = q_new / (self.q + self.SMALL)
        beta = np.log(np.max(div, 1) + self.SMALL)
        accum = .0
        for vertex in self.graph.vertices:
            delta = 1 if vertex in self.r else 0
            d_i = np.array(self.graph[vertex][self.graph.WEIGHTS_KEY]).sum()
            accum += (delta + d_i) * beta[vertex]
        return accum

    def optimize(self, graph, vertices_labels_dct):
        self.graph = graph
        self.r, self.nb_classes = self._labels_to_probabilities(vertices_labels_dct)
        self.p, self.q = self._init_probability_distributions()
        convergences = []
        for it in range(self.max_iter):
            convergences.append(self.alternate_minimization_step())
            if it > 0:
                change = (convergences[it - 1] - convergences[it]) / convergences[it]
                if change <= self.tol:
                    break

    def get_output_labels(self):
        return np.argmax(self.q, axis=1)

class MeasurePropagationSklearn(MeasurePropagation, BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        labeled_indices = np.where(y != -1)[0]
        vertices_labels_dct = {idx: y[idx] for idx in labeled_indices}
        return self.optimize(X, vertices_labels_dct)

    def predict(self):
        return self.get_output_labels()

# ========= Metrics =========
def compute_ccer(graph, labels):
    seen = set()
    total_edges = 0
    cross_edges = 0
    for u in graph.vertices:
        for v in graph[u][graph.NEIGHBOURS_KEY]:
            if u == v: continue
            e = (u, v) if u < v else (v, u)
            if e in seen: continue
            seen.add(e)
            total_edges += 1
            if labels[u] != labels[v]:
                cross_edges += 1
    ccer = (cross_edges / total_edges) if total_edges > 0 else 0.0
    return ccer, cross_edges, total_edges

# ========= Main Process =========
def process_file(file_path, ground_truth_path, K_values, param_grid, save_path):
    # Load Data
    data  = pd.read_excel(file_path)
    data1 = pd.read_csv(ground_truth_path)

    X = data.iloc[:, 1:-1].to_numpy()
    X_scaled = StandardScaler().fit_transform(X)
    y_true = data1['Class'].to_numpy()
    y_mask = data['Class'].to_numpy()

    # Determine Similarity
    fname = file_path.lower()
    if 'euclidean' in fname:
        similarity_func, dist_tag = euc_similarity, 'Euclidean'
    elif 'manhattan' in fname:
        similarity_func, dist_tag = manhattan_similarity, 'Manhattan'
    elif 'cosine' in fname:
        similarity_func, dist_tag = cosine_similarity, 'Cosine'
    else:
        raise ValueError("Similarity not recognized from filename.")

    base_name = os.path.basename(file_path).split('.')[0]
    print(f"\nProcessing: {base_name} | Metric: {dist_tag}")

    for K in tqdm(K_values, desc="K-Values"):
        summary_rows = []
        for params in tqdm(ParameterGrid(param_grid), desc="Params", leave=False):
            mu, nu, tol, mit = params['mu'], params['nu'], params['tol'], params['max_iter']

            acc_list, f1_list, auc_list, time_list = [], [], [], []
            ccer_list = []

            for _ in range(RUNS_PER_PARAM):
                t0 = time.perf_counter()
                knn_graph = KNNGraph(K=K, similarity_func=similarity_func).build(X_scaled)
                mp = MeasurePropagationSklearn(mu=mu, nu=nu, tol=tol, max_iter=mit)
                mp.fit(knn_graph, y_mask.copy())
                output_labels = mp.predict()
                t1 = time.perf_counter()

                acc_list.append(accuracy_score(y_true, output_labels))
                f1_list.append(f1_score(y_true, output_labels, average='macro'))
                
                try:
                    classes = np.unique(y_true)
                    y_bin = label_binarize(y_true, classes=classes)
                    auc_list.append(roc_auc_score(y_bin, mp.q, average='macro', multi_class='ovr'))
                except:
                    auc_list.append(np.nan)

                ccer, _, _ = compute_ccer(knn_graph, output_labels)
                ccer_list.append(ccer)
                time_list.append(t1 - t0)

            # Statistics
            summary_rows.append({
                "Distance": dist_tag, "K": K, "mu": mu, "nu": nu,
                "ACC_mean": np.mean(acc_list), "ACC_std": np.std(acc_list),
                "F1_mean": np.mean(f1_list), "F1_std": np.std(f1_list),
                "AUC_mean": np.nanmean(auc_list), "CCER_mean": np.mean(ccer_list),
                "Time_s": np.mean(time_list)
            })

        # Save Results
        os.makedirs(save_path, exist_ok=True)
        out_csv = os.path.join(save_path, f"result_{base_name}_K{K}.csv")
        pd.DataFrame(summary_rows).to_csv(out_csv, index=False)

if __name__ == "__main__":
    # --- GitHub Friendly Path Setup ---
    # These can be overridden by environment variables or CLI arguments
    BASE_DIR = os.path.dirname(__file__)
    DATA_DIR = os.path.join(BASE_DIR, "data")
    RESULT_DIR = os.path.join(BASE_DIR, "results")

    # Example: Place your files in a 'data' folder next to the script
    target_files = [
        os.path.join(DATA_DIR, 'pork_mst_Euclidean.xlsx'),
        os.path.join(DATA_DIR, 'pork_mst_Manhattan.xlsx'),
        os.path.join(DATA_DIR, 'pork_mst_Cosine.xlsx'),
    ]
    gt_file = os.path.join(DATA_DIR, 'pork_semi.csv')

    K_VALS = [3, 4, 5, 6]
    PARAMS = {
        'mu': [1e-4, 0.1, 10],
        'nu': [1e-4, 0.01],
        'max_iter': [1000],
        'tol': [1e-1]
    }

    # Run only if files exist
    if os.path.exists(gt_file):
        for f in target_files:
            if os.path.exists(f):
                process_file(f, gt_file, K_VALS, PARAMS, RESULT_DIR)
            else:
                print(f"Skipping missing file: {f}")
    else:
        print(f"Ground truth file not found at {gt_file}. Please check the 'data' directory.")