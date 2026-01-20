# Graph-based Semi-Supervised Approach to Food Adulteration Detection from Analytical Data with Limited Labels

## Core Modules

### 1. SJUS (Structural Joint-Uncertainty Selection)
**SJUS** is a budget-aware active learning algorithm. It is designed to identify the most informative samples for manual labeling when dealing with large-scale analytical datasets where labels are expensive or scarce.

* **Scoring Mechanism**:
    * **$\phi$ (MST-based Centrality)**: Measures the structural representativeness of a sample using Minimum Spanning Trees.
    * **$\psi$ (Embedding Boundary Uncertainty)**: Quantifies the uncertainty of a sample based on its proximity to decision boundaries.
* **Priority Logic**: Implements a strategic three-tier selection process:
    1.  **Intersection (Joint set)**: Prioritizes samples that are both structurally central and highly uncertain.
    2.  **Union Completion**: Fills the budget by selecting samples that excel in at least one metric.
    3.  **Alternating Supplementary Selection**: Ensures diverse coverage across the entire feature space.


### 2. Measure Propagation (MP)
**MP** is a semi-supervised classification algorithm that leverages the underlying geometry of the data represented as a Graph.

* **Graph Construction**: Builds $K$-Nearest Neighbor (KNN) graphs supporting multiple metrics: Euclidean, Manhattan, and Cosine similarity.
* **Label Propagation**: Propagates information from a small set of "seed" labels across the graph to predict unknown classes.
* **Evaluation Metric (CCER)**: Includes the **Cross-Class Edge Ratio (CCER)** to quantify the separation quality between different clusters and evaluate graph robustness.

---

## Repository Structure

```text
├── Code/                   # Primary Python scripts
│   ├── SJUS.py             # Module I: Key-Sample Selection
│   └── label_propagation.py # Module II: Semi-supervised Classification
├── data/                   # Input datasets (CSV/XLSX)
├── results/                # Output selection results, plots, and metrics
├── LICENSE                 # MIT License
└── README.md               # Project documentation
```

## 🛠️ Quick Start

### 1. Installation
Install the required Python libraries using `pip`:

```bash
pip install numpy pandas scikit-learn scipy networkx matplotlib tqdm
```

### 2. Running Sample Selection (SJUS)
Place your raw data in the data/ folder and run the script from the root directory:

```bash
python Code/SJUS.py
```
Output: Selected sample indices and MST (Minimum Spanning Tree) visualization plots will be generated in the results/ folder.

### 3. Running Semi-supervised Classification
Configure your labeled seeds in the dataset files and execute:

```bash
python Code/label_propagation.py
```
Output: Performance metrics including ACC, F1-score, and CCER will be saved as CSV files in the results/ folder.

## License
```
This project is licensed under the MIT License - see the LICENSE.md file for details.
```

## Acknowledgements
```
We thank all dataset providers and prior works related to graph-based semi-supervised learning.
```
