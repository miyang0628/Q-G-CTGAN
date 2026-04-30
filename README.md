# Q-G-CTGAN: Quality-Aware Cluster-Conditioned Oversampling via Intra-Cluster Synthetic Sample Filtering

> **Anonymous submission** — Author information will be released upon acceptance.

---

## Overview

**Q-G-CTGAN** is a generative oversampling framework designed for tabular class-imbalance problems.
It extends the cluster-conditioned CTGAN pipeline (G-CTGAN) with a two-stage quality filtering mechanism
that removes synthetic minority samples that drift from the true minority distribution or encroach on the majority boundary.

### Key contributions

- **GMM-based partitioning** — minority class is split into coherent sub-clusters; independent CTGANs are trained per cluster, eliminating inter-cluster interference.
- **Two-stage quality filter** — Stage 1 scores each candidate sample with a composite quality metric Q(x̃) combining density (S_density), separability (S_sep), and local outlier factor (S_lof); Stage 2 applies MMD-contribution selection for distributional fidelity.
- **Adaptive threshold α_c** — filtering aggressiveness scales inversely with per-cluster GMM log-likelihood, applying stricter rejection to unstable clusters.

---

## Repository Structure

```
├── 01_load_datasets.ipynb     # Dataset loading & preprocessing
├── 02_baselines.ipynb         # Baseline oversampling methods (SMOTE, ADASYN, G-SMOTE, CTGAN, TVAE, G-CTGAN)
├── 03_Q-G-CTGAN.ipynb         # Proposed method: training, filtering, evaluation
├── 04_analysis.ipynb          # Statistical tests, ranking, figures
└── results/
    ├── figures/               # Generated figures (fig01 – fig11)
    └── tables/                # Generated CSV tables
        ├── 04_auc_table_rf.csv
        ├── 04_auc_table_lgbm.csv
        ├── 04_auc_table_mlp.csv
        ├── 04_f1_table_rf.csv
        ├── 04_f1_table_lgbm.csv
        ├── 04_f1_table_mlp.csv
        ├── 04_statistical_tests.csv
        ├── 04_method_ranking.csv
        ├── 04_nemenyi_rf.csv
        ├── 04_nemenyi_lgbm.csv
        ├── 04_nemenyi_mlp.csv
        ├── 04_combined_comparison.csv
        ├── 03_qgctgan_mmd.csv
        ├── 03_ablation_quality.csv
        ├── 02_baselines_results.csv
        └── 03_qgctgan_results.csv
```

---

## Datasets

Experiments use 11 public benchmark datasets covering a wide range of imbalance ratios (IR 1.87 – 577.88).
All datasets are available from the UCI Machine Learning Repository or Kaggle and are loaded in `01_load_datasets.ipynb`.

| IR Range | Datasets |
|----------|----------|
| Low (IR < 10) | Pima Diabetes, Credit Default, IBM HR Attrition, Ecoli |
| Medium (10 ≤ IR < 50) | Wine Quality, Yeast ME2, Mammography |
| Extreme (IR ≥ 50) | Protein Homology, Abalone 19, PageBlocks, Fraud Detection |

---

## Compared Methods

| Method | Category |
|--------|----------|
| None (no oversampling) | Baseline |
| SMOTE | Interpolation |
| ADASYN | Adaptive interpolation |
| G-SMOTE | Geometric interpolation |
| CTGAN | Deep generative |
| TVAE | Deep generative |
| G-CTGAN | Cluster-conditioned generative |
| **Q-G-CTGAN** (ours) | Quality-filtered cluster-conditioned generative |

---

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: `torch`, `ctgan`, `scikit-learn`, `lightgbm`, `imbalanced-learn`, `scipy`, `pandas`, `matplotlib`, `seaborn`

---

## Reproducibility

Run notebooks in order:

```
01 → 02 → 03 → 04
```

All random seeds are fixed inside each notebook.
Results are saved automatically to `results/tables/` and `results/figures/`.

---

## Main Results

### Average AUC across 11 benchmark datasets

Average AUC across 11 benchmark datasets (bold = best per classifier):

| Method | RF | LightGBM | MLP |
|--------|:--:|:--------:|:---:|
| **Q-G-CTGAN (adaptive)** | **0.9027** | **0.8993** | **0.8680** |
| G-SMOTE | 0.8974 | 0.8950 | 0.8561 |
| SMOTE | 0.8954 | 0.8935 | 0.8604 |
| ADASYN | 0.8950 | 0.8904 | 0.8566 |
| TVAE | 0.8967 | 0.8913 | 0.8584 |
| G-CTGAN | 0.8850 | 0.8838 | 0.8598 |
| CTGAN | 0.8853 | 0.8879 | 0.8758 |
| None (no oversampling) | 0.9058 | 0.8956 | 0.8749 |

> Per-dataset breakdown: `results/tables/04_auc_table_[rf|lgbm|mlp].csv`

> Full per-dataset AUC and F1-score tables are in `results/tables/`.

### Average Rank (Friedman test, lower is better)

Average rank across 11 datasets (Friedman test, lower = better):

| Method | RF | LightGBM | MLP | Overall |
|--------|:--:|:--------:|:---:|:-------:|
| **Q-G-CTGAN (adaptive)** | **2.77** | **2.91** | **3.55** | **3.08** |
| SMOTE | 3.14 | 3.59 | 4.27 | 3.67 |
| G-SMOTE | 3.27 | 3.77 | 4.77 | 3.94 |
| ADASYN | 3.64 | 4.09 | 4.73 | 4.15 |
| G-CTGAN | 4.91 | 4.45 | 3.41 | 4.26 |
| CTGAN | 5.27 | 4.82 | 3.18 | 4.42 |
| TVAE | 5.00 | 4.36 | 4.09 | 4.48 |

> Full ranking table: `results/tables/04_method_ranking.csv`

### Q-G-CTGAN vs G-CTGAN (AUC improvement)

| Classifier | Wins (out of 11) | Mean ΔAUC |
|------------|:----------------:|:---------:|
| RF | 8 | +0.0176 |
| LightGBM | 7 | +0.0155 |
| MLP | 5 | +0.0082 |

### MMD Reduction (filtering before → after)

Adaptive α achieves an average MMD² reduction of **+0.0041** across all datasets,
confirming that quality filtering improves distributional fidelity of synthetic samples.

---

## License

This repository is released under the [MIT License](LICENSE).

> Code and data will be fully de-anonymized and archived on Zenodo upon paper acceptance.
