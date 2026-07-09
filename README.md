# Q-G-CTGAN: Quality-Aware Cluster-Conditioned Oversampling via Intra-Cluster Synthetic Sample Filtering

> **Anonymous submission** — Author information will be released upon acceptance.

---

## Overview

**Q-G-CTGAN** is a generative oversampling framework designed for tabular class-imbalance problems.
It extends the cluster-conditioned CTGAN pipeline (G-CTGAN) with a two-stage quality filtering mechanism
that removes synthetic minority samples that drift from the true minority distribution or encroach on the majority boundary.

This repository reflects the **major revision** of the manuscript, which substantially extends the original
experimental evaluation in response to reviewer feedback: the benchmark collection now spans **16 datasets**
(11 original + 5 newly added, covering remote sensing, telecommunications, semiconductor manufacturing,
medical diagnosis, and cybersecurity), and the comparison now includes **4 additional generative baselines**
(K-means CTGAN, CTGAN-MOS, ctdGAN, CTAB-GAN+) alongside the original non-generative baselines.

### Key contributions

- **GMM-based partitioning** — minority class is split into coherent sub-clusters; independent CTGANs are trained per cluster, eliminating inter-cluster interference.
- **Two-stage quality filter** — Stage 1 scores each candidate sample with a composite quality metric Q(x̃) combining density (S_density), separability (S_sep), and local outlier factor (S_lof); Stage 2 applies MMD-contribution selection for distributional fidelity.
- **Adaptive threshold α_c** — filtering aggressiveness scales inversely with per-cluster GMM log-likelihood, applying stricter rejection to unstable clusters. α_c is defined as a **rejection quantile** (fraction of the per-cluster candidate pool discarded), empirically validated to match measured Stage-1 rejection rates exactly (see `08_quality_ablation_and_alpha_viz.ipynb`).
- **Target-aware candidate generation** — candidate multiplier is computed per dataset to approach the full oversampling target Δ = n_maj − n_min, bounded to [5×, 50×] to keep runtime tractable on extreme-IR datasets.

---

## Repository Structure

```
├── 01_dataset_loading.ipynb              # Dataset loading & verification (16 datasets)
├── 02_baselines.ipynb                    # Non-generative baselines (None, SMOTE, ADASYN, G-SMOTE)
├── 03_Q-G-CTGAN.ipynb                    # Proposed method: GMM clustering, CTGAN training,
│                                          #   two-stage filtering, MMD ablation, evaluation
├── 04a_new_baselines_sdv.ipynb           # K-means CTGAN, CTGAN-MOS  (env: venv_sdv)
├── 04b_new_baselines_ctdgan.ipynb        # ctdGAN                    (env: venv_ctdgan)
├── 04c_new_baselines_ctabgan.ipynb       # CTAB-GAN+                 (env: venv_ctabgan)
├── 05_results_integration.ipynb          # Unifies results from Notebooks 02–04 into one schema
├── 06_statistical_analysis.ipynb         # Friedman test + Nemenyi post-hoc, CD diagrams
├── 07_computational_efficiency.ipynb     # Runtime comparison, Q-G-CTGAN stage breakdown
├── 08_quality_ablation_and_alpha_viz.ipynb  # Extended quality-score ablation (16 datasets);
│                                          #   empirical validation of adaptive α (Fig. 4)
└── results/
    ├── figures/                          # Generated figures (PNG + PDF)
    └── *.csv                             # Generated result/summary tables
```

### Multiple virtual environments

Several new baselines require mutually incompatible dependency versions and are therefore run in
separate virtual environments:

| Environment      | Used by              | Notes                                             |
|-------------------|-----------------------|----------------------------------------------------|
| `venv_base`       | Notebooks 01, 02       | `imbalanced-learn`, `imbalanced-learn-extra`       |
| `venv_sdv`        | Notebooks 03, 04a      | `sdv` / `ctgan`, GPU-enabled PyTorch (CUDA)        |
| `venv_ctdgan`     | Notebook 04b            | `artsyn` (official ctdGAN implementation)          |
| `venv_ctabgan`    | Notebook 04c            | official `Team-TUD/CTAB-GAN-Plus` repository       |

---

## Datasets

Experiments use **16 public benchmark datasets** spanning imbalance ratios from IR 1.87 to 577.88.

| IR Range               | Datasets |
|--------------------------|----------|
| Low (IR < 10)             | Pima Diabetes, Credit Default, IBM HR Attrition, Churn, Ecoli, Satellite |
| Medium (10 ≤ IR < 50)     | SECOM, Thyroid Sick, Wine Quality, Yeast ME2, Mammography |
| Extreme (IR ≥ 50)         | UNSW-NB15, Protein Homology, Abalone-19, PageBlocks, Fraud Detection |

**5 datasets added in this revision** (domain and source in parentheses):

- **Satellite** (remote sensing; OpenML, Statlog/Landsat)
- **Churn** (telecommunications; OpenML)
- **SECOM** (semiconductor manufacturing fault detection; UCI, contains realistic sensor-failure missing values)
- **Thyroid Sick** (medical diagnosis; imbalanced-learn built-in collection)
- **UNSW-NB15** (network intrusion detection / cybersecurity; Moustafa & Slay 2015, backdoor-attack detection task)

All datasets are loaded, one-hot encoded (for categorical features), and verified in `01_dataset_loading.ipynb`.

---

## Compared Methods

| Method | Category | Notebook |
|--------|----------|----------|
| None (no oversampling) | Baseline | 02 |
| SMOTE | Interpolation | 02 |
| ADASYN | Adaptive interpolation | 02 |
| G-SMOTE | Geometric interpolation | 02 |
| K-means CTGAN | Cluster-conditioned generative | 04a |
| CTGAN-MOS | Noise-filtered generative | 04a |
| ctdGAN | Cluster-conditioned generative | 04b |
| CTAB-GAN+ | Deep generative (mixed-type) | 04c |
| **Q-G-CTGAN** (ours) | Quality-filtered cluster-conditioned generative | 03 |

**Note on reimplementations:** K-means CTGAN, ctdGAN, and CTAB-GAN+ have official or source-paper-derived
implementations (see Notebook 04a/b/c docstrings for details). **CTGAN-MOS has no official implementation**
and no publicly accessible description of its exact "coin-throwing" noise-removal procedure beyond the
source paper's abstract; we reimplement a principled stochastic approximation (Bernoulli accept/reject
weighted by k-NN density similarity to the real minority class), disclosed explicitly in `04a_new_baselines_sdv.ipynb`.

**Note on dataset coverage:** CTAB-GAN+ was evaluated on 12 of 16 datasets. Three datasets
(Fraud Detection, Protein Homology, UNSW-NB15) were excluded due to confirmed/projected computational
infeasibility (21.7–42.3 projected hours under the standard 150-epoch schedule); Ecoli was excluded due to
a structural instability in conditional generation on its very small minority class (n=24). See
`04c_new_baselines_ctabgan.ipynb` for the runtime-screening methodology and full per-dataset accounting.

---

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies (spread across the four virtual environments above):
`torch`, `ctgan`, `sdv`, `artsyn`, `scikit-learn`, `lightgbm`, `imbalanced-learn`, `imbalanced-learn-extra`,
`scipy`, `pandas`, `matplotlib`, `seaborn`, `scikit-posthocs`

---

## Reproducibility

Run notebooks in order, switching virtual environments as indicated:

```
01 → 02 → 03 → [04a | 04b | 04c] → 05 → 06 → 07 → 08
```

Notebooks 04a, 04b, and 04c are mutually independent and can be run in any order (or in parallel across
separate environments); Notebook 05 requires all three to have completed first.

All random seeds are fixed inside each notebook (`RANDOM_STATE = 42`). Results are saved incrementally to
`results/*.csv` and `results/figures/*.png` / `*.pdf` after each dataset, so long-running notebooks (04b/04c
on large datasets) can be resumed if interrupted.

---

## Main Results

### Average rank across 16 datasets (Friedman–Nemenyi, 7 oversampling methods)

CTAB-GAN+ is excluded from this primary ranking analysis for complete-block validity (see Notebook 06);
a secondary 8-method / 12-dataset analysis including CTAB-GAN+ is also reported there.

| Method | RF | LightGBM | MLP | **Overall** |
|--------|:--:|:--------:|:---:|:------------:|
| **Q-G-CTGAN** | 3.25 | 2.44 | 3.78 | **3.16** |
| SMOTE | 2.75 | 4.00 | 4.25 | 3.67 |
| K-means CTGAN | 4.59 | 3.56 | 3.19 | 3.78 |
| G-SMOTE | 4.06 | 4.38 | 3.19 | 3.88 |
| CTGAN-MOS | 4.28 | 3.88 | 3.62 | 3.93 |
| ADASYN | 3.69 | 4.81 | 5.31 | 4.60 |
| ctdGAN | 5.38 | 4.94 | 4.66 | 4.99 |

> Friedman χ² = 23.02, p < 0.001. **3 of 21 pairwise Nemenyi comparisons are statistically significant**
> (α = 0.05) — a new result relative to the original 11-dataset analysis, where none were significant:
> Q-G-CTGAN significantly outperforms **ctdGAN** (p = 0.0006) and **ADASYN** (p = 0.0177).
> Full pairwise matrix and CD diagrams: `results/06_nemenyi_pvalues_b.csv`, `results/06_cd_diagram_b.png`.

### Computational efficiency (average combined runtime per dataset, seconds)

| Method | Mean (s) | Median (s) |
|--------|:--------:|:----------:|
| None | 22.14 | 5.75 |
| **Q-G-CTGAN** | **53.98** | 25.33 |
| CTGAN-MOS | 61.18 | 14.35 |
| K-means CTGAN | 87.02 | 32.25 |
| G-SMOTE | 95.18 | 8.70 |
| SMOTE | 114.16 | 6.96 |
| ADASYN | 121.86 | 7.03 |
| CTAB-GAN+ | 292.95 | 155.66 |
| ctdGAN | 1788.05 | 117.83 |

> Q-G-CTGAN is the fastest of all five generative baselines compared. A stage-by-stage breakdown
> (`results/07_qgctgan_stage_breakdown.csv`) shows CTGAN training accounts for 83.6% of total runtime,
> while the proposed quality-filtering stages (Stage 1 + Stage 2 combined) account for only 3.0%.

### MMD Stage-2 ablation (independent evaluation)

On the 10 of 16 datasets where the Stage-1 candidate pool exceeds the oversampling target (making the
MMD-selection toggle mechanically meaningful), Stage 2 shows a small, mixed effect on downstream AUC
(mean ΔAUC = −0.0001; improves 6/10 datasets, degrades 4/10). On the remaining 6 datasets (predominantly
extreme-IR cases), the Stage-1 pool does not exceed the target even at the maximum candidate multiplier,
making the toggle inapplicable regardless of implementation. Full results: `results/03_qgctgan_results.csv`
(columns `mmd_stage`, `AUC`).

### Extended metrics

In addition to AUC, all methods are evaluated on **PR-AUC, F1, Precision, Recall (macro), G-mean, and
Balanced Accuracy**. Minority-class recall (sensitivity) is additionally recovered algebraically from
macro-averaged Recall and G-mean and reported separately (`results/05_minority_recall_recovered.csv`):
Q-G-CTGAN achieves the highest average minority-class recall (0.503) among the five compared generative
methods.

---

## License

This repository is released under the [MIT License](LICENSE).

> Code and data will be fully de-anonymized and archived on Zenodo upon paper acceptance.
