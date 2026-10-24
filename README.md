# Quality-Aware Cluster-Conditioned Oversampling via Intra-Cluster Synthetic Sample Filtering

Reproducibility code for the manuscript *"Quality-Aware Cluster-Conditioned
Oversampling via Intra-Cluster Synthetic Sample Filtering"* (**Q-G-CTGAN**),
currently **under review** at a peer-reviewed journal.

> **Anonymous version.** This repository is released for double-blind review and
> contains no author- or affiliation-identifying information. Please do not add
> author names, institutional details, or private links until the review is complete.

Q-G-CTGAN addresses class imbalance in tabular classification. A separate CTGAN
is trained on each GMM-derived sub-cluster of the minority class, and generated
candidates pass a **two-stage quality filter**: (1) a composite quality score
`Q(x)` combining density fidelity, boundary separability, and local-outlier
behaviour, with a per-cluster rejection quantile; (2) Maximum Mean Discrepancy
(MMD)-based selection that aligns retained samples with the true minority
distribution.

---

## Repository layout

```
.
├── data/                      # place the 16 dataset CSVs here (see Data availability)
├── notebooks/
│   ├── 01_run_baselines_classic.ipynb     # None, SMOTE, ADASYN, G-SMOTE
│   ├── 02_run_baselines_ctgan.ipynb        # KMeans_CTGAN, CTGAN_MOS
│   ├── 03_run_qgctgan_and_gctgan.ipynb     # Q-G-CTGAN (proposed) + G-CTGAN (unfiltered)
│   ├── 04_run_ctdgan.ipynb                 # ctdGAN
│   ├── 05_run_ctabgan.ipynb                # CTAB-GAN+
│   ├── 06_aggregate_and_statistics.ipynb   # mean/std/95% CI, per-classifier Friedman, full-vs-unfiltered
│   └── 07_figures.ipynb                    # grayscale figures (PNG + PDF, 600 dpi)
└── results/
    ├── tables/                # all CSV outputs (raw per-seed, aggregated, statistics)
    └── figures/               # all figures (.png and .pdf, 600 dpi, grayscale)
```

Each notebook is a **single self-contained code cell** (English comments; notebook
name on the first line), so it can be run top-to-bottom in whichever environment
its method requires.

---

## Data availability

Place each dataset as `data/<name>.csv` with a `target` column (`1` = minority /
positive class). The 16 datasets are public:

- **UCI Machine Learning Repository** (https://archive.ics.uci.edu): Pima
  Diabetes, Credit Default, IBM HR Attrition, Ecoli, Wine Quality, Yeast ME2,
  Mammography, Abalone 19, PageBlocks, Protein Homology, SECOM (missing values
  median-imputed; `class`/`timestamp` and zero-variance columns dropped).
- **OpenML** (https://www.openml.org): Satellite (Statlog/Landsat, id 182),
  Churn (id 40701), PageBlocks (id 30).
- **Kaggle**: Fraud Detection (`mlg-ulb/creditcardfraud`), UNSW-NB15
  (`dhoogla/unswnb15`, training-set parquet).
- **imbalanced-learn** `fetch_datasets()`: Thyroid Sick and several of the above.

Expected keys (16): `credit_default, fraud_detection, pima_diabetes,
ibm_attrition, yeast_me2, mammography, abalone_19, wine_quality, ecoli,
pageblocks, protein_homo, satellite, churn, secom, thyroid_sick, unsw_nb15`.

---

## Environments

The generative baselines use mutually incompatible dependencies; use one
environment per notebook (Python 3.10+ recommended).

| Notebook | Key packages |
|----------|--------------|
| 01 | `scikit-learn`, `imbalanced-learn`, `imbalanced-learn-extra` (`imblearn_extra`), `lightgbm` |
| 02, 03 | `scikit-learn`, `ctgan`, `lightgbm`, `scipy`, `torch` |
| 04 | `artsyn==0.5.2` (ctdGAN), `scikit-learn`, `lightgbm`, `torch` |
| 05 | CTAB-GAN+ repo (`Team-TUD/CTAB-GAN-Plus`) on `sys.path`, `torch`, `scikit-learn`, `lightgbm` |
| 06, 07 | `pandas`, `numpy`, `scipy`, `scikit-posthocs`, `matplotlib`, `seaborn` |

For notebook 05, clone the CTAB-GAN+ repository and set `CTABGAN_REPO` (top of the
cell) to its path.

---

## How to run

1. Put the 16 CSVs in `data/`.
2. Run notebooks **01–05**, each in its matching environment. Each appends a
   `results/tables/raw_<family>.csv`. To test quickly, set `DATASETS_SUBSET`
   (e.g. `["ecoli", "pima_diabetes"]`) and/or shorten `SEEDS` at the top of the cell.
3. Run **06** to merge the raw files and produce the aggregated tables and
   statistics in `results/tables/`.
4. Run **07** to render all figures to `results/figures/` (PNG + PDF, 600 dpi).

---

## Evaluation protocol

- **5 repeated stratified 70/30 holdouts** (`SEEDS = [42, 43, 44, 45, 46]`), applied
  uniformly to every method; features standardized on the training split only;
  oversampling applied to the scaled training set; metrics on the held-out test set.
- Three classifiers: Random Forest, LightGBM, MLP. Metrics: AUC, PR-AUC, macro-F1,
  macro-Precision, macro-Recall, G-mean, Balanced Accuracy.
- Results are reported as **mean ± 95% CI** (Student-t) over the 5 seeds.
- **Statistics (notebook 06):** a **separate Friedman test is run for each
  classifier** (RF, LGBM, MLP), with Nemenyi post-hoc and critical-difference
  diagrams, computed on the across-seed mean AUC.
- **Q-G-CTGAN vs G-CTGAN (unfiltered):** both share the same generative base per
  (dataset, seed); G-CTGAN draws uniformly at random from the candidate pool with
  no filter, isolating the contribution of the two-stage filter.

### Notes on cost
- `ctdGAN` and `CTAB-GAN+` train on the full training set and are the slowest.
  On the three largest datasets (`fraud_detection`, `unsw_nb15`, `protein_homo`)
  a single `ctdGAN` seed already costs hours, so notebook 04 runs a **single
  seed** on those three (multiple seeds on the other 13); edit `LARGE`/`SEEDS`
  to change this.
- `CTAB-GAN+` is evaluated on **12 of 16** datasets (excludes `fraud_detection`,
  `protein_homo`, `unsw_nb15` for runtime and `ecoli` for near-zero minority
  yield), matching the study design.
- The CTGAN-based methods (`KMeans_CTGAN`, `CTGAN_MOS`, `Q-G-CTGAN`, `G-CTGAN`)
  train on the **minority class only** and are fast even on the large datasets.

---

## Suggested repository name

`quality-aware-cluster-oversampling` (descriptive and method-neutral). Alternatives:
`qgctgan-oversampling`, `intra-cluster-quality-filtering`.

For anonymous submission, host under a fresh anonymous account, or mirror to
`https://anonymous.4open.science/` and cite that link in the manuscript.
