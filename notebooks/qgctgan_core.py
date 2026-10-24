"""
qgctgan_core.py  --  Q-G-CTGAN pipeline functions, lifted verbatim from
notebook 03_Q-G-CTGAN.py (Section 2-8) for use by the multi-seed driver.

Only the generative pipeline (GMM clustering, per-cluster CTGAN training,
target-aware candidate generation, two-stage quality filtering) is here.
Classifiers / evaluation / data loading live in common_io.py.

Requires: numpy, pandas, scipy, scikit-learn, ctgan  (the standalone
`ctgan` package, same env as the original notebook 03 / 04a).
"""
import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from ctgan import CTGAN

RANDOM_STATE = 42

CANDIDATE_MULTIPLIER = 5.0

MAX_MULTIPLIER       = 50.0

ALPHA_FIXED          = [0.1, 0.2, 0.3]

ALPHA_MIN            = 0.1

ALPHA_MAX            = 0.3

TARGET_RATIO         = 0.50

def fit_gmm(X_min, k, random_state=RANDOM_STATE):
    gmm = GaussianMixture(
        n_components=k, covariance_type="full",
        random_state=random_state, max_iter=200,
    )
    gmm.fit(X_min)
    return gmm

def select_optimal_k(X_min, k_range=range(2, 11), random_state=RANDOM_STATE):
    bic_scores = {}
    for k in k_range:
        if len(X_min) < k * 2:
            break
        try:
            gmm = fit_gmm(X_min, k, random_state)
            bic_scores[k] = gmm.bic(X_min)
        except Exception:
            continue
    if not bic_scores:
        return 2, {}, fit_gmm(X_min, 2, random_state)
    best_k   = min(bic_scores, key=bic_scores.get)
    best_gmm = fit_gmm(X_min, best_k, random_state)
    return best_k, bic_scores, best_gmm

def s_density(X_syn, mu_c, sigma_c):
    """S_density = exp(-D_M(x̃, mu_c, Sigma_c)) — intra-cluster density score."""
    try:
        sigma_inv = np.linalg.pinv(sigma_c)
        diff      = X_syn - mu_c
        maha      = np.sqrt(np.einsum("ij,jk,ik->i", diff, sigma_inv, diff))
        return np.exp(-maha)
    except Exception:
        return np.ones(len(X_syn)) * 0.5

def s_sep(X_syn, mu_minority_all, mu_majority_clusters, eps=1e-8):
    """S_sep = d_maj / (d_maj + d_min + eps) — class separability score."""
    d_maj = cdist(X_syn, mu_majority_clusters).min(axis=1)
    d_min = cdist(X_syn, mu_minority_all).min(axis=1)
    return d_maj / (d_maj + d_min + eps)

def s_lof(X_syn, X_cluster_real, n_neighbors=5):
    """S_lof = 1 / LOF(x̃) — local outlier score (computed within cluster only)."""
    X_combined = np.vstack([X_cluster_real, X_syn])
    n_nb = min(n_neighbors, len(X_cluster_real) - 1)
    if n_nb < 1:
        return np.ones(len(X_syn)) * 0.5
    try:
        lof      = LocalOutlierFactor(n_neighbors=n_nb, novelty=False)
        lof.fit_predict(X_combined)
        lof_vals = -lof.negative_outlier_factor_[len(X_cluster_real):]
        return 1.0 / np.maximum(lof_vals, 1e-8)
    except Exception:
        return np.ones(len(X_syn)) * 0.5

def quality_score(X_syn, mu_c, sigma_c, mu_minority_all,
                  mu_majority_clusters, X_cluster_real,
                  w=(1/3, 1/3, 1/3)):
    """Q(x̃) = w1*S_density + w2*S_sep + w3*S_lof  (all normalised to [0,1])."""
    sd = s_density(X_syn, mu_c, sigma_c)
    ss = s_sep(X_syn, mu_minority_all, mu_majority_clusters)
    sl = s_lof(X_syn, X_cluster_real)

    def norm(v):
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + 1e-8)

    return w[0]*norm(sd) + w[1]*norm(ss) + w[2]*norm(sl)

def compute_adaptive_alpha(gmm, X_min, labels, best_k,
                           alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX):
    """
    alpha_c = alpha_max - (alpha_max - alpha_min) * ell_tilde_c
    High log-likelihood (stable cluster)   -> small alpha_c -> less filtering
        (a small quantile of candidates is rejected, R4-3)
    Low  log-likelihood (unstable cluster) -> large alpha_c -> more filtering
        (a large quantile of candidates is rejected, R4-3)
    """
    log_liks = []
    for c in range(best_k):
        X_c = X_min[labels == c]
        ll  = gmm.score_samples(X_c).mean() if len(X_c) >= 2 else -np.inf
        log_liks.append(ll)

    ll = np.array(log_liks, dtype=float)
    valid = ll[np.isfinite(ll)]
    if len(valid) == 0:
        return np.full(best_k, (alpha_min + alpha_max) / 2)
    ll = np.where(np.isfinite(ll), ll, valid.min())

    ll_min, ll_max = ll.min(), ll.max()
    ll_norm = (ll - ll_min) / (ll_max - ll_min + 1e-8)
    return np.clip(alpha_max - (alpha_max - alpha_min) * ll_norm,
                   alpha_min, alpha_max)

def rbf_kernel(X, Y, gamma=None):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    XX = np.sum(X**2, axis=1, keepdims=True)
    YY = np.sum(Y**2, axis=1, keepdims=True)
    return np.exp(-gamma * (XX + YY.T - 2 * X @ Y.T))

def median_bandwidth(X, Y, sample_size=500):
    """Median heuristic: gamma = 1 / (2 * median(pairwise_dist^2))."""
    rng = np.random.default_rng(RANDOM_STATE)
    Xs  = X[rng.choice(len(X), min(sample_size, len(X)), replace=False)]
    Ys  = Y[rng.choice(len(Y), min(sample_size, len(Y)), replace=False)]
    Z   = np.vstack([Xs, Ys])
    d   = cdist(Z, Z, metric="sqeuclidean")
    return 1.0 / (2.0 * np.median(d[d > 0]) + 1e-8)

def mmd_squared(X, Y, min_samples=10):
    """MMD^2(P,Q) with median heuristic bandwidth. Returns NaN if too few samples."""
    if len(X) < min_samples or len(Y) < min_samples:
        return np.nan
    gamma = median_bandwidth(X, Y)
    return float(rbf_kernel(X, X, gamma).mean()
                 - 2 * rbf_kernel(X, Y, gamma).mean()
                 + rbf_kernel(Y, Y, gamma).mean())

def mmd_contribution(x_new, X_ref, X_pool, gamma):
    """
    Approximate MMD^2 change when adding x_new to synthetic set.
    Lower score -> x_new reduces MMD -> closer to real minority distribution.
    """
    k_xref = rbf_kernel(x_new.reshape(1, -1), X_ref, gamma).mean()
    k_xsyn = rbf_kernel(x_new.reshape(1, -1), X_pool, gamma).mean() \
              if len(X_pool) > 0 else 0.0
    return float(-2 * k_xref + k_xsyn)

MULTIPLIER_SAFETY_FACTOR = 1.5

def compute_required_multiplier(n_min, n_maj, target_ratio=TARGET_RATIO,
                                 base_multiplier=CANDIDATE_MULTIPLIER,
                                 safety_factor=MULTIPLIER_SAFETY_FACTOR,
                                 max_multiplier=MAX_MULTIPLIER):
    """
    Returns (multiplier_to_use, n_syn_needed, achievable_ratio_at_cap).
    multiplier_to_use is clipped to [base_multiplier, max_multiplier].
    """
    n_syn_needed = max(int(np.round(
        (target_ratio * (n_maj + n_min) - n_min) / (1 - target_ratio)
    )), 0)
    if n_min == 0:
        return base_multiplier, n_syn_needed, target_ratio

    required = (n_syn_needed * safety_factor) / n_min
    multiplier_to_use = float(np.clip(required, base_multiplier, max_multiplier))

    max_possible_synthetic = multiplier_to_use * n_min
    achievable_ratio = (n_min + max_possible_synthetic) / (n_maj + n_min + max_possible_synthetic)

    return multiplier_to_use, n_syn_needed, round(achievable_ratio, 4)

def train_ctgan_per_cluster(X_min, labels, best_k, random_state=RANDOM_STATE):
    """Train one CTGAN model per GMM cluster."""
    ctgan_models = {}
    for c in range(best_k):
        X_c = X_min[labels == c]
        if len(X_c) < 2:
            ctgan_models[c] = None
            continue
        cols   = [f"f{i}" for i in range(X_c.shape[1])]
        X_c_df = pd.DataFrame(X_c, columns=cols)
        try:
            model = CTGAN(epochs=100, verbose=False)
            model.fit(X_c_df)
            ctgan_models[c] = model
            print(f"    Cluster {c}: n={len(X_c)}  OK")
        except Exception as e:
            ctgan_models[c] = None
            print(f"    Cluster {c}: failed ({e})")
    return ctgan_models

def generate_candidates(X_min, labels, best_k, ctgan_models, multiplier):
    """Generate N_c = round(n_min_c * multiplier) candidate samples per cluster."""
    candidates = {}
    for c in range(best_k):
        X_c   = X_min[labels == c]
        n_gen = max(int(np.round(len(X_c) * multiplier)), 10)
        if ctgan_models.get(c) is None or len(X_c) < 2:
            candidates[c] = np.empty((0, X_min.shape[1]))
            continue
        try:
            candidates[c] = ctgan_models[c].sample(n_gen).values.astype(float)
        except Exception:
            candidates[c] = np.empty((0, X_min.shape[1]))
    return candidates

def filter_by_quality(candidates, X_min, labels, X_maj, gmm, best_k,
                      alpha_vec, target_ratio=TARGET_RATIO,
                      mmd_stage_enabled=True):
    n_maj = len(X_maj)
    n_min = len(X_min)
    n_syn_needed = max(int(np.round(
        (target_ratio * (n_maj + n_min) - n_min) / (1 - target_ratio)
    )), 0)

    mu_minority_all      = gmm.means_
    mu_majority_clusters = np.array([X_maj.mean(axis=0)])

    # ── Stage 1: Q(x̃)-based per-cluster filtering (rejection quantile) ──
    filtered_per_cluster = {}
    quality_info         = {}
    stage1_time_start    = time.time()

    for c in range(best_k):
        X_cand = candidates[c]
        X_c    = X_min[labels == c]

        if len(X_cand) == 0 or len(X_c) < 2:
            filtered_per_cluster[c] = np.empty((0, X_min.shape[1]))
            quality_info[c] = {"n_candidate": 0, "n_after_filter": 0,
                               "alpha": alpha_vec[c], "mean_Q_before": np.nan,
                               "mean_Q_after": np.nan}
            continue

        Q = quality_score(
            X_cand, gmm.means_[c], gmm.covariances_[c],
            mu_minority_all, mu_majority_clusters, X_c,
        )
        alpha_c   = alpha_vec[c]
        threshold = np.quantile(Q, alpha_c)  # alpha_c = rejection quantile (R4-3)
        mask      = Q >= threshold
        X_filt    = X_cand[mask]

        quality_info[c] = {
            "n_candidate"   : len(X_cand),
            "n_after_filter": int(mask.sum()),
            "alpha"         : round(float(alpha_c), 3),
            "mean_Q_before" : round(float(Q.mean()), 4),
            "mean_Q_after"  : round(float(Q[mask].mean()), 4) if mask.sum() > 0 else np.nan,
        }
        filtered_per_cluster[c] = X_filt

    stage1_time = round(time.time() - stage1_time_start, 3)

    # ── Stage 2: MMD contribution-based final selection ─────────────
    stage2_time_start = time.time()
    all_filtered = [v for v in filtered_per_cluster.values() if len(v) > 0]
    if not all_filtered:
        return (np.empty((0, X_min.shape[1])), quality_info,
                {"stage1_time": stage1_time, "stage2_time": 0.0})

    X_pool = np.vstack(all_filtered)

    if not mmd_stage_enabled:
        X_final = X_pool[:n_syn_needed] if len(X_pool) > n_syn_needed else X_pool
        stage2_time = round(time.time() - stage2_time_start, 3)
        return X_final, quality_info, {"stage1_time": stage1_time, "stage2_time": stage2_time}

    if len(X_pool) <= n_syn_needed:
        stage2_time = round(time.time() - stage2_time_start, 3)
        return X_pool, quality_info, {"stage1_time": stage1_time, "stage2_time": stage2_time}

    gamma = median_bandwidth(X_min, X_pool)
    mmd_scores = np.array([
        mmd_contribution(X_pool[i], X_min, X_pool, gamma)
        for i in range(len(X_pool))
    ])
    top_idx = np.argsort(mmd_scores)[:n_syn_needed]
    stage2_time = round(time.time() - stage2_time_start, 3)
    return X_pool[top_idx], quality_info, {"stage1_time": stage1_time, "stage2_time": stage2_time}

def qgctgan_prepare(X_train, y_train, random_state=RANDOM_STATE):
    X_min = X_train[y_train == 1].astype(float)
    X_maj = X_train[y_train == 0].astype(float)

    t0 = time.time()
    best_k, bic_scores, gmm = select_optimal_k(X_min, random_state=random_state)
    labels        = gmm.predict(X_min)
    cluster_sizes = np.bincount(labels)
    gmm_time = round(time.time() - t0, 3)
    print(f"  GMM: best_k={best_k}  cluster_sizes={cluster_sizes.tolist()}  [{gmm_time:.1f}s]")

    required_mult, n_syn_needed, achievable_ratio = compute_required_multiplier(len(X_min), len(X_maj))
    capped = required_mult >= MAX_MULTIPLIER
    print(f"  Target synthetic samples (Delta, {TARGET_RATIO:.0%} balance): {n_syn_needed:,}")
    print(f"  Candidate multiplier: {required_mult:.1f}x "
          f"{'[CAPPED at ' + str(MAX_MULTIPLIER) + 'x]' if capped else '[within cap]'}")
    if capped:
        print(f"  NOTE: full {TARGET_RATIO:.0%} balance is not reachable within the multiplier cap; "
              f"best-case achievable minority ratio ~= {achievable_ratio:.1%}")

    t0 = time.time()
    print(f"  Training {best_k} CTGAN models...")
    ctgan_models = train_ctgan_per_cluster(X_min, labels, best_k, random_state)
    ctgan_train_time = round(time.time() - t0, 3)

    t0 = time.time()
    print(f"  Generating candidates (x{required_mult:.1f})...")
    candidates   = generate_candidates(X_min, labels, best_k, ctgan_models, multiplier=required_mult)
    n_cand_total = sum(len(v) for v in candidates.values())
    generation_time = round(time.time() - t0, 3)
    print(f"  Total candidates generated: {n_cand_total:,}  [{generation_time:.1f}s]  "
          f"(target={n_syn_needed:,}, {'sufficient' if n_cand_total >= n_syn_needed else 'insufficient even before filtering'})")

    return {
        "X_min": X_min, "X_maj": X_maj, "best_k": best_k, "gmm": gmm,
        "labels": labels, "cluster_sizes": cluster_sizes,
        "ctgan_models": ctgan_models, "candidates": candidates,
        "n_cand_total": n_cand_total,
        "candidate_multiplier_used": round(required_mult, 2),
        "multiplier_capped": capped,
        "n_syn_target": n_syn_needed,
        "achievable_ratio_at_cap": achievable_ratio,
        "timing": {
            "gmm_time": gmm_time,
            "ctgan_train_time": ctgan_train_time,
            "generation_time": generation_time,
        },
    }

def qgctgan_filter(prepared, alpha_mode="adaptive", alpha_fixed=0.2,
                    mmd_stage_enabled=True):
    X_min  = prepared["X_min"]; X_maj = prepared["X_maj"]
    best_k = prepared["best_k"]; gmm  = prepared["gmm"]
    labels = prepared["labels"]; candidates = prepared["candidates"]

    if alpha_mode == "adaptive":
        alpha_vec = compute_adaptive_alpha(gmm, X_min, labels, best_k)
        print(f"  Adaptive alpha: {np.round(alpha_vec, 3)}")
    else:
        alpha_vec = np.full(best_k, alpha_fixed)
        print(f"  Fixed alpha={alpha_fixed} for all clusters")

    print(f"  Filtering by quality score Q(x̃)... (MMD stage: {'ON' if mmd_stage_enabled else 'OFF'})")
    t_filter = time.time()
    X_syn, quality_info, stage_timing = filter_by_quality(
        candidates, X_min, labels, X_maj, gmm, best_k, alpha_vec,
        mmd_stage_enabled=mmd_stage_enabled,
    )
    filter_time = round(time.time() - t_filter, 2)
    print(f"  Filtered: {prepared['n_cand_total']:,} -> {len(X_syn):,} samples [{filter_time:.1f}s]")

    all_cand   = np.vstack([v for v in candidates.values() if len(v) > 0])
    mmd_before = mmd_squared(X_min, all_cand) if len(all_cand) > 0 else np.nan
    mmd_after  = mmd_squared(X_min, X_syn)    if len(X_syn)  > 0 else np.nan
    direction  = "improved" if (not np.isnan(mmd_after) and
                                 not np.isnan(mmd_before) and
                                 mmd_after < mmd_before) else "degraded"
    print(f"  MMD^2 before={mmd_before:.6f}  after={mmd_after:.6f}  ({direction})")

    if len(X_syn) > 0:
        X_out = np.vstack([X_maj, X_min, X_syn])
        y_out = np.concatenate([
            np.zeros(len(X_maj)), np.ones(len(X_min)), np.ones(len(X_syn))
        ]).astype(int)
    else:
        X_out = np.vstack([X_maj, X_min])
        y_out = np.concatenate([np.zeros(len(X_maj)), np.ones(len(X_min))]).astype(int)

    minority_ratio = round((y_out == 1).sum() / len(y_out), 4)
    target_gap = round(TARGET_RATIO - minority_ratio, 4)  # positive = fell short of 50% target

    info = {
        "best_k"        : best_k,
        "cluster_sizes" : prepared["cluster_sizes"].tolist(),
        "alpha_mode"    : alpha_mode,
        "alpha_vec"     : np.round(alpha_vec, 3).tolist(),
        "mmd_stage_enabled": mmd_stage_enabled,
        "n_candidates"  : prepared["n_cand_total"],
        "candidate_multiplier_used": prepared["candidate_multiplier_used"],
        "multiplier_capped": prepared["multiplier_capped"],
        "n_syn_target"  : prepared["n_syn_target"],
        "n_synthetic"   : len(X_syn),
        "minority_ratio": minority_ratio,
        "target_gap"    : target_gap,
        "mmd_before"    : round(mmd_before, 6) if not np.isnan(mmd_before) else None,
        "mmd_after"     : round(mmd_after,  6) if not np.isnan(mmd_after)  else None,
        "filter_time"   : filter_time,
        "stage1_time"   : stage_timing["stage1_time"],
        "stage2_time"   : stage_timing["stage2_time"],
        "gmm_time"      : prepared["timing"]["gmm_time"],
        "ctgan_train_time": prepared["timing"]["ctgan_train_time"],
        "generation_time" : prepared["timing"]["generation_time"],
        "quality_info"  : quality_info,
    }
    return X_out, y_out, info


# ════════════════════════════════════════════════════════════
#  R8-2 ADDITION: unfiltered G-CTGAN baseline
# ════════════════════════════════════════════════════════════
# "G-CTGAN" = the Q-G-CTGAN generative base (GMM clustering + per-cluster
# CTGAN + target-aware candidate generation) with the ENTIRE two-stage
# quality filter removed: synthetic samples are drawn uniformly at random
# from the generated candidate pool, with no Stage-1 Q(x) rejection and no
# Stage-2 MMD selection. It shares the SAME prepared CTGANs/candidates as
# Q-G-CTGAN (pass the same `prepared` dict from qgctgan_prepare), so the
# full-vs-unfiltered comparison (Reviewer #8, comment 2) isolates exactly
# the contribution of the filter and nothing else.

def gctgan_unfiltered(prepared, random_state=RANDOM_STATE):
    X_min = prepared["X_min"]; X_maj = prepared["X_maj"]
    candidates = prepared["candidates"]
    n_maj, n_min = len(X_maj), len(X_min)
    # identical target as the filter path (50% balance, rounded)
    n_syn_needed = max(int(np.round(
        (TARGET_RATIO * (n_maj + n_min) - n_min) / (1 - TARGET_RATIO)
    )), 0)

    pool = [v for v in candidates.values() if len(v) > 0]
    if not pool:
        X_out = np.vstack([X_maj, X_min])
        y_out = np.concatenate([np.zeros(n_maj), np.ones(n_min)]).astype(int)
        return X_out, y_out, {"n_synthetic": 0, "n_syn_target": n_syn_needed,
                              "selection": "none_available"}
    X_pool = np.vstack(pool)

    rng = np.random.default_rng(random_state)
    if len(X_pool) <= n_syn_needed:
        X_syn = X_pool                                  # take all; no surplus to choose from
    else:
        idx = rng.choice(len(X_pool), size=n_syn_needed, replace=False)  # uniform random, NO filtering
        X_syn = X_pool[idx]

    if len(X_syn) > 0:
        X_out = np.vstack([X_maj, X_min, X_syn])
        y_out = np.concatenate([np.zeros(n_maj), np.ones(n_min), np.ones(len(X_syn))]).astype(int)
    else:
        X_out = np.vstack([X_maj, X_min])
        y_out = np.concatenate([np.zeros(n_maj), np.ones(n_min)]).astype(int)

    info = {
        "best_k": prepared["best_k"],
        "cluster_sizes": prepared["cluster_sizes"].tolist(),
        "n_candidates": prepared["n_cand_total"],
        "candidate_multiplier_used": prepared["candidate_multiplier_used"],
        "n_syn_target": n_syn_needed,
        "n_synthetic": int(len(X_syn)),
        "minority_ratio": round((y_out == 1).sum() / len(y_out), 4),
        "selection": "uniform_random_unfiltered",
    }
    return X_out, y_out, info
