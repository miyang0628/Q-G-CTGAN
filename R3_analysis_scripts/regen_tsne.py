"""Regenerate Q-G-CTGAN run for one (dataset, seed) exactly as notebook 03 does,
save candidates/selected samples for the t-SNE figure."""
import sys, os, numpy as np, pandas as pd, pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, '/home/claude/rev')
import qgctgan_core as Q
ds, seed = sys.argv[1], int(sys.argv[2])
df = pd.read_csv(f'/home/claude/rev_data/{ds}.csv')
X = df.drop(columns=['target']).values.astype(float); y = df['target'].values.astype(int)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
Xtr = StandardScaler().fit_transform(Xtr)
np.random.seed(seed)
import torch; torch.manual_seed(seed)
prep = Q.qgctgan_prepare(Xtr, ytr, random_state=seed)
Xq, yq, info = Q.qgctgan_filter(prep, alpha_mode='adaptive', mmd_stage_enabled=True)
cand = np.vstack([v for v in prep['candidates'].values() if len(v) > 0])
n_real = len(prep['X_maj']) + len(prep['X_min'])
sel = Xq[n_real:]
pickle.dump(dict(X_min=prep['X_min'], X_maj=prep['X_maj'], cand=cand, sel=sel, info=info),
            open(f'/home/claude/rev/run_{ds}_{seed}.pkl', 'wb'))
print('RESULT', ds, seed, info['mmd_before'], info['mmd_after'], info['best_k'], info['cluster_sizes'])
