import pickle, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
names={'pima_diabetes':'Pima Diabetes','yeast_me2':'Yeast ME2'}
for ds in ['pima_diabetes','yeast_me2']:
    r=pickle.load(open(f'/home/claude/rev/run_{ds}_42.pkl','rb')); info=r['info']
    fig,axes=plt.subplots(1,2,figsize=(13,5.4))
    for ax,(S,lab,m) in zip(axes,[(r['cand'],'Before filtering (all candidates)',info['mmd_before']),
                                  (r['sel'],'After two-stage filtering',info['mmd_after'])]):
        Z=np.vstack([S,r['X_min']]); n=len(Z)
        E=TSNE(n_components=2,perplexity=min(30,(n-1)/3),random_state=42,init='pca').fit_transform(Z)
        ax.scatter(E[:len(S),0],E[:len(S),1],s=14,c='#F5C99B',alpha=.65,label=f'Synthetic (n={len(S)})',edgecolors='none')
        ax.scatter(E[len(S):,0],E[len(S):,1],s=26,c='#D7372A',alpha=.9,label=f'Real minority (n={len(r["X_min"])})',edgecolors='none')
        ax.set_title(f'{lab}\nMMD$^2$ = {m:.4f}',fontsize=12,fontweight='bold'); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        ax.legend(loc='upper right',fontsize=9,frameon=True)
    fig.suptitle(f'{names[ds]} — single run, split seed 42 (adaptive $\\alpha$, MMD stage on, multiplier {info["candidate_multiplier_used"]}$\\times$)',fontsize=12)
    fig.tight_layout(); fig.savefig(f'/home/claude/ms/figures/fig05_tsne_{ds}.png',dpi=200); plt.close(fig)
    print(ds,info['mmd_before'],info['mmd_after'],len(r['cand']),len(r['sel']))
