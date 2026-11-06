import pandas as pd, numpy as np
from scipy import stats
a=pd.read_csv('/home/claude/repo/results/tables/raw_ablation.csv')
comps=[('comp_full','comp_density'),('comp_full','comp_sep'),('comp_full','comp_lof'),
       ('alpha_adaptive','alpha_fixed_0.1'),('alpha_adaptive','alpha_fixed_0.2'),('alpha_adaptive','alpha_fixed_0.3'),
       ('mmd_on','mmd_off')]
rows=[]
for metric in ['AUC','F1','G_mean']:
  for clf in ['RF','LGBM','MLP']:
    s=a[a.classifier==clf].pivot_table(index=['dataset','seed'],columns='config',values=metric)
    for x,y in comps:
        d=(s[x]-s[y]).dropna()                    # paired at (dataset, seed)
        dd=d.groupby('dataset').mean()            # seed-averaged per dataset
        n=len(dd); m=dd.mean(); h=stats.t.ppf(.975,n-1)*dd.std(ddof=1)/np.sqrt(n)
        nz=dd[dd.abs()>1e-12]
        p=stats.wilcoxon(nz).pvalue if len(nz)>0 else 1.0
        w=(dd>5e-5).sum(); l=(dd<-5e-5).sum()
        rows.append(dict(metric=metric,clf=clf,comparison=f'{x} - {y}',n_pairs=len(d),n_ds=n,mean_diff=m,ci_lo=m-h,ci_hi=m+h,wilcoxon_p=p,W=w,T=n-w-l,L=l))
r=pd.DataFrame(rows)
# Holm within (metric, clf) family of 7 comparisons
def holm(p):
    o=np.argsort(p); adj=np.empty(len(p)); run=0
    for k,i in enumerate(o): run=max(run,(len(p)-k)*p[i]); adj[i]=min(1,run)
    return adj
r['p_holm']=r.groupby(['metric','clf']).wilcoxon_p.transform(lambda s: holm(s.values))
r.to_csv('/home/claude/rev/ablation_paired.csv',index=False)
pd.set_option('display.width',200)
print(r.round(4).to_string(index=False))
