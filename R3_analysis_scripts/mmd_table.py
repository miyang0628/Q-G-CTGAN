import json,re,pandas as pd,numpy as np
from scipy import stats
j=json.load(open('/home/claude/r2/notebooks/03_run_qgctgan_and_gctgan.ipynb'))
txt=''.join(''.join(o.get('text',[])) for c in j['cells'] for o in c.get('outputs',[]))
rows=[];last=None
for line in txt.splitlines():
    m=re.search(r'MMD\^2 before=([\d.]+)\s+after=([\d.]+)',line)
    if m: last=(float(m.group(1)),float(m.group(2)))
    m=re.search(r'\[ok\]\s+Q-G-CTGAN\s+seed=(\d+)\s+(\S+)',line)
    if m and last: rows.append((m.group(2),int(m.group(1)),*last)); last=None
d=pd.DataFrame(rows,columns=['dataset','seed','before','after']); d['delta']=d.after-d.before
assert d.groupby('dataset').seed.nunique().eq(5).all() and len(d)==80
d.to_csv('/home/claude/rev/mmd_per_seed.csv',index=False)
def ci(v): v=np.asarray(v);return stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))
g=d.groupby('dataset').agg(before=('before','mean'),before_ci=('before',ci),after=('after','mean'),after_ci=('after',ci),
    delta=('delta','mean'),delta_ci=('delta',ci),seeds_improved=('delta',lambda s:(s<0).sum())).sort_values('delta')
print(g.round(4).to_string())
print('improved datasets (mean delta<0):',(g.delta<0).sum(),'/16; seed-level improved:',(d.delta<0).sum(),'/80')
print('mean delta over datasets',round(g.delta.mean(),4),'CI',round(ci(g.delta),4),'wilcoxon p',stats.wilcoxon(g.delta).pvalue)
print(d[d.dataset.isin(['pima_diabetes','yeast_me2'])])
g.to_csv('/home/claude/rev/mmd_summary.csv')
