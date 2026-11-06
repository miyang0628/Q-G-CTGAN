import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
names={'pima_diabetes':'Pima Diabetes','churn':'Churn','mammography':'Mammography','ecoli':'Ecoli','yeast_me2':'Yeast ME2',
 'satellite':'Satellite','fraud_detection':'Fraud Detection','secom':'SECOM','wine_quality':'Wine Quality','ibm_attrition':'IBM HR Attrition',
 'pageblocks':'PageBlocks','abalone_19':'Abalone 19','credit_default':'Credit Default','protein_homo':'Protein Homology','unsw_nb15':'UNSW-NB15','thyroid_sick':'Thyroid Sick'}
g=pd.read_csv('/home/claude/rev/mmd_summary.csv',index_col=0).sort_values('delta',ascending=False)
y=np.arange(len(g)); h=0.38
fig,ax=plt.subplots(figsize=(8.5,8))
ax.barh(y-h/2,g.before,h,xerr=g.before_ci,color='#9CCBEB',label='MMD$^2$ before (all candidates)',error_kw=dict(lw=.8,capsize=2,ecolor='#555'))
ax.barh(y+h/2,g.after,h,xerr=g.after_ci,color='#E4574A',label='MMD$^2$ after (selected samples)',error_kw=dict(lw=.8,capsize=2,ecolor='#555'))
xm=(g[['before','after']].values+g[['before_ci','after_ci']].values).max()
for i,(idx,r) in enumerate(g.iterrows()):
    ax.text(xm*1.03,i,f'{r.delta:+.4f} ({int(r.seeds_improved)}/5)',va='center',fontsize=8.5,color='#1a7f37' if r.delta<0 else '#b42318')
ax.set_yticks(y); ax.set_yticklabels([names[i] for i in g.index]); ax.set_xlim(0,xm*1.33)
ax.set_xlabel('MMD$^2$ (mean over seeds 42–46, error bars = 95% CI)')
n=(g.delta<0).sum()
ax.set_title(f'MMD$^2$ before vs. after two-stage filtering (5 seeds, 16 datasets)\nmean $\\Delta$ lower on {n}/16 datasets; label = mean $\\Delta$ (seeds improved)',fontsize=11)
ax.grid(axis='x',alpha=.3); ax.legend(loc='upper center',bbox_to_anchor=(0.5,-0.08),ncol=2,fontsize=9,frameon=False)
fig.tight_layout(); fig.savefig('/home/claude/ms/figures/fig11_mmd_summary.png',dpi=200)
