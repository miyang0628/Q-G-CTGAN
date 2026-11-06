import pandas as pd
r=pd.read_csv('/home/claude/rev/ablation_paired.csv')
LAB={'comp_full - comp_density':r'Composite $-$ $S_{\text{density}}$ only','comp_full - comp_sep':r'Composite $-$ $S_{\text{sep}}$ only',
'comp_full - comp_lof':r'Composite $-$ $S_{\text{lof}}$ only','alpha_adaptive - alpha_fixed_0.1':r'Adaptive $-$ fixed $\alpha=0.1$',
'alpha_adaptive - alpha_fixed_0.2':r'Adaptive $-$ fixed $\alpha=0.2$','alpha_adaptive - alpha_fixed_0.3':r'Adaptive $-$ fixed $\alpha=0.3$',
'mmd_on - mmd_off':r'MMD stage on $-$ off'}
GRP=[('(A) Quality-score components (fixed $\\alpha=0.2$, MMD on)',list(LAB)[:3]),
     ('(B) Threshold (composite score, MMD on)',list(LAB)[3:6]),('(C) MMD Stage 2 (composite score, adaptive $\\alpha$)',list(LAB)[6:])]
def sgn(x):
    if abs(x)<5e-5: return '0.0000'
    return ('$+$' if x>=0 else '$-$')+f'{abs(x):.4f}'
def fp(p): return f'{p:.3f}' if p>=0.001 else '$<$0.001'
def table(metric,label,caption,pos='t'):
    out=[r'\begin{table}['+pos+']',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\resizebox{\textwidth}{!}{%',
         r'\begin{tabular}{llcccc}',r'\toprule',
         r'Comparison & Clf. & Mean $\Delta$ & 95\% CI of $\Delta$ & $p$ (Holm) & W/T/L \\',r'\midrule']
    for gi,(g,comps) in enumerate(GRP):
        if gi: out.append(r'\midrule')
        out.append(r'\multicolumn{6}{l}{\emph{'+g+r'}} \\')
        for c in comps:
            for k,clf in enumerate(['RF','LGBM','MLP']):
                x=r[(r.metric==metric)&(r.clf==clf)&(r.comparison==c)].iloc[0]
                name=LAB[c] if k==0 else ''
                star='$^{*}$' if x.wilcoxon_p<0.05 else ''
                out.append(f'{name} & {clf} & {sgn(x.mean_diff)} & [{sgn(x.ci_lo)}, {sgn(x.ci_hi)}] & '
                           f'{fp(x.wilcoxon_p)}{star} ({fp(x.p_holm)}) & {int(x['W'])}/{int(x['T'])}/{int(x['L'])} \\\\')
    out+= [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
    return '\n'.join(out)
CAP=(r'Paired ablation differences in {M}, separately per classifier. For every (dataset, seed) pair the two '
     r'configurations share the identical generative base (same split, GMM, per-cluster CTGANs and candidate pool), '
     r'so $\Delta$ (first minus second configuration) is computed within each pair, averaged over the five seeds of '
     r'each dataset, and summarised over the 16 datasets: mean, 95\% CI (Student-$t$, $\mathrm{{df}}=15$), Wilcoxon '
     r'signed-rank $p$ with the Holm-adjusted $p$ (seven comparisons per classifier) in parentheses, and win/tie/loss '
     r'counts ($|\Delta|\le5\times10^{{-5}}$ counted as a tie). $^{{*}}$Unadjusted $p<0.05$.')
open('/home/claude/rev/abl_auc.tex','w').write(table('AUC','tab:ablation_paired',CAP.format(M='AUC')))
open('/home/claude/rev/abl_f1.tex','w').write(table('F1','tab:ablation_paired_f1',CAP.format(M='macro-F1'),'htbp'))
open('/home/claude/rev/abl_gm.tex','w').write(table('G_mean','tab:ablation_paired_gmean',CAP.format(M='G-mean'),'htbp'))
print(open('/home/claude/rev/abl_auc.tex').read()[:2500])
