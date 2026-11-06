import pandas as pd, numpy as np, re, sys
from scipy import stats
d=pd.read_csv('/home/claude/repo/results/tables/05_multiseed_raw.csv'); d['method']=d.method.fillna('None')
METH=['None','SMOTE','ADASYN','G-SMOTE','KMeans_CTGAN','CTGAN_MOS','ctdGAN','CTAB-GAN+','Q-G-CTGAN']
AVG_METH=[m for m in METH if m!='CTAB-GAN+']
DS=[('credit_default','Credit Default'),('fraud_detection','Fraud Detection'),('pima_diabetes','Pima Diabetes'),
('ibm_attrition','IBM HR Attrition'),('churn','Churn'),('ecoli','Ecoli'),('satellite','Satellite'),('secom','SECOM'),
('thyroid_sick','Thyroid Sick'),('wine_quality','Wine Quality'),('yeast_me2','Yeast ME2'),('mammography','Mammography'),
('abalone_19','Abalone 19'),('pageblocks','PageBlocks'),('protein_homo','Protein Homology'),('unsw_nb15','UNSW-NB15')]
def tcrit(n): return stats.t.ppf(.975,n-1)
def cell(clf,metric):
    g=d[d.classifier==clf].groupby(['dataset','method'])[metric]
    return g.mean(), g.apply(lambda v: tcrit(len(v))*v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else np.nan), g.size()
def avg_ci(clf,metric,m):
    """CI of the 16-dataset average over the 5 seeds: average per seed across datasets, CI over seeds.
    Cells with fewer than 5 seeds contribute their available-seed mean to every seed's average."""
    s=d[(d.classifier==clf)&(d.method==m)]
    piv=s.pivot_table(index='dataset',columns='seed',values=metric)
    piv=piv.apply(lambda r: r.fillna(r.mean()),axis=1)
    ps=piv.mean(0)
    disp=s.groupby('dataset')[metric].mean().round(4).mean()   # paper convention: mean of rounded cell means
    return disp, tcrit(len(ps))*ps.std(ddof=1)/np.sqrt(len(ps))
def check(texfile,label,clf,metric):
    tex=open(texfile).read(); i=tex.find('\\label{'+label+'}'); body=tex[i:tex.find('\\end{table}',i)]
    mean,_,_=cell(clf,metric); bad=0
    for key,name in DS:
        row=re.search(re.escape(name)+r'\s*&(.*?)\\\\',body).group(1)
        vals=[re.sub(r'\\textbf\{|\\underline\{|\}','',v).strip() for v in row.split('&')]
        for m,v in zip(METH,vals):
            if v=='N/A': ok=(key,m) not in mean.index
            else: ok=abs(round(mean[(key,m)],4)-float(v))<1e-9
            if not ok: bad+=1; print('MISMATCH',label,name,m,v,mean.get((key,m)))
    print(label,'mismatches:',bad)
if __name__=='__main__':
    E='/home/claude/ms/sections/04_experiments.tex'; A='/home/claude/ms/sections/06_appendix.tex'
    for f,l,c,m in [(E,'tab:auc_rf','RF','AUC'),(E,'tab:auc_lgbm','LGBM','AUC'),(E,'tab:auc_mlp','MLP','AUC'),
                    (A,'tab:f1_rf','RF','F1'),(A,'tab:prauc_rf','RF','PR_AUC'),(A,'tab:gmean_rf','RF','G_mean'),(A,'tab:bacc_rf','RF','Balanced_Acc')]:
        check(f,l,c,m)

HDR=r"""\toprule
Dataset & None & SMOTE & ADASYN & G-SMOTE & K-means & CTGAN- & ctdGAN & CTAB- & \textbf{Q-G-} \\
        &      &       &        &         & CTGAN   & MOS    &        & GAN+  & \textbf{CTGAN} \\
\midrule
"""
NOTE=(r"Each cell gives the mean over the five seeds (42--46) and, below it, the 95\% CI half-width "
      r"(Student-$t$, $\mathrm{df}=n-1$). $^{a}$Single seed only (no CI; Section~\ref{subsubsec:protocol}); "
      r"$^{b}$three seeds ($\mathrm{df}=2$). N/A: not evaluated (Table~\ref{tab:ctabgan-coverage}).")
def build(label,clf,metric,caption,avg_note=True):
    mean,ci,n=cell(clf,metric)
    lines=[]
    for key,name in DS:
        vals={m:mean[(key,m)] for m in METH if (key,m) in mean.index}
        order=sorted(vals,key=lambda m:-vals[m]); best,sec=order[0],order[1]
        cells=[]
        for m in METH:
            if m not in vals: cells.append('N/A'); continue
            v=f'{vals[m]:.4f}'
            if m==best: v=r'\textbf{'+v+'}'
            elif m==sec: v=r'\underline{'+v+'}'
            k=int(n[(key,m)])
            if k==1: cells.append(r'\begin{tabular}[t]{@{}c@{}}'+v+r'$^{a}$\\[0pt]{\scriptsize ($n{=}1$)}\end{tabular}')
            else: cells.append(r'\mci{'+v+('$^{b}$' if k==3 else '')+'}{'+f'{ci[(key,m)]:.4f}'+'}')
        lines.append(name+' & '+' & '.join(cells)+r' \\[1pt]')
    av={m:avg_ci(clf,metric,m) for m in AVG_METH}
    bm=max(av,key=lambda m:av[m][0])
    acells=[]
    for m in METH:
        if m=='CTAB-GAN+': acells.append('---'); continue
        v=f'{av[m][0]:.4f}'; v=r'\textbf{'+v+'}' if m==bm else v
        acells.append(r'\mci{'+v+(r'$^{\ddagger}$' if m=='ctdGAN' else '')+'}{'+f'{av[m][1]:.4f}'+'}')
    foot=(r"$^{\dagger}$Average over all 16 datasets; CTAB-GAN+ is excluded from the averaged column since it is "
          r"not evaluated on all 16 datasets. The CI of the average is computed over the five per-seed "
          r"16-dataset averages. $^{\ddagger}$For ctdGAN the three single-seed cells enter every per-seed "
          r"average with their single value, so their run-to-run variability is not reflected in this CI.")
    return (r"\begin{sidewaystable}"+"\n"+r"\centering"+"\n"+r"\caption{"+caption+" "+NOTE+"}\n"+r"\label{"+label+"}\n"
            +r"\resizebox{\textheight}{!}{%"+"\n"+r"\begin{tabular}{lccccccccc}"+"\n"+HDR
            +"\n".join(lines)+"\n"+r"\midrule"+"\n"+r"\textbf{Average}$^{\dagger}$ & "+' & '.join(acells)+r" \\"+"\n"
            +r"\bottomrule"+"\n"+r"\end{tabular}}"+"\n"+r"\vspace{0.3em}"+"\n"+r"\footnotesize{"+foot+"}\n"+r"\end{sidewaystable}")
def replace_table(texfile,label,newtab):
    tex=open(texfile).read(); i=tex.find('\\label{'+label+'}')
    s=max(tex.rfind('\\begin{table}',0,i),tex.rfind('\\begin{sidewaystable}',0,i))
    endtag='\\end{sidewaystable}' if tex[s:s+20].startswith('\\begin{sideways') else '\\end{table}'
    e=tex.find(endtag,i)+len(endtag)
    open(texfile,'w').write(tex[:s]+newtab+tex[e:])
