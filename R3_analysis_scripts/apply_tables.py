import sys; sys.path.insert(0,'/home/claude/rev')
from make_tables import *
E='/home/claude/ms/sections/04_experiments.tex'; A='/home/claude/ms/sections/06_appendix.tex'
specs=[(E,'tab:auc_rf','RF','AUC',r'AUC results with Random Forest classifier, 16 datasets, 9 methods. Best mean per dataset in \textbf{bold}, second-best \underline{underlined}.'),
 (E,'tab:auc_lgbm','LGBM','AUC',r'AUC results with LightGBM classifier, 16 datasets, 9 methods. Best mean per dataset in \textbf{bold}, second-best \underline{underlined}.'),
 (E,'tab:auc_mlp','MLP','AUC',r'AUC results with MLP classifier, 16 datasets, 9 methods. Best mean per dataset in \textbf{bold}, second-best \underline{underlined}.'),
 (A,'tab:f1_rf','RF','F1',r'Macro-averaged F1-score with Random Forest classifier, 16 datasets, 9 methods. Best mean per dataset in \textbf{bold}, second-best \underline{underlined}.'),
 (A,'tab:prauc_rf','RF','PR_AUC',r'PR-AUC with Random Forest classifier, 16 datasets, 9 methods. Best mean per dataset in \textbf{bold}, second-best \underline{underlined}.'),
 (A,'tab:gmean_rf','RF','G_mean',r'G-mean with Random Forest classifier, 16 datasets, 9 methods. Best mean per dataset in \textbf{bold}, second-best \underline{underlined}.'),
 (A,'tab:bacc_rf','RF','Balanced_Acc',r'Balanced Accuracy with Random Forest classifier, 16 datasets, 9 methods. Best mean per dataset in \textbf{bold}, second-best \underline{underlined}.')]
for f,l,c,m,cap in specs: replace_table(f,l,build(l,c,m,cap))
# summary table 7 values
for c in ['RF','LGBM','MLP']:
    print(c,{m:tuple(round(x,4) for x in avg_ci(c,'AUC',m)) for m in METH+['G-CTGAN']})
