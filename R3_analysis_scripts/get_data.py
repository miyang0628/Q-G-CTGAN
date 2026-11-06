import pandas as pd, numpy as np, io, urllib.request
out='/home/claude/rev_data'
raw=urllib.request.urlopen('https://www.openml.org/data/get_csv/37/dataset_37_diabetes.arff',timeout=60).read().decode()
p=pd.read_csv(io.StringIO(raw)); print(p.columns.tolist(), p.shape)
p['target']=(p['class'].str.strip("'")=='tested_positive').astype(int); p=p.drop(columns='class')
p.to_csv(f'{out}/pima_diabetes.csv',index=False); print('pima',p.shape,p.target.sum())
from imblearn.datasets import fetch_datasets
d=fetch_datasets(filter_data=('yeast_me2',),data_home=out)['yeast_me2']
y=pd.DataFrame(d.data,columns=[f'x{i}' for i in range(d.data.shape[1])]); y['target']=(d.target==1).astype(int)
y.to_csv(f'{out}/yeast_me2.csv',index=False); print('yeast',y.shape,y.target.sum())
