import sys, numpy as np, pandas as pd
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
from psychohistory.data import get_panel
from psychohistory.features import *
from psychohistory.backtest import run_backtest, summarize
p=get_panel(); b=base_arrays(p); X=panel_features(p,b,'cuda'); Yl=labels(p,b)
new_groups={g:v for g,v in PASSTHROUGH.items()}
new_groups={"Climate shocks": GROUPS["Climate shocks"], **new_groups}
base_cols=[f for g,v in GROUPS.items() if g not in new_groups for f in v]
configs={'base':base_cols, **{f'+{g}':base_cols+v for g,v in new_groups.items()}, 'all':FEATURES}
res=[]
for name,cols in configs.items():
    Xa=X.copy(); drop=[FEATURES.index(f) for f in FEATURES if f not in cols]; Xa[...,drop]=np.nan
    bt=run_backtest(p,Xa,Yl,origins=range(1990,2016,5),members=6,verbose=False)
    s=summarize(bt); s['config']=name; res.append(s); print(name, 'done', flush=True)
    if name=='all': bt.to_parquet('outputs/backtest_all.parquet')
r=pd.concat(res); r.to_csv('outputs/ablation.csv',index=False)
pd.set_option('display.width',250)
for metric in ['auc','bss']:
    print(metric); print(r[r.model=='Seldon ensemble'].pivot(index='config',columns='task',values=metric).round(3).to_string())
print(r[r.config=='all'].pivot(index='model',columns='task',values='auc').round(3).to_string())
