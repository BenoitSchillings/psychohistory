"""Should the growth equation keep its squared income-gap term? (11 November 2026)

Background (docs/world_system.md, 11 November): the hindcasts make poor countries grow too slowly relative to rich
ones by 1.3 points a year, about 60% of it in the growth equation itself. An exploratory screen of 16 specifications
(this script's one-step check: the equation fitted at each of the 16 origins and fed the realised features and events
of the next 15 years) showed that the gap is mostly rich and middle countries over-forecast (+1.0 and +0.8 points a
year; poor +0.1), and that one change reduces it without costing accuracy: dropping the squared income gap
(poor minus rich -0.84 -> -0.50; mean absolute error of 15-year growth 1.69 -> 1.65). The squared term was fitted on
the divergence of 1960-1995, when the poorest countries fell further behind; it carries that forward.
The screen chose the candidate, so it is not evidence. This file is written and committed before the hindcasts.

Test. `bash scripts/hindcast_tag.sh v58_nosq --consts '{"simulate.GROWTH_SQUARES": []}'`, scored against v58_base with
scripts/adopt_score.py (docs/adopt_rule.md, unchanged), and scripts/convergence_bias_test.py on refit 0 against
v58c_s0.

Pre-registered reading.
  Adopt if the adopt rule passes (composite better than the floor, 3 of 4 decades, no weighted statistic worse by 3%).
  If the adopt rule does not pass but no guardrail is broken, the composite is not worse than the floor, AND the
  poor-minus-rich error on refit 0 shrinks by at least 0.3 points a year with the simulator's mean absolute error of
  country growth not worse: adopt as a bias fix (a known defect removed at no measured cost), labelled as such.
  Otherwise not adopted, and the squared term stays.
python -P scripts/growth_square_test.py -> outputs/growth_square_screen.json (the one-step check, both specifications)"""
import numpy as np, torch, warnings, sys
warnings.simplefilter("ignore")
from psychohistory.pipeline import prepare
from psychohistory import simulate as SIM
from psychohistory.features import FEATURES
p, base, X, Yl = prepare(); lg=base["lg"]; conf=base["conflict"]; coup=base["coup_attempts"]; inter=base["interstate"]
ORIG=(1970,1973,1976,1979,1980,1983,1986,1989,1990,1993,1996,1999,2000,2003,2006,2009)
BASEF=list(SIM.GROWTH_FEATURES)
def fit(O, cols, squares, alpha=10.0):
    C,Yn,F=X.shape; t=np.arange(Yn-1)
    years=np.broadcast_to(p.years[t],(C,len(t))); codes=np.broadcast_to(np.arange(C)[:,None],(C,len(t)))
    regions=np.broadcast_to(np.asarray(p.regions)[:,None],(C,len(t)))
    ok=(years>=1955)&(years+1<=O)
    on=np.where(np.isfinite(conf[:,1:])&np.isfinite(conf[:,:-1]),(conf[:,1:]==1)&(conf[:,:-1]==0),np.nan)
    ca=base["coup_attempts"][:,1:]; cn=np.where(np.isfinite(ca),ca>0,np.nan)
    it=base["interstate"][:,1:]; itn=np.where(np.isfinite(it),np.nan_to_num(it),np.nan)
    f=lambda a:a[ok]
    ev=np.stack([f(conf[:,1:]),f(on.astype(float)),f(cn.astype(float)),f(itn)],1)
    return SIM.Transition.fit(X[:,:-1][ok],ev,f(lg[:,1:]-lg[:,:-1]),f(years),f(codes),f(regions),cols,-0.4,0.3,
        bin_col="gdp_gap_frontier",grid=np.arange(1955,O),squares=squares,alpha=alpha,size=SIM.SIZE_NOISE,vol=SIM.VOL_NOISE,pool_state=SIM.POOL_STATE,
        shock_w=f(np.exp(SIM.YEAR_SHOCK_POW*(lg[:,:-1]+base["lpop"][:,:-1]-25))) if SIM.YEAR_SHOCK_POW else None)
OUT={}
def run(name, cols, squares, alpha=10.0, scale=None, decomp=False):
    tot=[];mae=[];dec={}
    for O in ORIG:
        gr=fit(O,cols,squares,alpha); gr.to("cpu"); i0=p.yi(O); T=15
        if scale:
            for c,s in scale.items():
                gr.w[cols.index(c)]*=s
                if c in squares: gr.w[len(cols)+squares.index(c)]*=s
        ok=np.isfinite(lg[:,i0:i0+T+1]).all(1)
        th=np.full(len(ok),-1); th[ok]=np.searchsorted(np.quantile(lg[ok,i0],[1/3,2/3]),lg[ok,i0])
        E=np.zeros((len(ok),T)); contrib=0
        for j in range(T):
            t=i0+j
            Xt=torch.as_tensor(X[:,t],dtype=torch.float32)[None]
            on=((conf[:,t+1]==1)&(conf[:,t]==0)).astype(float)
            ev=torch.as_tensor(np.nan_to_num(np.stack([conf[:,t+1],on,(coup[:,t+1]>0).astype(float),inter[:,t+1]],-1)),dtype=torch.float32)[None]
            Xs=Xt[...,gr.ci_t]; z=torch.nan_to_num(torch.minimum(torch.maximum((Xs-gr.mu_t)/gr.sd_t,gr.zlo_t),gr.zhi_t))
            D=gr._design(z,ev,gr.sq_t,torch)
            E[:,j]=(D@gr.w+gr.intercept)[0].numpy()-(lg[:,t+1]-lg[:,t])
            contrib=contrib+(D*gr.w)[0].numpy()/T
        e=E.mean(1)*100
        tot.append([e[th==k].mean() for k in range(3)]); mae.append(np.abs(e[ok]).mean())
        if decomp:
            names=cols+[c+"^2" for c in squares]+SIM.EVENTS
            d=(contrib[th==0].mean(0)-contrib[th==2].mean(0))*100
            for n_,v in zip(names,d): dec.setdefault(n_,[]).append(v)
            dec.setdefault("REALISED",[]).append((-(E.mean(1)*100)+contrib.sum(1)*100)[th==0].mean()-(-(E.mean(1)*100)+contrib.sum(1)*100)[th==2].mean())
    tot=np.array(tot); g=tot[:,0]-tot[:,2]
    dg=[g[i:i+4].mean() for i in (0,4,8,12)]
    OUT[name]={"poor_mid_rich":[round(float(v),3) for v in tot.mean(0)],"gap":round(float(g.mean()),3),"gap_by_decade":[round(float(v),3) for v in dg],"mae":round(float(np.mean(mae)),4)}
    print("%-28s poor %+.2f mid %+.2f rich %+.2f | gap %+.2f | by decade %s | MAE %.3f"%(name,*tot.mean(0),g.mean()," ".join("%+.2f"%v for v in dg),np.mean(mae)),flush=True)
    if decomp:
        for n_,v in dec.items(): print("    %-24s %+.2f   by decade %s"%(n_,np.mean(v)," ".join("%+.2f"%np.mean(v[i:i+4]) for i in (0,4,8,12))))
run("with the squared income gap", BASEF, ["gdp_gap_frontier"])
run("without", BASEF, [])
import json, os
json.dump(OUT, open(os.path.join(os.path.dirname(__file__), "..", "outputs", "growth_square_screen.json"), "w"), indent=1)
