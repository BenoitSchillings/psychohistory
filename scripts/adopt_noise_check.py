"""Where does the adopt rule's noise come from? (11 November 2026; a diagnostic, it changes nothing)

For every baseline / variant pair of hindcast tags on disk: the pooled composite, the composite on each of the three
ensemble refits, its spread between refits, a 90% interval from resampling the 16 origins, and the four decades.
python -P scripts/adopt_noise_check.py"""
import sys, json, numpy as np
import os
sys.path.insert(0, os.path.dirname(__file__))
import adopt_score as A
pairs=[("nf_base",v) for v in "nf_alignt nf_gr18 nf_mood20 nf_noabs nf_nofrail nf_nopool nf_oc nf_ocd nf_oilold nf_scope nf_split".split()]
pairs+=[("v55c_base","v55c_cslope"),("v55f_base","v55f_coupoff"),("v55f_base","v55f_harv"),("v55f_base","v55f_preston"),("v57_base","v57_pnet"),("v57_base","v57r_base"),("v57r_base","v58_base")]
O=A.ORIGINS; rng=np.random.default_rng(0)
print("%-28s pooled  per-refit              sd_refit  unpaired_floor  origin-boot 90%%      dec"%"pair")
sds=[]
for b,v in pairs:
    w,nf=A.active_weights(b); 
    if nf is None: w=dict(A.WEIGHTS)
    Ta={s:A.table(b,[s],O) for s in (0,1,2)}; Tb={s:A.table(v,[s],O) for s in (0,1,2)}
    Am,Bm=A.table(b,[0,1,2],O),A.table(v,[0,1,2],O)
    pooled=A.composite(Am,Bm,O,w)[1]
    per=[A.composite(Ta[s],Tb[s],O,w)[1] for s in (0,1,2)]
    boot=[]
    for _ in range(1000):
        oo=list(rng.choice(O,len(O))); boot.append(A.composite(Am,Bm,oo,w)[1])
    dec=[A.composite(Am,Bm,[o for o in O if o//10==d],w)[1] for d in (197,198,199,200)]
    sds.append(np.std(per,ddof=1))
    print("%-28s %+.2f%%  %s   %.2f%%     %s        [%+.2f, %+.2f]   %s"%(v+" / "+b[:9],pooled*100," ".join("%+.2f"%(x*100) for x in per),np.std(per,ddof=1)*100,("%.2f%%"%(nf["composite_floor"]*100)) if nf else "  -  ",*np.percentile(boot,[5,95])*100," ".join("%+.1f"%(x*100) for x in dec)))
print("median sd of the paired composite between refits: %.2f%%; 2*sd/sqrt(3) = %.2f%%"%(np.median(sds)*100,2*np.median(sds)/3**.5*100))
