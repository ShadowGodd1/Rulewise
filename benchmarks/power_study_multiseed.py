"""Multi-seed power comparison (holdout vs direct). Slow: ~15-25 min on one core. Run detached."""
import sys, time; sys.path.insert(0,'benchmarks')
import numpy as np
from power_study import make, evaluate
from rulewise import RuleMiner
def cell(n, s, n_items, seeds, ms):
    out={"holdout":[], "direct":[]}; fl={"holdout":0,"direct":0}
    for sd in seeds:
        tx,pats=make(n,s,seed=sd,n_items=n_items)
        for mode in out:
            m=RuleMiner(min_support=ms,min_confidence=0.2,mode=mode,n_bootstrap=0,random_state=sd).fit(tx)
            rec,nr,nf=evaluate(m,pats); out[mode].append(rec); fl[mode]+=nf
    return out, fl
print(f"{'setting':<34} {'holdout mean+/-sd':>16} {'direct mean+/-sd':>16} {'false h/d':>10}")
for label,(n,s,ni,seeds,ms) in {
  "n=5k  items=1k  strength=.002": (5000,0.002,1000,range(6),0.0005),
  "n=5k  items=1k  strength=.004": (5000,0.004,1000,range(6),0.0005),
  "n=20k items=1k  strength=.0006":(20000,0.0006,1000,range(4),0.0005),
  "n=20k items=10k strength=.001": (20000,0.001,10000,range(3),0.001),
  "n=20k items=10k strength=.002": (20000,0.002,10000,range(3),0.001),
}.items():
    t=time.time(); o,f=cell(n,s,ni,seeds,ms)
    h,d=np.array(o["holdout"]),np.array(o["direct"])
    print(f"{label:<34} {h.mean():>9.1f}+/-{h.std():<4.1f} {d.mean():>9.1f}+/-{d.std():<4.1f} {f['holdout']:>5}/{f['direct']:<4} ({time.time()-t:.0f}s)",flush=True)
