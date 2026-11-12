"""Empirical error control on pure-noise data: how often does rulewise report ANY rule?

Under a true null (no associations) the family-wise error rate should be at most alpha (0.05).
Each cell runs many independent null datasets and reports the share with at least one rule,
with a 95% Wilson interval. Usage: python error_control_simulation.py [n_runs_per_config]
"""
import sys, time, warnings
import numpy as np
from rulewise import RuleMiner, RuleClassifier
from rulewise.stats import wilson_interval
warnings.filterwarnings("ignore")
R = int(sys.argv[1]) if len(sys.argv) > 1 else 300

def null_baskets(seed, n=3000, m=25):
    rng = np.random.default_rng(seed); p = rng.uniform(0.03, 0.4, m)       # varied item frequencies
    M = rng.random((n, m)) < p
    return [[f"i{j}" for j in range(m) if row[j]] for row in M]
def null_xy(seed, n=3000, m=15):
    rng = np.random.default_rng(seed); X = rng.random((n, m)) < rng.uniform(0.05, 0.5, m)
    return X, rng.random(n) < rng.uniform(0.1, 0.5)                          # label independent of X

configs = {
  "RuleMiner holdout/holm": lambda s: len(RuleMiner(min_support=0.01, n_bootstrap=0, random_state=s).fit(null_baskets(s)).rules_),
  "RuleMiner holdout/BH(fdr)": lambda s: len(RuleMiner(min_support=0.01, n_bootstrap=0, correction="bh", random_state=s).fit(null_baskets(s)).rules_),
  "RuleMiner direct (layered)": lambda s: len(RuleMiner(min_support=0.01, n_bootstrap=0, mode="direct").fit(null_baskets(s)).rules_),
  "RuleMiner 2-item consequents": lambda s: len(RuleMiner(min_support=0.01, n_bootstrap=0, max_consequent=2, random_state=s).fit(null_baskets(s)).rules_),
  "RuleClassifier direct": lambda s: len(RuleClassifier(min_support=0.01, random_state=s).fit(*null_xy(s)).rules_),
  "RuleClassifier holdout": lambda s: len(RuleClassifier(min_support=0.01, mode="holdout", random_state=s).fit(*null_xy(s)).rules_),
}
print(f"{R} null datasets per configuration; nominal alpha = 0.05\n")
print(f"{'configuration':<32}{'datasets with >=1 rule':>24}{'rate':>8}{'95% CI':>14}{'mean rules':>12}{'sec':>7}", flush=True)
for name, fn in configs.items():
    t = time.time(); counts = np.array([fn(10_000 + i) for i in range(R)]); hits = int((counts > 0).sum())
    lo, hi = wilson_interval(hits, R, 0.05)
    print(f"{name:<32}{f'{hits}/{R}':>24}{hits/R:>8.3f}{f'{lo:.3f}-{hi:.3f}':>14}{counts.mean():>12.3f}{time.time()-t:>7.0f}", flush=True)
