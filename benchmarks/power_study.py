"""Power and false-discovery study: holdout vs direct mode across effect strength and sample size.

Ground truth: 20 planted 3-item patterns among 1000 Zipf-distributed items. A pattern is
'recovered' if at least one significant rule uses only that pattern's items. A rule is 'false'
if its items are not all inside one planted pattern.
"""
import sys, time
import numpy as np
from rulewise import RuleMiner


def make(n, strength, seed=0, n_items=1000, n_pat=20):
    rng = np.random.default_rng(seed)
    p = 1 / np.arange(1, n_items + 1) ** 0.9; p /= p.sum()
    pats = [tuple(int(x) for x in rng.choice(np.arange(5, 300), 3, replace=False)) for _ in range(n_pat)]
    lengths = np.maximum(1, rng.poisson(10, n))
    tx = []
    for L in lengths:
        t = set(rng.choice(n_items, size=L, p=p).tolist())
        tx.append(t)
    for pt in pats:
        for i in np.nonzero(rng.random(n) < strength)[0]:
            tx[i].update(pt)
    return [sorted(t) for t in tx], pats


def evaluate(m, pats):
    ps = [set(p) for p in pats]
    rules = [set(r.antecedent) | set(r.consequent) for r in m.rules_.itertuples()]
    rec = sum(any(r <= p for r in rules) for p in ps)
    false = sum(not any(r <= p for p in ps) for r in rules)
    return rec, len(rules), false


if __name__ == "__main__":
    sizes = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [20_000, 100_000]
    print(f"{'n':>8} {'strength':>9} {'mode':>8} {'recovered':>10} {'rules':>6} {'false':>6} {'sec':>6}")
    for n in sizes:
        for strength in (0.002, 0.005, 0.01, 0.03):
            tx, pats = make(n, strength)
            for mode in ("holdout", "direct"):
                t = time.time()
                m = RuleMiner(min_support=0.001, min_confidence=0.2, mode=mode, n_bootstrap=0).fit(tx)
                rec, nr, nf = evaluate(m, pats)
                print(f"{n:>8} {strength:>9} {mode:>8} {rec:>7}/20 {nr:>6} {nf:>6} {time.time()-t:>6.1f}", flush=True)
