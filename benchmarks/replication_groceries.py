"""Out-of-sample replication on real supermarket baskets (Groceries, 9,835 baskets).

Split baskets randomly into halves A and B. Each method finds single-consequent rules on A.
We then test every rule on B (one-sided Fisher exact test that the consequent is more likely
when the antecedent is present) and report the share that replicate (p < 0.05) and how much the
lift shrinks. Classical = support/confidence/lift thresholds with no statistical testing.
"""
import csv, os, sys, time, warnings
import numpy as np, pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from rulewise import RuleMiner
from rulewise.stats import fisher_greater
from rulewise.mining import build_vocab, encode
from rulewise.rules import TidCounter, productivity_pvalue
warnings.filterwarnings("ignore")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def load(path):
    out = []
    for row in csv.reader(open(path, encoding="utf-8")):
        items = sorted({x.strip() for x in row if x.strip()})
        if items: out.append(items)
    return out

def classical(tx, min_support=0.005, min_conf=0.25, max_ant=3):
    items = sorted({i for t in tx for i in t}); idx = {i: j for j, i in enumerate(items)}
    M = np.zeros((len(tx), len(items)), bool)
    for r, t in enumerate(tx):
        for i in t: M[r, idx[i]] = True
    fi = fpgrowth(pd.DataFrame(M, columns=items), min_support=min_support, use_colnames=True, max_len=max_ant + 1)
    r = association_rules(fi, metric="confidence", min_threshold=min_conf, num_itemsets=len(tx))
    r = r[(r.consequents.map(len) == 1) & (r.antecedents.map(len) <= max_ant) & (r.lift > 1)]
    return [(tuple(sorted(a)), next(iter(c)), l, cf) for a, c, l, cf in zip(r.antecedents, r.consequents, r.lift, r.confidence)]

def productive_share(tx, rules, vocab):
    """Share of rules that are still PRODUCTIVE on tx: beat every simpler version of themselves
    (Webb's productivity test, one-sided, p < 0.05), i.e. not explained by a shorter rule."""
    tids, n = encode(tx, vocab); tc = TidCounter(tids, n)
    ok = [productivity_pvalue(frozenset(vocab[i] for i in ant), frozenset([vocab[y]]), tc, 0.05) < 0.05
          for ant, y, *_ in rules]
    return float(np.mean(ok)) if ok else float("nan")

def evaluate_on(tx, rules):
    """rules: list of (antecedent tuple, consequent item, lift_on_A). Returns replication stats on tx."""
    n = len(tx); sets = [set(t) for t in tx]; ps, shrink = [], []
    cache = {}
    def cnt(s):
        if s not in cache: cache[s] = sum(1 for t in sets if s <= t)
        return cache[s]
    for ant, y, lift_a in rules:
        X = frozenset(ant); nx, ny, nxy = cnt(X), cnt(frozenset([y])), cnt(X | {y})
        ps.append(fisher_greater(nxy, nx - nxy, ny - nxy, n - nx - ny + nxy))
        lift_b = (nxy / nx) / (ny / n) if nx and ny else float("nan")
        shrink.append(lift_b / lift_a)
    ps = np.array(ps)
    return dict(n=len(rules), replicated=float((ps < 0.05).mean()) if len(ps) else float("nan"),
                prod_B=productive_share(tx, rules, VOCAB), prod_A=float("nan"),
                lift_ratio=float(np.nanmedian(shrink)) if shrink else float("nan"))

tx = load(os.path.join(DATA, "groceries.csv"))
VOCAB = build_vocab(tx)
print(f"{len(tx)} baskets, {len({i for t in tx for i in t})} distinct items, mean basket size {np.mean([len(t) for t in tx]):.1f}")
seeds = range(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
rows = {k: [] for k in ("classical (all)", "classical lift>=2", "classical top-K by lift", "classical top-K by confidence", "rulewise holdout", "rulewise direct")}
for sd in seeds:
    rng = np.random.default_rng(sd); perm = rng.permutation(len(tx)); half = len(tx) // 2
    A = [tx[i] for i in perm[:half]]; B = [tx[i] for i in perm[half:]]
    rw = {}
    for mode in ("holdout", "direct"):
        m = RuleMiner(min_support=0.005, max_antecedent=3, mode=mode, n_bootstrap=0, random_state=sd).fit(A)
        rw[mode] = [(tuple(sorted(r.antecedent)), r.consequent[0], r.lift) for r in m.rules_.itertuples()]
    cl = classical(A)
    K = max(len(rw["direct"]), 1)
    by_lift = sorted(cl, key=lambda r: -r[2])[:K]; by_conf = sorted(cl, key=lambda r: -r[3])[:K]
    hi = [r for r in cl if r[2] >= 2.0]
    for name, rs in (("classical (all)", cl), ("classical lift>=2", hi), ("classical top-K by lift", by_lift), ("classical top-K by confidence", by_conf),
                     ("rulewise holdout", rw["holdout"]), ("rulewise direct", rw["direct"])):
        rows[name].append(evaluate_on(B, [(a, y, l) for a, y, l, *_ in rs] if name.startswith("classical") else rs))
    print(f"seed {sd}: classical={len(cl)} rules, rulewise direct={len(rw['direct'])}, holdout={len(rw['holdout'])}", flush=True)
print(f"\n{'method':<32}{'rules':>8}{'lift>1 on B':>13}{'productive on B':>17}{'median lift B/A':>17}{'productive rules on B':>23}")
for name, rs in rows.items():
    n = np.mean([r["n"] for r in rs]); rep = np.nanmean([r["replicated"] for r in rs]); rep_sd = np.nanstd([r["replicated"] for r in rs])
    lr = np.nanmean([r["lift_ratio"] for r in rs]); pb = np.nanmean([r["prod_B"] for r in rs])
    print(f"{name:<32}{n:>8.0f}{rep:>13.2f}{pb:>17.2f}{lr:>17.2f}{n*pb:>23.0f}")
