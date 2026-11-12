"""Honest comparison of RuleClassifier against standard baselines on bundled sklearn datasets.

Settings are fixed in advance (not tuned on test folds). Numeric features are quantile-binned
inside each training fold. Repeated stratified 5-fold CV, 5 repeats.
"""
import sys, time, warnings
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_iris, load_digits
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KBinsDiscretizer, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from rulewise import RuleClassifier
warnings.filterwarnings("ignore")

def binner(bins): return KBinsDiscretizer(n_bins=bins, encode="onehot", strategy="quantile")
SETUPS = {  # name: (loader, bins, min_support, max_antecedent)
    "iris": (load_iris, 4, 0.05, 3), "wine": (load_wine, 4, 0.08, 3),
    "breast_cancer": (load_breast_cancer, 4, 0.10, 3), "digits": (load_digits, 3, 0.02, 2)}
def models(bins, ms, ma):
    return {
        "majority": DummyClassifier(strategy="prior"),
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "tree(d=4)": DecisionTreeClassifier(max_depth=4, random_state=0),
        "rules/holdout": make_pipeline(binner(bins), RuleClassifier(min_support=ms, max_antecedent=ma)),
        "rules/direct": make_pipeline(binner(bins), RuleClassifier(min_support=ms, max_antecedent=ma, mode="direct")),
    }
def brier(y, P, classes):
    Y = (y[:, None] == classes[None, :]).astype(float)
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))

names = sys.argv[1:] or list(SETUPS)
for name in names:
    load, bins, ms, ma = SETUPS[name]; d = load(); X, y = d.data, d.target
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=0)
    res = {k: dict(acc=[], ll=[], br=[], rules=[], cov=[], sec=[]) for k in models(bins, ms, ma)}
    for tr, te in cv.split(X, y):
        for k, m in models(bins, ms, ma).items():
            t = time.time(); m.fit(X[tr], y[tr]); sec = time.time() - t
            P = m.predict_proba(X[te]); cls = m.classes_ if hasattr(m, "classes_") else m[-1].classes_
            r = res[k]; r["acc"].append(accuracy_score(y[te], cls[P.argmax(1)]))
            r["ll"].append(log_loss(y[te], np.clip(P, 1e-6, 1), labels=cls)); r["br"].append(brier(y[te], P, cls)); r["sec"].append(sec)
            if k.startswith("rules"):
                rc = m[-1]; r["rules"].append(len(rc.rules_))
                r["cov"].append(np.mean([e is not None for e in rc.explain(m[0].transform(X[te]))]))
    print(f"\n== {name}: n={len(y)}, features={X.shape[1]}, classes={len(np.unique(y))} ==")
    print(f"{'model':<15}{'accuracy':>16}{'log-loss':>10}{'brier':>8}{'rules':>7}{'covered':>9}{'fit s':>7}")
    for k, r in res.items():
        a = np.array(r["acc"])
        print(f"{k:<15}{a.mean():>9.3f} +/-{a.std():<5.3f}{np.mean(r['ll']):>10.3f}{np.mean(r['br']):>8.3f}"
              f"{(np.mean(r['rules']) if r['rules'] else float('nan')):>7.1f}{(np.mean(r['cov']) if r['cov'] else float('nan')):>9.2f}{np.mean(r['sec']):>7.2f}", flush=True)
