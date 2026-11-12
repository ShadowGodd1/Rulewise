"""RuleClassifier vs baselines on real categorical data: Mushrooms (UCI) and Titanic."""
import os, sys, time, warnings
import numpy as np, pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from rulewise import RuleClassifier
warnings.filterwarnings("ignore")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

def mushrooms():
    path = os.path.join(DATA, "mushrooms.csv")
    if os.path.exists(path):                      # Kaggle-style file with a header
        d = pd.read_csv(path)
    else:                                         # UCI raw file: class first, no header
        d = pd.read_csv(os.path.join(DATA, "mushroom.data"), header=None)
        d.columns = ["type"] + [f"f{i}" for i in range(d.shape[1] - 1)]
    y = (d["type"] == "p").to_numpy()
    X = pd.get_dummies(d.drop(columns="type"), prefix_sep="=").astype(bool)
    return X, y, "poisonous", dict(min_support=0.01, max_antecedent=3)

def titanic():
    d = pd.read_csv(os.path.join(DATA, "titanic.csv"))
    rename = {c.lower(): c for c in d.columns}
    d = d.rename(columns=rename)
    f = pd.DataFrame({"class": "class=" + d["pclass"].astype(str), "sex": "sex=" + d["sex"],
        "age": "age=" + pd.cut(d["age"], [0, 12, 18, 60, 100]).astype(str).replace("nan", "unknown"),
        "sibsp": np.where(d["sibsp"] > 0, "has_sibsp_or_spouse", "no_sibsp_or_spouse"),
        "parch": np.where(d["parch"] > 0, "has_parent_or_child", "no_parent_or_child"),
        "embarked": "port=" + d["embarked"].fillna("unknown"),
        "fare": "fare=" + pd.qcut(d["fare"], 3, labels=["low", "mid", "high"]).astype(str)})
    X = pd.get_dummies(f, prefix="", prefix_sep="").astype(bool)
    return X, d["survived"].to_numpy().astype(bool), "survived", dict(min_support=0.03, max_antecedent=3)

for name, loader in (("mushrooms", mushrooms), ("titanic", titanic)):
    try:
        X, y, label, kw = loader(); A = X.to_numpy()
    except FileNotFoundError as e:
        print(f"skip {name}: {e}. Run python benchmarks/get_data.py and see data/README.md.")
        continue
    models = {"majority": DummyClassifier(strategy="prior"), "logistic": LogisticRegression(max_iter=3000),
              "tree(d=4)": DecisionTreeClassifier(max_depth=4, random_state=0), "rules/direct": RuleClassifier(**kw),
              "rules/holdout": RuleClassifier(mode="holdout", **kw)}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
    res = {k: dict(acc=[], ll=[], n=[]) for k in models}
    for tr, te in cv.split(A, y):
        for k, m in models.items():
            m.fit(A[tr], y[tr]); P = m.predict_proba(A[te])
            res[k]["acc"].append(float((m.classes_[P.argmax(1)] == y[te]).mean()))
            res[k]["ll"].append(log_loss(y[te], np.clip(P, 1e-6, 1), labels=m.classes_))
            if hasattr(m, "rules_"): res[k]["n"].append(len(m.rules_))
    print(f"\n== {name}: n={len(y)}, items={A.shape[1]}, positive rate ({label}) {y.mean():.2f} ==")
    print(f"{'model':<15}{'accuracy':>16}{'log-loss':>10}{'rules':>7}")
    for k, r in res.items():
        a = np.array(r["acc"]); print(f"{k:<15}{a.mean():>9.3f} +/-{a.std():<5.3f}{np.mean(r['ll']):>10.3f}{(np.mean(r['n']) if r['n'] else float('nan')):>7.0f}")
    m = RuleClassifier(**kw).fit(A, y, item_names=list(X.columns))
    print(f"\nTop rules for {name} (fit on all data, direct mode; {len(m.rules_)} rules):")
    top = m.rules_.sort_values(["confidence_low"], ascending=False).head(6)
    for r in top.itertuples():
        print(f"  {' & '.join(map(str, r.antecedent)):<55} -> {label}={bool(r.class_)}  conf={r.confidence:.3f} [{r.confidence_low:.3f},{r.confidence_high:.3f}] support={r.support:.3f}")
