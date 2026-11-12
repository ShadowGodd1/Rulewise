from itertools import combinations

import numpy as np
import pytest

pytest.importorskip("sklearn")

import scipy.sparse as sp
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer

from rulewise import RuleClassifier, RuleMiner
from rulewise.rules import targeted_space_sizes


def and_data(n=4000, seed=0, noise=0.05, n_feat=12):
    rng = np.random.default_rng(seed)
    X = rng.random((n, n_feat)) < 0.3
    y = (X[:, 0] & X[:, 1]) ^ (rng.random(n) < noise)
    return X, y


@pytest.mark.parametrize("m,k,ma", [(6, 2, 3), (7, 3, 2), (5, 1, 4)])
def test_targeted_space_matches_bruteforce(m, k, ma):
    brute = {}
    for a in range(1, ma + 1):
        for _ in combinations(range(m), a):
            brute[a + 1] = brute.get(a + 1, 0) + k
    assert targeted_space_sizes(m, k, ma) == brute


def test_recovers_planted_rule_and_accuracy():
    X, y = and_data()
    m = RuleClassifier(min_support=0.01).fit(X, y)
    top = m.rules_.iloc[0]
    assert tuple(top.antecedent) == (0, 1) and bool(top.class_) is True
    assert abs(top.confidence - 0.95) < 0.03
    Xt, yt = and_data(seed=1)
    assert m.score(Xt, yt) > 0.93  # Bayes-optimal is about 0.95


def test_binary_probability_equals_rule_confidence():
    X, y = and_data()
    m = RuleClassifier(min_support=0.01).fit(X, y)
    P = m.predict_proba(X[:300])
    assert np.allclose(P.sum(axis=1), 1.0)
    for row, p, rule in zip(X[:300], P, m.explain(X[:300])):
        if rule is not None:
            assert p[list(m.classes_).index(rule["class_"])] == pytest.approx(rule["confidence"])
        else:
            assert np.allclose(p, m.priors_)


@pytest.mark.parametrize("mode", ["holdout", "direct"])
def test_null_labels_give_no_rules_fwer(mode):
    hits, runs = 0, 40
    for s in range(runs):
        rng = np.random.default_rng(300 + s)
        X = rng.random((2500, 15)) < 0.25
        y = rng.random(2500) < 0.4  # independent of X
        m = RuleClassifier(min_support=0.01, mode=mode, random_state=s).fit(X, y)
        hits += len(m.rules_) > 0
        if len(m.rules_) == 0:
            assert (m.predict(X[:50]) == m.classes_[np.argmax(m.priors_)]).all()
    assert hits / runs <= 0.15, f"{mode}: {hits}/{runs} null datasets produced rules"


def test_multiclass_string_labels():
    rng = np.random.default_rng(5)
    n = 6000
    X = rng.random((n, 15)) < 0.25
    y = np.where(X[:, 0] & X[:, 1], "red", np.where(X[:, 2] & X[:, 3], "green", "blue"))
    y = np.where(rng.random(n) < 0.05, rng.choice(["red", "green", "blue"], n), y)
    m = RuleClassifier(min_support=0.005).fit(X, y)
    assert list(m.classes_) == ["blue", "green", "red"]
    P = m.predict_proba(X[:200])
    assert P.shape == (200, 3) and np.allclose(P.sum(axis=1), 1.0)
    base = max(np.mean(y == c) for c in m.classes_)
    assert m.score(X, y) > base + 0.05
    assert {"red", "green"} <= set(m.rules_.class_)


def test_input_formats_agree_and_validation():
    X, y = and_data(n=3000)
    lists = [[j for j in range(X.shape[1]) if row[j]] for row in X]
    a = RuleClassifier(min_support=0.01).fit(X, y)
    b = RuleClassifier(min_support=0.01).fit(lists, y)
    c = RuleClassifier(min_support=0.01).fit(sp.csr_matrix(X), y)
    for other in (b, c):
        np.testing.assert_allclose(a.predict_proba(X[:200]), other.predict_proba(X[:200]))
    np.testing.assert_allclose(a.predict_proba(X[:50]), a.predict_proba(lists[:50]))
    with pytest.raises(ValueError):
        RuleClassifier().fit(X, y[:-1])
    with pytest.raises((RuntimeError, ValueError)):
        RuleClassifier().predict(X)


def test_sklearn_compatibility():
    X, y = and_data(n=3000)
    est = RuleClassifier(min_support=0.01)
    assert clone(est).get_params() == est.get_params()
    scores = cross_val_score(est, X, y, cv=3)
    assert scores.mean() > 0.92
    gs = GridSearchCV(est, {"max_antecedent": [1, 3]}, cv=3).fit(X, y)
    assert gs.best_params_["max_antecedent"] == 3  # the signal needs a 2-item antecedent
    rng = np.random.default_rng(2)
    Xn = rng.random((3000, 5))
    yn = (Xn[:, 0] > 0.6) & (Xn[:, 1] > 0.6)
    pipe = Pipeline([("bin", KBinsDiscretizer(n_bins=5, encode="onehot", strategy="uniform")),
                     ("rules", RuleClassifier(min_support=0.01))])
    assert cross_val_score(pipe, Xn, yn, cv=3).mean() > 0.93


def test_recommend_options_and_miner_in_pipeline():
    rng = np.random.default_rng(6)
    tx = []
    for _ in range(5000):
        t = {f"n{j}" for j in range(15) if rng.random() < 0.15}
        if rng.random() < 0.2:
            t |= {"A", "B"}
        tx.append(sorted(t))
    m = RuleMiner(min_support=0.01).fit(tx)
    for by in ("confidence_low", "confidence", "lift", "leverage"):
        rec = m.recommend(["A"], k=3, by=by)
        assert rec and rec[0][0] == "B"
    df = m.recommend(["A"], k=3, explain=True)
    assert df.iloc[0]["item"] == "B" and "antecedent" in df.columns
    with pytest.raises(ValueError):
        m.recommend(["A"], by="nope")
    assert m.recommend(["zzz"]) == []
    F = Pipeline([("rules", RuleMiner(min_support=0.01))]).fit_transform(tx)
    assert F.shape[0] == 5000


def test_negations_let_rules_express_absence():
    rng = np.random.default_rng(0)
    n, F = 20000, 10
    X = rng.random((n, F)) < 0.3
    y = (X[:, 0] & X[:, 1]) | (X[:, 2] & ~X[:, 3] & X[:, 4])
    y = y ^ (rng.random(n) < 0.04)
    tr, te = slice(0, 14000), slice(14000, None)
    plain = RuleClassifier(min_support=0.005).fit(X[tr], y[tr])
    neg = RuleClassifier(min_support=0.005, add_negations=True).fit(X[tr], y[tr])
    from sklearn.metrics import log_loss
    ll = lambda m: log_loss(y[te], np.clip(m.predict_proba(X[te]), 1e-6, 1), labels=m.classes_)
    # same labels either way (the 0.72-confidence plain rule already predicts positive), but
    # the rule that names the absence gives far better probabilities
    assert ll(neg) < ll(plain) - 0.03, (ll(neg), ll(plain))
    assert neg.score(X[te], y[te]) >= plain.score(X[te], y[te]) > 0.93
    assert any(("not", 3) in a for a in neg.rules_.antecedent)
    # list-of-lists input is negated against the training universe
    lists = [[j for j in range(F) if row[j]] for row in X[te][:100]]
    np.testing.assert_allclose(neg.predict_proba(X[te][:100]), neg.predict_proba(lists))
    assert "add_negations" in neg.get_params()
