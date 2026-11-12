import importlib.util

import numpy as np
import pytest

from rulewise import RuleClassifier

HAS_FIM = importlib.util.find_spec("fim") is not None


def awkward_data(n=5000, seed=0):
    """Contains: items in every row, an item present in every row of one class, noise, a rule."""
    rng = np.random.default_rng(seed)
    X = rng.random((n, 14)) < 0.3
    y = (X[:, 0] & X[:, 1]) ^ (rng.random(n) < 0.04)
    rows = []
    for i in range(n):
        t = {f"f{j}" for j in range(14) if X[i, j]}
        t.add("ALWAYS")                                   # universal in the full data
        if y[i] or rng.random() < 0.3:
            t.add("SEEN_IF_POS")                          # present in every positive row
        if y[i]:
            t.add("ONLY_POS_1")                           # universal within the positive class
            t.add("ONLY_POS_2")
        rows.append(sorted(t))
    return rows, y


def keyed(m):
    return {(tuple(r.antecedent), r.class_): (round(r.p_adjusted, 12), r.count)
            for r in m.rules_.itertuples()}


@pytest.mark.parametrize("backend", ["python",
                                     pytest.param("pyfim", marks=pytest.mark.skipif(
                                         not HAS_FIM, reason="pyfim not installed"))])
@pytest.mark.parametrize("mode", ["holdout", "direct"])
def test_class_conditional_identical_to_general(backend, mode):
    rows, y = awkward_data()
    kw = dict(min_support=0.01, mode=mode, backend=backend, max_antecedent=3)
    a = RuleClassifier(class_conditional=True, **kw).fit(rows, y)
    b = RuleClassifier(class_conditional=False, **kw).fit(rows, y)
    assert len(a.rules_) > 0 and keyed(a) == keyed(b)
    assert a.diagnostics_["n_candidates"] == b.diagnostics_["n_candidates"]
    np.testing.assert_allclose(a.predict_proba(rows[:300]), b.predict_proba(rows[:300]))
    assert not any("ALWAYS" in r.antecedent for r in a.rules_.itertuples())


@pytest.mark.skipif(not HAS_FIM, reason="pyfim not installed")
def test_class_specific_universal_items_are_kept_as_predictors():
    rows, y = awkward_data()
    m = RuleClassifier(min_support=0.01, backend="pyfim").fit(rows, y)
    ants = [set(r.antecedent) for r in m.rules_.itertuples() if bool(r.class_)]
    assert any("ONLY_POS_1" in a for a in ants), "class-universal item was dropped"


def test_class_conditional_with_negations_and_multiclass():
    rng = np.random.default_rng(3)
    n = 4000
    X = rng.random((n, 8)) < 0.35
    y = np.where(X[:, 0] & ~X[:, 1], "a", np.where(X[:, 2] & X[:, 3], "b", "c"))
    kw = dict(min_support=0.01, add_negations=True, backend="python")
    a = RuleClassifier(class_conditional=True, **kw).fit(X, y)
    b = RuleClassifier(class_conditional=False, **kw).fit(X, y)
    assert keyed(a) == keyed(b) and len(a.rules_) > 0
