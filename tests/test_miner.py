import numpy as np

from rulewise import RuleMiner


def planted(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    tx = []
    for _ in range(n):
        t = {f"n{j}" for j in range(20) if rng.random() < 0.12}
        if rng.random() < 0.10:
            t |= {"A", "B", "C"}
        else:
            t |= {x for x in "ABC" if rng.random() < 0.04}
        tx.append(sorted(t))
    return tx


def test_recovers_planted_and_rejects_noise():
    m = RuleMiner(min_support=0.01).fit(planted())
    assert len(m.rules_) > 0
    items = {i for r in m.rules_.itertuples() for i in r.antecedent + r.consequent}
    assert items <= {"A", "B", "C"}, f"noise item in significant rules: {items}"
    pairs = {(r.antecedent, r.consequent) for r in m.rules_.itertuples()}
    assert (("A",), ("B",)) in pairs


def test_null_data_fwer_controlled():
    """Empirical FWER on independent-item data must stay near/below alpha=0.05."""
    hits = 0
    runs = 60
    for s in range(runs):
        rng = np.random.default_rng(1000 + s)
        tx = [[f"i{j}" for j in range(25) if rng.random() < 0.15] for _ in range(3000)]
        if len(RuleMiner(min_support=0.01, random_state=s).fit(tx).rules_) > 0:
            hits += 1
    assert hits / runs <= 0.13, f"FWER too high: {hits}/{runs}"


def test_transform_recommend_validate():
    tx = planted()
    m = RuleMiner(min_support=0.01).fit(tx)
    F = m.transform([["A"], ["A", "B", "C"], ["n1"]])
    assert F.shape == (3, len(m.rules_)) and F[1].any() and not F[2].any()
    rec = dict(m.recommend(["A"], k=3))
    assert "B" in rec or "C" in rec
    rep = m.validate(tx, n_null=8)
    assert rep["null_fwer"] <= 0.25 and rep["n_rules_real"] > 0


def test_bounded_cache_gives_identical_results():
    tx = planted()
    a = RuleMiner(min_support=0.01).fit(tx)
    b = RuleMiner(min_support=0.01, max_cache_mb=0.0).fit(tx)  # evicts constantly
    ka = {(r.antecedent, r.consequent): r.p_adjusted for r in a.rules_.itertuples()}
    kb = {(r.antecedent, r.consequent): r.p_adjusted for r in b.rules_.itertuples()}
    assert ka == kb and len(ka) > 0


def test_empty_rule_set_keeps_schema_and_transforms():
    """A fit that finds nothing must still expose usable columns and a (n, 0) matrix."""
    tx = [[], [], []]
    m = RuleMiner(min_support=0.5, n_bootstrap=0).fit(tx)
    assert len(m.rules_) == 0 and "antecedent" in m.rules_.columns
    assert m.transform(tx).shape == (3, 0)
    assert m.summary()
    assert m.query(min_lift=2).empty


def test_matrix_width_is_checked_after_fit():
    import pandas as pd
    import pytest

    df = pd.DataFrame([[1, 1, 0]] * 20 + [[1, 0, 0]] * 10 + [[0, 1, 1]] * 10,
                      columns=["a", "b", "c"])
    m = RuleMiner(min_support=0.1, n_bootstrap=0).fit(df)
    assert m.n_features_in_ == 3
    with pytest.raises(ValueError, match="expecting 3 features"):
        m.transform(df[["a", "b"]])
