from itertools import combinations

import numpy as np
import pytest

from rulewise import RuleMiner
from rulewise.rules import search_space_sizes
from tests.test_miner import planted


@pytest.mark.parametrize("m,ma,mc", [(6, 3, 1), (6, 2, 2), (7, 3, 2), (5, 1, 1), (6, 4, 3)])
def test_search_space_matches_bruteforce(m, ma, mc):
    items = range(m)
    brute = {}
    for l in range(2, ma + mc + 1):
        for S in combinations(items, l):
            for k in range(1, l):
                for Y in combinations(S, k):
                    if k <= mc and l - k <= ma:
                        brute[l] = brute.get(l, 0) + 1
    assert search_space_sizes(m, ma, mc) == brute


def test_direct_mode_null_fwer():
    hits, runs = 0, 60
    for s in range(runs):
        rng = np.random.default_rng(5000 + s)
        tx = [[f"i{j}" for j in range(25) if rng.random() < 0.15] for _ in range(3000)]
        if len(RuleMiner(min_support=0.01, mode="direct").fit(tx).rules_) > 0:
            hits += 1
    assert hits / runs <= 0.13, f"direct-mode FWER too high: {hits}/{runs}"


def test_direct_mode_finds_planted_not_noise():
    m = RuleMiner(min_support=0.01, mode="direct").fit(planted())
    assert len(m.rules_) > 0
    items = {i for r in m.rules_.itertuples() for i in r.antecedent + r.consequent}
    assert items <= {"A", "B", "C"}
    assert m.diagnostics_["mode"] == "direct" and m.diagnostics_["n_holdout"] == 6000


def test_duplicate_item_not_reported_as_extension():
    """A2 is a perfect copy of A. {A, A2} -> B adds nothing over {A} -> B and must be dropped."""
    tx = [sorted(set(t) | ({"A2"} if "A" in t else set())) for t in planted()]
    for mode in ("holdout", "direct"):
        m = RuleMiner(min_support=0.01, mode=mode).fit(tx)
        assert len(m.rules_) > 0
        bad = [r for r in m.rules_.itertuples() if "A" in r.antecedent and "A2" in r.antecedent]
        assert not bad, f"redundant superset antecedent reported in {mode} mode"


def test_multi_item_consequent_and_null():
    m = RuleMiner(min_support=0.01, max_consequent=2).fit(planted())
    assert any(len(r.consequent) == 2 for r in m.rules_.itertuples())
    items = {i for r in m.rules_.itertuples() for i in r.antecedent + r.consequent}
    assert items <= {"A", "B", "C"}
    hits, runs = 0, 40
    for s in range(runs):
        rng = np.random.default_rng(9000 + s)
        tx = [[f"i{j}" for j in range(20) if rng.random() < 0.2] for _ in range(2500)]
        if len(RuleMiner(min_support=0.01, max_consequent=2, random_state=s).fit(tx).rules_) > 0:
            hits += 1
    assert hits / runs <= 0.15, f"multi-consequent FWER too high: {hits}/{runs}"


def test_stability_report():
    rep = RuleMiner(min_support=0.01).stability(planted(), n_resamples=8)
    assert 0.5 <= rep["mean_jaccard"] <= 1.0
    f = rep["rule_frequency"]
    assert len(f) > 0 and f.frequency.between(0, 1).all()
