import numpy as np
import pytest

pytest.importorskip("statsmodels")

from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from rulewise.stats import adjust_pvalues, fisher_greater, wilson_interval


def test_fisher_matches_scipy():
    rng = np.random.default_rng(1)
    for _ in range(300):
        a, b, c, d = rng.integers(0, 40, size=4)
        ref = fisher_exact([[a, b], [c, d]], alternative="greater")[1]
        assert fisher_greater(a, b, c, d) == pytest.approx(ref, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize("method,sm", [("bonferroni", "bonferroni"), ("holm", "holm"),
                                       ("bh", "fdr_bh"), ("by", "fdr_by")])
def test_corrections_match_statsmodels(method, sm):
    rng = np.random.default_rng(2)
    for _ in range(20):
        p = np.concatenate([rng.uniform(0, 0.01, 10), rng.uniform(0, 1, 40)])
        ref = multipletests(p, method=sm)[1]
        np.testing.assert_allclose(adjust_pvalues(p, method), ref, rtol=1e-9, atol=1e-12)


def test_wilson_matches_statsmodels():
    for k, n in [(0, 10), (3, 10), (10, 10), (50, 200), (1, 1000)]:
        lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
        got = wilson_interval(k, n, 0.05)
        assert got[0] == pytest.approx(lo, abs=1e-9) and got[1] == pytest.approx(hi, abs=1e-9)


def test_fisher_capped_exact_below_cap_and_bounded_above():
    from rulewise.stats import fisher_greater_capped
    rng = np.random.default_rng(11)
    for _ in range(2000):
        a, b, c, d = (int(x) for x in rng.integers(0, 300, size=4))
        ref = fisher_exact([[a, b], [c, d]], alternative="greater")[1]
        # exact when uncapped
        assert fisher_greater_capped(a, b, c, d) == pytest.approx(ref, rel=1e-8, abs=1e-300)
        for cap in (0.05, 1e-6):
            got = fisher_greater_capped(a, b, c, d, cap)
            if ref < cap:
                assert got == pytest.approx(ref, rel=1e-8, abs=1e-300)
            else:
                assert cap <= got <= ref * (1 + 1e-8)


def test_fisher_capped_extreme_tables():
    from rulewise.stats import fisher_greater_capped
    for a, b, c, d in [(5000, 10, 10, 90000), (0, 5000, 5000, 90000), (2000, 3000, 1000, 94000),
                       (50000, 0, 0, 50000), (1, 99999, 99999, 1)]:
        ref = fisher_exact([[a, b], [c, d]], alternative="greater")[1]
        assert fisher_greater_capped(a, b, c, d) == pytest.approx(ref, rel=1e-6, abs=1e-300)
