import numpy as np

from rulewise import RuleMiner
from rulewise.stats import bootstrap_lift_leverage
from tests.test_miner import planted


def test_multinomial_bootstrap_matches_transaction_resampling():
    n, n_xy, n_x, n_y = 2000, 200, 400, 520
    x = np.zeros(n, bool); y = np.zeros(n, bool)
    x[:n_x] = True; y[:n_xy] = True; y[n_x:n_x + (n_y - n_xy)] = True
    rng = np.random.default_rng(0)
    lifts, levs = [], []
    for _ in range(4000):
        idx = rng.integers(0, n, n)
        xs, ys = x[idx], y[idx]
        a, b, c = (xs & ys).sum(), xs.sum(), ys.sum()
        lifts.append(a * n / (b * c)); levs.append(a / n - (b / n) * (c / n))
    ref = (*np.percentile(lifts, [2.5, 97.5]), *np.percentile(levs, [2.5, 97.5]))
    got = bootstrap_lift_leverage(n_xy, n_x, n_y, n, n_boot=4000, rng=np.random.default_rng(1))
    np.testing.assert_allclose(got[:2], ref[:2], atol=0.05)
    np.testing.assert_allclose(got[2:], ref[2:], atol=0.004)


def test_interval_coverage_near_nominal():
    """True lift 1.923 and leverage 0.048; 95% intervals should cover about 95% of the time."""
    rng = np.random.default_rng(7)
    true_lift, true_lev = 0.5 / 0.26, 0.1 - 0.2 * 0.26
    cov_l = cov_v = 0
    reps = 300
    for _ in range(reps):
        n = 2000
        X = rng.random(n) < 0.2
        Y = np.where(X, rng.random(n) < 0.5, rng.random(n) < 0.2)
        lo, hi, vlo, vhi = bootstrap_lift_leverage(int((X & Y).sum()), int(X.sum()), int(Y.sum()),
                                                   n, n_boot=300, rng=rng)
        cov_l += lo <= true_lift <= hi
        cov_v += vlo <= true_lev <= vhi
    assert 0.90 <= cov_l / reps <= 0.99, cov_l / reps
    assert 0.90 <= cov_v / reps <= 0.99, cov_v / reps


def test_rules_table_has_intervals():
    m = RuleMiner(min_support=0.01).fit(planted())
    r = m.rules_
    assert {"lift_low", "lift_high", "leverage_low", "leverage_high"} <= set(r.columns)
    assert (r.lift_low <= r.lift + 1e-9).all() and (r.lift_high >= r.lift - 1e-9).all()
    assert (r.leverage_low <= r.leverage_high).all()
    assert "lift_low" not in RuleMiner(min_support=0.01, n_bootstrap=0).fit(planted()).rules_.columns
