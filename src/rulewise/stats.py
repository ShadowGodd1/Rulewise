"""Statistical building blocks: exact tests, multiple-testing corrections, intervals.

References
----------
Webb, G.I. (2007). Discovering significant patterns. Machine Learning 68(1).
Hamalainen, W. & Webb, G.I. (2019). A tutorial on statistically sound pattern
    discovery. Data Mining and Knowledge Discovery 33(2).
Holm (1979); Benjamini & Hochberg (1995); Benjamini & Yekutieli (2001).
"""
from __future__ import annotations

from functools import lru_cache
from math import exp, lgamma

import numpy as np
from scipy.stats import hypergeom, norm


def fisher_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided Fisher exact p-value P(A >= a) for the table [[a, b], [c, d]].

    Tests whether the first row is enriched in the first column.
    """
    M = a + b + c + d
    n_col = a + c
    n_row = a + b
    if n_row == 0 or n_row == M or n_col == 0 or n_col == M:
        return 1.0  # degenerate margins: the table is fully determined
    return float(hypergeom.sf(a - 1, M, n_col, n_row))


def fisher_greater_capped(a: int, b: int, c: int, d: int, cap: float = 1.0) -> float:
    """Fast one-sided Fisher exact p-value with early exit.

    Returns the exact p-value P(A >= a) whenever it is below ``cap``. When the true p-value is
    at least ``cap`` it returns some value in [cap, p] (a partial tail sum), which is all a
    caller that only needs to know "is p < cap?" requires. With the default cap=1.0 the result
    is always exact. Sums the hypergeometric tail upward from ``a`` by the term ratio
    recurrence, which is much cheaper than a general distribution call for the many small
    tables a rule search produces and accurate for tiny tail probabilities.
    """
    M = a + b + c + d
    K = a + c
    n = a + b
    if n == 0 or n == M or K == 0 or K == M:
        return 1.0
    log_term = (lgamma(K + 1) - lgamma(a + 1) - lgamma(K - a + 1)
                + lgamma(M - K + 1) - lgamma(n - a + 1) - lgamma(M - K - n + a + 1)
                - lgamma(M + 1) + lgamma(n + 1) + lgamma(M - n + 1))
    if log_term < -700.0:  # the first term would underflow; use the general routine
        return float(hypergeom.sf(a - 1, M, K, n))
    term = exp(log_term)
    total = term
    k = a
    kmax = min(n, K)
    while k < kmax:
        if total >= cap:
            return total
        term *= (K - k) * (n - k) / ((k + 1) * (M - K - n + k + 1))
        k += 1
        total += term
        if term < total * 1e-17:
            break
    return total if total < 1.0 else 1.0


def adjust_pvalues(pvalues, method: str = "holm") -> np.ndarray:
    """Multiple-testing adjustment. Methods: none, bonferroni, holm, bh, by.

    bonferroni/holm control the family-wise error rate (FWER);
    bh/by control the false discovery rate (by holds under arbitrary dependence).
    """
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    method = method.lower()
    if method == "none":
        return p.copy()
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    order = np.argsort(p, kind="mergesort")
    ps = p[order]
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * ps)
    elif method in ("bh", "by"):
        scale = m / np.arange(1, m + 1)
        if method == "by":
            scale = scale * np.sum(1.0 / np.arange(1, m + 1))
        adj = np.minimum.accumulate((scale * ps)[::-1])[::-1]
    else:
        raise ValueError(f"unknown correction method: {method!r}")
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out


@lru_cache(maxsize=32)
def _z(alpha: float) -> float:
    return float(norm.ppf(1 - alpha / 2))


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 1.0)
    z = _z(alpha)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_lift_leverage(n_xy, n_x, n_y, n, n_boot=200, alpha=0.05, rng=None):
    """Percentile bootstrap intervals for lift and leverage of one rule.

    Resampling n transactions with replacement is equivalent to drawing the four cells of the
    rule's 2x2 table from a multinomial(n, observed cell proportions), so no transaction-level
    resampling is needed. Returns (lift_low, lift_high, leverage_low, leverage_high).
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    cells = np.array([n_xy, n_x - n_xy, n_y - n_xy, n - n_x - n_y + n_xy], dtype=float)
    draws = rng.multinomial(n, cells / cells.sum(), size=n_boot)
    xy = draws[:, 0].astype(float)
    x = xy + draws[:, 1]
    y = xy + draws[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        lift = xy * n / (x * y)
    lev = xy / n - (x / n) * (y / n)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    lift = lift[np.isfinite(lift)]
    l_lo, l_hi = (np.percentile(lift, q) if lift.size else (np.nan, np.nan))
    v_lo, v_hi = np.percentile(lev, q)
    return float(l_lo), float(l_hi), float(v_lo), float(v_hi)
