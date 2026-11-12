"""Candidate rule generation, rule measures, and the productivity test."""
from __future__ import annotations

from itertools import combinations
from math import comb

from .stats import fisher_greater_capped


def candidate_rules(itemsets, n, min_confidence, max_antecedent, max_consequent, targets=None):
    """Rules X -> Y from frequent itemsets with confidence >= min_confidence and lift > 1.

    If ``targets`` (a set of item ids) is given, the consequent must lie inside it and the
    antecedent must avoid it (used for class-association rules).
    """
    out = []
    for S, cnt_s in itemsets.items():
        if len(S) < 2:
            continue
        for k in range(1, min(max_consequent, len(S) - 1) + 1):
            for Y in combinations(sorted(S), k):
                Y = frozenset(Y)
                X = S - Y
                if len(X) > max_antecedent:
                    continue
                if targets is not None and (not Y <= targets or X & targets):
                    continue
                cnt_x, cnt_y = itemsets[X], itemsets[Y]
                conf = cnt_s / cnt_x
                if conf >= min_confidence and conf * n > cnt_y:  # lift > 1
                    out.append((X, Y))
    return out


class TidCounter:
    """Cached bitset intersections over a fixed set of transactions."""

    def __init__(self, tidsets, n, max_cache_mb: float = 128.0):
        self.tidsets, self.n = tidsets, n
        self.full = (1 << n) - 1
        self._cache = {}
        # each cached intersection is an n-bit integer; bound the total so memory stays flat
        self._max_entries = max(1000, int(max_cache_mb * 8e6 / max(n, 1)))

    def tid(self, itemset):
        key = frozenset(itemset)
        v = self._cache.get(key)
        if v is None:
            v = self.full
            for i in key:
                v &= self.tidsets[i]
            if len(self._cache) >= self._max_entries:
                self._cache.clear()
            self._cache[key] = v
        return v

    def count(self, itemset):
        return self.tid(itemset).bit_count()


def productivity_pvalue(X, Y, tc: TidCounter, cap: float = 1.0) -> float:
    """Webb's productivity test: p-value that X -> Y beats Z -> Y for EVERY proper subset Z of X
    (including the empty set). Each comparison is a one-sided Fisher test among transactions
    covering Z (does also covering X\\Z raise the rate of Y?). The rule's p-value is the maximum
    over comparisons (intersection-union test), so it is significant only if all comparisons are.

    With ``cap < 1`` the search stops as soon as a comparison reaches ``cap`` and returns a value
    in [cap, true p]; the result is exact whenever it is below ``cap``. Immediate subsets (the
    comparisons most likely to fail for a rule padded with an irrelevant item) are tried first.
    """
    X = sorted(X)
    tidY = tc.tid(Y)
    worst = 0.0
    for r in range(len(X) - 1, -1, -1):
        for Z in combinations(X, r):
            rest = [x for x in X if x not in Z]
            base = tc.tid(Z)
            with_rest = base & tc.tid(rest)
            without = base ^ with_rest
            a = (with_rest & tidY).bit_count()
            n_with = with_rest.bit_count()
            c = (without & tidY).bit_count()
            n_without = without.bit_count()
            p = fisher_greater_capped(a, n_with - a, c, n_without - c, cap)
            if p > worst:
                worst = p
                if worst >= cap:
                    return worst
    return worst


def itemset_partition_pvalue(S, tc: TidCounter, cap: float = 1.0) -> float:
    """Largest one-sided Fisher p-value over all bipartitions (Q, S\\Q) of the itemset S.

    Used for rules with multi-item consequents (Webb 2010, self-sufficient itemsets): every
    split of the itemset must show significant positive dependence, so an item that is merely
    carried along by the others (e.g. noise inside the consequent) makes the rule fail.
    """
    S = sorted(S)
    first, rest = S[0], S[1:]
    n, cs = tc.n, tc.count(S)
    worst = 0.0
    for r in range(0, len(rest)):  # r = len(rest) would make Q == S
        for sub in combinations(rest, r):
            Q = (first,) + sub
            R = [x for x in S if x not in Q]
            cq, cr = tc.count(Q), tc.count(R)
            p = fisher_greater_capped(cs, cq - cs, cr - cs, n - cq - cr + cs, cap)
            if p > worst:
                worst = p
                if worst >= cap:
                    return worst
    return worst


def rule_measures(n_xy, n_x, n_y, n):
    support = n_xy / n
    confidence = n_xy / n_x if n_x else 0.0
    p_y = n_y / n
    lift = confidence / p_y if p_y else float("nan")
    leverage = support - (n_x / n) * p_y
    conviction = float("inf") if confidence >= 1 else (1 - p_y) / (1 - confidence)
    return support, confidence, lift, leverage, conviction


def search_space_sizes(m: int, max_antecedent: int, max_consequent: int) -> dict:
    """Number of possible rules X -> Y over m items, grouped by level l = |X| + |Y|.

    This is S_l in Webb's layered critical values (Webb 2008): the correction at level l
    must account for every rule that *could* have been reported at that level, not only
    the ones the miner happened to examine.
    """
    sizes = {}
    for l in range(2, max_antecedent + max_consequent + 1):
        per_itemset = sum(comb(l, k) for k in range(1, min(max_consequent, l - 1) + 1)
                          if l - k <= max_antecedent)
        if per_itemset and comb(m, l):
            sizes[l] = comb(m, l) * per_itemset
    return sizes


def targeted_space_sizes(m_features: int, n_targets: int, max_antecedent: int) -> dict:
    """Search-space size per level l = |X| + 1 for rules X -> (one of n_targets class items)."""
    return {a + 1: n_targets * comb(m_features, a) for a in range(1, max_antecedent + 1)
            if comb(m_features, a)}
