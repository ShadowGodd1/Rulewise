"""Class-conditional candidate generation for class-association rules.

For rules X -> class, an itemset X can only matter if it is frequent *within* that class, so
we mine each class's transactions separately (itemset size <= max_antecedent) instead of
enumerating every feature combination in the whole data. The resulting candidate set is
identical to the general pipeline's (verified by tests) but is far smaller to enumerate on
dense data.
"""
from __future__ import annotations

from itertools import combinations

from .mining import encode, mine_itemsets


def class_conditional_candidates(expl, vocab, target_ids, min_count, max_antecedent,
                                 min_confidence, backend="auto"):
    """Return (candidates, n_itemsets_mined, backend_used).

    candidates: list of (frozenset(antecedent ids), frozenset({class item id})) with
    confidence >= min_confidence and lift > 1, exactly as the general pipeline would produce.
    """
    n = len(expl)
    tidsets, _ = encode(expl, vocab)
    full = (1 << n) - 1
    inv = {j: i for i, j in vocab.items()}
    # Items present in every transaction can never be in a significant rule (lift 1, no
    # added coverage); class items are never antecedents.
    drop = {j for j, t in enumerate(tidsets) if t.bit_count() == n} | set(target_ids)

    cache = {}

    def count_x(X):
        v = cache.get(X)
        if v is None:
            t = full
            for i in X:
                t &= tidsets[i]
            v = cache[X] = t.bit_count()
        return v

    cands, n_sets, used = [], 0, "python"
    for t in sorted(target_ids):
        cnt_c = tidsets[t].bit_count()
        if cnt_c < min_count:
            continue
        target_item = inv[t]
        sub = [[i for i in row if vocab[i] not in drop] for row in expl if target_item in row]
        n_sub = len(sub)
        # Items present in every row of THIS class are strong predictors but would be dropped
        # by the miner (and by PyFIM); mine without them and add them back exactly.
        per_item = {}
        for row in sub:
            for i in set(row):
                per_item[i] = per_item.get(i, 0) + 1
        universal = sorted(vocab[i] for i, c in per_item.items() if c == n_sub)
        base, used = mine_itemsets(sub, vocab, min_count, max_antecedent, backend)
        sets = dict(base)
        if universal:
            with_empty = dict(base)
            with_empty[frozenset()] = n_sub
            for r in range(1, min(len(universal), max_antecedent) + 1):
                for W in combinations(universal, r):
                    Wf = frozenset(W)
                    for S, c in with_empty.items():
                        if len(S) + r <= max_antecedent:
                            sets[S | Wf] = c
        n_sets += len(sets)
        for X, c_xc in sets.items():
            if not X:
                continue
            cx = count_x(X)
            conf = c_xc / cx
            if conf >= min_confidence and conf * n > cnt_c:  # same test as candidate_rules
                cands.append((X, frozenset({t})))
    return cands, n_sets, used
