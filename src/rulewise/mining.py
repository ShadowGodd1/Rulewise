"""Frequent itemset mining on bitsets (Eclat). Correctness-first reference backend."""
from __future__ import annotations

import warnings

import numpy as np


def build_vocab(transactions):
    vocab = {}
    for t in transactions:
        for item in t:
            if item not in vocab:
                vocab[item] = len(vocab)
    return vocab


def encode(transactions, vocab):
    """Return (tidsets, n): one Python-int bitset per vocab item (bit i = transaction i)."""
    n = len(transactions)
    positions = [[] for _ in vocab]
    for i, t in enumerate(transactions):
        for item in set(t):
            j = vocab.get(item)
            if j is not None:
                positions[j].append(i)
    tidsets = []
    for pos in positions:
        arr = np.zeros(n, dtype=bool)
        if pos:
            arr[pos] = True
        tidsets.append(int.from_bytes(np.packbits(arr, bitorder="little").tobytes(), "little"))
    return tidsets, n


def eclat(tidsets, min_count: int, max_len: int):
    """Mine all frequent itemsets (as frozensets of item ids) with their counts."""
    out: dict[frozenset, int] = {}
    items = [(i, t) for i, t in enumerate(tidsets) if t.bit_count() >= min_count]
    items.sort(key=lambda x: x[1].bit_count())

    def recurse(cands, prefix):
        for idx, (item, tid) in enumerate(cands):
            new = prefix + (item,)
            out[frozenset(new)] = tid.bit_count()
            if len(new) < max_len:
                ext = []
                for item2, tid2 in cands[idx + 1:]:
                    t = tid & tid2
                    if t.bit_count() >= min_count:
                        ext.append((item2, t))
                if ext:
                    recurse(ext, new)

    recurse(items, ())
    return out


def _closure_ok(itemsets) -> bool:
    """Downward closure: every immediate subset of a frequent itemset must be present."""
    for S in itemsets:
        if len(S) > 1:
            for x in S:
                if (S - {x}) not in itemsets:
                    return False
    return True


def mine_itemsets(transactions, vocab, min_count: int, max_len: int, backend: str = "auto"):
    """Frequent itemsets (frozenset of item ids -> count) and the backend actually used.

    backend: 'python' (built-in bitset Eclat), 'pyfim' (Borgelt's C implementation, optional
    dependency, MIT licence) or 'auto' (pyfim when installed, otherwise python).

    Items that occur in every transaction are excluded from both backends: they have lift 1 as
    consequents and add no coverage as antecedent items, so they can never appear in a
    significant rule. (PyFIM also omits itemsets made only of such items, so excluding them
    keeps the backends identical.) The PyFIM result is checked for downward closure and, if
    inconsistent, the exact Python miner is used instead with a warning.
    """
    if backend not in ("auto", "python", "pyfim"):
        raise ValueError("backend must be 'auto', 'python' or 'pyfim'")
    n = len(transactions)
    counts = {}
    for t in transactions:
        for i in set(t):
            j = vocab.get(i)
            if j is not None:
                counts[j] = counts.get(j, 0) + 1
    universal = {j for j, c in counts.items() if c == n}
    if backend in ("auto", "pyfim"):
        try:
            import fim
        except ImportError:
            if backend == "pyfim":
                raise ImportError("backend='pyfim' needs the optional dependency: pip install pyfim")
        else:
            tracts = [[vocab[i] for i in set(t) if i in vocab and vocab[i] not in universal]
                      for t in transactions]
            res = fim.eclat(tracts, target="s", supp=-min_count, zmin=1, zmax=max_len, report="a")
            out = {frozenset(items): int(c) for items, c in res}
            if _closure_ok(out):
                return out, "pyfim"
            warnings.warn("PyFIM returned an inconsistent itemset collection; using the exact "
                          "Python miner instead.", RuntimeWarning, stacklevel=2)
    tidsets, _ = encode(transactions, vocab)
    for j in universal:
        tidsets[j] = 0
    return eclat(tidsets, min_count, max_len), "python"
