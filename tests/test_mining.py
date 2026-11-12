import numpy as np
import pandas as pd
import pytest

pytest.importorskip("mlxtend")

from mlxtend.frequent_patterns import fpgrowth

from rulewise.mining import build_vocab, eclat, encode


def test_eclat_matches_mlxtend():
    rng = np.random.default_rng(3)
    tx = [[f"i{j}" for j in range(25) if rng.random() < 0.25] for _ in range(800)]
    vocab = build_vocab(tx)
    tids, n = encode(tx, vocab)
    inv = {j: it for it, j in vocab.items()}
    mine = {frozenset(inv[i] for i in s): c for s, c in eclat(tids, 20, 5).items()}
    df = pd.DataFrame([{it: True for it in t} for t in tx]).fillna(False).astype(bool)
    ref = fpgrowth(df, min_support=20 / n, use_colnames=True, max_len=5)
    ref = {frozenset(s): round(sup * n) for s, sup in zip(ref.itemsets, ref.support)}
    assert mine == ref
