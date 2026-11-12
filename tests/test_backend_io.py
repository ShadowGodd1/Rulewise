import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from rulewise import RuleMiner
from rulewise.mining import build_vocab, mine_itemsets
from tests.test_miner import planted

HAS_FIM = importlib.util.find_spec("fim") is not None
needs_fim = pytest.mark.skipif(not HAS_FIM, reason="pyfim not installed")


def keyed(m):
    return {(r.antecedent, r.consequent): (round(r.p_adjusted, 12), r.count) for r in m.rules_.itertuples()}


@needs_fim
@pytest.mark.parametrize("min_count,max_len", [(1, 3), (7, 4), (25, 2), (60, 5)])
def test_pyfim_and_python_itemsets_identical(min_count, max_len):
    rng = np.random.default_rng(4)
    tx = [[f"i{j}" for j in range(20) if rng.random() < 0.2] for _ in range(700)]
    vocab = build_vocab(tx)
    a, ua = mine_itemsets(tx, vocab, min_count, max_len, "python")
    b, ub = mine_itemsets(tx, vocab, min_count, max_len, "pyfim")
    assert (ua, ub) == ("python", "pyfim") and a == b


@needs_fim
@pytest.mark.parametrize("mode", ["holdout", "direct"])
def test_full_pipeline_identical_across_backends(mode):
    tx = planted()
    a = RuleMiner(min_support=0.01, mode=mode, backend="python").fit(tx)
    b = RuleMiner(min_support=0.01, mode=mode, backend="pyfim").fit(tx)
    assert a.diagnostics_["backend"] == "python" and b.diagnostics_["backend"] == "pyfim"
    assert len(a.rules_) > 0 and keyed(a) == keyed(b)


def test_backend_selection_and_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, "fim", None)  # makes `import fim` raise ImportError
    with pytest.raises(ImportError):
        RuleMiner(backend="pyfim").fit(planted()[:500])
    m = RuleMiner(min_support=0.02, backend="auto").fit(planted()[:2000])
    assert m.diagnostics_["backend"] == "python"
    with pytest.raises(ValueError):
        RuleMiner(backend="nope").fit(planted()[:500])


def one_hot(tx):
    names = sorted({i for t in tx for i in t})
    idx = {n: j for j, n in enumerate(names)}
    M = np.zeros((len(tx), len(names)), dtype=bool)
    for r, t in enumerate(tx):
        for i in t:
            M[r, idx[i]] = True
    return M, names


def test_input_formats_give_identical_rules():
    pl = pytest.importorskip("polars")
    tx = planted(n=4000)
    M, names = one_hot(tx)
    ref = keyed(RuleMiner(min_support=0.01).fit(tx))
    assert ref
    df = pd.DataFrame(M, columns=names)
    inputs = {
        "numpy": (M, names),
        "scipy": (sp.csr_matrix(M), names),
        "pandas dense": (df, None),
        "pandas sparse": (df.astype(pd.SparseDtype(bool, False)), None),
        "pandas series of lists": (pd.Series(tx), None),
        "polars one-hot": (pl.from_pandas(df), None),
        "polars list column": (pl.DataFrame({"basket": tx}), None),
        "generator": ((t for t in tx), None),
    }
    for label, (X, nm) in inputs.items():
        got = keyed(RuleMiner(min_support=0.01).fit(X, item_names=nm))
        assert got == ref, f"{label}: rules differ"


def test_transform_accepts_matrix_with_fitted_names():
    tx = planted(n=4000)
    M, names = one_hot(tx)
    m = RuleMiner(min_support=0.01).fit(M, item_names=names)
    np.testing.assert_array_equal(m.transform(M[:50]), m.transform(tx[:50]))


def test_string_transactions_rejected():
    with pytest.raises(TypeError):
        RuleMiner().fit(["a,b,c", "a,b"])
    with pytest.raises(ValueError):
        RuleMiner().fit(np.ones((5, 3), dtype=bool), item_names=["x", "y"])


@needs_fim
def test_universal_items_excluded_and_backends_agree():
    """Regression: PyFIM silently drops itemsets made only of items present in every row."""
    rng = np.random.default_rng(8)
    tx = [["U1", "U2", "U3"] + [f"i{j}" for j in range(15) if rng.random() < 0.4] for _ in range(600)]
    vocab = build_vocab(tx)
    a, _ = mine_itemsets(tx, vocab, 30, 3, "python")
    b, _ = mine_itemsets(tx, vocab, 30, 3, "pyfim")
    assert a == b and len(a) > 0
    universal = {vocab[u] for u in ("U1", "U2", "U3")}
    assert not any(S & universal for S in a)
    for mode in ("holdout", "direct"):
        ra = RuleMiner(min_support=0.05, mode=mode, backend="python").fit(tx).rules_
        rb = RuleMiner(min_support=0.05, mode=mode, backend="pyfim").fit(tx).rules_
        assert len(ra) == len(rb)


@needs_fim
def test_dense_data_backends_identical():
    rng = np.random.default_rng(9)
    tx = [[f"i{j}" for j in range(40) if rng.random() < 0.55] for _ in range(500)]
    vocab = build_vocab(tx)
    a, _ = mine_itemsets(tx, vocab, 60, 3, "python")
    b, _ = mine_itemsets(tx, vocab, 60, 3, "pyfim")
    assert a == b


@needs_fim
def test_inconsistent_pyfim_output_falls_back(monkeypatch):
    import fim as real_fim
    monkeypatch.setattr(real_fim, "eclat", lambda *a, **k: [((1, 2), 10)])  # missing subsets
    tx = [["a", "b", "c"], ["a", "b"], ["a", "c"], ["b", "c"]] * 20
    vocab = build_vocab(tx)
    with pytest.warns(RuntimeWarning):
        out, used = mine_itemsets(tx, vocab, 5, 3, "pyfim")
    assert used == "python" and len(out) > 0
