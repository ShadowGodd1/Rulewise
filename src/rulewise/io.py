"""Input adapters: turn common data containers into lists of transactions."""
from __future__ import annotations

import numpy as np


def _from_lists(seq):
    out = []
    for t in seq:
        if t is None:
            out.append([])
        elif isinstance(t, (str, bytes)):
            raise TypeError("each transaction must be a collection of items, not a string "
                            f"(got {t!r}); split it first")
        else:
            out.append(list(t))
    return out


def _names_for(width, names):
    if names is None:
        return list(range(width))
    names = list(names)
    if len(names) != width:
        raise ValueError(f"item_names has {len(names)} entries but the data has {width} columns")
    return names


def _from_dense(M, names):
    M = np.asarray(M)
    if M.ndim != 2:
        raise ValueError("matrix input must be 2-dimensional")
    if M.dtype.kind == "c":
        raise ValueError("Complex data not supported")
    if M.shape[1] == 0:
        raise ValueError(f"0 feature(s) (shape={M.shape}) while a minimum of 1 is required.")
    if M.dtype.kind == "O":
        try:
            M = M.astype(float)
        except (TypeError, ValueError) as e:
            raise TypeError("argument must be a string or a number (a 0/1 presence matrix), "
                            "or pass a collection of item collections instead") from e
    names = _names_for(M.shape[1], names)
    if M.dtype.kind == "f" and not np.isfinite(M).all():
        raise ValueError("Input contains NaN or infinity; encode missing values as 0")
    mask = M != 0
    return [[names[j] for j in np.flatnonzero(row)] for row in mask], names


def _from_sparse(M, names):
    M = M.tocsr()
    names = _names_for(M.shape[1], names)
    ind, ptr, dat = M.indices, M.indptr, M.data
    out = []
    for i in range(M.shape[0]):
        sl = slice(ptr[i], ptr[i + 1])
        cols = ind[sl][dat[sl] != 0]
        out.append([names[j] for j in cols])
    return out, names


def to_transactions(X, item_names=None, default_names=None):
    """Return (transactions, names).

    ``names`` is the list of column labels when X was matrix-like (so later calls can map a
    matrix back to the same items), otherwise None. Accepted inputs: iterables of item
    collections, numpy 0/1 or boolean matrices, scipy sparse matrices, pandas DataFrames
    (dense or sparse one-hot) and Series of lists, polars DataFrames (one-hot, or a single
    list column) and Series of lists.
    """
    mod = type(X).__module__.split(".")[0]
    if item_names is None:
        item_names = default_names

    if mod == "polars":
        import polars as pl
        if isinstance(X, pl.Series):
            return _from_lists(X.to_list()), None
        if isinstance(X, pl.DataFrame):
            if X.width == 1 and str(X.schema[X.columns[0]]).startswith(("List", "Array")):
                return _from_lists(X.get_column(X.columns[0]).to_list()), None
            return _from_dense(X.to_numpy(), item_names or X.columns)
    if mod == "pandas":
        import pandas as pd
        if isinstance(X, pd.Series):
            return _from_lists(X.tolist()), None
        if isinstance(X, pd.DataFrame):
            names = item_names or list(X.columns)
            if len(X.columns) and all(isinstance(dt, pd.SparseDtype) for dt in X.dtypes):
                return _from_sparse(X.sparse.to_coo(), names)
            return _from_dense(X.to_numpy(), names)
    if mod == "scipy":
        import scipy.sparse as sp
        if sp.issparse(X):
            return _from_sparse(X, item_names)
    if isinstance(X, np.ndarray) and X.ndim == 2:
        return _from_dense(X, item_names)
    return _from_lists(X), None
