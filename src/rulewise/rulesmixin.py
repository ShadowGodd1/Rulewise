"""Shared helpers for fitted estimators that expose a ``rules_`` DataFrame."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def fmt_item(item) -> str:
    if isinstance(item, tuple) and len(item) == 2 and item[0] == "not":
        return f"not {item[1]}"
    return str(item)


def _join(items) -> str:
    return " & ".join(fmt_item(i) for i in items)


def flatten_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Turn tuple columns into 'a & b' strings so the table exports cleanly."""
    df = df.copy()
    for col in ("antecedent", "consequent"):
        if col in df:
            df[col] = df[col].map(_join)
    return df


class RulesMixin:
    """query / export / summary / plot for estimators with ``rules_``."""

    def _check_width(self, X):
        """Reject a matrix whose column count differs from the one seen in fit()."""
        n = getattr(self, "n_features_in_", None)
        shape = getattr(X, "shape", None)
        if n is not None and shape is not None and len(shape) == 2 and shape[1] != n:
            raise ValueError(f"X has {shape[1]} features, but {type(self).__name__} is "
                             f"expecting {n} features as input")

    def _frame(self) -> pd.DataFrame:
        self._check()
        df = self.rules_.copy()
        if "consequent" not in df.columns and "class_" in df.columns:
            df.insert(1, "consequent", [(f"class={c}",) for c in df["class_"]])
        return df

    def query(self, min_lift=None, min_confidence=None, min_support=None, max_p=None,
              contains=None, antecedent_contains=None, consequent_contains=None,
              max_antecedent=None, sort_by=None, ascending=False, top=None) -> pd.DataFrame:
        """Filter and sort the significant rules.

        ``contains`` / ``antecedent_contains`` / ``consequent_contains`` take one item or a
        list of items that must all be present (anywhere / in the antecedent / in the
        consequent). ``max_p`` filters on the adjusted p-value.
        """
        df = self._frame()
        if not len(df):
            return df

        def as_set(x):
            return set(x) if isinstance(x, (list, set, frozenset)) else {x}

        keep = pd.Series(True, index=df.index)
        if min_lift is not None:
            keep &= df["lift"] >= min_lift
        if min_confidence is not None:
            keep &= df["confidence"] >= min_confidence
        if min_support is not None:
            keep &= df["support"] >= min_support
        if max_p is not None:
            keep &= df["p_adjusted"] <= max_p
        if max_antecedent is not None:
            keep &= df["antecedent"].map(len) <= max_antecedent
        if contains is not None:
            need = as_set(contains)
            keep &= pd.Series([need <= (set(a) | set(c))
                               for a, c in zip(df["antecedent"], df["consequent"])],
                              index=df.index)
        if antecedent_contains is not None:
            need = as_set(antecedent_contains)
            keep &= pd.Series([need <= set(a) for a in df["antecedent"]], index=df.index)
        if consequent_contains is not None:
            need = as_set(consequent_contains)
            keep &= pd.Series([need <= set(c) for c in df["consequent"]], index=df.index)
        df = df[keep]
        if sort_by is not None:
            df = df.sort_values(sort_by, ascending=ascending)
        if top is not None:
            df = df.head(top)
        return df.reset_index(drop=True)

    def export(self, path, **query_kwargs):
        """Write the (optionally filtered) rules to .csv, .json or .parquet and return the path."""
        flat = flatten_rules(self.query(**query_kwargs))
        ext = Path(path).suffix.lower()
        if ext == ".csv":
            flat.to_csv(path, index=False)
        elif ext == ".json":
            flat.to_json(path, orient="records", indent=2)
        elif ext == ".parquet":
            flat.to_parquet(path, index=False)
        else:
            raise ValueError("path must end in .csv, .json or .parquet")
        return path

    def to_polars(self, **query_kwargs):
        import polars as pl
        return pl.DataFrame(flatten_rules(self.query(**query_kwargs)).to_dict(orient="list"))

    def summary(self, top: int = 5) -> str:
        self._check()
        d = getattr(self, "diagnostics_", {})
        head = (f"{type(self).__name__}: {len(self.rules_)} significant rules "
                f"(alpha={self.alpha}, mode={d.get('mode')}, backend={d.get('backend')}); "
                f"{d.get('n_candidates')} candidates tested on {d.get('n_holdout')} transactions")
        if not len(self.rules_):
            return head + "\nNo rule passed the significance filter."
        df = flatten_rules(self._frame()).head(top)
        lines = [head, f"Top {len(df)} by adjusted p-value:"]
        for r in df.itertuples():
            tail = r.consequent if "consequent" in df.columns else r.class_
            lines.append(f"  {r.antecedent}  ->  {tail}   conf={r.confidence:.3f} "
                         f"[{r.confidence_low:.3f}, {r.confidence_high:.3f}]  lift={r.lift:.2f}  "
                         f"p_adj={r.p_adjusted:.2e}")
        return "\n".join(lines)

    def plot(self, kind="effect", **kwargs):
        """kind: 'effect' (lift vs significance), 'forest' (intervals), 'network'."""
        from . import plot as _plot
        fn = {"effect": _plot.plot_effect_sizes, "forest": _plot.plot_forest,
              "network": _plot.plot_network}.get(kind)
        if fn is None:
            raise ValueError("kind must be 'effect', 'forest' or 'network'")
        return fn(self, **kwargs)
