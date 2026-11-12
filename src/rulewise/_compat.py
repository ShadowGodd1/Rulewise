"""Optional scikit-learn integration: use its base classes when installed, else tiny fallbacks."""
from __future__ import annotations

import inspect

try:  # pragma: no cover - exercised implicitly depending on environment
    from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
    from sklearn.exceptions import NotFittedError
    HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    HAS_SKLEARN = False

    class NotFittedError(RuntimeError, ValueError):
        """Raised when a fitted attribute is needed before fit()."""

    class BaseEstimator:
        def get_params(self, deep=True):
            return {k: getattr(self, k) for k in inspect.signature(type(self).__init__).parameters
                    if k != "self"}

        def set_params(self, **params):
            valid = self.get_params()
            for k, v in params.items():
                if k not in valid:
                    raise ValueError(f"invalid parameter {k!r}")
                setattr(self, k, v)
            return self

    class TransformerMixin:
        def fit_transform(self, X, y=None, **fit_params):
            return self.fit(X, y, **fit_params).transform(X)

    class ClassifierMixin:
        def score(self, X, y):
            import numpy as np
            return float(np.mean(np.asarray(self.predict(X)) == np.asarray(y)))
