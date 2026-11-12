"""RuleClassifier: an associative classifier built only from statistically significant rules."""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._compat import BaseEstimator, ClassifierMixin, NotFittedError
from .io import to_transactions
from .miner import RuleMiner
from .report import ValidationReport
from .rulesmixin import RulesMixin
from .stats import wilson_interval

_CLASS = "__rulewise_class__"


class RuleClassifier(RulesMixin, ClassifierMixin, BaseEstimator):
    """Classify with class-association rules that survive honest statistical testing.

    Training mines rules ``antecedent -> class`` and keeps only those that are significantly
    productive after multiple-testing correction (see :class:`RuleMiner`). Prediction finds the
    matching rule with the highest Wilson lower bound on its confidence; that rule supplies the
    probability of its class, and the remaining probability is shared among the other classes
    in proportion to their training frequency. For two classes this is exactly the rule's
    estimated probability for each class. When no rule matches, class frequencies are returned.

    Because only significant rules are kept, the model can abstain from structure it cannot
    justify: on data with no real signal it predicts the majority class and says so
    (``rules_`` is empty).

    Notes
    -----
    * ``min_support`` is a fraction of ALL training transactions, so rules for rare classes
      need a lower value than for common ones.
    * ``mode='direct'`` (default) uses all data to find and test rules; in cross-validated
      benchmarks it predicted better than ``'holdout'`` on every dataset tried, most of all on
      small data, because holdout discards half the sample. Its in-sample rule confidences are
      optimistically biased, so judge probabilities by cross-validation, not by ``rules_``.
      ``mode='holdout'`` estimates rule confidences on data not used to find the rules.
    * Best suited to data where the class follows a few conjunctions of features (tabular,
      categorical, behavioural). It uses one best rule per prediction, so it is weak where
      evidence is spread thinly over many features (e.g. image pixels); see
      ``benchmarks/RESULTS.md``.
    * ``class_conditional=True`` (default) mines each class separately, enumerating only
      itemsets frequent within a class. It yields exactly the rules the general miner would
      (tested) but is much faster on dense data; ``False`` uses the general miner.
    * ``add_negations=True`` also creates an item ``('not', x)`` for every feature ``x`` a row
      lacks, so rules can express absence (e.g. ``a and b and not c``). Plain itemsets cannot.
      It doubles the item count and densifies the data, so use it with a modest number of
      features and a higher ``min_support``.
    * Numeric features must be discretised first (e.g. scikit-learn's KBinsDiscretizer with
      ``encode='onehot'``); a sparse matrix is accepted directly.
    """

    def __init__(self, min_support=0.01, min_confidence=0.0, max_antecedent=3, alpha=0.05,
                 correction="holm", mode="direct", holdout_fraction=0.5, backend="auto",
                 max_cache_mb=128.0, add_negations=False, class_conditional=True,
                 random_state=0):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_antecedent = max_antecedent
        self.alpha = alpha
        self.correction = correction
        self.mode = mode
        self.holdout_fraction = holdout_fraction
        self.backend = backend
        self.max_cache_mb = max_cache_mb
        self.add_negations = add_negations
        self.class_conditional = class_conditional
        self.random_state = random_state

    def __sklearn_tags__(self):
        try:
            tags = super().__sklearn_tags__()
        except AttributeError:  # scikit-learn < 1.6
            return {}
        tags.input_tags.sparse = True
        return tags

    def fit(self, X, y, item_names=None):
        tx, names = to_transactions(X, item_names)
        y = np.asarray(y)
        if len(y) != len(tx):
            raise ValueError(f"X has {len(tx)} transactions but y has {len(y)} labels")
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k = len(self.classes_)
        self.priors_ = np.bincount(y_idx, minlength=k) / len(y_idx)
        self._names = names
        self._universe = None
        if names is not None:
            self.n_features_in_ = len(names)
        if self.add_negations:
            self._universe = sorted({i for t in tx for i in t}, key=str)
            tx = self._negate(tx)
        class_items = [(_CLASS, c) for c in range(k)]
        aug = [t + [class_items[c]] for t, c in zip(tx, y_idx)]
        miner = RuleMiner(min_support=self.min_support, min_confidence=self.min_confidence,
                          max_antecedent=self.max_antecedent, max_consequent=1, alpha=self.alpha,
                          correction=self.correction, mode=self.mode,
                          holdout_fraction=self.holdout_fraction, n_bootstrap=0,
                          backend=self.backend, max_cache_mb=self.max_cache_mb,
                          random_state=self.random_state)
        miner._fit_tx(aug, targets=class_items, class_conditional=self.class_conditional)
        self.miner_ = miner
        self.diagnostics_ = dict(miner.diagnostics_)
        self._build_rules(miner)
        return self

    def _negate(self, tx):
        out = []
        for t in tx:
            present = set(t)
            out.append(list(present) + [("not", i) for i in self._universe if i not in present])
        return out

    def _build_rules(self, miner):
        r = miner.rules_
        if len(r) == 0:
            self.rules_ = pd.DataFrame(columns=["antecedent", "class_", "support", "confidence",
                                                "confidence_low", "confidence_high", "lift",
                                                "leverage", "count", "count_antecedent",
                                                "count_consequent", "p_value", "p_adjusted"])
            self._A = sp.csr_matrix((0, len(miner.vocab_)), dtype=np.int32)
            self._sizes = np.zeros(0, dtype=np.int32)
            self._rule_class = np.zeros(0, dtype=int)
            self._rule_conf = np.zeros(0)
            return
        r = r.copy()
        r["_k"] = [c[0][1] for c in r.consequent]
        r = r.sort_values(["confidence_low", "confidence", "count"],
                          ascending=[False, False, False]).reset_index(drop=True)
        vocab = miner.vocab_
        rows, cols = [], []
        for i, ant in enumerate(r.antecedent):
            for it in ant:
                rows.append(i)
                cols.append(vocab[it])
        self._A = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                                shape=(len(r), len(vocab)))
        self._sizes = r.antecedent.map(len).to_numpy().astype(np.int32)
        self._rule_class = r["_k"].to_numpy().astype(int)
        self._rule_conf = r["confidence"].to_numpy()
        r["class_"] = self.classes_[self._rule_class]
        keep = ["antecedent", "class_", "support", "confidence", "confidence_low",
                "confidence_high", "lift", "leverage", "count", "count_antecedent",
                "p_value", "p_adjusted"]
        self.rules_ = r[keep]

    def _check(self):
        if not hasattr(self, "miner_"):
            raise NotFittedError("call fit() first")

    def _best_rule(self, X):
        """Index of the highest-priority matching rule per row (-1 if none)."""
        self._check_width(X)
        tx, _ = to_transactions(X, default_names=self._names)
        if self._universe is not None:
            tx = self._negate(tx)
        vocab = self.miner_.vocab_
        rows, cols = [], []
        for i, t in enumerate(tx):
            for it in set(t):
                j = vocab.get(it)
                if j is not None:
                    rows.append(i)
                    cols.append(j)
        n = len(tx)
        best = np.full(n, -1, dtype=int)
        if self._A.shape[0] == 0 or n == 0:
            return best
        Xc = sp.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)),
                           shape=(n, len(vocab)))
        M = (Xc @ self._A.T).tocsr()
        M.data = (M.data == self._sizes[M.indices]).astype(np.int8)
        M.eliminate_zeros()
        M.sort_indices()
        nz = np.diff(M.indptr) > 0
        best[nz] = M.indices[M.indptr[:-1][nz]]  # rules are stored in priority order
        return best

    def predict_proba(self, X):
        self._check()
        best = self._best_rule(X)
        n, k = len(best), len(self.classes_)
        P = np.tile(self.priors_, (n, 1))
        hit = best >= 0
        if hit.any() and k > 1:
            c = self._rule_class[best[hit]]
            conf = self._rule_conf[best[hit]]
            rest = np.tile(self.priors_, (int(hit.sum()), 1))
            rest[np.arange(len(c)), c] = 0.0
            tot = rest.sum(axis=1, keepdims=True)
            rest = np.where(tot > 0, rest / np.where(tot > 0, tot, 1.0), 1.0 / (k - 1))
            P[hit] = rest * (1.0 - conf)[:, None]
            P[np.nonzero(hit)[0], c] = conf
        return P

    def predict(self, X):
        P = self.predict_proba(X)
        return self.classes_[np.argmax(P, axis=1)]

    def explain(self, X):
        """For each row, the rule that determined its probabilities (None if none matched)."""
        self._check()
        best = self._best_rule(X)
        return [self.rules_.iloc[b].to_dict() if b >= 0 else None for b in best]

    def validate(self, X, y, n_null=20, cv=5):
        """Validate on this data and return a :class:`ValidationReport`.

        * Null calibration: refit with the labels randomly permuted (features untouched, so any
          rule found is spurious); the share of permuted datasets with at least one rule
          estimates the empirical error rate.
        * ``cv``-fold cross-validated accuracy and log-loss against the majority-class and
          class-prior baselines (does the rule set beat knowing only the class frequencies?).
        """
        tx, names = to_transactions(X, default_names=getattr(self, "_names", None))
        y = np.asarray(y)
        n = len(tx)
        params = self.get_params()
        rng = np.random.default_rng(self.random_state)
        n_rules = []
        for _ in range(n_null):
            m = RuleClassifier(**{**params, "random_state": int(rng.integers(1 << 31))})
            m.fit(tx, rng.permutation(y))
            n_rules.append(len(m.rules_))
        n_rules = np.array(n_rules)
        accs, lls, base_accs, base_lls = [], [], [], []
        folds = np.array_split(rng.permutation(n), cv)
        for k in range(cv):
            test = folds[k]
            train = np.concatenate([folds[j] for j in range(cv) if j != k])
            m = RuleClassifier(**params).fit([tx[i] for i in train], y[train])
            P = m.predict_proba([tx[i] for i in test])
            yt = y[test]
            col = {c: j for j, c in enumerate(m.classes_)}
            p_true = np.array([P[r, col[c]] if c in col else 1e-6 for r, c in enumerate(yt)])
            prior = np.array([m.priors_[col[c]] if c in col else 1e-6 for c in yt])
            accs.append(float(np.mean(m.classes_[P.argmax(1)] == yt)))
            lls.append(float(-np.mean(np.log(np.clip(p_true, 1e-6, 1)))))
            base_accs.append(float(np.mean(yt == m.classes_[np.argmax(m.priors_)])))
            base_lls.append(float(-np.mean(np.log(np.clip(prior, 1e-6, 1)))))
        real = self.fit(tx, y)
        self._names = names
        hits = int((n_rules > 0).sum())
        ci = wilson_interval(hits, n_null, 0.05)
        return ValidationReport(
            mode=self.mode, alpha=self.alpha,
            honest_estimates=real.diagnostics_["honest_estimates"],
            n_rules_real=len(real.rules_), replication_rate=real.diagnostics_["replication_rate"],
            null_runs=n_null, null_kind="datasets with permuted labels",
            null_rule_counts=[int(c) for c in n_rules], null_mean_rules=float(n_rules.mean()),
            null_fwer=hits / n_null, null_fwer_ci=ci, calibrated=bool(ci[0] <= self.alpha),
            cv_folds=cv, cv_accuracy=float(np.mean(accs)), cv_accuracy_sd=float(np.std(accs)),
            majority_accuracy=float(np.mean(base_accs)), cv_log_loss=float(np.mean(lls)),
            prior_log_loss=float(np.mean(base_lls)))
