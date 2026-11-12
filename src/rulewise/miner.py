"""RuleMiner: fit / transform / recommend / validate."""
from __future__ import annotations

import inspect
import math
import warnings

import numpy as np
import pandas as pd

from ._compat import BaseEstimator, NotFittedError, TransformerMixin
from .report import ValidationReport
from .rulesmixin import RulesMixin
from .classmining import class_conditional_candidates
from .io import to_transactions
from .mining import build_vocab, encode, mine_itemsets
from .rules import (TidCounter, candidate_rules, itemset_partition_pvalue,
                    productivity_pvalue, rule_measures, search_space_sizes,
                    targeted_space_sizes)
from .stats import adjust_pvalues, bootstrap_lift_leverage, wilson_interval

_RULE_COLUMNS = ["antecedent", "consequent", "support", "confidence", "lift", "leverage",
                 "conviction", "count", "count_antecedent", "count_consequent",
                 "p_value", "p_adjusted"]


class RuleMiner(RulesMixin, TransformerMixin, BaseEstimator):
    """Mine association rules and keep only those that survive honest statistical testing.

    Data are split into an exploration part (mining) and a holdout part (testing), so the
    data that discovers a rule is never the data that tests it. Each candidate rule is tested
    with Webb's productivity test on the holdout, and p-values are corrected for the number of
    candidates tested.

    Parameters
    ----------
    min_support : fraction of exploration transactions an itemset must appear in.
    min_confidence : candidate rules below this confidence are not tested.
    max_antecedent, max_consequent : size limits for rule sides.
    alpha : significance level after correction.
    correction : 'holm' | 'bonferroni' (control FWER) or 'bh' | 'by' (control FDR) | 'none'.
    mode : 'holdout' (default) tests candidates on held-out data (Webb 2007) and corrects for
        the number of candidates tested. 'direct' uses all data for both mining and testing and
        applies Webb's layered critical values (2008): alpha is split equally across rule sizes
        and Bonferroni-corrected within each size over the FULL search space. It uses every
        transaction, so it is usually more powerful, but it is conservative when there are many
        items and its reported support/confidence/lift and their intervals are measured on the
        same data that selected the rules, so they are optimistically biased (winner's curse).
        Use holdout mode when you need honest effect-size estimates. `correction` is ignored.
    holdout_fraction : share of transactions reserved for testing in holdout mode. 0 disables
        the split (p-values are then NOT valid and a warning is raised).
    n_bootstrap : resamples for lift/leverage intervals on significant rules (0 disables).
    backend : 'auto' (use PyFIM if installed, else pure Python), 'pyfim' or 'python'.
    max_cache_mb : memory bound for cached itemset intersections during testing.
    random_state : seed for the split and for validation resampling.
    """

    def __init__(self, min_support=0.01, min_confidence=0.1, max_antecedent=3, max_consequent=1,
                 alpha=0.05, correction="holm", mode="holdout", holdout_fraction=0.5,
                 n_bootstrap=200, backend="auto", max_cache_mb=128.0, random_state=0):
        self.min_support = min_support
        self.min_confidence = min_confidence
        self.max_antecedent = max_antecedent
        self.max_consequent = max_consequent
        self.alpha = alpha
        self.correction = correction
        self.mode = mode
        self.n_bootstrap = n_bootstrap
        self.backend = backend
        self.max_cache_mb = max_cache_mb
        self.holdout_fraction = holdout_fraction
        self.random_state = random_state

    def __sklearn_tags__(self):
        try:
            tags = super().__sklearn_tags__()
        except AttributeError:  # scikit-learn < 1.6
            return {}
        tags.input_tags.sparse = True
        return tags

    def fit(self, X, y=None, item_names=None):
        """Fit on transactions: item collections, 0/1 matrices, sparse, pandas or polars data.

        ``y`` is ignored (present for scikit-learn pipeline compatibility).
        """
        tx, names = to_transactions(X, item_names)
        self._fit_tx(tx)
        self._names = names
        if names is not None:
            self.n_features_in_ = len(names)
        return self

    def _fit_tx(self, tx, targets=None, class_conditional=False):
        n = len(tx)
        if n < 2:
            raise ValueError("need at least 2 transactions")
        if self.mode not in ("holdout", "direct"):
            raise ValueError("mode must be 'holdout' or 'direct'")
        if not 0.0 <= self.holdout_fraction <= 1.0:
            raise ValueError("holdout_fraction must be between 0 and 1")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        rng = np.random.default_rng(self.random_state)
        if self.mode == "direct":
            expl = hold = tx
        elif self.holdout_fraction > 0:
            perm = rng.permutation(n)
            n_hold = max(1, int(round(n * self.holdout_fraction)))
            hold_idx, expl_idx = perm[:n_hold], perm[n_hold:]
            expl = [tx[i] for i in expl_idx]
            hold = [tx[i] for i in hold_idx]
        else:
            warnings.warn("holdout_fraction=0: rules are tested on the data that found them; "
                          "p-values are not valid.", stacklevel=2)
            expl = hold = tx
        self.vocab_ = build_vocab(tx)
        self.items_ = {j: it for it, j in self.vocab_.items()}

        n_expl = len(expl)
        min_count = max(1, math.ceil(self.min_support * n_expl))
        target_ids = ({self.vocab_[t] for t in targets if t in self.vocab_}
                      if targets is not None else None)
        if target_ids is not None and class_conditional:
            cands, n_itemsets, backend_used = class_conditional_candidates(
                expl, self.vocab_, target_ids, min_count, self.max_antecedent,
                self.min_confidence, self.backend)
        else:
            itemsets, backend_used = mine_itemsets(expl, self.vocab_, min_count,
                                                   self.max_antecedent + self.max_consequent,
                                                   self.backend)
            n_itemsets = len(itemsets)
            cands = candidate_rules(itemsets, n_expl, self.min_confidence,
                                    self.max_antecedent, self.max_consequent, target_ids)

        t_hold, n_hold = encode(hold, self.vocab_)
        tc = TidCounter(t_hold, n_hold, self.max_cache_mb)
        rows, pvals = [], []
        if self.mode == "direct":
            if target_ids is not None:
                sizes = targeted_space_sizes(len(self.vocab_) - len(target_ids), len(target_ids),
                                             self.max_antecedent)
            else:
                sizes = search_space_sizes(len(self.vocab_), self.max_antecedent,
                                           self.max_consequent)
            n_layers = len(sizes)
        for X, Y in cands:
            # A p-value at or above `cap` can never be significant after correction, so the
            # tests may stop there (the stored p_value is then a lower bound, not exact).
            cap = (self.alpha / (n_layers * sizes[len(X) + len(Y)])
                   if self.mode == "direct" else self.alpha)
            p = productivity_pvalue(X, Y, tc, cap)
            if len(Y) > 1 and p < cap:
                p = max(p, itemset_partition_pvalue(X | Y, tc, cap))
            n_xy, n_x, n_y = tc.count(X | Y), tc.count(X), tc.count(Y)
            sup, conf, lift, lev, conv = rule_measures(n_xy, n_x, n_y, n_hold)
            rows.append(dict(
                antecedent=tuple(sorted((self.items_[i] for i in X), key=str)),
                consequent=tuple(sorted((self.items_[i] for i in Y), key=str)),
                support=sup, confidence=conf,
                lift=lift, leverage=lev, conviction=conv, count=n_xy,
                count_antecedent=n_x, count_consequent=n_y, p_value=p,
                _X=X, _Y=Y))
            pvals.append(p)
        if self.mode == "direct":
            adj = [min(1.0, p * n_layers * sizes[len(r["_X"]) + len(r["_Y"])])
                   for p, r in zip(pvals, rows)]
        else:
            adj = adjust_pvalues(pvals, self.correction)
        for r, a in zip(rows, adj):
            r["p_adjusted"] = a
        df = pd.DataFrame(rows)
        n_cand = len(df)
        if n_cand:
            replicated = (float((df.p_value < self.alpha).mean())
                          if self.mode == "holdout" else float("nan"))
            sig = df[df.p_adjusted < self.alpha].sort_values(["p_adjusted", "lift"],
                                                              ascending=[True, False])
        else:
            replicated, sig = float("nan"), df
        if n_cand and len(sig):
            ci_w = [wilson_interval(r.count, r.count_antecedent, self.alpha)
                    for r in sig.itertuples()]
            sig = sig.assign(confidence_low=[c[0] for c in ci_w],
                             confidence_high=[c[1] for c in ci_w])
        if n_cand and len(sig) and self.n_bootstrap > 0:
            brng = np.random.default_rng(int(rng.integers(1 << 31)))
            ci = [bootstrap_lift_leverage(r.count, r.count_antecedent, r.count_consequent,
                                          n_hold, self.n_bootstrap, self.alpha, brng)
                  for r in sig.itertuples()]
            sig = sig.assign(lift_low=[c[0] for c in ci], lift_high=[c[1] for c in ci],
                             leverage_low=[c[2] for c in ci], leverage_high=[c[3] for c in ci])
        self.rules_ = sig.drop(columns=["_X", "_Y"], errors="ignore").reset_index(drop=True)
        if not len(self.rules_):
            self.rules_ = pd.DataFrame(columns=_RULE_COLUMNS)
        self.diagnostics_ = dict(mode=self.mode, backend=backend_used,
                                 honest_estimates=(self.mode == "holdout" and self.holdout_fraction > 0),
                                 n_transactions=n, n_explore=n_expl, n_holdout=n_hold,
                                 n_frequent_itemsets=n_itemsets, n_candidates=n_cand,
                                 n_significant=len(self.rules_),
                                 replication_rate=replicated)
        return self

    def _check(self):
        if not hasattr(self, "rules_"):
            raise NotFittedError("call fit() first")

    def transform(self, transactions):
        """Boolean matrix (n_transactions x n_rules): does the rule's antecedent hold?"""
        self._check()
        self._check_width(transactions)
        transactions, _ = to_transactions(transactions, default_names=self._names)
        if not len(self.rules_):
            return np.zeros((len(transactions), 0), dtype=bool)
        out = np.zeros((len(transactions), len(self.rules_)), dtype=bool)
        ants = [set(a) for a in self.rules_.antecedent]
        for i, t in enumerate(transactions):
            s = set(t)
            for j, a in enumerate(ants):
                out[i, j] = a <= s
        return out

    def recommend(self, basket, k=5, by="confidence_low", explain=False):
        """Rank items not in the basket using the significant rules that fire on it.

        Each candidate item is scored by its best matching rule on ``by``: 'confidence_low'
        (default; the Wilson lower bound, which discounts rules resting on little data),
        'confidence', 'lift' or 'leverage'. Returns [(item, score)], or with ``explain=True``
        a DataFrame that also names the rule behind each recommendation.
        """
        self._check()
        if by not in ("confidence_low", "confidence", "lift", "leverage"):
            raise ValueError("by must be 'confidence_low', 'confidence', 'lift' or 'leverage'")
        s = set(basket)
        best = {}
        if len(self.rules_):
            for r in self.rules_.itertuples():
                if set(r.antecedent) <= s:
                    score = getattr(r, by)
                    for item in r.consequent:
                        if item not in s and (item not in best or score > best[item][0]):
                            best[item] = (score, r)
        ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:k]
        if not explain:
            return [(item, sc) for item, (sc, _) in ranked]
        return pd.DataFrame([dict(item=item, score=sc, antecedent=r.antecedent,
                                  confidence=r.confidence, lift=r.lift,
                                  p_adjusted=r.p_adjusted) for item, (sc, r) in ranked])

    def stability(self, transactions, n_resamples=20, fraction=0.8):
        """Refit on random subsamples and measure how much the rule set changes.

        Returns the mean pairwise Jaccard similarity of the rule sets and, for every rule found
        on the full data, the share of subsamples in which it reappears. Subsamples hold less
        data than the full fit, so frequencies below 1 are expected even for genuine rules;
        compare rules with each other rather than reading the number as a probability.
        """
        tx, names = to_transactions(transactions, default_names=getattr(self, "_names", None))
        n = len(tx)
        base = self.fit(tx)
        self._names = names
        key = lambda r: (r.antecedent, r.consequent)
        base_keys = [key(r) for r in base.rules_.itertuples()]
        rng = np.random.default_rng(self.random_state)
        sets = []
        for _ in range(n_resamples):
            idx = rng.choice(n, size=max(2, int(fraction * n)), replace=False)
            m = RuleMiner(**{**self.get_params(), "random_state": int(rng.integers(1 << 31))})
            m.fit([tx[i] for i in idx])
            sets.append({key(r) for r in m.rules_.itertuples()})
        jac = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = sets[i] | sets[j]
                jac.append(len(sets[i] & sets[j]) / len(u) if u else 1.0)
        freq = pd.DataFrame(
            [dict(antecedent=a, consequent=c, frequency=sum((a, c) in s_ for s_ in sets) / len(sets))
             for a, c in base_keys])
        if len(freq):
            freq = freq.sort_values("frequency", ascending=False).reset_index(drop=True)
        return dict(n_resamples=n_resamples, fraction=fraction,
                    mean_jaccard=float(np.mean(jac)) if jac else float("nan"),
                    rule_frequency=freq)

    def validate(self, transactions, n_null=20, n_stability=0):
        """Validate the pipeline on this data and return a :class:`ValidationReport`.

        * Null calibration: refit on ``n_null`` datasets with the same item frequencies but no
          associations. A sound pipeline returns (almost) no rules; the share of null datasets
          with at least one rule estimates the empirical family-wise error rate.
        * Holdout replication (holdout mode) and, if ``n_stability > 0``, stability of the rule
          set across that many random subsamples.
        """
        tx, names = to_transactions(transactions, default_names=getattr(self, "_names", None))
        n = len(tx)
        counts = {}
        for t in tx:
            for item in set(t):
                counts[item] = counts.get(item, 0) + 1
        rng = np.random.default_rng(self.random_state)
        n_rules = []
        for _ in range(n_null):
            null = [[] for _ in range(n)]
            for item, c in counts.items():
                for i in rng.choice(n, size=c, replace=False):
                    null[i].append(item)
            m = RuleMiner(**{**self.get_params(), "random_state": int(rng.integers(1 << 31))})
            m._fit_tx(null)
            n_rules.append(len(m.rules_))
        n_rules = np.array(n_rules)
        real = self.fit(tx)
        self._names = names
        hits = int((n_rules > 0).sum())
        ci = wilson_interval(hits, n_null, 0.05)
        rep = ValidationReport(
            mode=self.mode, alpha=self.alpha,
            honest_estimates=real.diagnostics_["honest_estimates"],
            n_rules_real=len(real.rules_),
            replication_rate=real.diagnostics_["replication_rate"],
            null_runs=n_null, null_rule_counts=[int(c) for c in n_rules],
            null_mean_rules=float(n_rules.mean()), null_fwer=hits / n_null,
            null_fwer_ci=ci, calibrated=bool(ci[0] <= self.alpha))
        if n_stability > 0:
            st = self.stability(tx, n_resamples=n_stability)
            self._names = names
            rep["stability_resamples"] = n_stability
            rep["stability_mean_jaccard"] = st["mean_jaccard"]
            rep["rule_frequency"] = st["rule_frequency"]
        return rep
