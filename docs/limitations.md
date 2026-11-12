# Limitations

No method is free of trade-offs. These are the limits that matter when deciding whether rulewise
is appropriate, with the measured evidence where it exists.

## Recall is lower than a classical list

The error guarantee is conservative. On Groceries, rulewise returned about half as many
held-out-productive rules as a classical list in direct mode, and about a third in holdout mode.
If your goal is to collect every possible candidate and filter by hand, a classical miner is a
larger haystack.

The power study in `benchmarks/RESULTS.md` shows the same effect. With 20 planted 3-item patterns
among 1,000 items, n = 5,000 and minimum support 0.001, rulewise recovered 9 of 20 patterns in
holdout mode and 11 of 20 in direct mode at the weakest signal. It recovered 20 of 20 once the
signal was stronger (strength 0.005 and above), with zero false rules. Weak effects need more
data.

Mitigations:

* use direct mode if you need power and can accept optimistic effect sizes;
* lower `min_support` and `min_confidence`, at the cost of more candidates and a slower fit;
* increase the sample size;
* use `alpha=0.1` if a 10% family-wise error rate is acceptable for exploration, and confirm on
  a fresh sample.

## Redundancy is reduced, not eliminated

The productivity test removes rules whose extra antecedent items add nothing. It does not remove
nested sub-rules or reverse rules. In the planted benchmark, four planted structures produced 35
returned rules: the four planted rules, their sub-rules, and their reverse directions. All 35
were productive on held-out data, so they are not false positives, but the set is not minimal.

If you need a smaller set, use `query` to filter by antecedent size or confidence, or add a
non-redundant or self-sufficient-itemset post-processing step. A built-in minimal-rule mode is a
natural next feature, not something the current release guarantees.

## Basket size and other confounders

The test asks whether two items are independent. Items that co-occur only because some baskets
are larger are dependent, so a rule can be reported even though the extra item does not cause the
other. The same applies to a third variable that drives both items. The shuffle null in
`validate` preserves item frequencies but does not model basket-size effects or confounding.

This is a property of the null hypothesis, not a bug: the rule is a real association in the
sample. It is not evidence of causation or of a useful intervention.

## Independent and identically distributed rows

All p-values assume exchangeable transactions. Repeated baskets from the same household, grouped
or clustered data, time series, and survey weights violate this. The effective sample size is
smaller than the row count, and the p-values can be too small.

Mitigations:

* split by group, not by row, when fitting and validating;
* use a holdout part that is independent in the same sense;
* treat the output as descriptive, not confirmatory, when the rows are not exchangeable.

## Direct mode effect sizes are optimistic

In direct mode, the same data selects the rules and estimates their confidence and lift. Rules
are selected partly because their effect is large in that sample, so the reported effects are
biased upward. This is the winner's curse. The p-values remain valid under the layered
correction, but the confidence and lift columns do not.

Use holdout mode when you need honest effect sizes, or validate direct-mode rules on a fresh
sample.

## Small samples and rare items

A rule with a small `count` can have a large lift and a wide interval. The Wilson interval and the
bootstrap interval show this, but the point estimate can still mislead. Items that occur in every
transaction are excluded because they cannot appear in a significant rule. Rare items may never
reach the support floor.

Raise `min_support` for dense data to keep the candidate count manageable. Lower it only when
the sample is large enough for the resulting counts to be meaningful.

## Dense numeric data

rulewise predicts or describes with conjunctions of discrete items. Numeric features must be
binned first. Signal spread thinly across many numeric features is a weak case: each rule covers
a small region, and prediction uses one best rule rather than accumulating evidence. On digits,
the classifier reached about 0.61 accuracy against 0.97 for logistic regression. Use a linear
model or a tree ensemble when accuracy is the goal.

Binning also matters. Quantile bins inside each training fold avoid leakage. Too many bins
produce sparse items; too few lose signal.

## Positive associations only

Candidate rules require lift above 1. rulewise does not report negative associations, rules with
lift below 1, or independence rules. It also does not mine closed or maximal itemsets, does not
support item constraints, and does not implement top-k mining. mlxtend and other libraries cover
some of those cases.

## Recommendation quality

`recommend` is an explainable interface, not a state-of-the-art recommender. On Groceries with one
hidden item per basket, hit@5 was 18.7% in holdout mode and 19.7% in direct mode. A popularity
baseline reached 26.4%, and a classical rule list reached 20.1%. The value of `recommend` is that
every recommendation names the rule behind it, not that it maximizes hit rate.

## Classification is competitive, not superior

On categorical data with a few strong conjunctions, such as Mushrooms, the classifier is close to
logistic regression: 0.9970 accuracy against 0.9998. On Titanic it is a few points behind
logistic regression and a depth-4 tree. The advantage is readability and an interval on every
rule, not accuracy.

## Multiple testing is conservative

The empirical family-wise error rate on null data is about 0.8% at a nominal 5%, not 5%. The
correction and the productivity test are both conservative, and discreteness of the Fisher test
adds further conservatism. This is deliberate: the method is designed to avoid false rules rather
than to maximize discoveries.

The practical consequence is that a null result does not prove independence. It means the data
did not provide enough evidence at the chosen error rate. Report the candidate count, the sample
size and the support floor alongside the result.

## PyFIM is optional and legacy

PyFIM is a C extension with wheels only for old Python versions. It is not installed by default
and the core package uses the pure-Python backend. The pure-Python backend produced identical
itemsets in the tests and was faster than mlxtend on the benchmark datasets, so PyFIM is not
needed for typical use.

## No causal claims

A significant rule is an association in the sample. It is not evidence that changing the
antecedent will change the consequent. Confounding, selection and reverse causation remain
possible. Use domain knowledge and, when possible, an experiment before acting on a rule.

## What to report

When publishing results, report:

* the mode and `alpha`;
* the correction method;
* `min_support`, `min_confidence`, `max_antecedent` and `max_consequent`;
* the number of transactions and the number of candidates tested;
* the null calibration result from `validate`, with the number of null runs;
* the recall limitation, especially if the rule set is used as a complete catalogue.

These are the numbers a reader needs to judge whether the guarantee applies to the analysis.
