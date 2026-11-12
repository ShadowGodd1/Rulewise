# API reference

This page documents the public objects exported from `rulewise`:

```python
from rulewise import RuleMiner, RuleClassifier, ValidationReport
from rulewise import adjust_pvalues, fisher_greater, wilson_interval
```

## RuleMiner

```python
RuleMiner(
    min_support=0.01,
    min_confidence=0.1,
    max_antecedent=3,
    max_consequent=1,
    alpha=0.05,
    correction="holm",
    mode="holdout",
    holdout_fraction=0.5,
    n_bootstrap=200,
    backend="auto",
    max_cache_mb=128.0,
    random_state=0,
)
```

### Parameters

| parameter | type | default | meaning |
|---|---|---|---|
| `min_support` | float | `0.01` | minimum fraction of exploration transactions containing an itemset |
| `min_confidence` | float | `0.1` | candidate rules below this confidence are not tested |
| `max_antecedent` | int | `3` | maximum number of items on the left side |
| `max_consequent` | int | `1` | maximum number of items on the right side |
| `alpha` | float | `0.05` | significance level after correction, in `(0, 1]` |
| `correction` | str | `"holm"` | `"holm"`, `"bonferroni"`, `"bh"`, `"by"` or `"none"` |
| `mode` | str | `"holdout"` | `"holdout"` or `"direct"` |
| `holdout_fraction` | float | `0.5` | share of transactions reserved for testing in holdout mode |
| `n_bootstrap` | int | `200` | resamples for lift and leverage intervals; `0` disables them |
| `backend` | str | `"auto"` | `"auto"`, `"python"` or `"pyfim"` |
| `max_cache_mb` | float | `128.0` | memory bound for cached itemset intersections during testing |
| `random_state` | int | `0` | seed for the split, bootstrap and validation resampling |

`correction="none"` disables multiple-testing correction. It is useful for diagnostics, not for
claims of significance.

### Attributes

| attribute | type | meaning |
|---|---|---|
| `rules_` | pandas.DataFrame | significant rules, one row per rule |
| `diagnostics_` | dict | mode, backend, counts and replication rate |
| `vocab_` | dict | item name to integer id |
| `items_` | dict | integer id to item name |
| `n_features_in_` | int | number of matrix columns, when fitted on a matrix |

`diagnostics_` contains:

| key | meaning |
|---|---|
| `mode` | `"holdout"` or `"direct"` |
| `backend` | backend actually used |
| `honest_estimates` | True when holdout mode used a non-empty holdout part |
| `n_transactions` | total transactions |
| `n_explore`, `n_holdout` | sizes of the exploration and holdout parts |
| `n_frequent_itemsets` | itemsets enumerated |
| `n_candidates` | candidate rules tested |
| `n_significant` | rules returned |
| `replication_rate` | share of tested candidates with unadjusted p below alpha, holdout mode only |

### Methods

#### fit

```python
fit(X, y=None, item_names=None) -> RuleMiner
```

Fit on transactions: an iterable of item collections, a 0/1 matrix, scipy sparse, pandas or
polars. `y` is ignored and exists for scikit-learn pipeline compatibility. `item_names` labels
matrix columns with strings.

#### transform

```python
transform(transactions) -> numpy.ndarray
```

Boolean matrix of shape `(n_transactions, n_rules)`. Entry `[i, j]` is True when rule `j`'s
antecedent is contained in transaction `i`. Returns shape `(n, 0)` when no rule was found.

#### recommend

```python
recommend(basket, k=5, by="confidence_low", explain=False)
```

Rank items not in the basket using the significant rules that fire on it. Each item is scored by
its best matching rule on `by`: `"confidence_low"` (Wilson lower bound, the default),
`"confidence"`, `"lift"` or `"leverage"`. Returns `[(item, score)]`, or a DataFrame with the rule
behind each recommendation when `explain=True`.

`confidence_low` discounts rules that rest on little data. It is not a probability calibration
guarantee; evaluate recommendations on held-out data.

#### stability

```python
stability(transactions, n_resamples=20, fraction=0.8) -> dict
```

Refit on random subsamples and measure how much the rule set changes. Returns `mean_jaccard`
(mean pairwise Jaccard similarity of the rule sets) and `rule_frequency`, a DataFrame with the
share of subsamples in which each rule from the full fit reappears. Subsamples hold less data
than the full fit, so frequencies below 1 are expected even for genuine rules. Compare rules
with each other rather than reading a frequency as a probability.

#### validate

```python
validate(transactions, n_null=20, n_stability=0) -> ValidationReport
```

Refit on `n_null` datasets with the same item frequencies but no associations. A sound pipeline
returns almost nothing; the share of null datasets with at least one rule estimates the empirical
family-wise error rate. Also reports holdout replication and, when `n_stability > 0`, stability.
See [ValidationReport](#validationreport).

#### query

```python
query(
    min_lift=None,
    min_confidence=None,
    min_support=None,
    max_p=None,
    contains=None,
    antecedent_contains=None,
    consequent_contains=None,
    max_antecedent=None,
    sort_by=None,
    ascending=False,
    top=None,
) -> pandas.DataFrame
```

Filter and sort the significant rules. `contains`, `antecedent_contains` and
`consequent_contains` take one item or a list of items that must all be present. `max_p` filters
on the adjusted p-value. `sort_by` takes any column name. The result is a new DataFrame.

#### export

```python
export(path, **query_kwargs) -> path
```

Write the filtered rules to `.csv`, `.json` or `.parquet` and return the path. Tuple columns are
flattened to readable strings such as `"milk & bread"` and `"not eggs"`. Parquet needs pyarrow.

#### to_polars

```python
to_polars(**query_kwargs) -> polars.DataFrame
```

Same as `export`, but returns a polars DataFrame. Requires polars.

#### summary

```python
summary(top=5) -> str
```

Human-readable summary: number of significant rules, candidates tested, and the top rules by
adjusted p-value.

#### plot

```python
plot(kind="effect", **kwargs)
```

`kind` is `"effect"` (lift against significance), `"forest"` (intervals) or `"network"`.
Requires matplotlib. Returns the matplotlib axes.

## RuleClassifier

```python
RuleClassifier(
    min_support=0.01,
    min_confidence=0.0,
    max_antecedent=3,
    alpha=0.05,
    correction="holm",
    mode="direct",
    holdout_fraction=0.5,
    backend="auto",
    max_cache_mb=128.0,
    add_negations=False,
    class_conditional=True,
    random_state=0,
)
```

`RuleClassifier` mines rules `antecedent -> class`, keeps only significant and productive ones,
and predicts with the best matching rule ordered by the Wilson lower bound on confidence. When no
rule matches, it returns the class priors.

### Additional parameters

| parameter | type | default | meaning |
|---|---|---|---|
| `add_negations` | bool | `False` | add a `("not", item)` item for every feature a row lacks |
| `class_conditional` | bool | `True` | mine each class separately for speed; yields the same rules as the general miner |

`min_support` is a fraction of all training transactions, so rules for rare classes need a lower
value than rules for common ones.

### Attributes

| attribute | type | meaning |
|---|---|---|
| `rules_` | pandas.DataFrame | significant class-association rules |
| `classes_` | numpy.ndarray | class labels in sorted order |
| `priors_` | numpy.ndarray | training class frequencies |
| `miner_` | RuleMiner | the fitted internal miner |
| `diagnostics_` | dict | same keys as `RuleMiner.diagnostics_` |
| `n_features_in_` | int | number of matrix columns, when fitted on a matrix |

### Methods

#### fit

```python
fit(X, y, item_names=None) -> RuleClassifier
```

`X` accepts the same formats as `RuleMiner.fit`. `y` must have one label per row.

#### predict

```python
predict(X) -> numpy.ndarray
```

Class label for each row.

#### predict_proba

```python
predict_proba(X) -> numpy.ndarray
```

Class probabilities. When a rule fires for class `c` with confidence `conf`, `P(c)` is `conf` and
the remaining probability is shared among the other classes in proportion to their training
priors. For two classes this is exactly the rule's estimated probability for each class. When no
rule fires, the class priors are returned.

#### score

```python
score(X, y) -> float
```

Mean accuracy, as in scikit-learn.

#### explain

```python
explain(X) -> list[dict | None]
```

For each row, a dict describing the rule that determined its probabilities, or `None` when no
rule matched.

#### validate

```python
validate(X, y, n_null=20, cv=5) -> ValidationReport
```

Permute the labels `n_null` times and refit. A sound pipeline finds almost no rules when the
labels are random. Also runs `cv`-fold cross-validation and compares accuracy and log-loss with
the majority-class and class-prior baselines.

#### query, export, to_polars, summary, plot

The same helpers as `RuleMiner`. On a classifier, `query` adds a `consequent` column rendered as
`class=<label>` so filters and exports work the same way.

## Output columns

### RuleMiner.rules_

| column | meaning |
|---|---|
| `antecedent` | tuple of item names |
| `consequent` | tuple of item names |
| `support` | support in the measurement data |
| `confidence` | confidence in the measurement data |
| `lift` | confidence divided by the consequent base rate |
| `leverage` | support minus the independence expectation |
| `conviction` | `(1 - P(Y)) / (1 - confidence)`, or infinity at confidence 1 |
| `count` | transactions containing antecedent and consequent |
| `count_antecedent` | transactions containing the antecedent |
| `count_consequent` | transactions containing the consequent |
| `p_value` | exact productivity p-value |
| `p_adjusted` | p-value after correction |
| `confidence_low`, `confidence_high` | Wilson interval |
| `lift_low`, `lift_high` | bootstrap interval, when `n_bootstrap > 0` |
| `leverage_low`, `leverage_high` | bootstrap interval, when `n_bootstrap > 0` |

In holdout mode the counts and measures come from the holdout part. In direct mode they come from
the full data and are in-sample estimates.

When a fit returns no significant rules, `rules_` keeps the core columns but omits the interval
columns. `transform` returns an array of shape `(n, 0)` in that case.

### RuleClassifier.rules_

Same measures as `RuleMiner.rules_`, with `class_` instead of `consequent` and without
`conviction`. `confidence_low`, `confidence_high`, `lift`, `leverage`, `count_antecedent` and
`count_consequent` are always present. Use `query` or `explain` to see the class label.

## ValidationReport

`ValidationReport` is a `dict` with a readable `summary()` and `__str__`. Keys depend on the
estimator and options.

### RuleMiner.validate

| key | meaning |
|---|---|
| `mode`, `alpha` | settings used |
| `honest_estimates` | True when holdout mode used a non-empty holdout part |
| `n_rules_real` | rules found on the real data |
| `replication_rate` | share of candidates with unadjusted p below alpha, holdout mode |
| `null_runs` | number of null datasets |
| `null_rule_counts` | rules found in each null dataset |
| `null_mean_rules` | mean rules per null dataset |
| `null_fwer` | share of null datasets with at least one rule |
| `null_fwer_ci` | Wilson interval on that share |
| `calibrated` | True when the lower confidence bound is at or below alpha |
| `null_kind` | description of the null model |
| `stability_resamples` | number of stability subsamples, when requested |
| `stability_mean_jaccard` | mean pairwise Jaccard overlap, when requested |
| `rule_frequency` | DataFrame of per-rule reappearance rates, when requested |

### RuleClassifier.validate

Adds `cv_folds`, `cv_accuracy`, `cv_accuracy_sd`, `majority_accuracy`, `cv_log_loss` and
`prior_log_loss` to the null-calibration keys.

## Statistical helpers

```python
fisher_greater(a, b, c, d) -> float
```

One-sided Fisher exact p-value for the table `[[a, b], [c, d]]`, testing whether the first row is
enriched in the first column. Degenerate margins return 1.0.

```python
adjust_pvalues(pvalues, method="holm") -> numpy.ndarray
```

Multiple-testing adjustment. Methods: `"none"`, `"bonferroni"`, `"holm"`, `"bh"`, `"by"`.
`bonferroni` and `holm` control the family-wise error rate. `bh` and `by` control the false
discovery rate; `by` holds under arbitrary dependence.

```python
wilson_interval(k, n, alpha=0.05) -> tuple[float, float]
```

Wilson score interval for a binomial proportion.

## Input formats

| input | accepted | notes |
|---|---|---|
| list or generator of lists, sets or tuples | yes | items must be hashable |
| 2D numpy array | yes | any numeric dtype; nonzero means present |
| scipy sparse matrix | yes | any sparse format |
| pandas DataFrame | yes | dense or sparse one-hot |
| pandas Series of lists | yes | each element is an item collection |
| polars DataFrame | yes | one-hot, or a single list column |
| polars Series of lists | yes | each element is an item collection |
| a bare string | no | split or encode it first |

Matrix input sets `n_features_in_` and checks the column count at `transform` and `predict` time.
NaN, infinity, complex numbers and zero-width matrices are rejected with a `ValueError`.

## Scikit-learn integration

`RuleMiner` is a transformer and `RuleClassifier` is a classifier. They work in pipelines,
`GridSearchCV`, `cross_val_score` and `RepeatedStratifiedKFold`. Numeric features must be
discretized first, for example with `KBinsDiscretizer(encode="onehot")`.

The estimator tags declare sparse input support. `NotFittedError` is raised when a method that
needs a fitted model is called before `fit`.
