# rulewise

Repository: https://github.com/ShadowGodd1/Rulewise
Documentation: https://shadowgodd1.github.io/Rulewise/

Association rule mining you can trust: every rule that comes back has survived exact statistical
testing, correction for the number of rules tried, and a test that it still beats simpler versions
of itself. The library also reports how far to trust the result.

Classical rule mining returns thousands of rules ranked by support and confidence. Many are
redundant, explained by a simpler rule; on pure noise, all of them are chance. rulewise mines the
candidates, tests each one with an exact test, corrects for the number of rules tried, drops rules
whose extra antecedent items add nothing over a simpler version of the same rule, and reports
validation metrics. On data with no real structure it almost never returns a rule: across 1,800
pure-noise datasets it reported a spurious rule in at most 1.7% of them (nominal level 5%).

The guarantee is conservative. The price of almost never reporting a spurious rule is that weak
associations need more data to be detected, and recall is lower than a classical list. See
"Limitations" below and `docs/limitations.md`.

## Install

```bash
pip install rulewise             # core, pure-Python Eclat
pip install "rulewise[sklearn]"  # plus scikit-learn integration
pip install "rulewise[plot]"     # plus matplotlib plots
```

PyFIM is an optional legacy C backend (`backend="pyfim"`). It has no wheel for modern Python, so
it is not installed by default. Install it manually if your platform can build it; the pure-Python
backend is the default.

## Find rules

```python
from rulewise import RuleMiner

baskets = [["milk", "bread", "butter"], ["beer", "chips"], ...]   # or a 0/1 matrix,
                                                                  # scipy sparse, pandas, polars
miner = RuleMiner(min_support=0.01).fit(baskets)

print(miner.summary())
miner.rules_                       # DataFrame of significant rules and their measures
miner.query(min_lift=2, contains="milk", sort_by="lift", top=10)
miner.recommend(["milk"], k=5, explain=True)
miner.export("rules.csv")          # .csv, .json or .parquet
miner.to_polars()                  # requires polars
miner.plot("effect")               # also "forest" and "network"; requires matplotlib
```

`RuleMiner` splits the data by default: it finds candidates on one part and tests and measures
them on the other (`mode="holdout"`). That makes the reported confidence, lift and intervals
honest. Set `mode="direct"` to use all data and Webb's layered critical values; direct mode is
usually more powerful on small data, but its effect sizes are measured on the same data that
selected the rules and are therefore optimistic. See "Modes" below.

## What comes back

`miner.rules_` contains one row per significant rule:

| column | meaning |
|---|---|
| `antecedent`, `consequent` | tuples of item names |
| `support` | share of transactions containing antecedent and consequent |
| `confidence` | P(consequent present given antecedent present) |
| `lift` | confidence divided by the consequent's base rate |
| `leverage` | support minus the support expected under independence |
| `conviction` | how much more confident the rule is than a rule with no antecedent |
| `count`, `count_antecedent`, `count_consequent` | raw counts |
| `p_value` | exact productivity p-value |
| `p_adjusted` | p-value after the chosen correction |
| `confidence_low`, `confidence_high` | Wilson interval on confidence |
| `lift_low`, `lift_high` | bootstrap interval on lift (when `n_bootstrap > 0`) |
| `leverage_low`, `leverage_high` | bootstrap interval on leverage (when `n_bootstrap > 0`) |

In holdout mode all measures are computed on the holdout part. In direct mode they are computed
on the full data and are in-sample estimates.

## Validate the pipeline

```python
report = miner.validate(baskets, n_null=50, n_stability=20)
print(report)
```

`validate` refits the pipeline on shuffled data with the same item frequencies. A sound pipeline
returns almost nothing there; the share of null datasets with at least one rule estimates the
empirical family-wise error rate. It also reports holdout replication and, if `n_stability > 0`,
how stable the rule set is across random subsamples. The result is a `ValidationReport`, a dict
with a readable `summary()`.

## Classify with rules

```python
from rulewise import RuleClassifier

clf = RuleClassifier(min_support=0.01).fit(X, y)   # X: transactions, 0/1 matrix or sparse
clf.predict(X_new)
clf.predict_proba(X_new)
clf.score(X_test, y_test)
clf.explain(X_new[:1])                              # the rule behind each prediction
clf.validate(X, y, cv=5)                            # permuted-label null plus cross-validation
```

`RuleClassifier` mines rules of the form `antecedent -> class`, keeps only significant ones, and
predicts with the best matching rule ordered by the Wilson lower bound on confidence. When no
rule matches, it returns the class priors. It works in scikit-learn pipelines and
cross-validation.

Numeric features must be binned first, for example with
`KBinsDiscretizer(encode="onehot")`. `add_negations=True` adds a `("not", item)` item for every
feature a row lacks, so rules can express "feature absent".

## How it works

1. **Mine** frequent itemsets with a bitset Eclat implementation, or PyFIM when installed.
2. **Generate candidates** `X -> Y` with confidence above `min_confidence` and lift above 1.
3. **Test productivity.** A rule must beat every simpler version of itself: for every proper
   subset `Z` of `X`, adding `X - Z` must raise the rate of `Y`. Each comparison is a one-sided
   Fisher exact test; the rule's p-value is the maximum over comparisons, so a rule is
   significant only if all comparisons are. Rules with multi-item consequents must also show
   positive dependence across every bipartition of the itemset.
4. **Correct for multiple testing.** Holm or Bonferroni control the family-wise error rate;
   Benjamini-Hochberg or Benjamini-Yekutieli control the false discovery rate.
5. **Estimate honestly.** Holdout mode finds rules on one part and tests and measures them on the
   other. Direct mode uses all data and Webb's layered critical values.

## Modes

| mode | where candidates are found | where they are tested and measured | use when |
|---|---|---|---|
| `holdout` (default for `RuleMiner`) | exploration part | independent holdout part | you need honest effect sizes and intervals |
| `direct` (default for `RuleClassifier`) | all data | all data, with layered critical values | data is small, or you need maximum power, and you accept optimistic effect sizes |

In direct mode, `alpha` is split equally across rule sizes and Bonferroni-corrected within each
size over the full search space. The `correction` parameter is ignored. In holdout mode,
`holdout_fraction=0` disables the split; p-values are then not valid and a warning is raised.

## Statistical guarantees and assumptions

The guarantee is family-wise or false-discovery control over the rules actually tested, under
these assumptions:

* transactions are independent and identically distributed;
* the null model is independence between the items in a rule;
* in holdout mode, the holdout part is independent of the part used to select candidates;
* in direct mode, the correction covers the full search space, not only the candidates that
  happened to pass the other filters.

The testing is conservative, which is why the null-data error rate is well below 5%. It does not
model dependence caused by basket size or by other variables. Items can be associated only
because some baskets are larger, or because a third variable drives both, and such rules can be
reported. It also does not establish causation. See `docs/methodology.md` and
`docs/limitations.md`.

## Input formats

`RuleMiner` and `RuleClassifier` accept:

* an iterable of item collections, including a generator;
* a 2D numpy array or a 0/1 matrix (any numeric dtype, nonzero means present);
* a scipy sparse matrix;
* a pandas DataFrame (dense or sparse one-hot) or a Series of lists;
* a polars DataFrame (one-hot or a single list column) or a Series of lists.

Use `item_names` to label matrix columns with strings. Strings as transactions are rejected
because a single string is ambiguous; split or encode them first. Matrix input rejects NaN,
infinity, complex numbers and zero-width data.

## When to use it, and when not to

Good for:

* sparse transaction-style or categorical data where a few rules carry the structure;
* explainable models and auditable discovery;
* separating real associations from noise in small or exploratory samples;
* reporting an interval and a p-value with every rule.

Weak for:

* signal spread thinly over many dense numeric features; bin first, and expect low accuracy
  compared with regularized linear models or trees;
* maximum recall: a classical list contains more true rules, mixed with more redundancy;
* recommendation hit rate: a popularity baseline can beat rule-based recommendations;
* negative associations, closed or maximal itemsets, constrained mining, and the full metric
  catalogue that mlxtend provides.

Measured results and the head-to-head against mlxtend are in `benchmarks/COMPARISON.md`. The
methodology and limits are in `docs/methodology.md` and `docs/limitations.md`.

## Documentation

* `docs/index.md`: documentation home.
* `docs/api.md`: parameters, attributes, methods and output columns.
* `docs/methodology.md`: the statistical method and its assumptions.
* `docs/limitations.md`: known failure modes and power.
* `docs/comparison.md`: rulewise compared with existing Python libraries.
* `docs/tutorials/groceries.md`: a real supermarket basket walkthrough.
* `docs/tutorials/classification.md`: rules as a classifier on Mushrooms and Titanic.
* `docs/publishing.md`: release steps.

## Development

```bash
pip install -e ".[dev]"
python benchmarks/get_data.py
python benchmarks/library_comparison.py --section all --runs 500 --seeds 5
python -m pytest
```

The test suite checks p-values and corrections against scipy and statsmodels, itemsets against
mlxtend and across backends, error control by simulation on null data, and recovery of planted
rules. Optional-dependency tests skip when the dependency is absent, so the core install is
tested without PyFIM, scikit-learn, matplotlib, mlxtend, statsmodels or polars.

## References

* Webb, G.I. (2007). Discovering significant patterns. Machine Learning 68(1).
* Webb, G.I. (2008). Layered critical values: a powerful direct-adjustment approach to
  discovering significant patterns. Machine Learning 71.
* Hamalainen, W. and Webb, G.I. (2019). A tutorial on statistically sound pattern discovery.
  Data Mining and Knowledge Discovery 33.
* Webb, G.I. (2010). Self-sufficient itemsets. ACM TKDD 4(1).
* Holm (1979); Benjamini and Hochberg (1995); Benjamini and Yekutieli (2001).
* Borgelt, C. PyFIM and Eclat (MIT licence).

## License

MIT. See `LICENSE`.
