# Methodology

This page explains what rulewise computes and why the result can be called statistically sound.
The implementation follows the productive-rule framework of Webb (2007, 2008, 2010) and the
tutorial of Hamalainen and Webb (2019).

## The problem

A rule `X -> Y` is a claim that transactions containing `X` are more likely to contain `Y` than
transactions that do not contain `X`. A classical miner applies minimum support and minimum
confidence and returns every rule that passes. That procedure tests an enormous number of
hypotheses without accounting for the number tested, so the chance of returning at least one
spurious rule rises with the search space. On pure noise, a classical miner still returns
thousands of rules.

rulewise treats rule discovery as multiple hypothesis testing. It tests each candidate, corrects
for the number of tests, and reports the corrected p-value.

## The 2x2 table

For a rule `X -> Y`, count the four cells over the measurement data:

|  | Y present | Y absent |
|---|---|---|
| X present | a | b |
| X absent | c | d |

`a` is the number of transactions containing both sides, `a + b` the number containing `X`,
`a + c` the number containing `Y`, and `n = a + b + c + d` the total.

Support, confidence, lift, leverage and conviction are computed from these counts:

```text
support    = a / n
confidence = a / (a + b)
lift       = confidence / ((a + c) / n)
leverage   = support - ((a + b) / n) * ((a + c) / n)
conviction = (1 - (a + c) / n) / (1 - confidence)
```

## The productivity test

A frequent and confident rule can still be redundant. If `A` already predicts `Y`, then
`A and B -> Y` may add nothing: the extra item `B` may simply ride along. Webb calls a rule
productive if its confidence is higher than the confidence of every proper subset of its
antecedent:

```text
productive(X -> Y) = for every proper subset Z of X:
                         confidence(X -> Y) > confidence(Z -> Y)
```

rulewise tests each comparison with a one-sided Fisher exact test on the measurement data. For a
subset `Z`, the test asks whether adding `X - Z` raises the rate of `Y` among transactions that
already contain `Z`.

The rule's p-value is the maximum over all comparisons. This is an intersection-union test: the
rule is declared significant only if every comparison is significant. It controls the type I
error for the null "this rule is not productive" without an additional correction across the
comparisons, because the maximum of the p-values is itself a valid p-value for the intersection
of the null hypotheses.

For a rule with more than one item in the consequent, rulewise also requires significant positive
dependence across every bipartition of the full itemset. An item that is merely carried along by
the others makes the rule fail. This is the self-sufficient-itemset idea of Webb (2010). The
reported p-value is the maximum of the productivity p-value and the partition p-value.

Immediate subsets are tested first. When a comparison already reaches the cap, the search stops
and returns a value at or above the cap. Any p-value below the cap is exact; a value above the
cap is a lower bound, which is enough to decide that the rule is not significant.

## Multiple testing

The candidates are not independent, and the number of possible rules is large. rulewise offers
two families of corrections.

### Family-wise error rate

`correction="holm"` (default) or `"bonferroni"` controls the probability of returning at least one
spurious rule at or below `alpha`. Holm is uniformly more powerful than Bonferroni and is the
recommended default.

### False discovery rate

`correction="bh"` (Benjamini-Hochberg) or `"by"` (Benjamini-Yekutieli) controls the expected share
of false discoveries among the returned rules. `by` is valid under arbitrary dependence and is
more conservative than `bh`.

Use FDR when you can tolerate a small share of false rules in exchange for more true ones. Use
FWER when a single false rule is unacceptable.

`correction="none"` disables adjustment. It is for diagnostics only.

## Holdout mode

In holdout mode, rulewise splits the transactions into an exploration part and a holdout part.
Candidates are generated on the exploration part, using minimum support and confidence. Each
candidate is then tested with the productivity test on the holdout part, and all reported
measures and intervals are computed there. Because the holdout part was not used to choose the
candidates, the tests and effect sizes are honest.

The cost is statistical power: the miner sees only part of the data, and rules that are frequent
in the full data may not reach the support floor in the exploration part.

`holdout_fraction` controls the split. The default is 0.5. A value of 0 disables the split and
raises a warning because the p-values are then not valid.

## Direct mode

In direct mode, all data are used for both candidate generation and testing. To keep the
family-wise error rate valid when the hypotheses are selected from the same data, rulewise uses
Webb's layered critical values:

* rules are grouped by total size `l = |X| + |Y|`;
* `alpha` is divided equally across the layers;
* within a layer, a Bonferroni correction is applied over the number of rules that could possibly
  be reported at that size, not only the candidates that passed the filters.

For a rule at level `l`, the critical value is:

```text
alpha / (number_of_layers * search_space_size[l])
```

This is why direct mode remains valid even though the support and confidence filters are applied
to the same data. It is conservative when there are many items, because the search space grows
combinatorially. On small data it is often more powerful than holdout mode because it uses every
transaction.

The reported confidence, lift and intervals in direct mode are measured on the same data that
selected the rules. They are optimistically biased by the winner's curse. Use holdout mode when
effect sizes matter; use direct mode when detection matters more.

## Intervals

Confidence intervals use the Wilson score interval, which behaves better than the normal
approximation for small counts and proportions near 0 or 1.

Lift and leverage intervals use a percentile bootstrap. Resampling transactions with replacement
is equivalent to drawing the four cell counts from a multinomial distribution with the observed
cell proportions, so no transaction-level resampling is needed. The interval reflects the
uncertainty in the 2x2 table, not model uncertainty.

## Candidate generation

Frequent itemsets are mined with a bitset Eclat implementation. For every frequent itemset `S` and
every split `S = X union Y` with `1 <= |Y| <= max_consequent` and `|X| <= max_antecedent`, a rule
is a candidate when its confidence is at least `min_confidence` and its lift is above 1. Rules
with an empty antecedent or consequent are not generated.

The implementation excludes items that occur in every transaction. They cannot appear in a
significant rule: as a consequent they have lift 1, and as an antecedent item they add no
coverage. This also keeps the Python and PyFIM backends identical.

Class-conditional mining, used by `RuleClassifier` by default, mines each class separately. It
produces the same candidates as the general miner and is much faster on dense data. Items that
occur in every row of one class are re-added after mining because they are strong predictors.

## Assumptions and what they mean

The error guarantee is conditional on these assumptions:

1. **Independent and identically distributed transactions.** Repeated baskets, clustering,
   time dependence or survey weights violate this.
2. **Independence null.** The test asks whether `X` and `Y` are independent. It does not adjust
   for basket size, a third variable, or selection of the data.
3. **Independent holdout.** In holdout mode, the holdout part must not be used to choose
   candidates. The default split satisfies this, but do not tune thresholds on the holdout part
   and then report the same p-values.
4. **Full search space in direct mode.** The layered correction counts every rule that could be
   reported at a size, not only the candidates that passed the filters.
5. **Correct model of the data.** The Fisher test assumes the 2x2 table is a random sample from
   the population of interest. If rows are not exchangeable, the p-value is not meaningful.

Under these assumptions, the probability of one or more false rules is at most `alpha` with Holm
or Bonferroni, and the expected false discovery proportion is at most `alpha` with BH or BY.

The procedure is conservative. Empirical error rates on null data are well below 5%, which means
the method pays for its guarantee with power. This is visible in the low recall on real data and
in the power study.

## References

* Webb, G.I. (2007). Discovering significant patterns. Machine Learning 68(1), 1-33.
* Webb, G.I. (2008). Layered critical values: a powerful direct-adjustment approach to
  discovering significant patterns. Machine Learning 71(2-3), 307-323.
* Webb, G.I. (2010). Self-sufficient itemsets: an approach to screening potentially interesting
  associations. ACM Transactions on Knowledge Discovery from Data 4(1), 1-20.
* Hamalainen, W. and Webb, G.I. (2019). A tutorial on statistically sound pattern discovery.
  Data Mining and Knowledge Discovery 33, 325-377.
* Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal
  of Statistics 6, 65-70.
* Benjamini, Y. and Hochberg, Y. (1995). Controlling the false discovery rate. Journal of the
  Royal Statistical Society B 57, 289-300.
* Benjamini, Y. and Yekutieli, D. (2001). The control of the false discovery rate in multiple
  testing under dependency. Annals of Statistics 29, 1165-1188.
