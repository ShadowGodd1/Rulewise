# Benchmark results (rulewise dev)

All numbers measured in a 1-core, 3 GB sandbox. Reproduce with the scripts in this folder.
Settings were fixed before looking at results; nothing was tuned on test folds.

## RuleClassifier vs baselines (`classifier_eval.py`)

Repeated stratified 5-fold CV x 5 repeats. Numeric features quantile-binned inside each
training fold. Rule settings: iris (4 bins, support 0.05), wine (4 bins, 0.08),
breast_cancer (4 bins, 0.10), digits (3 bins, 0.02, max_antecedent 2).

| dataset (n, classes) | model | accuracy | log-loss | Brier |
|---|---|---|---|---|
| iris (150, 3) | logistic | 0.956 | 0.153 | 0.072 |
| | tree depth 4 | 0.943 | 0.704 | 0.111 |
| | rules / holdout | 0.809 | 0.708 | 0.271 |
| | rules / direct | 0.937 | 0.388 | 0.129 |
| wine (178, 3) | logistic | 0.982 | 0.062 | 0.028 |
| | tree depth 4 | 0.897 | 1.335 | 0.200 |
| | rules / holdout | 0.909 | 0.558 | 0.145 |
| | rules / direct | 0.936 | 0.274 | 0.114 |
| breast_cancer (569, 2) | logistic | 0.977 | 0.074 | 0.038 |
| | tree depth 4 | 0.929 | 0.563 | 0.128 |
| | rules / holdout | 0.936 | 0.360 | 0.119 |
| | rules / direct | 0.935 | 0.295 | 0.115 |
| digits (1797, 10) | logistic | 0.969 | 0.108 | 0.050 |
| | tree depth 4 | 0.558 | 1.333 | 0.515 |
| | rules / holdout | 0.576 | 1.643 | 0.659 |
| | rules / direct | 0.606 | 1.535 | 0.614 |

Findings
* Direct mode beat holdout on every dataset, on accuracy and log-loss (largest gap on tiny data,
  where holdout discards half the sample). The classifier therefore defaults to `direct`.
* Rules beat a depth-4 tree on wine and breast_cancer and are far better calibrated than the
  tree, but lose to logistic regression on these smooth, linear-friendly datasets.
* Digits (signal spread thinly over many pixels) is a clear weakness: prediction uses one best
  rule, so weak evidence cannot accumulate. Use RuleClassifier where the class follows a few
  conjunctions of features, not diffuse numeric signal.

## Where rules are the ground truth (100k rows, 60 binary features)

Class = (f0 and f1) or (f2 and f3 and f4) or (f5 and not f6 and f7), 5% label noise
(Bayes-optimal accuracy about 0.95). 70k train / 30k test.

| model | accuracy | log-loss | fit |
|---|---|---|---|
| logistic | 0.855 | 0.374 | 0.1 s |
| tree depth 6 | 0.946 | 0.212 | 0.2 s |
| rules / direct (presence only) | 0.897 | 0.399 | 15.2 s |
| rules / holdout (presence only) | 0.893 | 0.402 | 10.5 s |

Presence-only itemsets cannot express "not f6", so accuracy stops at 0.897. With negated items
(`add_negations`) the classifier recovers all three planted rules at about 0.95 confidence. On a
smaller version (40k rows, 14 features) accuracy rose 0.900 -> 0.934 (tree 0.950); on a 2-rule
version with a 0.72-confidence presence-only rule, accuracy was identical but log-loss fell
0.344 -> 0.177.

### Class-conditional mining (`class_conditional=True`, default)

Mines each class separately and enumerates only itemsets frequent within a class. Produces
exactly the same candidates and rules as the general miner (tested for both backends, both
modes, multiclass and negations), but enumerates far fewer itemsets.

| 100k rows x 60 features, support 0.002 | fit | itemsets enumerated | candidates | rules | accuracy |
|---|---|---|---|---|---|
| general miner | 14.2 s | 595,728 | 36,050 | 44 | 0.897 |
| class-conditional | 5.5 s | 72,041 | 36,050 | 44 | 0.897 |
| class-conditional, 120 items (with negations) | 79.0 s | 561,861 | 280,960 | 107 | **0.948** |

The last row did not finish in 280 s with the general miner. It recovers all three planted
rules, including `(f5, f7, not_f6)`, and matches the depth-6 tree (0.946).

## Error control on pure-noise data (`error_control_simulation.py`)

300 independent null datasets per configuration (items independent of each other, or labels
independent of features). Share of datasets in which any rule was reported; nominal level 0.05.

| configuration | datasets with >= 1 rule | rate | 95% CI |
|---|---|---|---|
| RuleMiner holdout / Holm | 1/300 | 0.003 | 0.001-0.019 |
| RuleMiner holdout / BH (FDR) | 2/300 | 0.007 | 0.002-0.024 |
| RuleMiner direct (layered critical values) | 1/300 | 0.003 | 0.001-0.019 |
| RuleMiner, 2-item consequents | 0/300 | 0.000 | 0.000-0.013 |
| RuleClassifier direct | 5/300 | 0.017 | 0.007-0.038 |
| RuleClassifier holdout | 2/300 | 0.007 | 0.002-0.024 |

Every configuration is below the nominal level (the method is conservative). The null here
has independent items; it does not cover dependence that arises from basket size (see limits).
The weekly CI job `error-control.yml` repeats this with 1,000 datasets per cell.

## Power at weak signal (`power_study.py 5000`)

20 planted 3-item patterns among 1,000 Zipf-distributed items, n=5,000, `min_support=0.001`.
"Recovered" = at least one significant rule uses only that pattern's items; "false" = a rule
whose items are not all inside one planted pattern.

| strength | holdout recovered | direct recovered | false rules |
|---|---|---|---|
| 0.002 | 9/20 | 11/20 | 0 |
| 0.005 | 20/20 | 20/20 | 0 |
| 0.010 | 20/20 | 20/20 | 0 |
| 0.030 | 20/20 | 20/20 | 0 |

Direct mode is more powerful at the weakest signal; neither mode produced a false rule. This is
the honest cost of the guarantee: at very weak effects the method abstains.

## Real data

### Out-of-sample replication, Groceries (`replication_groceries.py`, 6 random splits)

Rules found on half the baskets, tested on the other half.

| method | rules | lift > 1 on B | productive on B | productive rules on B |
|---|---|---|---|---|
| classical (all) | 709 | 0.91 | 0.47 | 331 |
| classical, lift >= 2 | 376 | 0.97 | 0.44 | 166 |
| classical, top-K by lift | 176 | 0.99 | 0.43 | 76 |
| classical, top-K by confidence | 176 | 0.97 | 0.37 | 65 |
| rulewise holdout | 114 | 0.99 | 0.98 | 112 |
| rulewise direct | 176 | 0.98 | 0.98 | 173 |

Classical rules at conventional thresholds are already mostly real on this data (91% have
lift > 1 held out). The difference is redundancy: only 47% of the classical list is still
productive on held-out data, against 98% for rulewise, and no lift or confidence threshold
fixes it. The cost is recall: rulewise recovers about half (direct) or a third (holdout) of the
held-out-productive rules that the full classical list contains.

### Classification (`real_classification.py`, 5-fold x 3 repeats)

| dataset | logistic | tree (depth 4) | rules / direct | rules / holdout |
|---|---|---|---|---|
| Mushrooms (accuracy) | 1.000 | 0.991 | 0.997 (1,373 rules) | 0.996 (1,042 rules) |
| Titanic (accuracy) | 0.790 | 0.818 | 0.777 (14 rules) | 0.773 (7 rules) |

Rules are competitive on Mushrooms and a few points behind on Titanic; their advantage is
readability and an interval on every rule, not accuracy.

## Known limits

* rulewise tests statistical dependence between items. Items that co-occur only because some
  baskets are larger (basket-size effects) are dependent and can be reported; the shuffle null in
  `validate` does not model this.
* Stability across 80% subsamples is moderate on real data (Groceries: mean Jaccard 0.54) because
  borderline rules flip; use `rule_frequency` to separate stable rules.

* Very dense data x low support still produces many itemsets. Class-conditional mining helps a
  lot (above) but the testing stage grows with the number of candidate rules (281k candidates
  took about 70 s). Raise `min_support` or lower `max_antecedent` if a fit is slow.
* RuleMiner (descriptive mining, no class) still uses the general miner.
* PyFIM omits itemsets made only of items present in every transaction; rulewise excludes such
  items in both backends (they cannot appear in significant rules) and falls back to the exact
  Python miner if PyFIM output is ever inconsistent. Items present in every row of one class
  are kept (they are strong predictors) and re-added exactly after mining.
