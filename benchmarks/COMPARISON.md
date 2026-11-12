# rulewise vs the existing Python ARM stack

Generated 2026-11-12 by `benchmarks/library_comparison.py`. mlxtend 0.25.0, numpy 2.3.5, pandas 3.1.1, scipy 1.18.1.

## 1. Correctness: identical output to mlxtend

  support 0.02: 122 itemsets, identical=True
  support 0.01: 333 itemsets, identical=True
  support 0.005: 1001 itemsets, identical=True
  support 0.002: 4223 itemsets, identical=True

| min_support | mlxtend itemsets | rulewise itemsets | identical | mlxtend s | rulewise s |
|---|---|---|---|---|---|
| 0.02 | 122 | 122 | yes | 0.09 | 0.01 |
| 0.01 | 333 | 333 | yes | 0.11 | 0.02 |
| 0.005 | 1001 | 1001 | yes | 0.20 | 0.03 |
| 0.002 | 4223 | 4223 | yes | 0.38 | 0.07 |

Frequent itemsets are exact (counts and membership), so the statistical layer sits on top of a correct miner, not a different one.

## 2. Pure noise: 500 independent null datasets (n=3000, 25 items)

| method | mean rules | median | datasets with >=1 rule | rate | 95% CI | fit s |
|---|---|---|---|---|---|---|
| mlxtend all | 1929.31 | 1629 | 500/500 | 1.000 | 0.992-1.000 | 0.196 |
| rulewise holdout | 0.02 | 0 | 4/500 | 0.008 | 0.003-0.020 | 0.072 |
| rulewise direct | 0.01 | 0 | 4/500 | 0.008 | 0.003-0.020 | 0.076 |

Out-of-sample replication of mlxtend's rules on a fresh null sample (one-sided Fisher p<0.05):
  all rules: mean 0.042 replicate (expected 0.05 by chance)
  top-10 by lift: mean 0.040 replicate (expected 0.05 by chance)

This is the core difference: on data with no associations mlxtend returns hundreds to thousands of rules and essentially none of them hold on new data; rulewise returns none.

## 3. Planted rules: precision and recall (5 datasets, n=20000, 30 items)

Ground truth: 4 rules, background items independent, 15% label noise.

  seed 0: classical 14658 rules, rulewise direct 35; evaluated on 10000 held-out baskets
  seed 1: classical 13265 rules, rulewise direct 35; evaluated on 10000 held-out baskets
  seed 2: classical 15420 rules, rulewise direct 35; evaluated on 10000 held-out baskets
  seed 3: classical 16014 rules, rulewise direct 35; evaluated on 10000 held-out baskets
  seed 4: classical 19069 rules, rulewise direct 35; evaluated on 10000 held-out baskets

| method | rules | productive on test | lift>1 on test | planted recall | exact planted precision | reliable F1 |
|---|---|---|---|---|---|---|
| classical all | 15685 | 1.4% | 63.3% | 100.0% | 0.0% | 0.03 |
| classical top-35 by lift | 35 | 9.1% | 100.0% | 65.0% | 7.4% | 0.16 |
| classical top-35 by confidence | 35 | 1.1% | 100.0% | 0.0% | 0.0% | 0.00 |
| classical lift>=2 | 129 | 8.7% | 100.0% | 75.0% | 2.3% | 0.16 |
| mlxtend + Fisher/Holm | 3486 | 3.1% | 100.0% | 100.0% | 0.1% | 0.06 |
| rulewise holdout | 35 | 100.0% | 100.0% | 100.0% | 11.4% | 1.00 |
| rulewise direct | 35 | 100.0% | 100.0% | 100.0% | 11.4% | 1.00 |

'productive on test' = the rule still beats every simpler version of itself on data it never saw (Webb test, p<0.05). 'reliable F1' is the harmonic mean of that share and planted recall: it asks whether the returned set is both real and complete. 'exact planted precision' is stricter than useful: rulewise also returns genuine sub-rules and reverse rules of the planted structure, which do not match the planted rule verbatim but are still true on the test half. Classical rows are one library run split into views; they are not tuned on the test half.

## 4. Real baskets: Groceries (9835 transactions)

  seed 0: classical 799 rules, rulewise 134 (holdout) / 185 (direct)
  seed 1: classical 652 rules, rulewise 103 (holdout) / 172 (direct)
  seed 2: classical 624 rules, rulewise 113 (holdout) / 171 (direct)
  seed 3: classical 650 rules, rulewise 124 (holdout) / 171 (direct)
  seed 4: classical 623 rules, rulewise 106 (holdout) / 172 (direct)

| method | rules | productive on held-out | lift>1 on held-out | productive vs classical | exact coverage | fit s |
|---|---|---|---|---|---|---|
| classical all | 670 | 48.8% | 98.5% | 100.0% | 100.0% | 0.19 |
| classical lift>=2 | 350 | 48.9% | 99.9% | 52.1% | 52.1% | - |
| classical top-K by lift | 174 | 47.0% | 100.0% | 25.1% | 25.1% | - |
| classical top-K by confidence | 174 | 37.5% | 99.9% | 20.1% | 20.1% | - |
| mlxtend + Fisher/Holm | 185 | 64.9% | 100.0% | 37.0% | 35.8% | - |
| rulewise holdout | 116 | 98.7% | 100.0% | 35.3% | 15.3% | 0.05 |
| rulewise direct | 174 | 99.3% | 100.0% | 53.6% | 20.9% | 0.07 |

'productive vs classical' = number of productive rules returned divided by the number in the classical list (the README's 'recovers about half as many'). 'exact coverage' = share of the classical productive rules the method returned verbatim; it is a lower bound on recall because a simpler or differently-scoped rule can express the same association. Top-K uses K = the number of rules returned by rulewise direct on the same split (mean 174).

Recommendation hit@5 (hide one item from each held-out basket):
| method | hit@5 | coverage |
|---|---|---|
| popularity | 26.4% | 100.0% |
| classical all | 20.1% | 92.7% |
| classical top-K | 4.9% | 36.9% |
| rulewise holdout | 18.7% | 85.3% |
| rulewise direct | 19.7% | 90.2% |

## 5. Rule classifier vs standard baselines (UCI Mushroom, 5-fold x 3)

| model | accuracy | log-loss | rules | fit s |
|---|---|---|---|---|
| majority | 0.5180 | 0.6925 | - | 0.00 |
| logistic (one-hot) | 0.9998 | 0.0007 | - | 0.11 |
| tree (d=6) | 0.9959 | 0.0141 | - | 0.11 |
| rules / holdout | 0.9957 | 0.0209 | 1042 | 0.91 |
| rules / direct | 0.9970 | 0.0113 | 1373 | 1.18 |

Mushroom is categorical and rule-friendly: a fair best case for an associative classifier, and the place where it is competitive. The README already reports that it loses badly on dense numeric data (digits).

## 6. Fit time (single core, this machine)

| data | method | seconds |
|---|---|---|
| groceries n=9.8k, 169 items, sup=0.005 | mlxtend (661 rules) | 0.35 |
| groceries n=9.8k, 169 items, sup=0.005 | efficient-apriori (673 rules) | 0.17 |
| groceries n=9.8k, 169 items, sup=0.005 | rulewise holdout (250 rules) | 0.08 |
| groceries n=9.8k, 169 items, sup=0.005 | rulewise direct (344 rules) | 0.09 |
| synthetic n=20k, 30 items, sup=0.01 | mlxtend (14607 rules) | 2.62 |
| synthetic n=20k, 30 items, sup=0.01 | efficient-apriori (37963 rules) | 8.25 |
| synthetic n=20k, 30 items, sup=0.01 | rulewise holdout (35 rules) | 0.62 |
| synthetic n=20k, 30 items, sup=0.01 | rulewise direct (35 rules) | 0.75 |
| dense null n=50k, 60 items, sup=0.0005 | mlxtend (49134 rules) | 8.39 |
| dense null n=50k, 60 items, sup=0.0005 | efficient-apriori (50641 rules) | 119.33 |
| dense null n=50k, 60 items, sup=0.0005 | rulewise holdout (0 rules) | 2.75 |
| dense null n=50k, 60 items, sup=0.0005 | rulewise direct (0 rules) | 4.26 |

rulewise is not slower in these runs even though it tests every candidate: it never materialises the full rule list (mlxtend returns up to 51k rules where rulewise returns none or a few dozen), and its bitset Eclat is fast on these sizes. On dense data at low support the statistical pass grows with the number of candidates (281k candidates took about 70 s in RESULTS.md). The optional PyFIM backend could not be tested: the `pyfim` C extension does not build on Python 3.12 / Windows in this environment, so the core install uses the pure-Python backend.
