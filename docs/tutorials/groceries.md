# Tutorial: rules from real supermarket baskets (Groceries)

The Groceries dataset has 9,835 real baskets over 169 items (mean basket size 4.4), from the
arules R package. It is not bundled with rulewise; download it once:

```bash
curl -L -o groceries.csv \
  https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv
```

All output below was produced by running this code (`benchmarks/` has the scripts).

## Fit

```python
import csv
from rulewise import RuleMiner

tx = [sorted({x.strip() for x in row if x.strip()}) for row in csv.reader(open("groceries.csv"))]
tx = [t for t in tx if t]

miner = RuleMiner(min_support=0.005, max_antecedent=3, random_state=0).fit(tx)
print(miner.summary(top=6))
```

```
RuleMiner: 250 significant rules (alpha=0.05, mode=holdout, backend=python); 1563 candidates tested on 4918 transactions
Top 6 by adjusted p-value:
  root vegetables  ->  other vegetables   conf=0.436 [0.395, 0.479]  lift=2.30  p_adj=2.62e-41
  other vegetables  ->  root vegetables   conf=0.249 [0.223, 0.278]  lift=2.30  p_adj=2.62e-41
  root vegetables  ->  beef   conf=0.176 [0.146, 0.211]  lift=3.19  p_adj=2.71e-24
  beef  ->  root vegetables   conf=0.347 [0.293, 0.405]  lift=3.19  p_adj=2.71e-24
  pip fruit  ->  tropical fruit   conf=0.282 [0.239, 0.329]  lift=2.73  p_adj=1.66e-21
  tropical fruit  ->  pip fruit   conf=0.211 [0.177, 0.248]  lift=2.73  p_adj=1.66e-21
```

Rules were found on one half of the baskets and tested and measured on the other (holdout mode),
so the confidences and lifts are honest estimates, with intervals.

## Strongest associations, with uncertainty

```python
miner.query(min_lift=2.5, sort_by="lift", top=6)
```

```
sugar               -> flour          conf=0.13 [0.09,0.19] lift=8.74 [6.12,11.26]
flour               -> sugar          conf=0.29 [0.20,0.40] lift=8.74 [6.09,11.60]
processed cheese    -> white bread    conf=0.22 [0.14,0.32] lift=4.97 [2.86,7.21]
white bread         -> processed cheese conf=0.07 [0.05,0.12] lift=4.97 [3.05,6.81]
whipped/sour cream  -> berries        conf=0.13 [0.10,0.17] lift=3.68 [2.77,4.60]
berries             -> whipped/sour cream conf=0.27 [0.21,0.34] lift=3.68 [2.89,4.49]
```

Note how wide the lift intervals are for rare items: sugar and flour have a lift of 8.7, but the
data only pin it between 6.1 and 11.3. A ranking by lift alone hides that.

## Recommendations with the rule behind them

```python
miner.recommend(["yogurt", "whole milk"], k=4, explain=True)
```

```
               item     score            antecedent
0  other vegetables  0.343362  (whole milk, yogurt)
1        rolls/buns  0.212485             (yogurt,)
2    tropical fruit  0.185759             (yogurt,)
3   root vegetables  0.167799         (whole milk,)
```

The score is the Wilson lower bound on confidence, so a rule resting on little data is discounted.

## Validate

```python
print(miner.validate(tx, n_null=30, n_stability=10))
```

```
rulewise validation report (mode=holdout, alpha=0.05)
  rules found on the real data:        250
  holdout replication (p < alpha):      0.53 of tested candidates
  null calibration (30 shuffled datasets with the same item frequencies):
    datasets producing >= 1 rule:      0/30  -> empirical error rate 0.00 (95% CI 0.00-0.11), consistent with the nominal level
    mean rules per null dataset:       0.00
    note: with only 30 null runs this detects gross miscalibration only
  stability (10 subsamples): mean Jaccard overlap 0.54
```

How to read it:

* **Null calibration** passes: on shuffled copies of this data (same item frequencies, no
  associations) the pipeline almost never reports anything.
* **Stability 0.54** is a real signal, not a defect: each subsample has only 80% of the data, so
  borderline rules enter and leave. Use `rule_frequency` in the report to separate rules that
  appear in nearly every subsample from borderline ones.

## Does it beat the classical approach? A held-out test

`benchmarks/replication_groceries.py` splits the baskets in halves A and B, finds rules on A,
and checks them on B (6 random splits). "Classical" is support >= 0.5%, confidence >= 25%,
lift > 1, with no statistical testing.

| method | rules | lift > 1 on B | still productive on B | productive rules on B |
|---|---|---|---|---|
| classical (all) | 709 | 0.91 | 0.47 | 331 |
| classical, lift >= 2 | 376 | 0.97 | 0.44 | 166 |
| classical, top-K by lift | 176 | 0.99 | 0.43 | 76 |
| classical, top-K by confidence | 176 | 0.97 | 0.37 | 65 |
| rulewise holdout | 114 | 0.99 | **0.98** | 112 |
| rulewise direct | 176 | 0.98 | **0.98** | 173 |

(K is set to rulewise's rule count, which gives the classical baselines an advantage.)

What this does and does not show:

* On this data the **classical rules are already mostly real**: 91% have lift > 1 on held-out
  baskets. Raising the lift threshold does not change that much. A supermarket has strong
  genuine associations, and a 0.5% support floor already removes most noise.
* The difference is **redundancy**: only 47% of the classical list is still productive on
  held-out data (the rest is explained by simpler rules), and no lift or confidence threshold
  fixes it. 98% of rulewise's rules are productive on baskets they never saw.
* The price is **recall**. The classical list contains 331 productive-on-B rules among 378
  redundant ones; rulewise direct recovers 173 and holdout 112. If you want every candidate
  and will filter by hand, the classical list is a bigger haystack.
* The metric rewards what rulewise optimises (productivity), so read it as validation that the
  error control generalises, not as a neutral ranking of all uses of rules.

## Where to next

* [API reference](../api.md) for every parameter and output column.
* [Methodology](../methodology.md) for the statistical test and the two modes.
* [Limitations](../limitations.md) for power, redundancy and basket-size effects.
* [Comparison](../comparison.md) for the head-to-head against mlxtend.
