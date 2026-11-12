# Tutorial: classification with rules (Mushrooms and Titanic)

Two real categorical datasets, downloaded rather than bundled. In a checkout of the repository,
run `python benchmarks/get_data.py`; otherwise download them directly:

```bash
curl -L -o mushrooms.csv https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/mushrooms.csv
curl -L -o titanic.csv   https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv
```

## Mushrooms: 8,124 mushrooms, 22 categorical features, edible or poisonous

```python
import pandas as pd
from rulewise import RuleClassifier

d = pd.read_csv("mushrooms.csv")
y = (d["type"] == "p").to_numpy()                       # poisonous
X = pd.get_dummies(d.drop(columns="type"), prefix_sep="=").astype(bool)

clf = RuleClassifier(min_support=0.01, max_antecedent=3)
clf.fit(X.to_numpy(), y, item_names=list(X.columns))
clf.rules_[["antecedent", "class_", "confidence", "confidence_low", "support"]]
clf.explain(X.to_numpy()[:1])                           # the rule behind a prediction
```

Strong rules it finds (fit on all data; the interval is the 95% Wilson interval). This is a
selection, not the full list:

```
gill_size=b & odor=n & ring_number=o         -> poisonous=False  conf=1.000 [0.999,1.000] support=0.331
odor=n & stalk_shape=t                       -> poisonous=False  conf=1.000 [0.998,1.000] support=0.307
gill_spacing=c & stalk_surface_above_ring=k  -> poisonous=True   conf=1.000 [0.998,1.000] support=0.274
odor=f                                       -> poisonous=True   conf=1.000 [0.998,1.000] support=0.266
```

`odor=f` (a foul smell) being a perfect poison indicator is the classic finding for this dataset.

Cross-validated (5-fold x 3 repeats; `benchmarks/real_classification.py`):

| model | accuracy | log-loss | rules |
|---|---|---|---|
| majority class | 0.518 | 0.693 | |
| logistic regression | 1.000 | 0.006 | |
| decision tree (depth 4) | 0.991 | 0.043 | |
| **RuleClassifier (direct)** | 0.997 | 0.011 | 1,373 |
| RuleClassifier (holdout) | 0.996 | 0.021 | 1,042 |

Rules come close to logistic regression but do not beat it, and 1,373 rules is a lot to read:
many are different routes to the same conclusion because this dataset has many redundant,
correlated attributes. Use `query` (for example `min_confidence`, `max_antecedent=1`) to look at
the short, strong ones first.

## Titanic: 891 passengers, survived or not

Features are one-hot: class, sex, age band, having siblings/spouse, having parents/children,
port, fare tercile.

```
fare=high & no_parent_or_child & sex=female  -> survived=True   conf=0.984 [0.917,0.997] support=0.071
class=1 & sex=female                         -> survived=True   conf=0.968 [0.910,0.989] support=0.102
fare=low & sex=male                          -> survived=False  conf=0.894 [0.850,0.926] support=0.255
class=2 & sex=female                         -> survived=True   conf=0.921 [0.838,0.963] support=0.079
sex=male                                     -> survived=False  conf=0.811 [0.777,0.841] support=0.525
class=3 & port=S                             -> survived=False  conf=0.810 [0.766,0.848] support=0.321
```

| model | accuracy | log-loss | rules |
|---|---|---|---|
| majority class | 0.616 | 0.666 | |
| logistic regression | 0.790 | 0.464 | |
| decision tree (depth 4) | 0.818 | 0.493 | |
| **RuleClassifier (direct)** | 0.777 | 0.529 | 14 |
| RuleClassifier (holdout) | 0.773 | 0.543 | 7 |

Here the rules are a few points behind logistic regression and the tree. What you get in
exchange is a model you can read in a minute, with an interval on every rule.

## When to use it

Use rules where you need an explanation or a short list of statistically defensible patterns on
categorical or behavioural data. If you only want maximum accuracy, logistic regression or a
tree-based model is as good or better on all of these datasets.

## Reproducing the numbers

The tables are from `benchmarks/real_classification.py` on the local data files. Download them
once with `python benchmarks/get_data.py`. Exact values can move by a few thousandths across
library versions; the conclusions do not.

## Where to next

* [API reference](../api.md) for `RuleClassifier`, `predict_proba`, `explain` and `validate`.
* [Methodology](../methodology.md) for class-conditional mining and direct mode.
* [Limitations](../limitations.md) for the numeric-data and accuracy gap.
