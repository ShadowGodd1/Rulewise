# rulewise documentation

rulewise mines association rules and keeps only those that survive exact statistical testing,
multiple-testing correction and a simplification (productivity) test. It also reports honest
intervals and a validation report.

Start with the five-minute example below, then use these pages:

| page | what it covers |
|---|---|
| [API reference](api.md) | parameters, attributes, methods, output columns and input formats |
| [Methodology](methodology.md) | exact tests, productivity, corrections, holdout and direct modes |
| [Limitations](limitations.md) | assumptions, power, redundancy, basket-size effects and other failure modes |
| [Comparison](comparison.md) | measured evidence against mlxtend and the classical workflow |
| [Groceries tutorial](tutorials/groceries.md) | real supermarket baskets end to end |
| [Classification tutorial](tutorials/classification.md) | rule-based classification on Mushrooms and Titanic |
| [Publishing](publishing.md) | release checklist, build and upload |

## Five-minute example

```python
from rulewise import RuleMiner

baskets = (
    [["a", "b", "c"]] * 150
    + [["a", "b"]] * 50
    + [["c"]] * 50
    + [["d", "e"]] * 50
)

miner = RuleMiner(min_support=0.2, n_bootstrap=0).fit(baskets)
print(miner.summary())
```

This toy example finds six rules, including the symmetric `a -> b` and `b -> a` and the
partially productive `a -> c` and `c -> a`. On real data use a support floor that leaves enough
candidates, for example 0.005 to 0.01 for supermarket baskets.

## The one idea

A classical miner answers: which rules are frequent and confident? rulewise answers: which rules
are frequent, confident, **and still useful after accounting for every rule we tried**?

That difference is measured, not asserted. On 500 pure-noise datasets, mlxtend returned rules in
500 of them (mean 1,929 rules); rulewise returned a rule in 4 (0.8%). On Groceries, 98.7% to
99.3% of rulewise's rules were still productive on held-out baskets, against 48.8% for a
classical list. The price is recall. See [Comparison](comparison.md) and [Limitations](limitations.md).
