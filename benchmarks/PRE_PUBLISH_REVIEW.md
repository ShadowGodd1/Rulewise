# Pre-publication review: is rulewise better than what Python already has?

Date: 2026-11-12. Environment: Windows, Python 3.12.4, mlxtend 0.25.0, numpy 2.3.5,
pandas 3.1.1, scipy 1.18.1, scikit-learn 1.9.1.

The question was whether publishing rulewise adds something the community does not already
have. The answer is **yes, for one specific job: trustworthy discovery of associations**.
It is not a faster or more general replacement for mlxtend, and it should not be marketed as
one. The evidence and the remaining risks are below. All tables are reproducible from the
scripts in `benchmarks/`; the generated evidence is `benchmarks/COMPARISON.md`.

## Bottom line

| claim | verdict | evidence |
|---|---|---|
| Finds the same frequent itemsets as mlxtend | **confirmed** | identical sets and counts at 4 supports on Groceries (122 / 333 / 1,001 / 4,223 itemsets) |
| Controls false discoveries on noise | **confirmed, and conservative** | 4/500 null datasets returned any rule (0.8%, 95% CI 0.3-2.0%) vs 500/500 for mlxtend (mean 1,929 rules) |
| Returns rules that keep working out of sample | **strong advantage** | Groceries: 98.7% (holdout) / 99.3% (direct) of returned rules productive on held-out data vs 48.8% for a classical list |
| Recovers the real structure | **yes, at moderate signal** | 4/4 planted rules recovered from 35 returned rules; classical needs 15,685 rules to recover 4/4 |
| Higher recall than a classical list | **no** | 35% (holdout) / 54% (direct) as many held-out-productive rules |
| Better than logistic/tree on general classification | **no** | Mushroom: 0.997 vs 0.9998 logistic; digits already documented at 0.61 vs 0.97 |
| Better recommendations | **no** | Groceries hit@5: popularity 26.4%, classical 20.1%, rulewise 18.7-19.7% |
| Faster than mlxtend | **yes in these runs** | Groceries 0.08 s vs 0.35 s; 20k synthetic 0.62 s vs 2.62 s; dense 50k 2.75 s vs 8.39 s |
| A gap in the Python ecosystem | **yes** | mlxtend, efficient-apriori and apyori are classical; Webb's OPUS/Magnum Opus is Java/closed. We found no maintained Python package that does statistically sound rule discovery with FWER control. |

## What already exists (and why it is not this)

| tool | what it does | why it is not a substitute |
|---|---|---|
| mlxtend | apriori / fpgrowth / fpmax, many rule metrics | classical thresholds only; no significance, no FWER/FDR, no productivity test |
| efficient-apriori, apyori | classical ARM | same: frequent itemsets + confidence/lift, no statistical error control |
| ruleminer | validation rules over pandas DataFrames, many metrics | rule templates for validation; no multiple-testing control and no held-out calibration |
| pysubgroup | subgroup discovery with chi-squared / standard quality functions, beam search | closest neighbour, but it searches for a small set of subgroups against one target, not all rules; p-values are per-subgroup, not corrected across the whole search |
| FASM / FAST-YB (ICDM 2023) | research algorithms for FDR-controlled significant pattern mining | research code, not a maintained Python package; targets significant itemsets, not a general rule-mining API |
| OPUS / Magnum Opus / Skopus (Webb) | the reference statistically sound pattern-discovery systems | Java/commercial, not Python, not scikit-learn-style |
| R `arules` | classical ARM and visualization | no FWER control across the search |

The gap is therefore **not** "Python has no association-rule mining". It is narrower and more
defensible: Python has no maintained, scikit-learn-style package that turns the statistically
sound pattern-discovery literature into a usable rule miner with error control across the whole
search and honest out-of-sample reporting. That is the gap rulewise fills.

## 1. Correctness is not in question

`rulewise.mining.mine_itemsets` returned exactly the same itemsets and absolute counts as
mlxtend's `fpgrowth` on Groceries at support 0.02, 0.01, 0.005 and 0.002. The statistical
layer therefore sits on a correct miner, not a different one. `fisher_greater` also matched
`scipy.stats.fisher_exact` on 20,000 random 2x2 tables (worst absolute error 4.4e-16), and the
capped version was exact whenever the p-value was below the cap.

## 2. Error control: the real differentiator

500 independent null datasets (3,000 transactions, 25 items, varied frequencies):

| method | mean rules | datasets with >=1 rule | rate (95% CI) |
|---|---|---|---|
| mlxtend (support 0.01, confidence 0.25, lift>1) | 1,929 | 500/500 | 1.000 |
| rulewise holdout (Holm) | 0.02 | 4/500 | 0.008 (0.003-0.020) |
| rulewise direct (layered) | 0.01 | 4/500 | 0.008 (0.003-0.020) |

The 4 false-positive datasets each returned **one association as two symmetric rules**
(`A -> B` and `B -> A`), with adjusted p-values 0.005-0.038. That is a legitimate 5%-level
false discovery, not a bug. The independent `error_control_simulation.py 300` run reproduced
the README table exactly:

| configuration | rate |
|---|---|
| RuleMiner holdout / Holm | 0.003 |
| RuleMiner holdout / BH (FDR) | 0.007 |
| RuleMiner direct (layered) | 0.003 |
| RuleMiner, 2-item consequents | 0.000 |
| RuleClassifier direct | 0.017 |
| RuleClassifier holdout | 0.007 |

The guarantee is deliberately conservative (0.8% at a nominal 5%), which is the source of the
recall cost. This is a defensible trade-off, but it must be stated plainly to users.

## 3. Out-of-sample precision: a large, reproducible advantage

Planted-rule benchmark (5 datasets, 20,000 baskets, 30 items, four planted rules, 15% noise).
All rules were mined on half and evaluated on the untouched half.

| method | rules | productive on test | lift>1 on test | planted recall | reliable F1 |
|---|---|---|---|---|---|
| classical all (mlxtend) | 15,685 | 1.4% | 63.3% | 100% | 0.03 |
| classical top-35 by lift | 35 | 9.1% | 100% | 65% | 0.16 |
| classical top-35 by confidence | 35 | 1.1% | 100% | 0% | 0.00 |
| mlxtend + Fisher/Holm | 3,486 | 3.1% | 100% | 100% | 0.06 |
| rulewise holdout | 35 | **100.0%** | 100% | **100%** | **1.00** |
| rulewise direct | 35 | **100.0%** | 100% | **100%** | **1.00** |

The `mlxtend + Fisher/Holm` row is important: even after a statistically valid association
test with Holm correction, only 3.1% of the returned rules survive the productivity test.
Significance alone is not enough; the simplification test is what removes the redundant
`A, B -> C` variants that a classical miner emits.

The "reliable F1" uses productive precision (is the returned rule real?) and planted recall
(did we find the planted rule?). The stricter "exact planted precision" is 11.4% for rulewise:
the other 31 rules are genuine sub-rules and reverse rules of the four planted structures, not
false positives. That is worth stating in the docs, because a user who expects four rules gets
35.

On real Groceries baskets (5 random halves):

| method | rules | productive on held-out | lift>1 | productive vs classical | fit |
|---|---|---|---|---|---|
| classical all | 670 | 48.8% | 98.5% | 100% | 0.19 s |
| classical lift>=2 | 350 | 48.9% | 99.9% | 52.1% | - |
| classical top-K by lift | 174 | 47.0% | 100% | 25.1% | - |
| mlxtend + Fisher/Holm | 185 | 64.9% | 100% | 37.0% | - |
| rulewise holdout | 116 | **98.7%** | 100% | 35.3% | 0.05 s |
| rulewise direct | 174 | **99.3%** | 100% | 53.6% | 0.07 s |

This reproduces the README: about half (direct) to a third (holdout) as many productive rules,
with 98-99% of what is returned still productive versus 49% for the classical list. Classical
rules on this data are mostly real (98.5% have lift>1 held out); their problem is redundancy,
not statistical falsehood. The README wording now says "redundant or do not replicate" rather
than "noise", which matches the evidence.

## 4. Where rulewise is not better

* **Recall / power.** It returns fewer true rules. At n=5,000 with 20 planted 3-item patterns
  among 1,000 items, it recovered 9/20 (holdout) and 11/20 (direct) at the weakest effect;
  20/20 once the effect was stronger (strength >= 0.005), with zero false rules. Weak effects
  need more data. Direct mode is consistently more powerful at low signal.
* **Recommendation.** Hit@5 on Groceries: popularity 26.4%, classical all 20.1%, rulewise
  18.7% (holdout) / 19.7% (direct). Do not market `recommend` as beating a popularity baseline;
  it is an explainable interface, not a state-of-the-art recommender.
* **Classification.** On Mushroom (categorical, rule-friendly), direct rules reach 0.9970
  accuracy and 0.0113 log-loss; logistic reaches 0.9998 / 0.0007. On dense numeric data the
  README's digits result (0.61 vs 0.97) is the honest picture.
* **Feature breadth.** mlxtend offers apriori, fpgrowth, fpmax, closed itemsets, many rule
  metrics (cosine, jaccard, certainty, kulczynski, ...), and a mature API. rulewise offers one
  mining strategy, positive associations only (lift>1), and five measures. That is a deliberate
  focus, but it is narrower.
* **Redundancy is reduced, not eliminated.** Four planted structures produced 35 rules because
  productivity removes *irrelevant extra antecedent items*, not nested sub-rules or reverse
  rules. If the goal is a minimal rule set, a self-sufficient-itemset / non-redundant filter
  (Webb 2010) would be the next feature.
* **The `pyfim` backend is not usable on modern Python.** It has no wheel beyond cp37; the
  source build failed on Python 3.12 / Windows. The README no longer advertises a `[fast]`
  extra or a 14x speedup. The pure-Python bitset Eclat was faster than mlxtend on every dataset
  tested, so the extra is not needed for competitiveness.

## 5. Publication blockers and what was fixed in this review

Fixed:

* `RuleMiner.transform` crashed on a fit with zero significant rules (`rules_` had no
  `antecedent` column). It now returns an `(n, 0)` matrix, and an empty fit keeps the standard
  schema.
* `predict`/`transform` before `fit` now raise scikit-learn's `NotFittedError` (still a
  `RuntimeError`/`ValueError` for the fallback), not a bare `RuntimeError`.
* Matrices now set `n_features_in_` and reject a different column count with the standard
  scikit-learn message. `transform`/`predict`/`explain` check it.
* Matrix input rejects complex data, NaN/inf, and empty feature sets with clear errors; sparse
  input is declared in the sklearn tags.
* The class-conditional equivalence test no longer skips the pure-Python backend when `pyfim`
  is absent; the non-pyfim input/transform tests in `test_backend_io.py` now run too.
* `pyfim` was removed from `[dev]` (it broke `pip install -e ".[dev]"` on Python 3.12/Windows)
  and the `[fast]` extra was removed. The `backend="pyfim"` code path remains for platforms
  where it can be installed manually.
* Benchmark data paths are local (`benchmarks/data/`), with `benchmarks/get_data.py` and a data
  README. `library_comparison.py` and `COMPARISON.md` make the head-to-head reproducible.
* README claims were tightened: "most of them noise" -> "redundant or do not replicate";
  "drops rules that add nothing" -> "extra antecedent items add nothing"; added the power
  limitation and the comparison report.

Still to do before `twine upload`:

1. Version is now `0.1.0` (was `0.1.0.dev0`), so `pip install rulewise` will pick it up without
   `--pre`. Bump it before each release.
2. Author and classifiers are set. Add `project.urls` (repository, issues, documentation) once the
   repository URL is known. The `rulewise` name is free on PyPI (checked).
3. Run the CI matrix on Linux and macOS. The local run was Windows-only. The current
   `no-optional-deps` job is now green locally (25 passed, 17 skipped) and the full suite is
   56 passed, 12 skipped. `twine check` passes with `packaging>=24.2`; pin or upgrade if the
   release environment is older.
4. Consider a short paper or a JOSS submission. The method is a careful Python implementation
   of published statistical results (Webb 2007/2008/2010), so the contribution is the usable,
   validated implementation and the empirical comparison, not a new theorem. Reviewers will
   ask for exactly the null-data and held-out-probability evidence above; it now exists.
5. Decide whether to add a non-redundant/self-sufficient output mode. It is the clearest gap
   between the README promise and the returned rule set (35 rules for 4 structures).

## 6. Recommended positioning

Lead with the job, not the algorithm:

> rulewise is for analysts who need association rules they can defend: it controls the chance
> of reporting a spurious rule, keeps only rules that still beat their own simplifications on
> held-out data, and tells you when the data cannot support a claim. Use mlxtend when you want
> the full classical catalogue, many metrics, or closed/maximal itemsets; use rulewise when a
> short, statistically vetted list matters more than recall.

Do not claim:

* "better association rule mining" in general (it loses on recall and feature breadth);
* "faster than mlxtend" as a headline (it happened to be faster here, but speed is not the
  contribution and PyFIM is not available);
* "better recommendations" or "better classifier" (the benchmarks do not support either).

Do claim:

* exact, correct frequent-itemset mining;
* empirical FWER control on null data;
* 98-99% held-out productivity vs ~49% for a classical list on Groceries;
* honest intervals and a validation report;
* a scikit-learn-compatible API and a small, readable rule set.

## 7. Reproduce

```bash
pip install -e ".[dev]"
python benchmarks/get_data.py
python benchmarks/library_comparison.py --section all --runs 500 --seeds 5 \
       --out benchmarks/COMPARISON.md
python benchmarks/error_control_simulation.py 300
python benchmarks/power_study.py 5000
pytest -q
```

Core-only environment (no optional dependencies):

```bash
pip install -e . pytest
pytest -q          # 25 passed, 17 skipped locally
```
