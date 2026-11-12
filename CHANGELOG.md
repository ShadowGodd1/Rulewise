# Changelog

## 0.1.0 (2026-11-12)

First public version.

* `RuleMiner`: mining (bitset Eclat; optional PyFIM backend), exact one-sided Fisher tests,
  Webb productivity test, Holm / Bonferroni / Benjamini-Hochberg / Benjamini-Yekutieli
  correction, holdout mode and direct mode (layered critical values), Wilson and bootstrap
  intervals, multi-item consequents, stability and null-calibration validation.
* `RuleClassifier`: class-association rules, class-conditional mining, optional negated items,
  probabilities from the best matching rule, `explain`, permuted-label and cross-validated
  validation.
* scikit-learn compatibility; input from lists, numpy, scipy sparse, pandas and polars.
* `query`, `export` (csv/json/parquet), `to_polars`, `summary`, plots.
* Documentation site under `docs/`, with an API reference, methodology, limitations,
  comparison, two tutorials and a release guide. Build with `python -m mkdocs build --strict`.
* Known limits are documented in `docs/limitations.md` and `benchmarks/RESULTS.md`.
  Head-to-head evidence against mlxtend is in `benchmarks/COMPARISON.md`, with a
  pre-publication review in `benchmarks/PRE_PUBLISH_REVIEW.md`.

### Packaging

* Removed the `[fast]` extra and `pyfim` from `[dev]`: PyFIM has no wheel for modern Python
  and failed to build on Python 3.12/Windows. The `backend="pyfim"` option remains for users
  who install it themselves; the pure-Python backend is the default.
* Benchmark data is local and reproducible via `benchmarks/get_data.py`.

### Fixes

* `RuleMiner.transform` no longer crashes when a fit returns zero rules; it returns an
  `(n, 0)` matrix and the empty `rules_` table keeps the standard columns.
* `predict`/`transform` before `fit` raise scikit-learn's `NotFittedError`.
* Matrix input sets `n_features_in_` and checks the column count after fit; complex data,
  NaN/inf and zero-width matrices are rejected with clear errors.
* Optional-dependency tests now skip individually, so the pure-Python backend and the input
  adapters are tested even when PyFIM, scikit-learn, matplotlib, mlxtend or statsmodels are
  absent.
