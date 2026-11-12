# Benchmark data

These files are public and are not shipped in the package. Fetch them with:

```bash
python benchmarks/get_data.py
```

| file | source | used by |
|---|---|---|
| `groceries.csv` | [Groceries](https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/groceries.csv) (9,835 baskets) | `library_comparison.py`, `replication_groceries.py`, `docs/tutorials/groceries.md` |
| `mushroom.data` | [UCI Mushroom](https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data) (8,124 rows, 22 categorical features) | `library_comparison.py` |
| `mushrooms.csv` | same data with a header, from the R datasets mirror | `real_classification.py`, `docs/tutorials/classification.md` |
| `titanic.csv` | [seaborn-data](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv) (891 rows) | `real_classification.py`, `docs/tutorials/classification.md` |
