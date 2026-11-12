"""Download the public datasets used by the benchmark scripts.

    python benchmarks/get_data.py

Files are written to benchmarks/data/ (not bundled in the wheel).
"""
from __future__ import annotations

import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

FILES = {
    "groceries.csv": "https://raw.githubusercontent.com/stedy/"
                     "Machine-Learning-with-R-datasets/master/groceries.csv",
    "mushroom.data": "https://archive.ics.uci.edu/ml/machine-learning-databases/"
                     "mushroom/agaricus-lepiota.data",
    "mushrooms.csv": "https://raw.githubusercontent.com/stedy/"
                     "Machine-Learning-with-R-datasets/master/mushrooms.csv",
    "titanic.csv": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv",
}


def main():
    os.makedirs(DATA, exist_ok=True)
    for name, url in FILES.items():
        path = os.path.join(DATA, name)
        if os.path.exists(path):
            print(f"have {path}")
            continue
        print(f"downloading {url} -> {path}")
        with urllib.request.urlopen(url, timeout=60) as fh, open(path, "wb") as out:
            out.write(fh.read())
        print(f"  {os.path.getsize(path) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
