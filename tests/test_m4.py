import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from rulewise import RuleClassifier, RuleMiner, ValidationReport, plot  # noqa: E402
from tests.test_miner import planted  # noqa: E402


@pytest.fixture(scope="module")
def fitted():
    tx = planted()
    return tx, RuleMiner(min_support=0.01).fit(tx)


def test_miner_validation_report(fitted):
    tx, m = fitted
    rep = m.validate(tx, n_null=8, n_stability=4)
    assert isinstance(rep, ValidationReport) and isinstance(rep, dict)
    for k in ("n_rules_real", "null_fwer", "null_fwer_ci", "null_rule_counts", "calibrated",
              "stability_mean_jaccard", "rule_frequency", "mode", "alpha"):
        assert k in rep
    assert len(rep["null_rule_counts"]) == 8 and rep["n_rules_real"] > 0
    text = str(rep)
    assert "empirical error rate" in text and "stability" in text and "only 8 null runs" in text
    assert rep["null_fwer"] == rep.get("null_fwer")  # dict access works


def test_report_flags_miscalibration():
    rep = ValidationReport(mode="holdout", alpha=0.05, honest_estimates=True, n_rules_real=3,
                           replication_rate=0.5, null_runs=200, null_rule_counts=[1] * 100 + [0] * 100,
                           null_mean_rules=0.5, null_fwer=0.5, null_fwer_ci=(0.43, 0.57),
                           calibrated=False)
    assert "ABOVE the nominal level" in rep.summary()
    rep["honest_estimates"] = False
    assert "direct mode" in rep.summary()


def test_classifier_validation_report():
    rng = np.random.default_rng(0)
    X = rng.random((3000, 10)) < 0.3
    y = (X[:, 0] & X[:, 1]) ^ (rng.random(3000) < 0.05)
    rep = RuleClassifier(min_support=0.01).validate(X, y, n_null=8, cv=4)
    assert rep["null_rule_counts"] == [0] * 8 or sum(rep["null_rule_counts"]) <= 2
    assert rep["cv_accuracy"] > rep["majority_accuracy"] + 0.05
    assert rep["cv_log_loss"] < rep["prior_log_loss"]
    assert "cross-validated accuracy" in str(rep) and "permuted labels" in str(rep)


def test_query_filters_and_sorting(fitted):
    _, m = fitted
    allr = m.query()
    assert len(allr) == len(m.rules_) > 2
    hi = m.query(min_lift=allr.lift.median())
    assert 0 < len(hi) <= len(allr) and (hi.lift >= allr.lift.median()).all()
    assert all("A" in set(a) | set(c) for a, c in zip(m.query(contains="A").antecedent,
                                                       m.query(contains="A").consequent))
    q = m.query(antecedent_contains=["A", "B"])
    assert len(q) and all({"A", "B"} <= set(a) for a in q.antecedent)
    q = m.query(consequent_contains="C")
    assert len(q) and all("C" in c for c in q.consequent)
    top = m.query(sort_by="lift", top=2)
    assert len(top) == 2 and top.lift.iloc[0] >= top.lift.iloc[1]
    assert len(m.query(max_antecedent=1)) < len(allr)
    assert len(m.query(contains="nonexistent")) == 0


def test_export_roundtrip_and_polars(fitted, tmp_path):
    _, m = fitted
    csv, js = tmp_path / "r.csv", tmp_path / "r.json"
    m.export(csv)
    m.export(js, min_lift=1.0)
    df = pd.read_csv(csv)
    assert len(df) == len(m.rules_)
    assert df.antecedent.str.contains(r"\w").all() and df.consequent.map(lambda v: isinstance(v, str)).all()
    recs = json.loads(js.read_text())
    assert len(recs) == len(m.query(min_lift=1.0)) and "p_adjusted" in recs[0]
    with pytest.raises(ValueError):
        m.export(tmp_path / "r.xlsx")
    pl = pytest.importorskip("polars")
    pdf = m.to_polars()
    assert isinstance(pdf, pl.DataFrame) and pdf.height == len(m.rules_)


def test_classifier_query_export_summary(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.random((3000, 8)) < 0.3
    y = (X[:, 0] & ~X[:, 1]) ^ (rng.random(3000) < 0.04)
    c = RuleClassifier(min_support=0.01, add_negations=True).fit(X, y)
    q = c.query(min_confidence=0.8)
    assert len(q) and "consequent" in q.columns
    c.export(tmp_path / "c.csv")
    flat = pd.read_csv(tmp_path / "c.csv")
    assert flat.antecedent.str.contains("not 1").any()  # negations rendered readably
    assert "significant rules" in c.summary()


def test_summary_text(fitted):
    _, m = fitted
    s = m.summary(top=3)
    assert "significant rules" in s and "conf=" in s and "lift=" in s
    empty = RuleMiner(min_support=0.01).fit([["a"], ["b"]] * 50)
    assert "No rule passed" in empty.summary()


def test_plots_render(fitted, tmp_path):
    tx, m = fitted
    for kind, kw in (("effect", {}), ("forest", {"top": 8}), ("network", {"top": 10})):
        ax = m.plot(kind, **kw)
        ax.figure.savefig(tmp_path / f"{kind}.png")
        plt.close(ax.figure)
        assert (tmp_path / f"{kind}.png").stat().st_size > 1000
    fig = plot.plot_validation(m.validate(tx, n_null=4, n_stability=3))
    fig.savefig(tmp_path / "v.png")
    plt.close(fig)
    with pytest.raises(ValueError):
        m.plot("nope")


def test_plots_handle_empty_and_classifier_and_negations():
    empty = RuleMiner(min_support=0.01).fit([["a"], ["b"]] * 50)
    for kind in ("effect", "forest", "network"):
        ax = empty.plot(kind)
        plt.close(ax.figure)
    rng = np.random.default_rng(0)
    X = rng.random((3000, 8)) < 0.3
    y = (X[:, 0] & ~X[:, 1]) ^ (rng.random(3000) < 0.04)
    c = RuleClassifier(min_support=0.01, add_negations=True).fit(X, y)
    for kind in ("effect", "forest", "network"):
        ax = c.plot(kind)
        plt.close(ax.figure)
