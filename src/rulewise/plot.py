"""Plots for rule sets and validation reports (matplotlib is an optional dependency)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .rulesmixin import fmt_item


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("plotting needs matplotlib: pip install matplotlib") from e
    return plt


def _rules(obj) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        df = obj._frame()
    return df


def _label(r, width=48):
    s = " & ".join(fmt_item(i) for i in r.antecedent) + "  ->  " + " & ".join(
        fmt_item(i) for i in r.consequent)
    return s if len(s) <= width else s[: max(0, width - 3)] + "..."


def plot_effect_sizes(rules, ax=None):
    """Effect size against significance: lift (x, log) vs -log10 adjusted p (y)."""
    plt = _plt()
    df = _rules(rules)
    ax = ax or plt.subplots(figsize=(6.5, 4.5))[1]
    if not len(df):
        ax.text(0.5, 0.5, "no significant rules", ha="center", va="center", transform=ax.transAxes)
        return ax
    floor = 1e-300
    p = df["p_adjusted"].to_numpy(float)
    y = -np.log10(np.clip(p, floor, 1.0))
    size = 25 + 250 * df["support"].to_numpy(float) / max(df["support"].max(), 1e-12)
    sc = ax.scatter(df["lift"], y, s=size, c=df["confidence"], cmap="viridis", vmin=0, vmax=1,
                    alpha=0.8, edgecolor="k", linewidth=0.3)
    ax.set_ylim(0, max(float(y.max()), 1.0) * 1.08)
    if (p < floor).any():
        ax.text(0.99, 0.02, f"{int((p < floor).sum())} rules with p < 1e-300 shown at the cap",
                ha="right", va="bottom", transform=ax.transAxes, fontsize=7, color="grey")
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_formatter(fmt)
    ax.tick_params(axis="x", which="minor", labelsize=7)
    ax.axvline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("lift (log scale)")
    ax.set_ylabel("-log10 adjusted p-value")
    ax.set_title("Significant rules: effect size vs evidence")
    ax.figure.colorbar(sc, ax=ax, label="confidence")
    return ax


def plot_forest(rules, top=15, metric="lift", ax=None):
    """Top rules with confidence intervals (lift if bootstrap intervals exist, else confidence)."""
    plt = _plt()
    df = _rules(rules).head(top)
    ax = ax or plt.subplots(figsize=(7, 0.35 * max(len(df), 3) + 1.2))[1]
    if not len(df):
        ax.text(0.5, 0.5, "no significant rules", ha="center", va="center", transform=ax.transAxes)
        return ax
    if metric == "lift" and "lift_low" in df:
        val, lo, hi, xl = df["lift"], df["lift_low"], df["lift_high"], "lift (95% bootstrap interval)"
    else:
        val, lo, hi = df["confidence"], df["confidence_low"], df["confidence_high"]
        xl = "confidence (95% Wilson interval)"
    ypos = np.arange(len(df))[::-1]
    ax.errorbar(val, ypos, xerr=[val - lo, hi - val], fmt="o", color="C0", ecolor="C0",
                capsize=2, ms=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels([_label(r) for r in df.itertuples()], fontsize=8)
    if metric == "lift" and "lift_low" in df:
        ax.axvline(1.0, color="grey", lw=0.8, ls=":")
    ax.set_xlabel(xl)
    ax.set_title(f"Top {len(df)} rules")
    return ax


def plot_network(rules, top=25, ax=None):
    """Items on a circle; an arrow from each antecedent item to each consequent item, width ~ lift."""
    plt = _plt()
    df = _rules(rules)
    if len(df):
        df = df.sort_values("lift", ascending=False).head(top)
    ax = ax or plt.subplots(figsize=(8.5, 6.5))[1]
    ax.set_axis_off()
    if not len(df):
        ax.text(0.5, 0.5, "no significant rules", ha="center", va="center", transform=ax.transAxes)
        return ax
    items = sorted({i for r in df.itertuples() for i in tuple(r.antecedent) + tuple(r.consequent)},
                   key=fmt_item)
    adj = {i: set() for i in items}
    for r in df.itertuples():
        grp = list(r.antecedent) + list(r.consequent)
        for a in grp:
            adj[a].update(x for x in grp if x != a)
    seen, order = set(), []
    for start in items:  # keep items of the same connected group adjacent on the circle
        if start in seen:
            continue
        stack = [start]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            order.append(u)
            stack.extend(sorted(adj[u] - seen, key=fmt_item, reverse=True))
    items = order
    ang = np.linspace(0, 2 * np.pi, len(items), endpoint=False)
    pos = {it: (np.cos(a), np.sin(a)) for it, a in zip(items, ang)}
    lmax = max(df["lift"].max(), 1e-9)
    for r in df.itertuples():
        for a in r.antecedent:
            for c in r.consequent:
                ax.annotate("", xy=pos[c], xytext=pos[a],
                            arrowprops=dict(arrowstyle="->", lw=0.6 + 3.0 * r.lift / lmax,
                                            color="C0", alpha=0.45, shrinkA=8, shrinkB=8))
    xs, ys = zip(*pos.values())
    ax.scatter(xs, ys, s=60, color="C1", zorder=3)
    for it, (x, y) in pos.items():  # labels sit outside the circle, aligned away from it
        ha = "left" if x > 0.15 else ("right" if x < -0.15 else "center")
        va = "bottom" if (abs(x) <= 0.15 and y > 0) else ("top" if abs(x) <= 0.15 else "center")
        ax.text(1.07 * x, 1.07 * y, fmt_item(it), ha=ha, va=va, fontsize=8)
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.35, 1.35)
    ax.set_title(f"Rule network (top {len(df)} by lift; arrow width ~ lift)")
    return ax


def plot_validation(report, ax=None):
    """Null-calibration counts, plus stability or cross-validation panels when available."""
    plt = _plt()
    has_stab = "rule_frequency" in report and len(report["rule_frequency"])
    has_cv = "cv_accuracy" in report
    n = 1 + int(bool(has_stab or has_cv))
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 3.8))
    axes = np.atleast_1d(axes)
    counts = report["null_rule_counts"]
    if max(counts) == 0:
        axes[0].scatter(range(len(counts)), counts, color="C2", zorder=3)
        axes[0].axhline(0, color="C2", lw=1)
        axes[0].set_ylim(-0.5, 3)
        axes[0].text(0.5, 0.55, f"0 rules in all {len(counts)} null datasets\n(shuffled data: nothing to find)",
                     ha="center", va="center", transform=axes[0].transAxes, color="C2")
    else:
        axes[0].bar(range(len(counts)), counts, color="C3")
        axes[0].set_ylim(0, max(1, max(counts) + 1))
    axes[0].set_xlabel("null dataset")
    axes[0].set_ylabel("rules found")
    axes[0].set_title(f"Null calibration (real data: {report['n_rules_real']} rules)")
    if has_stab:
        f = report["rule_frequency"].head(15)
        labels = [" & ".join(fmt_item(i) for i in a) + " -> " + " & ".join(fmt_item(i) for i in c)
                  if isinstance(c, tuple) else str(a) for a, c in zip(f.antecedent, f.consequent)]
        axes[1].barh(range(len(f))[::-1], f["frequency"], color="C0")
        axes[1].set_yticks(range(len(f))[::-1])
        axes[1].set_yticklabels([s[:40] for s in labels], fontsize=7)
        axes[1].set_xlim(0, 1)
        axes[1].set_xlabel("share of subsamples with the rule")
        axes[1].set_title("Rule stability")
    elif has_cv:
        vals = [report["majority_accuracy"], report["cv_accuracy"]]
        axes[1].bar(["majority\nbaseline", "rule\nclassifier"], vals, color=["grey", "C0"])
        axes[1].errorbar([1], [report["cv_accuracy"]], yerr=[report["cv_accuracy_sd"]], color="k")
        axes[1].set_ylim(0, 1)
        axes[1].set_title("Cross-validated accuracy")
    fig.tight_layout()
    return fig
