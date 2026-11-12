"""ValidationReport: validation metrics as a dict with a readable summary."""
from __future__ import annotations

import numpy as np


def _fmt_ci(ci):
    return f"{ci[0]:.2f}-{ci[1]:.2f}"


class ValidationReport(dict):
    """A dict of validation metrics. ``print(report)`` or ``report.summary()`` explains them."""

    def summary(self) -> str:
        r = self
        out = [f"rulewise validation report (mode={r.get('mode')}, alpha={r.get('alpha')})"]
        out.append(f"  rules found on the real data:        {r['n_rules_real']}")
        rep = r.get("replication_rate")
        if rep is not None and not (isinstance(rep, float) and np.isnan(rep)):
            out.append(f"  holdout replication (p < alpha):      {rep:.2f} of tested candidates")
        runs = r["null_runs"]
        hits = int(round(r["null_fwer"] * runs))
        kind = r.get("null_kind", "shuffled datasets with the same item frequencies")
        out.append(f"  null calibration ({runs} {kind}):")
        verdict = ("consistent with the nominal level" if r["calibrated"]
                   else "ABOVE the nominal level - investigate")
        out.append(f"    datasets producing >= 1 rule:      {hits}/{runs}  -> empirical error rate "
                   f"{r['null_fwer']:.2f} (95% CI {_fmt_ci(r['null_fwer_ci'])}), {verdict}")
        out.append(f"    mean rules per null dataset:       {r['null_mean_rules']:.2f}")
        if runs < 100:
            out.append(f"    note: with only {runs} null runs this detects gross miscalibration only")
        if "stability_mean_jaccard" in r:
            out.append(f"  stability ({r['stability_resamples']} subsamples): mean Jaccard overlap "
                       f"{r['stability_mean_jaccard']:.2f}")
        if "cv_accuracy" in r:
            out.append(f"  cross-validated accuracy ({r['cv_folds']} folds): {r['cv_accuracy']:.3f} "
                       f"+/- {r['cv_accuracy_sd']:.3f}  (majority baseline {r['majority_accuracy']:.3f})")
            out.append(f"  cross-validated log-loss:             {r['cv_log_loss']:.3f}  "
                       f"(class-prior baseline {r['prior_log_loss']:.3f})")
        if not r.get("honest_estimates", True):
            out.append("  caution: direct mode - reported confidences/lifts are measured on the same "
                       "data that selected the rules")
        return "\n".join(out)

    __str__ = summary
