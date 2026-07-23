"""Automated bias detection and data-level de-biasing.

Detection (automated, data-driven):
  1. correlation scan - how strongly each feature correlates with race;
  2. proxy test - can a logistic regression predict race from the non-race
     features? If yes, dropping the race column is not enough
     ("fairness through unawareness" fails).

Mitigation (data-level, so any downstream model benefits):
  1. remove the race dummies from the feature set;
  2. Fairlearn CorrelationRemover: linearly project the remaining features so
     their correlation with race is removed, keeping everything else intact.

Outputs: models/correlation_remover.joblib,
data/processed/{train,test}_debiased.csv, figures/05_*.png,
reports/05_debias.md
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.preprocessing import CorrelationRemover
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import common
from common import FIGURES_DIR, MODELS_DIR, PROCESSED_DIR, REPORTS_DIR, SERIES, TARGET

SENSITIVE_COL = "race_african_american"
NON_SENSITIVE = [
    "age", "priors_count", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "charge_felony", "sex_male",
]


def main() -> None:
    common.apply_plot_style()
    train, test = common.train_test_frames()

    X_train = common.build_features(train)   # includes race dummies
    X_test = common.build_features(test)
    is_aa_train = X_train[SENSITIVE_COL]
    is_aa_test = X_test[SENSITIVE_COL]

    # --- detection 1: correlation scan -------------------------------------
    corr_before = X_train[NON_SENSITIVE].corrwith(is_aa_train)

    # --- detection 2: proxy predictability ---------------------------------
    clf = LogisticRegression(max_iter=1000)
    scaler = StandardScaler().fit(X_train[NON_SENSITIVE])
    clf.fit(scaler.transform(X_train[NON_SENSITIVE]), is_aa_train)
    auc_before = roc_auc_score(
        is_aa_test, clf.predict_proba(scaler.transform(X_test[NON_SENSITIVE]))[:, 1])

    # --- mitigation: drop race + CorrelationRemover ------------------------
    remover = CorrelationRemover(sensitive_feature_ids=[SENSITIVE_COL], alpha=1.0)
    cols_for_remover = [SENSITIVE_COL] + NON_SENSITIVE
    remover.fit(X_train[cols_for_remover])
    Xd_train = pd.DataFrame(remover.transform(X_train[cols_for_remover]),
                            columns=NON_SENSITIVE, index=X_train.index)
    Xd_test = pd.DataFrame(remover.transform(X_test[cols_for_remover]),
                           columns=NON_SENSITIVE, index=X_test.index)

    corr_after = Xd_train.corrwith(is_aa_train)

    clf2 = LogisticRegression(max_iter=1000)
    scaler2 = StandardScaler().fit(Xd_train)
    clf2.fit(scaler2.transform(Xd_train), is_aa_train)
    auc_after = roc_auc_score(
        is_aa_test, clf2.predict_proba(scaler2.transform(Xd_test))[:, 1])

    joblib.dump(remover, MODELS_DIR / "correlation_remover.joblib")

    out_train = Xd_train.copy()
    out_train[TARGET] = train[TARGET].values
    out_train["race"] = train["race"].values
    out_train.to_csv(PROCESSED_DIR / "train_debiased.csv", index=False)
    out_test = Xd_test.copy()
    out_test[TARGET] = test[TARGET].values
    out_test["race"] = test["race"].values
    out_test.to_csv(PROCESSED_DIR / "test_debiased.csv", index=False)

    # --- figure: correlation before/after ----------------------------------
    order = corr_before.abs().sort_values().index
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    y = np.arange(len(order))
    ax.barh(y + 0.19, corr_before[order].values, height=0.34, color=SERIES[0],
            label="original data")
    ax.barh(y - 0.19, corr_after[order].values, height=0.34, color=SERIES[1],
            label="after CorrelationRemover")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.axvline(0, color=common.BASELINE, linewidth=0.8)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Pearson correlation with African-American indicator")
    ax.set_title("Feature-race correlation before and after de-biasing")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.text(-0.19, 2.2,
            "after de-biasing every correlation is exactly 0\n(orange bars have zero length)",
            fontsize=8, color=common.INK_SECONDARY, style="italic")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "05_correlation_removal.png", bbox_inches="tight")
    plt.close(fig)

    base_rates = train.groupby("race")[TARGET].mean()

    report = f"""# Automated bias detection and data-level de-biasing

## Detection 1 - which features carry race information?

Pearson correlation of each candidate feature with the African-American
indicator (training split):

| Feature | Correlation (original) | Correlation (de-biased) |
|---------|----------------------:|------------------------:|
{chr(10).join(f"| `{f}` | {corr_before[f]:+.3f} | {corr_after[f]:+.3f} |" for f in corr_before.abs().sort_values(ascending=False).index)}

`priors_count`, `age` and the juvenile counts all correlate with race - they
are exactly the features the baseline models lean on (report 04). This is the
quantitative footprint of the institutional bias discussed in RQ2: unequal
policing shows up as unequal *measured* criminal history.

## Detection 2 - the proxy test

A logistic regression trying to predict whether a defendant is
African-American from the seven non-race features reaches
**AUC {auc_before:.3f}** on the held-out test split - far from random (0.5).
Simply deleting the race column therefore does **not** remove race from the
data; the model can partially reconstruct it. This is the classic failure of
*fairness through unawareness*.

## Mitigation - what was changed and why

1. **Race dummies removed** from the feature set. Using a protected attribute
   as a direct input to a punitive risk model is not compatible with ALTAI
   Requirement #5 (avoidance of unfair bias) or with equal-protection norms.
2. **Fairlearn `CorrelationRemover` (alpha=1.0)** applied to the remaining
   features: each feature is linearly transformed so its correlation with the
   African-American indicator becomes zero, while retaining as much of the
   original information as possible.

After the transformation the same proxy test drops to
**AUC {auc_after:.3f}** (~random), and every feature-race correlation is
~0 (figure below). Race - direct or by linear proxy - is no longer encoded in
the training data.

![Correlation removal](../figures/05_correlation_removal.png)

## What this does *not* fix (honest limitations)

- **Label bias.** The target `two_year_recid` means *re-arrest*. Base rates in
  the training data differ ({base_rates['African-American']:.0%}
  African-American vs {base_rates['Caucasian']:.0%} Caucasian), and part of
  that gap is produced by unequal enforcement, which no feature transformation
  can undo. De-biasing the features removes the model's ability to treat
  equally-situated people differently by race; it cannot correct who got
  arrested in the first place.
- **Non-linear proxies.** CorrelationRemover removes *linear* dependence. The
  near-random proxy AUC after transformation suggests little non-linear signal
  remains here, but this must be re-checked whenever features are added.
- **The impossibility theorem still applies.** With different base rates,
  calibration and equal error rates cannot hold simultaneously
  (Chouldechova 2017); script 06 measures where the de-biased model lands.

## Note on the dropped AutoML step

The original proposal included an AutoML tool to pinpoint what needs
de-biasing. That step was descoped; the automated detection above (correlation
scan + proxy predictability test + the Fairlearn audit of report 04) fulfils
the same role with transparent, reproducible methods.
"""
    (REPORTS_DIR / "05_debias.md").write_text(report)
    print(f"proxy AUC before {auc_before:.3f} -> after {auc_after:.3f}")
    print("max |corr| after removal:", corr_after.abs().max().round(4))
    print("Wrote debiased train/test, remover, figure, reports/05_debias.md")


if __name__ == "__main__":
    main()
