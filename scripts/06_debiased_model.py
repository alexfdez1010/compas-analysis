"""Train the Logistic Regression on the de-biased data and compare fairness
with the baseline.

Same architecture as the biased reference model (StandardScaler + Logistic
Regression), trained on the CorrelationRemover-transformed features without
race. Race is kept alongside ONLY as an audit attribute, never as an input.

Outputs: models/lr_debiased.joblib, figures/06_fairness_comparison.png,
reports/06_debiased_model.md
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
    selection_rate,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common
from common import (
    FIGURES_DIR,
    INK_SECONDARY,
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS_DIR,
    SERIES,
    TARGET,
)

FEATURES = [
    "age", "priors_count", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "charge_felony", "sex_male",
]


def rates(y_true, y_pred, race, group):
    m = race == group
    return (false_positive_rate(y_true[m], y_pred[m]),
            false_negative_rate(y_true[m], y_pred[m]))


def main() -> None:
    common.apply_plot_style()

    train_d = pd.read_csv(PROCESSED_DIR / "train_debiased.csv")
    test_d = pd.read_csv(PROCESSED_DIR / "test_debiased.csv")
    train, test = common.train_test_frames()

    lr_d = make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=1000, random_state=42))
    lr_d.fit(train_d[FEATURES], train_d[TARGET])
    joblib.dump(lr_d, MODELS_DIR / "lr_debiased.joblib")

    # Canonical deployment is RACE-BLIND inference: the CorrelationRemover is a
    # training-time intervention only, and at prediction time the model receives
    # the raw (untransformed) features, which contain no race column. This makes
    # the de-biased model invariant to race by construction (counterfactual
    # fairness at the individual level) and means no protected attribute is
    # needed at decision time. The race-aware alternative (transforming the
    # incoming features with the person's race) is evaluated below for
    # comparison; it yields better group metrics but lets a person's stated
    # race move their individual score.
    X_test_raw = common.build_features(test, include_race=False)[FEATURES]
    pred_d = lr_d.predict(X_test_raw)
    acc_d = accuracy_score(test[TARGET], pred_d)
    auc_d = roc_auc_score(test[TARGET], lr_d.predict_proba(X_test_raw)[:, 1])

    # race-aware variant (transform test features with the person's race)
    pred_d_aware = lr_d.predict(test_d[FEATURES])

    # biased reference predictions on the same test rows
    lr_b = joblib.load(MODELS_DIR / "lr_biased.joblib")
    X_test_b = common.build_features(test)
    pred_b = lr_b.predict(X_test_b)
    acc_b = accuracy_score(test[TARGET], pred_b)
    auc_b = roc_auc_score(test[TARGET], lr_b.predict_proba(X_test_b)[:, 1])
    acc_d_aware = accuracy_score(test[TARGET], pred_d_aware)

    y_test = test[TARGET]
    race = test.race

    mask = race.isin(["African-American", "Caucasian"]).values
    dpd_b = demographic_parity_difference(y_test[mask], pred_b[mask],
                                          sensitive_features=race[mask])
    eod_b = equalized_odds_difference(y_test[mask], pred_b[mask],
                                      sensitive_features=race[mask])
    dpd_d = demographic_parity_difference(y_test[mask], pred_d[mask],
                                          sensitive_features=race[mask])
    eod_d = equalized_odds_difference(y_test[mask], pred_d[mask],
                                      sensitive_features=race[mask])
    dpd_d_aware = demographic_parity_difference(y_test[mask], pred_d_aware[mask],
                                                sensitive_features=race[mask])
    eod_d_aware = equalized_odds_difference(y_test[mask], pred_d_aware[mask],
                                            sensitive_features=race[mask])

    mf_d = MetricFrame(
        metrics={"accuracy": accuracy_score, "selection rate": selection_rate,
                 "false positive rate": false_positive_rate,
                 "false negative rate": false_negative_rate},
        y_true=y_test, y_pred=pred_d, sensitive_features=race,
    )
    by_d = mf_d.by_group.loc[["African-American", "Caucasian", "Hispanic"]]

    # --- comparison figure ---------------------------------------------------
    groups = ["African-American", "Caucasian"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), sharey=True)
    for ax, (metric_name, idx) in zip(axes, [("False positive rate", 0),
                                             ("False negative rate", 1)]):
        x = np.arange(2)  # biased, de-biased
        width = 0.32
        for i, g in enumerate(groups):
            vals = [rates(y_test, pred_b, race, g)[idx],
                    rates(y_test, pred_d, race, g)[idx]]
            bars = ax.bar(x + (i - 0.5) * width, vals, width=width - 0.04,
                          color=SERIES[i], label=g if idx == 0 else None)
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                        f"{b.get_height():.0%}", ha="center", va="bottom",
                        fontsize=8.5, color=INK_SECONDARY)
        ax.set_xticks(x)
        ax.set_xticklabels(["LR on\noriginal data", "LR on\nde-biased data"])
        ax.set_title(metric_name)
        ax.set_ylim(0, 0.85)
    fig.legend(frameon=False, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Error-rate gaps before and after de-biasing", y=1.14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_fairness_comparison.png", bbox_inches="tight")
    plt.close(fig)

    fpr_b_aa, fnr_b_aa = rates(y_test, pred_b, race, "African-American")
    fpr_b_c, fnr_b_c = rates(y_test, pred_b, race, "Caucasian")
    fpr_d_aa, fnr_d_aa = rates(y_test, pred_d, race, "African-American")
    fpr_d_c, fnr_d_c = rates(y_test, pred_d, race, "Caucasian")

    report = f"""# Logistic Regression on the de-biased data

Identical architecture to the reference model (StandardScaler + Logistic
Regression), trained on the de-biased features from script 05. Race enters the
pipeline only as an **audit attribute**.

## Deployment decision: race-blind inference

The CorrelationRemover needs the sensitive attribute to transform a row, which
poses a choice for prediction time:

| Deployment | Accuracy | Demographic parity diff. | Equalized odds diff. | Individual race-invariance |
|------------|---------:|-------------------------:|---------------------:|:--:|
| **Race-blind** (raw features at inference, chosen) | {acc_d:.1%} | {dpd_d:.3f} | {eod_d:.3f} | yes - exact |
| Race-aware (transform with the person's race) | {acc_d_aware:.1%} | {dpd_d_aware:.3f} | {eod_d_aware:.3f} | no |

The race-aware variant achieves markedly better *group* fairness - with a
linear model on top of the linear CorrelationRemover the demographic-parity
difference falls to **{dpd_d_aware:.3f}** (near-parity), the pairing script 03
anticipated. But a person's stated race then moves their individual score: it
breaks counterfactual fairness, requires collecting the protected attribute at
decision time, and amounts to explicit differential treatment. We therefore
deploy **race-blind**: the de-biasing is a training-time intervention (the
model's *coefficients* were learned from race-neutralized data), and at
prediction time the model never sees race, so flipping race provably cannot
change any suggestion. All numbers below use race-blind inference.

## Performance

| Model | Accuracy | ROC-AUC |
|-------|---------:|--------:|
| LR, original data | {acc_b:.1%} | {auc_b:.3f} |
| LR, de-biased data | **{acc_d:.1%}** | **{auc_d:.3f}** |

De-biasing costs {acc_b - acc_d:+.1%} accuracy - essentially within noise. The
"fairness tax" on predictive performance is negligible here, consistent with
the finding that most of the usable signal (priors, age) is retained after the
transformation.

## Fairness comparison (African-American vs Caucasian, test set)

![Comparison](../figures/06_fairness_comparison.png)

| Metric | LR original | LR de-biased |
|--------|------------:|-------------:|
| FPR African-American | {fpr_b_aa:.1%} | {fpr_d_aa:.1%} |
| FPR Caucasian | {fpr_b_c:.1%} | {fpr_d_c:.1%} |
| **FPR gap** | **{fpr_b_aa - fpr_b_c:.1%}** | **{fpr_d_aa - fpr_d_c:.1%}** |
| FNR African-American | {fnr_b_aa:.1%} | {fnr_d_aa:.1%} |
| FNR Caucasian | {fnr_b_c:.1%} | {fnr_d_c:.1%} |
| **FNR gap** | **{fnr_b_c - fnr_b_aa:.1%}** | **{fnr_d_c - fnr_d_aa:.1%}** |
| Demographic parity difference | {dpd_b:.3f} | **{dpd_d:.3f}** |
| Equalized odds difference | {eod_b:.3f} | **{eod_d:.3f}** |

Full per-group metrics of the de-biased model:

| Metric | African-American | Caucasian | Hispanic |
|--------|----------------:|----------:|---------:|
| Accuracy | {by_d.loc['African-American', 'accuracy']:.1%} | {by_d.loc['Caucasian', 'accuracy']:.1%} | {by_d.loc['Hispanic', 'accuracy']:.1%} |
| Selection rate | {by_d.loc['African-American', 'selection rate']:.1%} | {by_d.loc['Caucasian', 'selection rate']:.1%} | {by_d.loc['Hispanic', 'selection rate']:.1%} |
| False positive rate | {by_d.loc['African-American', 'false positive rate']:.1%} | {by_d.loc['Caucasian', 'false positive rate']:.1%} | {by_d.loc['Hispanic', 'false positive rate']:.1%} |
| False negative rate | {by_d.loc['African-American', 'false negative rate']:.1%} | {by_d.loc['Caucasian', 'false negative rate']:.1%} | {by_d.loc['Hispanic', 'false negative rate']:.1%} |

## Reading the result honestly

Under race-blind inference the error-rate gaps shrink but do **not** vanish.
Two forces bound how far a feature-side intervention can go: (1) the different
*base rates* of the re-arrest label - the part of the disparity that lives in
the outcome variable itself (Chouldechova 2017); and (2) the deliberate choice
of race-blind inference, which trades the near-parity of the race-aware variant
(DPD {dpd_d_aware:.3f}) for an exact individual-level guarantee. Closing the
residual gap entirely would require either post-processing per-group thresholds
(explicit differential treatment) or better labels (measuring reoffending
rather than re-arrest).

This residual gap is a further argument for the project's central claim: the
system must remain a **suggestion** presented to accountable humans, together
with its known error profile - not an automated decision.
"""
    (REPORTS_DIR / "06_debiased_model.md").write_text(report)
    print(f"debiased lr acc {acc_d:.3f} auc {auc_d:.3f} | biased lr acc {acc_b:.3f} auc {auc_b:.3f}")
    print(f"DPD {dpd_b:.3f}->{dpd_d:.3f}  EOD {eod_b:.3f}->{eod_d:.3f}  "
          f"(race-aware DPD {dpd_d_aware:.3f})")
    print(f"FPR gap {fpr_b_aa - fpr_b_c:.3f}->{fpr_d_aa - fpr_d_c:.3f}  "
          f"FNR gap {fnr_b_c - fnr_b_aa:.3f}->{fnr_d_c - fnr_d_aa:.3f}")
    print("Wrote model, figure, reports/06_debiased_model.md")


if __name__ == "__main__":
    main()
