"""Baseline models on the original (biased) COMPAS data.

Trains
  1. a shallow decision tree - chosen for transparency: the whole decision
     logic can be printed and inspected, satisfying the interpretability goal
     of the initial assessment;
  2. a Logistic Regression - the reference "biased model" used in the rest of
     the project, selected empirically in script 03 (tied-best accuracy, most
     interpretable, well-calibrated).

The split is created and persisted by script 03; this script reuses it. Both
models are audited with Fairlearn (accuracy, selection rate, FPR, FNR per
racial group; demographic-parity and equalized-odds differences). Outputs:
  models/tree_biased.joblib, models/lr_biased.joblib
  figures/04_*.png, reports/04_baseline.md
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
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

import common
from common import (
    FIGURES_DIR,
    INK_SECONDARY,
    MODELS_DIR,
    REPORTS_DIR,
    SERIES,
    TARGET,
)

AUDIT_GROUPS = ["African-American", "Caucasian", "Hispanic"]


def audit(y_true, y_pred, race) -> MetricFrame:
    return MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection rate": selection_rate,
            "false positive rate": false_positive_rate,
            "false negative rate": false_negative_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=race,
    )


def fig_fairness(mf: MetricFrame, title: str, path: str) -> None:
    frame = mf.by_group.loc[AUDIT_GROUPS]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    x = np.arange(len(frame.columns))
    width = 0.26
    for i, group in enumerate(frame.index):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, frame.loc[group].values, width=width - 0.03,
                      color=SERIES[i], label=group)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{b.get_height():.0%}", ha="center", va="bottom",
                    fontsize=7.5, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(frame.columns)
    ax.set_ylim(0, min(1.0, frame.values.max() * 1.3))
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    common.apply_plot_style()
    for d in (MODELS_DIR, FIGURES_DIR, REPORTS_DIR):
        d.mkdir(exist_ok=True)

    train, test = common.train_test_frames()  # persisted by script 03

    X_train = common.build_features(train)
    X_test = common.build_features(test)
    y_train, y_test = train[TARGET], test[TARGET]

    # --- interpretable decision tree ---------------------------------------
    tree = DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)
    tree.fit(X_train, y_train)
    tree_pred = tree.predict(X_test)
    tree_acc = accuracy_score(y_test, tree_pred)
    tree_auc = roc_auc_score(y_test, tree.predict_proba(X_test)[:, 1])
    tree_rules = export_text(tree, feature_names=list(X_train.columns), decimals=1)

    fig, ax = plt.subplots(figsize=(26, 9))
    plot_tree(tree, feature_names=list(X_train.columns), filled=False,
              class_names=["no recid", "recid"], fontsize=8, ax=ax,
              impurity=False, proportion=True, rounded=True)
    ax.set_title("Decision tree on the original data (depth 4)", fontsize=14)
    fig.savefig(FIGURES_DIR / "04_decision_tree.png", bbox_inches="tight")
    plt.close(fig)

    importances = pd.Series(tree.feature_importances_, index=X_train.columns)
    importances = importances[importances > 0].sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.barh(importances.index, importances.values, color=SERIES[0], height=0.6)
    ax.set_title("Decision-tree feature importance (Gini)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "04_tree_importance.png", bbox_inches="tight")
    plt.close(fig)

    # --- reference Logistic Regression ("biased model") --------------------
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=1000, random_state=42))
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    lr_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

    # standardized coefficients (comparable magnitudes; the model is linear and
    # therefore directly readable - the point of choosing it over the SVM)
    coefs = pd.Series(lr.named_steps["logisticregression"].coef_[0],
                      index=X_train.columns).sort_values(key=np.abs, ascending=False)

    joblib.dump(tree, MODELS_DIR / "tree_biased.joblib")
    joblib.dump(lr, MODELS_DIR / "lr_biased.joblib")

    # --- fairness audit -----------------------------------------------------
    mf_lr = audit(y_test, lr_pred, test.race)
    mf_tree = audit(y_test, tree_pred, test.race)
    fig_fairness(mf_lr, "Logistic Regression on original data - fairness by race",
                 "04_fairness_lr.png")

    mask = test.race.isin(["African-American", "Caucasian"]).values
    dpd = demographic_parity_difference(
        y_test[mask], lr_pred[mask], sensitive_features=test.race[mask])
    eod = equalized_odds_difference(
        y_test[mask], lr_pred[mask], sensitive_features=test.race[mask])

    by = mf_lr.by_group.loc[AUDIT_GROUPS]
    by_tree = mf_tree.by_group.loc[AUDIT_GROUPS]

    coef_table = "\n".join(
        f"| `{name}` | {val:+.3f} |" for name, val in coefs.items())

    report = f"""# Baseline models on the original data

Split: 70/30 train/test, stratified jointly on outcome and race
({len(train):,} train / {len(test):,} test), created and persisted by script 03
so every later script scores the identical test set. Features: age, priors
count, juvenile counts (felony/misdemeanor/other), charge degree, sex, **and
race** (one-hot).

**On including race.** Deliberately keeping race in this baseline is itself an
ethical decision that requires justification: the point of the reference model
is to *expose* how much predictive weight the data assigns to race, so that the
de-biasing step has a measurable target. A deployed system should not use race
as an input - but silently dropping it does not produce fairness either
("fairness through unawareness"), because priors count, charge degree and age
act as proxies. This is exactly what the de-biasing step (script 05) addresses.

**On the model.** The reference classifier is a Logistic Regression, selected
in script 03: on this data every model family is tied within cross-validation
noise, so the project picks the estimator that is simultaneously tied-for-best
on accuracy and directly interpretable. The depth-4 decision tree below is an
additional, even more transparent sanity check.

## Interpretable decision tree

Accuracy **{tree_acc:.1%}**, ROC-AUC **{tree_auc:.3f}**.

![Decision tree](../figures/04_decision_tree.png)

![Feature importance](../figures/04_tree_importance.png)

The learned rules are dominated by `priors_count` and `age`:

```text
{tree_rules}```

Two observations relevant to the ethics assessment:

1. The tree achieves essentially the same accuracy as COMPAS itself (~65%),
   echoing Dressel & Farid (2018): a transparent model with a handful of
   features matches the proprietary 137-question instrument. There is no
   accuracy argument for opacity.
2. Race dummies barely appear in the split rules, yet the fairness audit below
   still shows large error-rate gaps - the bias travels through `priors_count`
   and `age`, which are products of unequal policing intensity (see RQ2).

## Reference Logistic Regression ("biased model")

Accuracy **{lr_acc:.1%}**, ROC-AUC **{lr_auc:.3f}**. Because the model is
linear its logic is fully readable - the standardized coefficients (log-odds
impact per one-SD change in each feature) are:

| Feature | Std. coefficient |
|---------|-----------------:|
{coef_table}

![Fairness metrics](../figures/04_fairness_lr.png)

| Metric | African-American | Caucasian | Hispanic |
|--------|----------------:|----------:|---------:|
| Accuracy | {by.loc['African-American', 'accuracy']:.1%} | {by.loc['Caucasian', 'accuracy']:.1%} | {by.loc['Hispanic', 'accuracy']:.1%} |
| Selection rate | {by.loc['African-American', 'selection rate']:.1%} | {by.loc['Caucasian', 'selection rate']:.1%} | {by.loc['Hispanic', 'selection rate']:.1%} |
| False positive rate | {by.loc['African-American', 'false positive rate']:.1%} | {by.loc['Caucasian', 'false positive rate']:.1%} | {by.loc['Hispanic', 'false positive rate']:.1%} |
| False negative rate | {by.loc['African-American', 'false negative rate']:.1%} | {by.loc['Caucasian', 'false negative rate']:.1%} | {by.loc['Hispanic', 'false negative rate']:.1%} |

Decision tree for comparison (same test set): FPR
{by_tree.loc['African-American', 'false positive rate']:.1%} vs
{by_tree.loc['Caucasian', 'false positive rate']:.1%}, FNR
{by_tree.loc['African-American', 'false negative rate']:.1%} vs
{by_tree.loc['Caucasian', 'false negative rate']:.1%}
(African-American vs Caucasian).

Aggregate disparity of the Logistic Regression restricted to African-American
vs Caucasian:

- **Demographic parity difference: {dpd:.3f}** (gap in the share of people
  flagged as likely recidivists)
- **Equalized odds difference: {eod:.3f}** (largest gap in FPR or TPR)

The model reproduces the asymmetry found in the COMPAS scores themselves
(report 02): African-American defendants face a much higher false positive
rate, Caucasian defendants a much higher false negative rate. Training a fresh
model on the raw data *reproduces* the injustice pattern of the data-generating
system - the reference point the de-biasing step must improve on.

## ALTAI Requirement #2 - accuracy in context

An accuracy of ~{lr_acc:.0%} means roughly one in three suggestions is wrong.
For a system that could influence detention decisions this error rate is only
acceptable - if at all - in a decision-support setting with a human weighing
independent evidence (see reports/08_reflection.md).
"""
    (REPORTS_DIR / "04_baseline.md").write_text(report)
    print(f"tree acc {tree_acc:.3f} auc {tree_auc:.3f} | lr acc {lr_acc:.3f} auc {lr_auc:.3f}")
    print(f"LR AA-vs-C: DPD {dpd:.3f}, EOD {eod:.3f}")
    print("Wrote reports/04_baseline.md, 3 figures, 2 models")


if __name__ == "__main__":
    main()
