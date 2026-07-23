"""Baseline models on the original (biased) COMPAS data.

Trains
  1. a shallow decision tree - chosen for transparency: the whole decision
     logic can be printed and inspected, satisfying the interpretability goal
     of the initial assessment;
  2. an RBF-kernel SVM - the reference "biased model" used in the rest of the
     project.

Both are audited with Fairlearn (accuracy, selection rate, FPR, FNR per racial
group; demographic-parity and equalized-odds differences). Outputs:
  models/tree_biased.joblib, models/svm_biased.joblib
  data/processed/train.csv, data/processed/test.csv (persisted split)
  figures/03_*.png, reports/03_baseline.md
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
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

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
    for d in (MODELS_DIR, PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR):
        d.mkdir(exist_ok=True)

    df = common.load_filtered()
    train, test = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df[[TARGET, "race"]].astype(str).agg("|".join, axis=1)
    )
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

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
    fig.savefig(FIGURES_DIR / "03_decision_tree.png", bbox_inches="tight")
    plt.close(fig)

    importances = pd.Series(tree.feature_importances_, index=X_train.columns)
    importances = importances[importances > 0].sort_values()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.barh(importances.index, importances.values, color=SERIES[0], height=0.6)
    ax.set_title("Decision-tree feature importance (Gini)")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_tree_importance.png", bbox_inches="tight")
    plt.close(fig)

    # --- reference SVM (the "biased model") --------------------------------
    svm = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True,
                                              random_state=42))
    svm.fit(X_train, y_train)
    svm_pred = svm.predict(X_test)
    svm_acc = accuracy_score(y_test, svm_pred)
    svm_auc = roc_auc_score(y_test, svm.predict_proba(X_test)[:, 1])

    joblib.dump(tree, MODELS_DIR / "tree_biased.joblib")
    joblib.dump(svm, MODELS_DIR / "svm_biased.joblib")

    # --- fairness audit -----------------------------------------------------
    mf_svm = audit(y_test, svm_pred, test.race)
    mf_tree = audit(y_test, tree_pred, test.race)
    fig_fairness(mf_svm, "SVM on original data - fairness metrics by race",
                 "03_fairness_svm.png")

    mask = test.race.isin(["African-American", "Caucasian"]).values
    dpd = demographic_parity_difference(
        y_test[mask], svm_pred[mask], sensitive_features=test.race[mask])
    eod = equalized_odds_difference(
        y_test[mask], svm_pred[mask], sensitive_features=test.race[mask])

    by = mf_svm.by_group.loc[AUDIT_GROUPS]
    by_tree = mf_tree.by_group.loc[AUDIT_GROUPS]

    report = f"""# Baseline models on the original data

Split: 70/30 train/test, stratified jointly on outcome and race
({len(train):,} train / {len(test):,} test). Features: age, priors count,
juvenile counts (felony/misdemeanor/other), charge degree, sex, **and race**
(one-hot).

**On including race.** Deliberately keeping race in this baseline is itself an
ethical decision that requires justification: the point of the reference model
is to *expose* how much predictive weight the data assigns to race, so that the
de-biasing step has a measurable target. A deployed system should not use race
as an input - but silently dropping it does not produce fairness either
("fairness through unawareness"), because priors count, charge degree and age
act as proxies. This is exactly what the de-biasing step (script 04) addresses.

## Interpretable decision tree

Accuracy **{tree_acc:.1%}**, ROC-AUC **{tree_auc:.3f}**.

![Decision tree](../figures/03_decision_tree.png)

![Feature importance](../figures/03_tree_importance.png)

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

## Reference SVM ("biased model")

RBF-kernel SVM. Accuracy **{svm_acc:.1%}**, ROC-AUC **{svm_auc:.3f}**.

![Fairness metrics](../figures/03_fairness_svm.png)

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

Aggregate disparity of the SVM restricted to African-American vs Caucasian:

- **Demographic parity difference: {dpd:.3f}** (gap in the share of people
  flagged as likely recidivists)
- **Equalized odds difference: {eod:.3f}** (largest gap in FPR or TPR)

The model reproduces the asymmetry found in the COMPAS scores themselves
(report 02): African-American defendants face a much higher false positive
rate, Caucasian defendants a much higher false negative rate. Training a fresh
model on the raw data *reproduces* the injustice pattern of the data-generating
system - the reference point the de-biasing step must improve on.

## ALTAI Requirement #2 - accuracy in context

An accuracy of ~{svm_acc:.0%} means roughly one in three suggestions is wrong.
For a system that could influence detention decisions this error rate is only
acceptable - if at all - in a decision-support setting with a human weighing
independent evidence (see reports/07_reflection.md).
"""
    (REPORTS_DIR / "03_baseline.md").write_text(report)
    print(f"tree acc {tree_acc:.3f} auc {tree_auc:.3f} | svm acc {svm_acc:.3f} auc {svm_auc:.3f}")
    print(f"SVM AA-vs-C: DPD {dpd:.3f}, EOD {eod:.3f}")
    print("Wrote reports/03_baseline.md, 3 figures, 2 models, persisted split")


if __name__ == "__main__":
    main()
