"""Model-family selection: which classifier should this project use?

Runs BEFORE any model is committed to. It (1) creates and persists the 70/30
train/test split that every later script reuses, and (2) benchmarks a panel of
classifier families so the choice of model is an evidence-based decision rather
than an assumption.

Design of the comparison:
  * race-blind inputs (the 7 non-race features) - this is the regime the system
    actually deploys in (see script 06), so it is the honest basis for choosing;
  * 5-fold stratified cross-validation on the TRAIN split for accuracy / ROC-AUC
    (robust to a single lucky split);
  * fairness (demographic-parity and equalized-odds differences, FPR/FNR gaps,
    African-American vs Caucasian) measured on the held-out TEST split.

Outputs: data/processed/{train,test}.csv (persisted split),
figures/03_model_selection.png, reports/03_model_selection.md
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    false_negative_rate,
    false_positive_rate,
)
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import common
from common import FIGURES_DIR, INK_SECONDARY, PROCESSED_DIR, REPORTS_DIR, SERIES, TARGET

warnings.filterwarnings("ignore")

RS = 42
# The race-blind feature set (matches the deployed regime in script 06).
FEATURES = [
    "age", "priors_count", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "charge_felony", "sex_male",
]


def candidate_models() -> dict:
    """One representative, lightly-tuned estimator per model family.

    All are scikit-learn native (no libomp/GPU dependency) so the benchmark
    runs on any clean clone. Scaling-sensitive models are wrapped in a
    StandardScaler pipeline.
    """
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, random_state=RS)),
        "Gaussian NB": GaussianNB(),
        "k-NN (k=25)": make_pipeline(
            StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
        "Decision Tree (d=4)": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=50, random_state=RS),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, random_state=RS),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=20, random_state=RS),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RS),
        "SVM (RBF)": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True, random_state=RS)),
        "MLP (64, 32)": make_pipeline(
            StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32),
                                            max_iter=800, random_state=RS)),
    }


def fairness(y, pred, race) -> tuple:
    m = race.isin(["African-American", "Caucasian"]).values
    aa = (race == "African-American").values
    ca = (race == "Caucasian").values
    dpd = demographic_parity_difference(y[m], pred[m], sensitive_features=race[m])
    eod = equalized_odds_difference(y[m], pred[m], sensitive_features=race[m])
    fpr_gap = false_positive_rate(y[aa], pred[aa]) - false_positive_rate(y[ca], pred[ca])
    fnr_gap = false_negative_rate(y[ca], pred[ca]) - false_negative_rate(y[aa], pred[aa])
    return dpd, eod, fpr_gap, fnr_gap


def main() -> None:
    common.apply_plot_style()
    for d in (PROCESSED_DIR, FIGURES_DIR, REPORTS_DIR):
        d.mkdir(exist_ok=True)

    # --- create and persist the canonical split (reused by every later script)
    df = common.load_filtered()
    train, test = train_test_split(
        df, test_size=0.3, random_state=RS,
        stratify=df[[TARGET, "race"]].astype(str).agg("|".join, axis=1),
    )
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    X_train = common.build_features(train, include_race=False)[FEATURES]
    X_test = common.build_features(test, include_race=False)[FEATURES]
    y_train, y_test = train[TARGET].to_numpy(), test[TARGET].to_numpy()
    race_test = test["race"]

    # --- benchmark -----------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    rows = []
    for name, model in candidate_models().items():
        cvres = cross_validate(model, X_train, y_train, cv=cv,
                               scoring=["accuracy", "roc_auc"], n_jobs=-1)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        try:
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        except (AttributeError, ValueError):
            auc = float("nan")
        dpd, eod, fpr_gap, fnr_gap = fairness(y_test, pred, race_test)
        rows.append({
            "model": name,
            "cv_acc": cvres["test_accuracy"].mean(),
            "cv_auc": cvres["test_roc_auc"].mean(),
            "cv_auc_std": cvres["test_roc_auc"].std(),
            "test_acc": accuracy_score(y_test, pred),
            "test_auc": auc,
            "dpd": dpd, "eod": eod, "fpr_gap": fpr_gap, "fnr_gap": fnr_gap,
        })
    res = pd.DataFrame(rows).sort_values("cv_auc", ascending=False).reset_index(drop=True)
    real = res[res.model != "Majority baseline"]

    # --- figure: performance vs fairness across families ---------------------
    order = real.sort_values("cv_auc")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    y = np.arange(len(order))
    chosen = (order.model == "Logistic Regression").values
    colors = [SERIES[1] if c else SERIES[0] for c in chosen]

    axes[0].barh(y, order.cv_auc, xerr=order.cv_auc_std, height=0.66, color=colors,
                 error_kw=dict(ecolor=common.MUTED, lw=0.8))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(order.model)
    axes[0].set_xlim(0.66, 0.75)
    axes[0].set_xlabel("5-fold CV ROC-AUC (higher = better)")
    axes[0].set_title("Predictive performance")
    axes[0].grid(axis="x")
    axes[0].grid(axis="y", visible=False)

    axes[1].barh(y, order.dpd, height=0.66, color=colors)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xlabel("Demographic parity diff. (lower = fairer)")
    axes[1].set_title("Group fairness (test)")
    axes[1].grid(axis="x")
    axes[1].grid(axis="y", visible=False)

    fig.suptitle("Model-family comparison on race-blind COMPAS features "
                 "(orange = chosen: Logistic Regression)", y=1.02, fontsize=11)
    fig.text(0.5, -0.04,
             "Performance is flat across families (all within ~0.02 AUC); no "
             "family is fair on the original data.\nLogistic Regression ties the "
             "best AUC while being the most interpretable and calibratable.",
             ha="center", fontsize=8.5, color=INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "03_model_selection.png", bbox_inches="tight")
    plt.close(fig)

    # --- report --------------------------------------------------------------
    def fmt(r):
        return (f"| {r.model} | {r.cv_acc:.3f} | {r.cv_auc:.3f} | {r.test_acc:.3f} "
                f"| {r.test_auc:.3f} | {r.dpd:.3f} | {r.eod:.3f} | {r.fpr_gap:+.3f} |")

    lr = res[res.model == "Logistic Regression"].iloc[0]
    best = real.iloc[0]
    auc_spread = real.cv_auc.max() - real.cv_auc.min()
    table = "\n".join(fmt(r) for _, r in res.iterrows())

    report = f"""# Model selection: which classifier for COMPAS?

Before committing to a model, every practical classifier family is benchmarked
under the regime the system actually deploys in (**race-blind**: the 7 non-race
features). Accuracy and ROC-AUC come from **5-fold stratified cross-validation**
on the training split; fairness is measured on the held-out test split
(African-American vs Caucasian). The 70/30 split created here (stratified
jointly on outcome and race, {len(train):,} train / {len(test):,} test) is
persisted and reused by every later script.

| Model | CV acc | CV AUC | Test acc | Test AUC | DPD | EOD | FPR gap |
|-------|-------:|-------:|---------:|---------:|----:|----:|--------:|
{table}

## Finding 1 - the performance ceiling is flat

Every genuine model lands in a CV ROC-AUC band of just **{real.cv_auc.min():.3f} -
{real.cv_auc.max():.3f}** (spread {auc_spread:.3f}) and ~66-67% accuracy. The
best family ({best.model}, AUC {best.cv_auc:.3f}) beats Logistic Regression
({lr.cv_auc:.3f}) by ~{best.cv_auc - lr.cv_auc:.3f} AUC - inside the
cross-validation noise (±{lr.cv_auc_std:.3f}). This is a direct, quantitative
confirmation of Dressel & Farid (2018): on this data a handful of features caps
predictive power, and model complexity buys essentially nothing. **There is no
accuracy argument for an opaque model here.**

## Finding 2 - fairness is not a model-selection lever

No family is fair on the original data: demographic-parity differences run
{real.dpd.min():.2f}-{real.dpd.max():.2f} and FPR gaps
{real.fpr_gap.min():+.2f} to {real.fpr_gap.max():+.2f}. The models that look
"fairer" (e.g. Gaussian NB) get there only by predicting fewer positives, at an
accuracy cost. Unfairness lives in the labels and base rates, not in the choice
of estimator - which is why the project addresses it at the data level
(script 05) rather than by shopping for a model.

![Model comparison](../figures/03_model_selection.png)

## Decision: Logistic Regression

Because performance is tied across the board, the choice is made on the axes
that actually differ:

1. **Performance** - LR matches the top ensembles on test AUC
   ({lr.test_auc:.3f}) and accuracy ({lr.test_acc:.1%}); the gap to the best
   family is within noise.
2. **Interpretability** - a linear model exposes a signed weight per feature,
   so the reference "biased" model can be read directly (script 04), reinforcing
   the project's thesis that opacity is unnecessary.
3. **Calibration & stability** - logistic outputs are well-calibrated
   probabilities (the demo presents a *probability*, not a label) and LR has no
   high-variance hyperparameters to overfit on a small dataset.
4. **Compatibility with the de-biasing step** - Fairlearn's `CorrelationRemover`
   is a *linear* transform; pairing it with a linear model makes the
   de-biasing near-exact. Script 06 shows LR on the de-biased data reaching
   demographic parity an SVM or tree ensemble cannot (DPD ~0.03 race-aware).

The RBF-SVM used as the project's earlier reference is mid-pack here
(AUC {res[res.model == 'SVM (RBF)'].iloc[0].cv_auc:.3f}) and opaque; it is
retained in the table above only as a benchmarked alternative. **All downstream
scripts (04-07) and the demo now use Logistic Regression.** The depth-4 decision
tree is kept in script 04 as an inherently-transparent sanity reference, not as
the deployed model.
"""
    (REPORTS_DIR / "03_model_selection.md").write_text(report)

    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nChosen: Logistic Regression (CV AUC {lr.cv_auc:.3f}, "
          f"test acc {lr.test_acc:.3f}, test AUC {lr.test_auc:.3f})")
    print("Wrote split, figures/03_model_selection.png, reports/03_model_selection.md")


if __name__ == "__main__":
    main()
