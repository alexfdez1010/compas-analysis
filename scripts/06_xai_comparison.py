"""Explainability comparison of the biased and de-biased SVMs.

Three lenses:
  1. SHAP (permutation explainer) - which features drive each model's risk
     estimates, before vs after de-biasing;
  2. counterfactual race-flip - change ONLY race for every African-American /
     Caucasian test defendant and measure how the suggested risk moves;
  3. DiCE - actionable counterfactuals for one high-risk individual under the
     de-biased model ("what would need to change for a low-risk suggestion?").

Outputs: figures/06_*.png, reports/06_xai.md
"""

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import common
from common import FIGURES_DIR, MODELS_DIR, REPORTS_DIR, SERIES, TARGET

warnings.filterwarnings("ignore")

FEATURES_D = [
    "age", "priors_count", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "charge_felony", "sex_male",
]
RNG = np.random.RandomState(42)
N_EXPLAIN = 150
N_BACKGROUND = 30
MAX_EVALS = 300  # permutation budget per row; >= 2*n_features+2


def shap_values_for(model, X_bg: pd.DataFrame, X_expl: pd.DataFrame):
    f = lambda X: model.predict_proba(X)[:, 1]  # noqa: E731
    masker = shap.maskers.Independent(X_bg.values, max_samples=N_BACKGROUND)
    explainer = shap.PermutationExplainer(f, masker, feature_names=list(X_expl.columns))
    return explainer(X_expl.values, max_evals=MAX_EVALS)


def save_beeswarm(sv, title: str, path: str) -> None:
    plt.figure()
    shap.plots.beeswarm(sv, show=False, max_display=11)
    fig = plt.gcf()
    fig.set_size_inches(8, 4.2)
    ax = plt.gca()
    ax.set_title(title, fontsize=11)
    ax.set_facecolor(common.SURFACE)
    fig.patch.set_facecolor(common.SURFACE)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> None:
    np.random.seed(42)  # shap's permutation sampling uses the global RNG
    common.apply_plot_style()
    train, test = common.train_test_frames()
    svm_b = joblib.load(MODELS_DIR / "svm_biased.joblib")
    svm_d = joblib.load(MODELS_DIR / "svm_debiased.joblib")

    X_train_b = common.build_features(train)
    X_test_b = common.build_features(test)
    X_train_d = common.build_features(train, include_race=False)[FEATURES_D]
    X_test_d = common.build_features(test, include_race=False)[FEATURES_D]

    idx_bg = RNG.choice(len(X_train_b), N_BACKGROUND, replace=False)
    idx_ex = RNG.choice(len(X_test_b), N_EXPLAIN, replace=False)

    # --- SHAP ----------------------------------------------------------------
    print("computing SHAP values (biased model)...")
    sv_b = shap_values_for(svm_b, X_train_b.iloc[idx_bg], X_test_b.iloc[idx_ex])
    print("computing SHAP values (de-biased model)...")
    sv_d = shap_values_for(svm_d, X_train_d.iloc[idx_bg], X_test_d.iloc[idx_ex])

    save_beeswarm(sv_b, "SHAP - SVM trained on original data (P(recidivism))",
                  "06_shap_biased.png")
    save_beeswarm(sv_d, "SHAP - SVM trained on de-biased data (P(recidivism))",
                  "06_shap_debiased.png")

    imp_b = pd.Series(np.abs(sv_b.values).mean(axis=0), index=X_test_b.columns)
    imp_d = pd.Series(np.abs(sv_d.values).mean(axis=0), index=FEATURES_D)
    race_share_b = imp_b[common.RACE_DUMMIES].sum() / imp_b.sum()

    # mean-|SHAP| comparison chart
    order = imp_b.sort_values().index
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    y = np.arange(len(order))
    ax.barh(y + 0.19, imp_b[order].values, height=0.34, color=SERIES[0],
            label="SVM, original data")
    ax.barh(y - 0.19, [imp_d.get(f, 0.0) for f in order], height=0.34,
            color=SERIES[1], label="SVM, de-biased data")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("mean |SHAP value| (impact on P(recidivism))")
    ax.set_title("Global feature importance before vs after de-biasing")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_shap_importance_comparison.png", bbox_inches="tight")
    plt.close(fig)

    # --- counterfactual race-flip -------------------------------------------
    def flip(df_feat: pd.DataFrame, source: str, target: str) -> pd.DataFrame:
        flipped = df_feat.copy()
        flipped[f"race_{source}"] = 0
        flipped[f"race_{target}"] = 1
        return flipped

    results = {}
    for src, tgt, label in [("african_american", "caucasian", "African-American -> Caucasian"),
                            ("caucasian", "african_american", "Caucasian -> African-American")]:
        rows = X_test_b[X_test_b[f"race_{src}"] == 1]
        p0 = svm_b.predict_proba(rows)[:, 1]
        p1 = svm_b.predict_proba(flip(rows, src, tgt))[:, 1]
        # de-biased model is race-blind at inference: identical rows in, so the
        # counterfactual difference is exactly zero for every individual.
        results[label] = {
            "n": len(rows),
            "mean_dp": float(np.mean(p1 - p0)),
            "flipped": float((np.sign(p0 - 0.5) != np.sign(p1 - 0.5)).mean()),
            "dp": p1 - p0,
        }

    aa2c = results["African-American -> Caucasian"]
    c2aa = results["Caucasian -> African-American"]

    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.hist(aa2c["dp"], bins=40, color=SERIES[0])
    ax.axvline(0, color=common.INK, linewidth=0.8)
    ax.set_xlabel("change in P(recidivism) when race is flipped to Caucasian")
    ax.set_ylabel("defendants")
    ax.set_title("Biased SVM - effect of flipping race for African-American defendants\n"
                 "(de-biased SVM: exactly 0 for every defendant, by construction)")
    ax.text(0.02, 0.92,
            f"mean shift {aa2c['mean_dp']:+.3f}\n"
            f"{aa2c['flipped']:.1%} of suggestions change side",
            transform=ax.transAxes, fontsize=9, va="top",
            color=common.INK_SECONDARY)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_counterfactual_race_flip.png", bbox_inches="tight")
    plt.close(fig)

    # --- DiCE actionable counterfactual (de-biased model) --------------------
    dice_section = ""
    try:
        import dice_ml

        # DiCE requires float dtypes for its sampling to work with sklearn
        train_dice = X_train_d.astype(float)
        train_dice[TARGET] = train[TARGET].values
        d = dice_ml.Data(dataframe=train_dice,
                         continuous_features=list(FEATURES_D),
                         outcome_name=TARGET)
        m = dice_ml.Model(model=svm_d, backend="sklearn")
        exp = dice_ml.Dice(d, m, method="random")

        X_test_f = X_test_d.astype(float)
        high_risk = X_test_f[svm_d.predict_proba(X_test_f)[:, 1] > 0.7]
        query = high_risk.iloc[[0]]
        cf = exp.generate_counterfactuals(query, total_CFs=3, desired_class=0,
                                          random_seed=42)
        cf_df = cf.cf_examples_list[0].final_cfs_df
        q = query.iloc[0]
        dice_section = f"""## DiCE - "what would have to change?"

For one defendant the de-biased model rates high-risk
(age {q.age:.0f}, {q.priors_count:.0f} priors, felony charge:
{'yes' if q.charge_felony else 'no'}), DiCE searches for minimal feature
changes that would flip the suggestion to low-risk (binary columns are
treated as continuous by the sampler, so values like 0.1 read as
"switched off"):

```text
{cf_df.to_string(index=False)}
```

The counterfactuals identify small changes - dropping the felony charge
degree or the juvenile record - that would move this person across the
decision boundary. Counterfactuals like these are what a human
decision-maker should see next to every score: they expose *why* the
suggestion is what it is and how close the person is to the boundary
(see the Streamlit demo).
"""
    except Exception as e:  # noqa: BLE001 - DiCE is an optional enrichment
        dice_section = f"""## DiCE

DiCE counterfactual generation was skipped in this run ({type(e).__name__}:
{e}). The race-flip analysis above already provides the project's central
counterfactual result.
"""
        print(f"DiCE skipped: {e}")

    report = f"""# Explainability comparison: biased vs de-biased model

SHAP values computed with the permutation explainer on {N_EXPLAIN} sampled test
defendants ({N_BACKGROUND} background samples, seed 42), explaining each
model's predicted probability of recidivism.

## What drives each model

![SHAP biased](../figures/06_shap_biased.png)

![SHAP de-biased](../figures/06_shap_debiased.png)

![Importance comparison](../figures/06_shap_importance_comparison.png)

Both models rely primarily on `priors_count` and `age`. In the model trained
on original data the race dummies contribute
**{race_share_b:.1%} of the total attribution mass** - direct evidence that
the model uses race itself, on top of whatever flows through proxies. In the
de-biased model this contribution is structurally zero (race is not an input),
and the remaining features were decorrelated from race, so their attributions
no longer secretly encode it (proxy AUC ~0.51, report 04).

Note how the de-biased model's attributions are not merely the biased model's
minus race: the importance of `priors_count` changes as well, because the
CorrelationRemover shifted each defendant's priors relative to their group
mean. The model still uses criminal history - it just can no longer use the
*racial component* of criminal history.

## Counterfactual race-flip

Changing **only** the race field and re-scoring every test defendant:

| Counterfactual | n | Biased SVM: mean shift in P(recid) | Biased SVM: suggestions that change side | De-biased SVM |
|---------------|---:|----------------------------------:|-----------------------------------------:|--------------:|
| African-American -> Caucasian | {aa2c['n']} | {aa2c['mean_dp']:+.3f} | {aa2c['flipped']:.1%} | 0 (exact) |
| Caucasian -> African-American | {c2aa['n']} | {c2aa['mean_dp']:+.3f} | {c2aa['flipped']:.1%} | 0 (exact) |

![Race flip](../figures/06_counterfactual_race_flip.png)

For the biased model, relabelling an African-American defendant as Caucasian
lowers the estimated recidivism probability for most individuals and flips
roughly **one suggestion in eight** - people would receive a different risk
label for no reason other than race. The de-biased model is race-blind at
inference, so the same experiment cannot change any suggestion - not as an
empirical observation but **by construction**, which is the stronger guarantee
(ALTAI Requirement #5).

{dice_section}
"""
    (REPORTS_DIR / "06_xai.md").write_text(report)
    print(f"race share of biased model attribution: {race_share_b:.3f}")
    print(f"AA->C: mean dp {aa2c['mean_dp']:+.4f}, flipped {aa2c['flipped']:.3%}")
    print("Wrote reports/06_xai.md and 4 figures")


if __name__ == "__main__":
    main()
