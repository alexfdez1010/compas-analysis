"""Streamlit demo: COMPAS-style risk SUGGESTION, biased vs de-biased model.

Run with:  uv run streamlit run app/demo.py

The app deliberately frames the model output as a *suggestion* to a human
decision-maker (ALTAI Requirement #1): it always shows why, how uncertain the
models are per group, what happens under a race counterfactual, and requires
an explicit human decision with a recorded rationale.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FIGURES = ROOT / "figures"

FEATURES_D = [
    "age", "priors_count", "juv_fel_count", "juv_misd_count",
    "juv_other_count", "charge_felony", "sex_male",
]
RACES = ["African-American", "Caucasian", "Hispanic", "Other"]

# human-readable labels for the per-feature explanation
FEATURE_LABELS = {
    "age": "Age at screening",
    "priors_count": "Prior offenses",
    "juv_fel_count": "Juvenile felonies",
    "juv_misd_count": "Juvenile misdemeanors",
    "juv_other_count": "Other juvenile offenses",
    "charge_felony": "Current charge is a felony",
    "sex_male": "Sex = male",
    "race_african_american": "Race = African-American",
    "race_caucasian": "Race = Caucasian",
    "race_hispanic": "Race = Hispanic",
    "race_other": "Race = Other",
}

# per-group error rates of each model on the held-out test set (report 06)
ERROR_TABLE = pd.DataFrame(
    {
        "Model": ["LR, original data"] * 2 + ["LR, de-biased data"] * 2,
        "Group": ["African-American", "Caucasian"] * 2,
        "False positive rate": ["32%", "10%", "26%", "10%"],
        "False negative rate": ["34%", "66%", "40%", "65%"],
    }
)


@st.cache_resource
def load_models():
    return (
        joblib.load(MODELS / "lr_biased.joblib"),
        joblib.load(MODELS / "lr_debiased.joblib"),
    )


def biased_features(age, priors, juv_fel, juv_misd, juv_other, felony, male, race):
    row = {
        "age": age, "priors_count": priors, "juv_fel_count": juv_fel,
        "juv_misd_count": juv_misd, "juv_other_count": juv_other,
        "charge_felony": int(felony), "sex_male": int(male),
        "race_african_american": int(race == "African-American"),
        "race_caucasian": int(race == "Caucasian"),
        "race_hispanic": int(race == "Hispanic"),
        "race_other": int(race == "Other"),
    }
    return pd.DataFrame([row])


def contributions(model, X_row: pd.DataFrame) -> pd.Series:
    """Per-feature contribution to the log-odds of re-arrest for one defendant.

    The deployed models are linear (StandardScaler + LogisticRegression), so the
    score decomposes exactly: log-odds = intercept + sum_j (coef_j * z_j), where
    z_j is the standardized value of feature j. Each term is that feature's
    signed push on this specific suggestion - the reason the model is auditable
    at the individual level, not just globally (report 07).
    """
    scaler = model.named_steps["standardscaler"]
    clf = model.named_steps["logisticregression"]
    z = (X_row.values[0] - scaler.mean_) / scaler.scale_
    return pd.Series(z * clf.coef_[0], index=X_row.columns)


def contribution_chart(contrib: pd.Series):
    order = contrib.reindex(contrib.abs().sort_values().index)
    labels = [FEATURE_LABELS.get(f, f) for f in order.index]
    colors = ["#c0504d" if v > 0 else "#4f81bd" for v in order.values]
    fig, ax = plt.subplots(figsize=(5.2, 0.42 * len(order) + 0.6))
    ax.barh(range(len(order)), order.values, color=colors)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_xlabel("contribution to log-odds of re-arrest", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    st.set_page_config(page_title="COMPAS bias demo", page_icon="⚖️", layout="wide")
    model_b, model_d = load_models()

    st.title("Recidivism risk: AI-based *suggestion*, not AI-based decision")
    st.markdown(
        "Compare a **logistic-regression** model trained on the **original "
        "COMPAS data** (race included) with the same model trained on "
        "**de-biased data** (race removed and proxy correlations stripped, "
        "race-blind at prediction time). "
        "Every output below is an **algorithmic suggestion with a known error "
        "profile** — the decision belongs to an accountable human."
    )

    with st.sidebar:
        st.header("Defendant profile")
        age = st.slider("Age at screening", 18, 80, 27)
        priors = st.slider("Prior offenses", 0, 30, 3)
        juv_fel = st.slider("Juvenile felonies", 0, 10, 0)
        juv_misd = st.slider("Juvenile misdemeanors", 0, 10, 0)
        juv_other = st.slider("Other juvenile offenses", 0, 10, 0)
        felony = st.selectbox("Current charge degree", ["Felony", "Misdemeanor"]) == "Felony"
        sex = st.radio("Sex", ["Male", "Female"], horizontal=True)
        race = st.selectbox("Race (as recorded by the system)", RACES)
        st.caption(
            "Race is collected here only to demonstrate the two models' "
            "behavior. The de-biased model never receives it."
        )

    Xb = biased_features(age, priors, juv_fel, juv_misd, juv_other,
                         felony, sex == "Male", race)
    Xd = Xb[FEATURES_D]

    p_b = float(model_b.predict_proba(Xb)[0, 1])
    p_d = float(model_d.predict_proba(Xd)[0, 1])

    col1, col2 = st.columns(2)
    for col, name, p, note in (
        (col1, "Model trained on original data", p_b,
         "uses race directly + racial proxies"),
        (col2, "Model trained on de-biased data", p_d,
         "race-blind by construction"),
    ):
        with col:
            st.subheader(name)
            st.caption(note)
            st.metric("Suggested P(re-arrest within 2 years)", f"{p:.1%}")
            if p >= 0.5:
                st.error("Suggestion: HIGHER risk")
            else:
                st.success("Suggestion: LOWER risk")

    # --- why: per-feature explanation for THIS defendant ---------------------
    st.divider()
    st.subheader("How was this suggestion computed? (per-feature reasons)")
    st.markdown(
        "Both models are **logistic regressions** — a linear model chosen "
        "precisely because its reasoning is fully auditable (see the "
        "model-selection benchmark, report 03). For this exact defendant the "
        "score decomposes into one signed contribution per feature: "
        "**red bars push the suggestion toward higher risk, blue bars toward "
        "lower risk**, and their sum (plus a constant) is the log-odds behind "
        "the probability above."
    )
    contrib_b = contributions(model_b, Xb)
    contrib_d = contributions(model_d, Xd)
    ecol1, ecol2 = st.columns(2)
    with ecol1:
        st.caption("Model trained on original data (uses race)")
        st.pyplot(contribution_chart(contrib_b))
    with ecol2:
        st.caption("Model trained on de-biased data (race-blind)")
        st.pyplot(contribution_chart(contrib_d))
    top_b = contrib_b.reindex(contrib_b.abs().sort_values(ascending=False).index)
    lead = top_b.index[0]
    race_push = contrib_b[[c for c in contrib_b.index if c.startswith("race_")]].abs().sum()
    st.markdown(
        f"For this profile the biggest driver of the original-data model is "
        f"**{FEATURE_LABELS.get(lead, lead)}** "
        f"({'+' if top_b.iloc[0] > 0 else ''}{top_b.iloc[0]:.2f} log-odds). "
        f"The race fields contribute **{race_push:.2f}** of log-odds movement "
        "in that model; in the de-biased model they contribute exactly 0, "
        "because race is not an input."
    )

    st.divider()

    # --- counterfactual: what if only race were different? -------------------
    st.subheader("Counterfactual check: change *only* race")
    rows = []
    for r in RACES:
        Xb_r = biased_features(age, priors, juv_fel, juv_misd, juv_other,
                               felony, sex == "Male", r)
        rows.append({
            "Race": r + (" (selected)" if r == race else ""),
            "Original-data model": f"{float(model_b.predict_proba(Xb_r)[0, 1]):.1%}",
            "De-biased model": f"{float(model_d.predict_proba(Xb_r[FEATURES_D])[0, 1]):.1%}",
        })
    st.table(pd.DataFrame(rows))
    st.markdown(
        "The original-data model returns a **different suggestion for the same "
        "person depending on recorded race**. The de-biased model cannot: race "
        "is not among its inputs."
    )

    # --- transparency: error profile and explanation -------------------------
    st.divider()
    st.subheader("Known error profile (held-out test set)")
    st.markdown(
        "Both models are wrong for roughly **1 person in 3**. A suggestion "
        "from either model must never be treated as a fact."
    )
    st.table(ERROR_TABLE)

    with st.expander("Why does the model suggest this? (global explanation)"):
        shap_fig = FIGURES / "07_shap_importance_comparison.png"
        if shap_fig.exists():
            st.image(str(shap_fig),
                     caption="SHAP global feature importance for both models "
                             "(script 07). Priors count and age dominate; the "
                             "de-biased model assigns race zero weight by design.")
        else:
            st.info("Run scripts/07_xai_comparison.py to generate the SHAP figure.")

    # --- human in the loop ----------------------------------------------------
    st.divider()
    st.subheader("Human decision (required)")
    st.markdown(
        "ALTAI Requirement #1 — *Human Agency and Oversight*: the operator "
        "must make and justify the decision; the score is one input among many."
    )
    decision = st.radio(
        "Your decision as the accountable human:",
        ["No determination yet", "Follow suggestion", "Override suggestion"],
    )
    rationale = st.text_area(
        "Rationale (required, recorded for audit)",
        placeholder="e.g. stable employment and family support outweigh the score; "
                    "the model cannot see either.",
    )
    if decision != "No determination yet":
        if rationale.strip():
            st.success(
                f"Recorded: **{decision}** with rationale. In a real deployment "
                "this record would go to an audit log reviewed for automation "
                "bias (systematic rubber-stamping of suggestions)."
            )
        else:
            st.warning("A rationale is required — a decision without one would "
                       "be automation, not oversight.")

    st.caption(
        "Demo for research/education only. Trained on Broward County FL "
        "2013–2014 re-arrest data, which records criminal-justice decisions, "
        "not criminal behavior. Not for operational use."
    )


if __name__ == "__main__":
    main()
