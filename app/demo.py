"""Streamlit demo: COMPAS-style risk SUGGESTION, biased vs de-biased SVM.

Run with:  uv run streamlit run app/demo.py

The app deliberately frames the model output as a *suggestion* to a human
decision-maker (ALTAI Requirement #1): it always shows why, how uncertain the
models are per group, what happens under a race counterfactual, and requires
an explicit human decision with a recorded rationale.
"""

from pathlib import Path

import joblib
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

# per-group error rates of each model on the held-out test set (report 05)
ERROR_TABLE = pd.DataFrame(
    {
        "Model": ["SVM, original data"] * 2 + ["SVM, de-biased data"] * 2,
        "Group": ["African-American", "Caucasian"] * 2,
        "False positive rate": ["30%", "8%", "25%", "9%"],
        "False negative rate": ["38%", "74%", "46%", "70%"],
    }
)


@st.cache_resource
def load_models():
    return (
        joblib.load(MODELS / "svm_biased.joblib"),
        joblib.load(MODELS / "svm_debiased.joblib"),
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


def main() -> None:
    st.set_page_config(page_title="COMPAS bias demo", page_icon="⚖️", layout="wide")
    svm_b, svm_d = load_models()

    st.title("Recidivism risk: AI-based *suggestion*, not AI-based decision")
    st.markdown(
        "Compare an SVM trained on the **original COMPAS data** (race included) "
        "with the same SVM trained on **de-biased data** (race removed and proxy "
        "correlations stripped, race-blind at prediction time). "
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

    p_b = float(svm_b.predict_proba(Xb)[0, 1])
    p_d = float(svm_d.predict_proba(Xd)[0, 1])

    col1, col2 = st.columns(2)
    for col, name, p, note in (
        (col1, "SVM trained on original data", p_b,
         "uses race directly + racial proxies"),
        (col2, "SVM trained on de-biased data", p_d,
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

    st.divider()

    # --- counterfactual: what if only race were different? -------------------
    st.subheader("Counterfactual check: change *only* race")
    rows = []
    for r in RACES:
        Xb_r = biased_features(age, priors, juv_fel, juv_misd, juv_other,
                               felony, sex == "Male", r)
        rows.append({
            "Race": r + (" (selected)" if r == race else ""),
            "Original-data model": f"{float(svm_b.predict_proba(Xb_r)[0, 1]):.1%}",
            "De-biased model": f"{float(svm_d.predict_proba(Xb_r[FEATURES_D])[0, 1]):.1%}",
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
        shap_fig = FIGURES / "06_shap_importance_comparison.png"
        if shap_fig.exists():
            st.image(str(shap_fig),
                     caption="SHAP global feature importance for both models "
                             "(script 06). Priors count and age dominate; the "
                             "de-biased model assigns race zero weight by design.")
        else:
            st.info("Run scripts/06_xai_comparison.py to generate the SHAP figure.")

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
