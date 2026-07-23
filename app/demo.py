"""Streamlit demo: COMPAS-style risk SUGGESTION, biased vs de-biased model.

Run with:  uv run streamlit run app/demo.py

The app deliberately frames the model output as a *suggestion* to a human
decision-maker (ALTAI Requirement #1): it shows the suggestion, a per-feature
explanation of how it was computed, a global explanation of what drives each
model, and requires an explicit human decision with a recorded rationale.
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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


def probability_steps(model, X_row: pd.DataFrame):
    """Waterfall decomposition of one suggestion, in probability space.

    The deployed models are linear (StandardScaler + LogisticRegression), so the
    score decomposes exactly in *log-odds*: log-odds = intercept + sum_j
    (coef_j * z_j), where z_j is the standardized value of feature j. Log-odds
    are additive but unreadable, so we walk them into probability: start from the
    baseline (every feature at its average, z = 0), then apply features one at a
    time, largest effect first, converting to probability after each step. The
    running probability lands *exactly* on model.predict_proba.

    Returns (baseline_prob, [(feature, delta_fraction, cumulative_prob), ...]).
    Note: because the sigmoid is non-linear, the per-feature percentage-point
    split depends on the order features are applied (largest-first here); the
    baseline and the final total do not. Log-odds remain the order-free view.
    """
    scaler = model.named_steps["standardscaler"]
    clf = model.named_steps["logisticregression"]
    z = (X_row.values[0] - scaler.mean_) / scaler.scale_
    contrib = z * clf.coef_[0]  # per-feature log-odds push (order-free, exact)
    intercept = float(clf.intercept_[0])

    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    order = np.argsort(-np.abs(contrib))  # largest absolute effect first
    baseline = sigmoid(intercept)
    running_logodds = intercept
    prev_p = baseline
    steps = []
    for j in order:
        running_logodds += contrib[j]
        p = sigmoid(running_logodds)
        steps.append((X_row.columns[j], p - prev_p, p))
        prev_p = p
    return baseline, steps


def waterfall_chart(baseline: float, steps):
    labels = (["Baseline (average profile)"]
              + [FEATURE_LABELS.get(f, f) for f, _, _ in steps]
              + ["Suggested probability"])
    n = len(labels)
    y = np.arange(n)[::-1]  # first row at the top
    fig, ax = plt.subplots(figsize=(5.8, 0.42 * n + 0.6))

    ax.barh(y[0], baseline, color="#8c8c8c")
    ax.text(baseline + 0.01, y[0], f"{baseline:.0%}", va="center", fontsize=8)

    prev = baseline
    for i, (_, delta, cum) in enumerate(steps, start=1):
        color = "#c0504d" if delta > 0 else "#4f81bd"
        ax.barh(y[i], delta, left=prev, color=color, height=0.6)
        # dashed connector from the previous cumulative level
        ax.plot([prev, prev], [y[i] + 0.3, y[i - 1] - 0.3],
                color="#bbb", linewidth=0.6, linestyle=(0, (2, 2)))
        sign = "+" if delta >= 0 else "−"
        ax.text(max(prev, cum) + 0.01, y[i],
                f"{sign}{abs(delta) * 100:.1f} pp", va="center", fontsize=7.5)
        prev = cum

    final = steps[-1][2] if steps else baseline
    ax.barh(y[-1], final, color="#333333")
    ax.text(final + 0.01, y[-1], f"{final:.0%}", va="center",
            fontsize=8, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.18)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("suggested P(re-arrest within 2 years)", fontsize=8)
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
        "Every output below is an **algorithmic suggestion** — the decision "
        "belongs to an accountable human."
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

    # --- decision: the suggestion from each model ----------------------------
    st.subheader("Suggestion")
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
        "chart below reads as a **waterfall in plain probability**: start from "
        "the baseline (an average profile), then apply each feature one at a "
        "time — **red bars push the suggestion toward higher risk, blue bars "
        "toward lower risk (in percentage points)** — and you land *exactly* on "
        "the suggested probability shown above."
    )
    baseline_b, steps_b = probability_steps(model_b, Xb)
    baseline_d, steps_d = probability_steps(model_d, Xd)
    ecol1, ecol2 = st.columns(2)
    with ecol1:
        st.caption("Model trained on original data (uses race)")
        st.pyplot(waterfall_chart(baseline_b, steps_b))
    with ecol2:
        st.caption("Model trained on de-biased data (race-blind)")
        st.pyplot(waterfall_chart(baseline_d, steps_d))
    lead_f, lead_delta, _ = steps_b[0]
    race_pp = sum(abs(d) for f, d, _ in steps_b if f.startswith("race_"))
    st.markdown(
        f"For this profile the biggest driver of the original-data model is "
        f"**{FEATURE_LABELS.get(lead_f, lead_f)}** "
        f"({'+' if lead_delta >= 0 else '−'}{abs(lead_delta) * 100:.1f} "
        "percentage points). The race fields together move the suggestion by "
        f"**{race_pp * 100:.1f} pp** in that model; in the de-biased model they "
        "move it exactly 0, because race is not an input. "
        "*(Percentage-point splits depend on the order features are applied, "
        "since probability is non-linear; the baseline and the final total do "
        "not.)*"
    )

    # --- global explanation: what drives each model --------------------------
    st.divider()
    st.subheader("Why does the model suggest this? (global explanation)")
    st.markdown(
        "Both models are **logistic regressions** — a linear model chosen "
        "precisely because its reasoning is fully auditable. The SHAP summary "
        "below shows which features drive each model's suggestions overall."
    )
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
