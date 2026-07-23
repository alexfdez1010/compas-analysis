# COMPAS Bias Analysis — From AI-based Decision to AI-based Suggestion

**Full final report: [REPORT.md](REPORT.md)** — abstract, key results, all
figures, decisions made, reproducibility, and limitations.

This project examines how bias manifests in the COMPAS dataset and how it can
be identified and mitigated. The dataset records interactions with the criminal
justice system of Broward County, Florida (2013–2014) — judicial processing,
criminal history, algorithmic risk scores — and is therefore a record of
criminal-justice *decisions*, not an objective representation of crime. The
project detects the embedded bias, mitigates it at the data level, and reframes
the model as a *suggestion* to an accountable human (ALTAI Requirements #1, #2,
#5).

A model-selection benchmark (step 3) shows every classifier family — from
logistic regression to gradient-boosted trees — lands within ~0.02 ROC-AUC, so
the project uses a **logistic regression**: tied-best accuracy and fully
interpretable, with no accuracy argument for an opaque model.

Headline result: de-biasing cut the model's demographic parity difference from
0.300 to 0.238 and equalized odds difference from 0.315 to 0.250 at a
negligible accuracy cost (67.2% to 67.1%), and made suggestions provably
invariant to race (race-flip changes: 6.7% of biased suggestions vs exactly 0).

Data source: [ProPublica compas-analysis](https://github.com/propublica/compas-analysis)

## Project pipeline

| Step | Script | Output |
|------|--------|--------|
| 1. Download data | `scripts/01_download_data.py` | `data/raw/*.csv` |
| 2. Exploratory analysis (RQ1–RQ3) | `scripts/02_eda.py` | `figures/`, `reports/02_eda.md` |
| 3. Model selection (family benchmark) + persisted split | `scripts/03_model_selection.py` | `reports/03_model_selection.md`, `data/processed/` |
| 4. Baseline models (decision tree + logistic regression) + fairness audit | `scripts/04_baseline_model.py` | `reports/04_baseline.md`, `models/` |
| 5. Automated bias detection & data de-biasing | `scripts/05_debias.py` | `data/processed/`, `reports/05_debias.md` |
| 6. Logistic regression on de-biased data | `scripts/06_debiased_model.py` | `reports/06_debiased_model.md`, `models/` |
| 7. Explainability comparison (SHAP + counterfactuals) | `scripts/07_xai_comparison.py` | `figures/`, `reports/07_xai.md` |
| 8. Interactive demo | `app/demo.py` (Streamlit) | side-by-side biased vs de-biased suggestion |

Ethical reflection (ALTAI): `reports/08_reflection.md`. Literature landscape &
novelty: `reports/09_literature_and_novelty.md`.

## Setup and usage

Dependencies are managed with [uv](https://docs.astral.sh/uv/); all seeds are
fixed (42) and the train/test split is persisted, so the pipeline is fully
reproducible:

```bash
uv sync
uv run python scripts/01_download_data.py
uv run python scripts/02_eda.py
uv run python scripts/03_model_selection.py
uv run python scripts/04_baseline_model.py
uv run python scripts/05_debias.py
uv run python scripts/06_debiased_model.py
uv run python scripts/07_xai_comparison.py
```

Run the interactive demo locally:

```bash
uv run streamlit run app/demo.py
```

Or try the live version: **[compas-analysis.streamlit.app](https://compas-analysis.streamlit.app)**

## Tools

- **scikit-learn** — model-selection benchmark across 10 classifier families;
  logistic regression (main classifier, chosen empirically) and a decision tree
  (interpretability reference)
- **Fairlearn** — fairness metrics and `CorrelationRemover` de-biasing
- **SHAP** — post-hoc explainability before/after de-biasing
- **DiCE / counterfactual analysis** — race-flip and "what would have to change" tests
- **Streamlit** — interactive demo comparing the two models
