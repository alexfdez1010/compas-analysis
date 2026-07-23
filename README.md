# COMPAS Bias Analysis — From AI-based Decision to AI-based Suggestion

This project examines how bias manifests within the COMPAS (Correctional Offender
Management Profiling for Alternative Sanctions) dataset and how it can be identified
and mitigated through data-driven approaches.

The COMPAS dataset does not directly measure criminal behavior. It records
individuals' interactions with the criminal justice system of Broward County,
Florida during 2013–2014 — judicial processing, criminal history, and algorithmic
risk assessments — and is therefore a product of a particular criminal justice
system, not an objective representation of crime itself.

Data source: [ProPublica compas-analysis](https://github.com/propublica/compas-analysis)

## Project pipeline

| Step | Script | Output |
|------|--------|--------|
| 1. Download data | `scripts/01_download_data.py` | `data/raw/*.csv` |
| 2. Exploratory analysis (RQ1–RQ3) | `scripts/02_eda.py` | `figures/`, `reports/02_eda.md` |
| 3. Baseline models on original data (decision tree + SVM) + fairness audit | `scripts/03_baseline_model.py` | `reports/03_baseline.md`, `models/` |
| 4. Automated bias detection & data de-biasing | `scripts/04_debias.py` | `data/processed/`, `reports/04_debias.md` |
| 5. SVM on de-biased data | `scripts/05_debiased_model.py` | `reports/05_debiased_model.md`, `models/` |
| 6. Explainability comparison (SHAP + counterfactuals) | `scripts/06_xai_comparison.py` | `figures/xai/`, `reports/06_xai.md` |
| 7. Interactive demo | `app/demo.py` (Streamlit) | side-by-side biased vs de-biased model |

## Research questions

1. **Is the COMPAS dataset representative and what does it represent?**
   Data source, time range, covered cases, collected variables, and why it is a
   canonical dataset in AI-fairness research.
2. **Does the dataset reflect historical and institutional inequalities? Why?**
   The dataset is a record of criminal justice *decisions*, not of crime —
   policing bias, sentencing bias, socioeconomic and racial inequality may be
   embedded in it.
3. **What demographic disparities exist in the dataset?**
   Race/sex proportions, age distribution, risk-score distributions, false
   positive/negative rates per group.

## Ethical framing (ALTAI)

- **Requirement #1 — Human Agency and Oversight:** the system is framed as a
  *decision-support suggestion*, never an automated decision; measures against
  automation bias are discussed in `reports/07_reflection.md`.
- **Requirement #2 — Technical Robustness and Safety:** accuracy and error-rate
  audits per group.
- **Requirement #5 — Diversity, Non-discrimination and Fairness:** avoidance of
  unfair bias via automated detection (Fairlearn) and data-level mitigation.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run python scripts/01_download_data.py
uv run python scripts/02_eda.py
# ... remaining scripts in order
uv run streamlit run app/demo.py
```

## Tools

- **scikit-learn** — decision trees (interpretability) and SVMs (main classifier)
- **Fairlearn** — fairness metrics (demographic parity, equalized odds, per-group
  error rates) and automated bias detection/mitigation
- **SHAP** — post-hoc explainability of the models before/after de-biasing
- **DiCE / counterfactual analysis** — changing race to observe how biased and
  de-biased models react
- **Streamlit** — interactive demo comparing the two models
