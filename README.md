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

Headline result: de-biasing cut the SVM's demographic parity difference from
0.317 to 0.233 and equalized odds difference from 0.361 to 0.249 at a
negligible accuracy cost (66.0% to 65.3%), and made suggestions provably
invariant to race (race-flip changes: 12.6% of biased suggestions vs exactly 0).

Data source: [ProPublica compas-analysis](https://github.com/propublica/compas-analysis)

## Project pipeline

| Step | Script | Output |
|------|--------|--------|
| 1. Download data | `scripts/01_download_data.py` | `data/raw/*.csv` |
| 2. Exploratory analysis (RQ1–RQ3) | `scripts/02_eda.py` | `figures/`, `reports/02_eda.md` |
| 3. Baseline models (decision tree + SVM) + fairness audit | `scripts/03_baseline_model.py` | `reports/03_baseline.md`, `models/` |
| 4. Automated bias detection & data de-biasing | `scripts/04_debias.py` | `data/processed/`, `reports/04_debias.md` |
| 5. SVM on de-biased data | `scripts/05_debiased_model.py` | `reports/05_debiased_model.md`, `models/` |
| 6. Explainability comparison (SHAP + counterfactuals) | `scripts/06_xai_comparison.py` | `figures/`, `reports/06_xai.md` |
| 7. Interactive demo | `app/demo.py` (Streamlit) | side-by-side biased vs de-biased suggestion |

Ethical reflection (ALTAI): `reports/07_reflection.md`.

## Setup and usage

Dependencies are managed with [uv](https://docs.astral.sh/uv/); all seeds are
fixed (42) and the train/test split is persisted, so the pipeline is fully
reproducible:

```bash
uv sync
uv run python scripts/01_download_data.py
uv run python scripts/02_eda.py
uv run python scripts/03_baseline_model.py
uv run python scripts/04_debias.py
uv run python scripts/05_debiased_model.py
uv run python scripts/06_xai_comparison.py
```

Run the interactive demo:

```bash
uv run streamlit run app/demo.py
```

## Tools

- **scikit-learn** — decision tree (interpretability) and SVM (main classifier)
- **Fairlearn** — fairness metrics and `CorrelationRemover` de-biasing
- **SHAP** — post-hoc explainability before/after de-biasing
- **DiCE / counterfactual analysis** — race-flip and "what would have to change" tests
- **Streamlit** — interactive demo comparing the two models
