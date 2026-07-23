# Baseline models on the original data

Split: 70/30 train/test, stratified jointly on outcome and race
(4,320 train / 1,852 test), created and persisted by script 03
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

Accuracy **66.3%**, ROC-AUC **0.709**.

![Decision tree](../figures/04_decision_tree.png)

![Feature importance](../figures/04_tree_importance.png)

The learned rules are dominated by `priors_count` and `age`:

```text
|--- priors_count <= 1.5
|   |--- age <= 22.5
|   |   |--- age <= 20.5
|   |   |   |--- class: 1
|   |   |--- age >  20.5
|   |   |   |--- race_caucasian <= 0.5
|   |   |   |   |--- class: 1
|   |   |   |--- race_caucasian >  0.5
|   |   |   |   |--- class: 0
|   |--- age >  22.5
|   |   |--- age <= 35.5
|   |   |   |--- priors_count <= 0.5
|   |   |   |   |--- class: 0
|   |   |   |--- priors_count >  0.5
|   |   |   |   |--- class: 0
|   |   |--- age >  35.5
|   |   |   |--- charge_felony <= 0.5
|   |   |   |   |--- class: 0
|   |   |   |--- charge_felony >  0.5
|   |   |   |   |--- class: 0
|--- priors_count >  1.5
|   |--- age <= 33.5
|   |   |--- priors_count <= 7.5
|   |   |   |--- age <= 23.5
|   |   |   |   |--- class: 1
|   |   |   |--- age >  23.5
|   |   |   |   |--- class: 1
|   |   |--- priors_count >  7.5
|   |   |   |--- race_african_american <= 0.5
|   |   |   |   |--- class: 1
|   |   |   |--- race_african_american >  0.5
|   |   |   |   |--- class: 1
|   |--- age >  33.5
|   |   |--- priors_count <= 6.5
|   |   |   |--- priors_count <= 2.5
|   |   |   |   |--- class: 0
|   |   |   |--- priors_count >  2.5
|   |   |   |   |--- class: 0
|   |   |--- priors_count >  6.5
|   |   |   |--- priors_count <= 9.5
|   |   |   |   |--- class: 1
|   |   |   |--- priors_count >  9.5
|   |   |   |   |--- class: 1
```

Two observations relevant to the ethics assessment:

1. The tree achieves essentially the same accuracy as COMPAS itself (~65%),
   echoing Dressel & Farid (2018): a transparent model with a handful of
   features matches the proprietary 137-question instrument. There is no
   accuracy argument for opacity.
2. Race dummies barely appear in the split rules, yet the fairness audit below
   still shows large error-rate gaps - the bias travels through `priors_count`
   and `age`, which are products of unequal policing intensity (see RQ2).

## Reference Logistic Regression ("biased model")

Accuracy **67.2%**, ROC-AUC **0.724**. Because the model is
linear its logic is fully readable - the standardized coefficients (log-odds
impact per one-SD change in each feature) are:

| Feature | Std. coefficient |
|---------|-----------------:|
| `priors_count` | +0.763 |
| `age` | -0.498 |
| `juv_other_count` | +0.184 |
| `charge_felony` | +0.105 |
| `sex_male` | +0.094 |
| `race_other` | -0.043 |
| `juv_fel_count` | +0.040 |
| `race_african_american` | +0.039 |
| `race_hispanic` | -0.029 |
| `juv_misd_count` | -0.008 |
| `race_caucasian` | -0.003 |

![Fairness metrics](../figures/04_fairness_lr.png)

| Metric | African-American | Caucasian | Hispanic |
|--------|----------------:|----------:|---------:|
| Accuracy | 66.8% | 68.0% | 67.3% |
| Selection rate | 49.8% | 19.8% | 26.8% |
| False positive rate | 32.2% | 10.4% | 17.7% |
| False negative rate | 34.1% | 65.6% | 57.9% |

Decision tree for comparison (same test set): FPR
41.6% vs
16.4%, FNR
28.5% vs
59.1%
(African-American vs Caucasian).

Aggregate disparity of the Logistic Regression restricted to African-American
vs Caucasian:

- **Demographic parity difference: 0.300** (gap in the share of people
  flagged as likely recidivists)
- **Equalized odds difference: 0.315** (largest gap in FPR or TPR)

The model reproduces the asymmetry found in the COMPAS scores themselves
(report 02): African-American defendants face a much higher false positive
rate, Caucasian defendants a much higher false negative rate. Training a fresh
model on the raw data *reproduces* the injustice pattern of the data-generating
system - the reference point the de-biasing step must improve on.

## ALTAI Requirement #2 - accuracy in context

An accuracy of ~67% means roughly one in three suggestions is wrong.
For a system that could influence detention decisions this error rate is only
acceptable - if at all - in a decision-support setting with a human weighing
independent evidence (see reports/08_reflection.md).
