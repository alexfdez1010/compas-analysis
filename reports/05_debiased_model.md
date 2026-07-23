# SVM on the de-biased data

Identical architecture to the reference model (StandardScaler + RBF SVM,
C=1.0), trained on the de-biased features from script 04. Race enters the
pipeline only as an **audit attribute**.

## Deployment decision: race-blind inference

The CorrelationRemover needs the sensitive attribute to transform a row, which
poses a choice for prediction time:

| Deployment | Accuracy | Demographic parity diff. | Equalized odds diff. | Individual race-invariance |
|------------|---------:|-------------------------:|---------------------:|:--:|
| **Race-blind** (raw features at inference, chosen) | 65.3% | 0.233 | 0.249 | yes - exact |
| Race-aware (transform with the person's race) | 65.3% | 0.168 | 0.168 | no |

The race-aware variant achieves better *group* fairness, but a person's stated
race then moves their individual score - it breaks counterfactual fairness,
requires collecting the protected attribute at decision time, and amounts to
explicit differential treatment. We therefore deploy **race-blind**: the
de-biasing is a training-time intervention (the model's *coefficients* were
learned from race-neutralized data), and at prediction time the model never
sees race, so flipping race provably cannot change any suggestion. All numbers
below use race-blind inference.

## Performance

| Model | Accuracy | ROC-AUC |
|-------|---------:|--------:|
| SVM, original data | 66.0% | 0.720 |
| SVM, de-biased data | **65.3%** | **0.712** |

De-biasing costs +0.6% accuracy - essentially within noise. The
"fairness tax" on predictive performance is negligible here, consistent with
the finding that most of the usable signal (priors, age) is retained after the
transformation.

## Fairness comparison (African-American vs Caucasian, test set)

![Comparison](../figures/05_fairness_comparison.png)

| Metric | SVM original | SVM de-biased |
|--------|-------------:|--------------:|
| FPR African-American | 29.5% | 25.3% |
| FPR Caucasian | 7.6% | 9.4% |
| **FPR gap** | **22.0%** | **16.0%** |
| FNR African-American | 38.4% | 45.6% |
| FNR Caucasian | 74.5% | 70.4% |
| **FNR gap** | **36.1%** | **24.9%** |
| Demographic parity difference | 0.317 | **0.233** |
| Equalized odds difference | 0.361 | **0.249** |

Full per-group metrics of the de-biased model:

| Metric | African-American | Caucasian | Hispanic |
|--------|----------------:|----------:|---------:|
| Accuracy | 64.1% | 66.7% | 66.7% |
| Selection rate | 40.5% | 17.3% | 26.1% |
| False positive rate | 25.3% | 9.4% | 17.7% |
| False negative rate | 45.6% | 70.4% | 59.6% |

## Reading the result honestly

The error-rate gaps shrink substantially but do **not** vanish. The remaining
gap is driven by the different *base rates* of the re-arrest label - the part
of the disparity that lives in the outcome variable itself and that no
feature-side intervention can remove (Chouldechova 2017). Closing it entirely
would require either post-processing per-group thresholds (a policy decision
with its own ethical cost: explicit differential treatment) or better labels
(measuring reoffending rather than re-arrest).

This residual gap is a further argument for the project's central claim: the
system must remain a **suggestion** presented to accountable humans, together
with its known error profile - not an automated decision.
