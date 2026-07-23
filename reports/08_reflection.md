# Reflection: From AI-Based Decision to AI-Based Suggestion

*Ethical and governance reflections on the COMPAS analysis project (ProPublica dataset, Broward County FL, 2013–2014), structured around the EU's Assessment List for Trustworthy AI (ALTAI).*

This project benchmarked a panel of classifier families, selected a logistic-regression model, trained it on the original COMPAS data and again on a de-biased version, compared the two with SHAP attributions and counterfactual (race-flip) analysis, and packaged the result in a Streamlit demo. The technical work answers *what the models do*; this document asks *what role such models should be allowed to play*. The central thesis is simple: **COMPAS-like systems must move from AI-based decision to AI-based suggestion** — the model proposes, and accountable humans (judges, case workers, with defendants' interests explicitly in scope) dispose.

---

## 1. Why risk scores must remain decision-*support* under human oversight

ALTAI Requirement #1 (Human Agency and Oversight) asks whether an AI system respects human autonomy — both of the people who use it and of the people it is used *on*. In the pretrial context, both groups are at risk.

**Automation bias and over-reliance.** Decades of human-factors research show that people defer to automated recommendations even when those recommendations are wrong, especially under time pressure and cognitive load — exactly the conditions of a busy bail hearing. A judge who sees "Risk: HIGH" anchors on it. The number arrives with an aura of objectivity that a probation officer's verbal assessment never had, even though Dressel and Farid (2018) showed that COMPAS is no more accurate than untrained Mechanical Turk workers given two sentences of information, and no more accurate than a two-feature logistic regression. When a score performs like a coin toss with a modest edge but *reads* like laboratory instrumentation, deference is a design failure, not a user failure.

**Are subjects aware the score is algorithmic?** ALTAI explicitly asks whether end users and affected persons are made aware that they are interacting with, or being scored by, an algorithm. In practice, many defendants scored by COMPAS never learned that a proprietary statistical model influenced their detention, and — because Northpointe's model is a trade secret — could not meaningfully contest it (a point at the heart of *Loomis v. Wisconsin*, 2016). Rudin (2019) argues this is exactly backwards: for high-stakes decisions we should demand inherently interpretable models rather than post-hoc explanations of black boxes, and there is no evidence the black box buys accuracy that a transparent model cannot match — our own ~65%-accuracy models (a fully interpretable logistic regression and decision tree, with a formal model-selection step confirming that opaque ensembles do no better), and Dressel and Farid's two-feature baseline, support that.

**Procedures against over-reliance.** ALTAI asks not just for a "human in the loop" but for procedures that make the human's role real. Concretely:

- **Display uncertainty**, not a bare label. "62% ± 8% probability of rearrest within two years" invites scrutiny; "HIGH RISK" invites rubber-stamping.
- **Display explanations** with every score (our SHAP panel does this), so the human can check *why* the model said what it said and notice when the reasoning is spurious.
- **Require a recorded human justification** both for *following* and for *overriding* the score. If only overrides need justification, the path of least resistance is deference, and the "human in the loop" degrades into a human rubber stamp — oversight in name only.

A human-in-the-loop requirement without these procedures is what critics call "moral crumple zones": the human absorbs blame for the machine's errors without having real control.

---

## 2. Technical robustness and safety: 65% accuracy deciding liberty

ALTAI Requirement #2 covers accuracy, reliability, and reproducibility. Two problems dominate here.

**The accuracy ceiling.** COMPAS — and our logistic-regression and decision-tree replications — achieve roughly 65% accuracy at predicting two-year rearrest (Angwin et al. 2016; Dressel & Farid 2018). Framed as an ML benchmark, 65% is a weak baseline. Framed as public policy, it means that **for roughly one person in three, the system's prediction is wrong**, and each wrong prediction has an asymmetric, irreversible cost: a false positive can mean weeks or months of detention for someone who would not have reoffended (with cascading job loss, housing loss, family disruption — factors that themselves *increase* recidivism risk); a false negative can mean a preventable crime. No accuracy threshold makes autonomous operation acceptable here, but 65% makes it indefensible. This is the strongest purely technical argument for the suggestion-not-decision framing: a tool this uncertain can inform a judgment; it cannot *be* the judgment.

**Data quality and representativeness.** Our training data is a single cohort: Broward County, Florida, 2013–2014, roughly 7,000 defendants. That is one county's demographics, one sheriff's office's arrest practices, one state attorney's charging policies, at one moment in time. A model fit to this distribution has no warrant to generalize to Milwaukee in 2026 — yet risk instruments are routinely transported across jurisdictions and years without revalidation. Distribution shift here is not an abstract concern: policing priorities change (e.g., drug enforcement waxes and wanes), and with them the very definition of the label. Any deployment claim from this project must be scoped to "models like these, on data like this," never "defendants in general."

---

## 3. Fairness: the ProPublica findings and why de-biasing cannot finish the job

ALTAI Requirement #5 (Diversity, Non-discrimination and Fairness) is where COMPAS became infamous. Angwin et al. (2016) found that among defendants who did *not* recidivate, Black defendants were roughly twice as likely as white defendants to be labeled high risk (false positive rates around 45% vs. 23%), while among defendants who *did* recidivate, white defendants were more likely to be labeled low risk (false negatives around 48% vs. 28%). The errors are not just frequent; they are *patterned*: the system errs against Black defendants and in favor of white defendants.

Northpointe's rebuttal — that COMPAS is *calibrated* (a score of 7 means roughly the same recidivism probability for both groups) — turned out to be technically true and beside the point. Chouldechova (2017) and Kleinberg, Mullainathan and Raghavan (2016) proved, independently, that **when base rates differ between groups, no non-trivial classifier can simultaneously satisfy calibration and equal false positive/false negative rates**. Both ProPublica and Northpointe were right about their own metric; the impossibility theorem says you cannot have both. This transforms fairness from an engineering problem into a *policy choice*: which error do we care about more, and who bears it? That choice is a value judgment about the distribution of harm — precisely the kind of judgment that belongs with accountable humans, not inside a loss function.

**Why our de-biasing helps but cannot solve it.** Our de-biased model narrows the FPR gap, and the counterfactual race-flip analysis shows reduced sensitivity to protected attributes. This is genuinely useful — it removes the most direct channels of disparate treatment. But data-level de-biasing operates on the dataset we have, and the dataset itself encodes upstream disparities (Section 4). Repairing the features cannot repair the label, and repairing the label would require knowing who actually offended rather than who was rearrested — information we do not have. Bias in criminal justice is a **socio-technical problem**: the technical component can be mitigated; the social component re-enters through the label, the deployment context, and the base rates themselves. (The same lesson recurs across ML: Tao et al. (2024) show that LLMs carry measurable cultural bias inherited from their training distributions — the artifact faithfully reflects the world that produced its data, which is exactly the problem.)

---

## 4. The label problem: a record of criminal justice *decisions*, not of crime

The single most important caveat in this project is what the target variable actually measures. `two_year_recid` is **rearrest**, not reoffending. Between an offense and an arrest stand a series of human decisions: where police patrol, whom they stop, whom they arrest rather than warn, whom prosecutors charge, and what prior record follows a person into the next encounter.

- **Policing intensity is not uniform.** Neighborhoods that are more heavily policed generate more arrests *for the same underlying behavior*. Drug offenses are the canonical example: survey data consistently show similar drug-use rates across racial groups, while arrest rates differ severalfold.
- **Priors are outputs of the same biased pipeline.** "Number of prior arrests" — the strongest feature in our SHAP analysis — is itself a cumulative record of past policing decisions. Using it as a neutral measure of criminal history launders historical enforcement disparity into a seemingly objective covariate.
- **The feedback loop.** High predicted risk → detention or heavier supervision → higher chance of technical violations and rearrest → "confirmation" of the prediction and more biased training data for the next model.

So the honest description of our models is not "predicting reoffending" but "predicting the future behavior of the criminal justice system toward this person." That reframing alone justifies human oversight: a judge can, in principle, recognize that a defendant's long arrest record reflects an over-policed neighborhood; a hyperplane cannot.

---

## 5. Concrete human-in-the-loop design recommendations

For the Streamlit demo — and for any hypothetical deployment descended from it — we recommend the following, all implementable with components the project already has:

1. **Never show a score alone.** Every score is rendered with its SHAP explanation (which features pushed it up or down) and an uncertainty interval. The demo's side-by-side original/de-biased comparison should stay in the interface: showing that two defensible models disagree is itself an anti-over-reliance measure.
2. **Show per-group error rates on-screen.** A persistent panel with FPR/FNR by race for the active model turns the impossibility trade-off (Section 3) from a footnote into something the decision-maker confronts every time.
3. **Mandatory human justification, symmetric.** The interface requires a free-text reason before a recommendation is *accepted or overridden*, and logs it. This creates an audit trail and forces engagement (see Section 1).
4. **A contestation path for the affected person.** Defendants must be told a score was used, shown the explanation in plain language, and given a mechanism to challenge input errors (a wrong prior, a misattributed charge) and the score itself.
5. **Periodic audits.** Scheduled re-evaluation of calibration, per-group error rates, and drift against the deployment population — with authority to suspend the tool if disparities exceed agreed thresholds. Our counterfactual race-flip test belongs in that audit suite as a regression test.
6. **One input among many, by design.** The score should appear in the decision-maker's workflow alongside — not above — the case file, counsel's arguments, and individual circumstances, and policy should state explicitly that no adverse decision may rest on the score alone (in the spirit of GDPR Article 22).

---

## 6. Which variables belong in the model?

**Race** obviously must not be a predictive input: using it directly is disparate treatment. But the harder questions are elsewhere.

**Proxies.** Zip code correlates with race strongly enough in most US metros to function as a race variable with plausible deniability; the same holds for features downstream of residential segregation (school, employment sector). Our race-flip counterfactuals are informative but insufficient — a model can be invariant to the race *field* while remaining highly sensitive to its proxies.

**Prior arrests.** As argued in Section 4, prior-arrest counts are biased *measurements*, not neutral facts. Excluding them entirely may be too costly (they carry most of the predictive signal), but they should at minimum be treated as what they are — enforcement history — and candidates like "prior convictions" (further from street-level policing discretion) are less contaminated than "prior arrests."

**The auditing trade-off.** Here is the genuinely counterintuitive point: **removing race from the pipeline entirely makes bias harder to detect, not easier to commit.** Every analysis in this project — the ProPublica replication, the per-group error rates, the counterfactual tests, the de-biasing itself — *required* the race column. "Fairness through unawareness" fails twice: it does not prevent discrimination (proxies remain), and it destroys the ability to measure it. The right architecture is therefore: race **excluded from the model's inputs**, but **retained in the dataset** for auditing, monitoring, and fairness-aware training. Blindness is not fairness; measured, audited awareness is closer.

---

## 7. Conclusion

Everything this project measured points the same way. The models are modestly accurate (Section 2), their errors are racially patterned in ways no algorithm can fully dissolve (Section 3), and the target they predict is a record of institutional behavior rather than individual conduct (Section 4). None of this means the models are useless — a calibrated, explained, audited risk estimate can be a legitimate *input* to a human decision. It means the models cannot be the *decider*.

That is what the demo is for. The Streamlit app deliberately shows two models disagreeing, shows SHAP explanations instead of bare labels, and shows what happens under a race flip — it is built to provoke the question "should I trust this number?" rather than to answer it. In ALTAI's terms, it operationalizes human agency and oversight (Req. #1) while making robustness limits (Req. #2) and fairness trade-offs (Req. #5) visible rather than buried. AI-based suggestion, human-based decision: the score advises, the explanation informs, the error rates warn — and a person who can be held accountable, looking at a person whose liberty is at stake, decides.

---

## References

- Angwin, J., Larson, J., Mattu, S., & Kirchner, L. (2016). *Machine Bias.* ProPublica.
- Chouldechova, A. (2017). Fair prediction with disparate impact: A study of bias in recidivism prediction instruments. *Big Data*, 5(2), 153–163.
- Dressel, J., & Farid, H. (2018). The accuracy, fairness, and limits of predicting recidivism. *Science Advances*, 4(1), eaao5580.
- Kleinberg, J., Mullainathan, S., & Raghavan, M. (2016). Inherent trade-offs in the fair determination of risk scores. *arXiv:1609.05807*.
- Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206–215.
- Tao, Y., Viberg, O., Baker, R. S., & Kizilcec, R. F. (2024). Cultural bias and cultural alignment of large language models. *PNAS Nexus*, 3(9).
- European Commission, High-Level Expert Group on AI (2020). *Assessment List for Trustworthy Artificial Intelligence (ALTAI).*
