# Literature Landscape & Novelty Assessment

*Where the COMPAS dataset has already been analyzed, and where this project
adds something the literature has not settled.*

This report situates the work in [REPORT.md](../REPORT.md) against the published
COMPAS / recidivism-fairness literature. It was assembled with the `paperhound`
CLI (arXiv, OpenAlex, DBLP, Crossref, Semantic Scholar, CORE) across ~15
targeted queries covering the project's distinct angles: the ProPublica
controversy, impossibility results, data-quality audits, pre-processing
de-biasing, XAI on recidivism, counterfactual/individual fairness, and
human-in-the-loop decision support. A companion BibTeX file with the 21 core
references is at [reports/related_work.bib](related_work.bib) — keys are cited
inline below as `[key]` and drop straight into `paper/user/biblio.bib`.

> **Namespace warning for future searches.** "COMPAS" is also the name of a
> widely-cited *astrophysics* code (rapid stellar/binary population synthesis,
> arXiv:2109.10352). Bare `paperhound search "COMPAS ..."` returns that as a top
> hit. Always disambiguate with `recidivism`, `defendants`, `ProPublica`, or
> `criminal justice`.

---

## 1. TL;DR

- **The COMPAS dataset is one of the most heavily analyzed benchmarks in
  algorithmic fairness.** The core empirical findings the project reproduces
  (the ~2× FPR disparity, the ~65% accuracy ceiling, the calibration-vs-error-rate
  impossibility) are *settled science* with thousands of citations. Reproducing
  them is correct and expected, but it is not where novelty can live.
- **The project's contribution is not any single technique** — each of its
  building blocks (CorrelationRemover, SHAP, DiCE, Fairlearn audits) exists in
  the literature — **but the *integrated, honesty-first framing*: de-bias →
  audit the de-biasing with paired explanations → commit to an explicit,
  defensible deployment stance (race-blind, individual counterfactual fairness
  *by construction*) → operationalize it as decision-support with forced human
  justification.** That end-to-end pipeline, tied to ALTAI, is rare in the
  literature, which tends to isolate one link.
- **Five concrete, publishable research openings** emerge where the literature
  is thin or where the project is one experiment away from a real contribution
  (Section 4). The strongest is **re-running the whole pipeline on Barenstein's
  corrected COMPAS data** [barenstein2019propublica] — a known data-processing
  error inflates the very base-rate gap the project calls "irreducible."

---

## 2. What the literature has already settled

Organized by theme, with the strongest anchors and their approximate citation
counts (a proxy for "how done is this").

### 2.1 The original controversy and the impossibility results — *closed*
- **ProPublica's *Machine Bias*** (Angwin et al. 2016) launched the field: Black
  defendants who did not recidivate were ~2× as likely to be flagged high-risk.
  The project reproduces this exactly (42.3% vs 22.0% FPR).
- **Northpointe's calibration rebuttal**, then **Chouldechova (2017)**
  [chouldechova2017fair] and **Kleinberg, Mullainathan & Raghavan (2016)**
  [kleinberg2016inherent] proved *independently* that calibration and equal
  error rates cannot co-hold when base rates differ. **~700 / large citation
  counts.** The project cites both correctly as the reason its residual gap is
  irreducible.
- **Corbett-Davies et al. (2017)** [corbettdavies2017algorithmic] framed fairness
  as a cost/threshold policy choice (~715 cites). Directly supports the project's
  "fairness is a value judgment for humans, not a loss function" stance.

**Verdict:** fully established. The project's job here is faithful replication,
which it does.

### 2.2 The accuracy ceiling and the human-comparison literature — *closed, but with a twist the project under-uses*
- **Dressel & Farid (2018)** [dressel2018accuracy]: COMPAS ≈ untrained Mechanical
  Turk workers ≈ a 2-feature logistic regression, all near 65%. (~690 cites.)
  The project's ~65% SVM/tree is a deliberate echo.
- **Twist:** **Lin, Jung & Goel (2020)** [lin2020limits] and **Tan, Adebayo &
  Inkpen (2018)** [tan2018investigating] show human predictions of recidivism are
  *also* limited and *also* biased — humans are not a clean fallback. This
  complicates the "keep a human in the loop" conclusion and is a tension the
  project's reflection should engage rather than assume the human is better.

### 2.3 Data quality of the ProPublica file itself — *under-appreciated, directly actionable*
- **Barenstein (2019)** [barenstein2019propublica]: ProPublica applied the
  two-year observation cutoff to non-recidivists but **not** to recidivists,
  keeping ~40% too many recidivists and inflating the two-year recidivism rate
  from a corrected **36.2%** to the widely-quoted **45.1%**. Almost every COMPAS
  fairness paper — including this project — uses the *uncorrected*
  `compas-scores-two-years.csv`. **This is the single most important gap for the
  project** (see 4.1).

### 2.4 Bias-mitigation methods on recidivism data — *crowded*
- **Surveys:** Hort et al. (2024) [hort2024bias] catalogue pre-/in-/post-processing
  mitigations (~196 cites); Fairlearn's `CorrelationRemover` (the project's tool)
  sits in a large family of pre-processing methods.
- **In-processing on COMPAS specifically:** Wadsworth, Vera & Piech (2018)
  [wadsworth2018achieving] use adversarial de-biasing on this exact dataset.
- **Post-processing for RAIs:** Mishler, Kennedy & Chouldechova (2021)
  [mishler2021fairness] target *counterfactual* equalized odds.
- **Causes of the bias:** Miron, Tolan & Gómez (2020) [miron2020evaluating]
  decompose sources of algorithmic bias in juvenile recidivism.

**Verdict:** the *act* of de-biasing COMPAS is well-trodden. The project's
CorrelationRemover run is a competent instance, not a new method. Novelty must
come from what it does *around* the de-biasing (auditing it, deploying it), not
the de-biasing itself.

### 2.5 XAI on recidivism — *common tools, an uncommon question*
- **SHAP/DiCE are standard:** Mothilal et al. (2020) [mothilal2020explaining] is
  DiCE itself (~1,100 cites); Ingram, Gursoy & Kakadiaris (2022) [ingram2022accuracy]
  is the closest single paper to this project — it evaluates recidivism models
  jointly on **accuracy, fairness, and interpretability** (on a Georgia parole
  dataset, not COMPAS) and finds explicit trade-offs.
- **Critical caveat the project should absorb:** Slack et al. (2020)
  [slack2020fooling] ("Fooling LIME and SHAP", ~790 cites) show post-hoc
  attributions can be adversarially manipulated and are not ground truth — and
  Rudin (2019) [rudin2019stop] (~8,000 cites) argues for inherently interpretable
  models over post-hoc explanations in exactly this high-stakes setting. The
  project already leans on Rudin; it relies on SHAP as *evidence* of de-biasing,
  so the Slack caveat matters (see 4.4).

### 2.6 Human-in-the-loop & automation bias — *active, mostly problem-diagnosis not tool-building*
- **De-Arteaga, Fogliato & Chouldechova (2020)** [dearteaga2020case] (~170 cites):
  humans *can* catch erroneous algorithmic scores — the empirical case for HITL.
- **Alon-Barkat & Busuioc (2022)** [alonbarkat2022human] (~360 cites): "automation
  bias" and *selective adherence* — humans over-trust algorithmic advice,
  especially when it confirms priors.

**Verdict:** the literature richly documents *that* automation bias is a problem
but offers far fewer *evaluated interface mechanisms* that fix it. The project's
Streamlit demo (forced symmetric rationale, two disagreeing models, on-screen
error rates) is a concrete design response — currently asserted, not tested
(see 4.3).

### 2.7 The sociotechnical / label critique — *established as argument, rarely quantified in the same paper*
- **Selbst et al. (2019)** [selbst2019fairness] ("Fairness and Abstraction in
  Sociotechnical Systems", ~1,400 cites): fairness cannot be solved inside the
  model boundary — the project's "socio-technical problem" language.
- **Obermeyer et al. (2019)** [obermeyer2019dissecting] (*Science*, ~6,000 cites):
  the canonical label-proxy failure (cost as a proxy for need) — the exact
  structure of the project's "re-arrest ≠ re-offense" argument, in healthcare.
- **Ensign et al. (2017)** [ensign2017runaway]: runaway feedback loops in
  predictive policing — the mechanism behind the project's biased-label claim.

**Verdict:** the *argument* is canonical. What most papers do **not** do, and
the project does, is pair it with a quantified in-pipeline demonstration (proxy
AUC 0.682, race SHAP share 15.9%, 12.6% counterfactual flips) *and* let it drive
the deployment decision.

---

## 3. Where this project sits — component-by-component

| Project component | Closest prior work | How settled | This project's distinctive move |
|---|---|---|---|
| ProPublica FPR/FNR replication | Angwin 2016; Chouldechova 2017 | **Settled** | Faithful reproduction as a baseline (not a contribution) |
| ~65% accuracy = no cost to transparency | Dressel & Farid 2018; Rudin 2019 | **Settled** | Reframed as an *argument against opacity + for suggestion framing* |
| CorrelationRemover de-biasing | Hort 2024 survey; Wadsworth 2018 | Common | Combined with a **proxy-predictability falsification test** (AUC 0.682→0.508) |
| "Fairness through unawareness fails" | Standard result | Settled | Quantified *on this data* as a headline audit metric |
| SHAP attribution of race share | Ingram 2022; SHAP lit | Common tool | **Paired biased-vs-de-biased attribution audit**: shows de-biased ≠ biased−race |
| DiCE counterfactual recourse | Mothilal 2020 | Common tool | Positioned as decision-support content, not just explanation |
| Race-flip counterfactual (12.6% flips) | Counterfactual-fairness lit | Known idea | Turned into a **deployment guarantee**: race-blind ⇒ *exactly 0 by construction* |
| Race-blind vs race-aware deployment choice | Binns 2019 [binns2019apparent]; Mishler 2021 | Discussed in theory | An **explicit, documented engineering decision** with the group-vs-individual trade-off tabled honestly |
| HITL demo w/ forced symmetric rationale | De-Arteaga 2020; Alon-Barkat 2022 | Problem well-known | A **built, runnable anti-automation-bias interface** (rare) |
| ALTAI-structured reflection | EU HLEG ALTAI | Framework exists | Applied end-to-end to a concrete pipeline |

**Bottom line:** no single cell is "world-first." The project's real, defensible
originality is **the integration and the intellectual honesty**: it is one of
the few COMPAS treatments that de-biases, *then audits whether the de-biasing
actually worked using paired explanations*, *then makes and documents a
principled deployment decision*, *then instruments the human-oversight layer* —
and reports the residual, irreducible gap rather than declaring victory.

---

## 4. Concrete research openings (ranked)

These are where the project could produce a genuinely novel result with modest
additional work. Ranked by impact-to-effort.

### 4.1 ⭐ Re-run everything on Barenstein-corrected data — *highest value, low effort*
The project's central "irreducible gap" argument rests on base rates of **52%
(AA) vs 39% (Cauc.)**. But Barenstein (2019) [barenstein2019propublica] shows the
ProPublica two-year file **over-counts recidivists by ~40%** due to an
asymmetric cutoff bug, inflating the overall rate from 36.2% to 45.1%.
**Open question no one has cleanly answered:** does the fairness gap — and the
"impossibility-driven irreducibility" — *shrink* on the corrected labels? Re-run
scripts 02–06 on a Barenstein-corrected cohort and report the delta. If the gap
persists, the project's thesis is *strengthened* on clean data; if it shrinks,
that is itself a publishable finding about how much of "COMPAS bias" is a
dataset-construction artifact. Either way it is novel because almost the entire
literature uses the uncorrected file.

### 4.2 ⭐ The "does de-biasing survive its own explanation audit?" method
The paired SHAP comparison (biased vs de-biased) showing that de-biased
attributions are **not** simply "biased minus race" — because CorrelationRemover
moves each defendant's priors relative to their group mean — is an
under-explored *methodological* idea. Most mitigation papers report group
metrics before/after; few ask *whether the explanation structure confirms the
bias channel actually closed*. Formalize this as a reusable **"attribution-shift
audit"** for any pre-processing de-biaser, validated on 2–3 mitigations
(CorrelationRemover vs reweighing vs adversarial [wadsworth2018achieving]). This
is a clean methods contribution.

### 4.3 ⭐ Turn the Streamlit demo into an actual user study
The demo asserts that forced symmetric rationale + two disagreeing models + live
race-flip table reduce over-reliance. The HITL literature
[dearteaga2020case; alonbarkat2022human] documents the *problem* extensively but
tests *interventions* rarely. A small controlled study (does the interface change
override rates / decision quality vs a bare-score baseline?) would be a real HCI
contribution — the project already has the instrument built.

### 4.4 Robustness of the SHAP-based de-biasing evidence
Because the project uses SHAP as *evidence* that the race channel closed, the
Slack et al. (2020) [slack2020fooling] result (SHAP is manipulable / not ground
truth) is a live threat. Add a robustness check: corroborate the 15.9%→0 race
attribution finding with a second, independent method (permutation importance
already partly there; add a causal/ablation check, or the InfoGram admissibility
lens). Small effort, closes an obvious reviewer objection.

### 4.5 Non-linear proxy leakage beyond CorrelationRemover
The report honestly flags that CorrelationRemover is *linear* and that unmeasured
proxies (zip code) would re-introduce race. The proxy test (AUC→0.508) only
rules out residual *linear-model-detectable* signal. Fit a **non-linear** proxy
predictor (gradient boosting) on the de-biased features to test whether a
stronger adversary recovers race — a direct, quantitative stress test of the
"fairness through unawareness" boundary that the current linear proxy test
leaves open.

### 4.6 Engage the "humans aren't better" tension
The reflection assumes human oversight improves outcomes. Lin et al. (2020)
[lin2020limits] and Tan et al. (2018) [tan2018investigating] show humans predict
recidivism poorly and with their own biases. The strongest version of the
project's argument is not "humans are more accurate" but "humans are
*accountable and contestable* in ways a model is not" — sharpen the reflection to
make that the load-bearing claim, and cite the human-limits literature to show
the argument survives even when the human is *not* more accurate.

---

## 5. How to use this in `paper/`

- `reports/related_work.bib` → merge into `paper/user/biblio.bib`.
- Section 2 above → the Related Work section (themes 2.1–2.7 map to subsections).
- Section 3 table → a "positioning" paragraph/table distinguishing replication
  from contribution.
- Section 4.1 (Barenstein re-analysis) is the recommended **new experiment** to
  run before writing the paper, so the manuscript reports a result the field
  does not yet have.

## 6. Core references (keys in `related_work.bib`)

- **Impossibility / foundations:** `chouldechova2017fair`, `kleinberg2016inherent`,
  `corbettdavies2017algorithmic`
- **Accuracy ceiling / human comparison:** `dressel2018accuracy`, `lin2020limits`,
  `tan2018investigating`
- **Data quality:** `barenstein2019propublica`
- **Mitigation & closest overlap:** `hort2024bias`, `wadsworth2018achieving`,
  `mishler2021fairness`, `miron2020evaluating`, `ingram2022accuracy`
- **XAI & interpretability:** `mothilal2020explaining`, `slack2020fooling`,
  `rudin2019stop`
- **HITL / automation bias:** `dearteaga2020case`, `alonbarkat2022human`
- **Individual vs group fairness:** `binns2019apparent`
- **Sociotechnical / label critique:** `selbst2019fairness`,
  `obermeyer2019dissecting`, `ensign2017runaway`

*Generated with `paperhound` across arXiv, OpenAlex, DBLP, Crossref, Semantic
Scholar, and CORE.*
