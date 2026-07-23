"""Exploratory analysis of the COMPAS dataset (research questions 1-3).

Produces figures in figures/ and a written report in reports/02_eda.md
answering:
  RQ1 - Is the dataset representative and what does it represent?
  RQ2 - Does it reflect historical and institutional inequalities?
  RQ3 - What demographic disparities exist in the dataset?
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
from common import FIGURES_DIR, INK_SECONDARY, MUTED, REPORTS_DIR, SEQUENTIAL, SERIES

MAIN_GROUPS = ["African-American", "Caucasian", "Hispanic"]


def fig_demographics(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))

    race = df.race.value_counts()
    axes[0].barh(race.index[::-1], race.values[::-1], color=SERIES[0], height=0.62)
    axes[0].set_title("Defendants by race")
    axes[0].grid(axis="x")
    axes[0].grid(axis="y", visible=False)
    for i, v in enumerate(race.values[::-1]):
        axes[0].text(v, i, f" {v:,}", va="center", fontsize=8, color=INK_SECONDARY)

    sex = df.sex.value_counts()
    axes[1].bar(sex.index, sex.values, color=SERIES[0], width=0.55)
    axes[1].set_title("Defendants by sex")
    for i, v in enumerate(sex.values):
        axes[1].text(i, v, f"{v:,}\n({v / len(df):.0%})", ha="center", va="bottom",
                     fontsize=8, color=INK_SECONDARY)
    axes[1].set_ylim(0, sex.values[0] * 1.25)

    axes[2].hist(df.age, bins=np.arange(18, 80, 2), color=SERIES[0])
    axes[2].set_title("Age distribution")
    axes[2].set_xlabel("Age at screening")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_demographics.png", bbox_inches="tight")
    plt.close(fig)


def fig_decile_by_race(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 3.4))
    deciles = np.arange(1, 11)
    width = 0.38
    for i, group in enumerate(["African-American", "Caucasian"]):
        sub = df[df.race == group]
        share = sub.decile_score.value_counts(normalize=True).reindex(deciles, fill_value=0)
        offset = (i - 0.5) * width
        ax.bar(deciles + offset, share.values, width=width - 0.04,
               color=SERIES[i], label=group)
    ax.set_xticks(deciles)
    ax.set_xlabel("COMPAS decile score")
    ax.set_ylabel("Share of group")
    ax.set_title("Distribution of COMPAS decile scores by race")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_decile_by_race.png", bbox_inches="tight")
    plt.close(fig)


def fig_score_text_by_race(df: pd.DataFrame) -> None:
    levels = ["Low", "Medium", "High"]
    groups = MAIN_GROUPS
    shares = np.array([
        [
            (df[df.race == g].score_text == level).mean()
            for level in levels
        ]
        for g in groups
    ])
    fig, ax = plt.subplots(figsize=(7, 3.0))
    left = np.zeros(len(groups))
    for j, level in enumerate(levels):
        vals = shares[:, j]
        ax.barh(groups[::-1], vals[::-1], left=left[::-1], color=SEQUENTIAL[j],
                label=level, height=0.55, edgecolor=common.SURFACE, linewidth=2)
        for i, g in enumerate(groups[::-1]):
            v = vals[::-1][i]
            if v > 0.08:
                ax.text(left[::-1][i] + v / 2, i, f"{v:.0%}", ha="center",
                        va="center", fontsize=8, color="#0b0b0b" if j == 0 else "#ffffff")
        left += vals
    ax.set_xlim(0, 1)
    ax.set_title("COMPAS risk category by race")
    ax.grid(visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_score_category_by_race.png", bbox_inches="tight")
    plt.close(fig)


def error_rates(df: pd.DataFrame, group: str) -> tuple[float, float]:
    """FPR and FNR of the binary COMPAS classification (Medium/High = positive)
    against observed two-year recidivism, within a racial group."""
    sub = df[df.race == group]
    fp = ((sub.score_binary == 1) & (sub.two_year_recid == 0)).sum()
    tn = ((sub.score_binary == 0) & (sub.two_year_recid == 0)).sum()
    fn = ((sub.score_binary == 0) & (sub.two_year_recid == 1)).sum()
    tp = ((sub.score_binary == 1) & (sub.two_year_recid == 1)).sum()
    return fp / (fp + tn), fn / (fn + tp)


def fig_error_rates(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    rates = {g: error_rates(df, g) for g in ["African-American", "Caucasian"]}
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(2)
    width = 0.34
    for i, (g, (fpr, fnr)) in enumerate(rates.items()):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, [fpr, fnr], width=width - 0.04, color=SERIES[i], label=g)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{b.get_height():.0%}", ha="center", va="bottom",
                    fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels([
        "False positive rate\n(labelled risky, did not recidivate)",
        "False negative rate\n(labelled low risk, did recidivate)",
    ])
    ax.set_title("COMPAS error rates by race (Medium/High = positive)")
    ax.set_ylim(0, 0.62)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "02_error_rates.png", bbox_inches="tight")
    plt.close(fig)
    return rates


def main() -> None:
    common.apply_plot_style()
    FIGURES_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    raw = common.load_raw()
    df = common.load_filtered()

    fig_demographics(df)
    fig_decile_by_race(df)
    fig_score_text_by_race(df)
    rates = fig_error_rates(df)

    race_counts = df.race.value_counts()
    race_share = df.race.value_counts(normalize=True)
    sex_share = df.sex.value_counts(normalize=True)
    recid_by_race = df.groupby("race").two_year_recid.mean()
    score_by_race = df.groupby("race").decile_score.mean()
    aa_fpr, aa_fnr = rates["African-American"]
    c_fpr, c_fnr = rates["Caucasian"]

    # Broward County demographics (2013 US Census estimates) for comparison.
    broward_black_share = 0.28

    report = f"""# Exploratory analysis of the COMPAS dataset

Dataset: `compas-scores-two-years.csv` from ProPublica's
[compas-analysis](https://github.com/propublica/compas-analysis) repository.
Raw file: **{len(raw):,} defendants, {raw.shape[1]} variables**. After applying
ProPublica's cohort filter (screening within ±30 days of arrest, valid COMPAS
record, no ordinary traffic offenses, non-missing score):
**{len(df):,} defendants**.

## RQ1 — Is the dataset representative and what does it represent?

**What it represents.** Every row is a *criminal defendant* screened with the
COMPAS risk-assessment tool in **Broward County, Florida, in 2013–2014**, joined
with their criminal history and a two-year follow-up flag (`two_year_recid`)
indicating whether the person was **re-arrested** within two years. Key variable
groups:

- demographics: `sex`, `age`, `age_cat`, `race`
- criminal history: `priors_count`, `juv_fel_count`, `juv_misd_count`, `juv_other_count`
- current case: `c_charge_degree` (felony/misdemeanor), `c_charge_desc`
- COMPAS outputs: `decile_score` (1–10), `score_text` (Low/Medium/High)
- outcome: `is_recid`, `two_year_recid`

**What it does not represent.** The outcome label is *re-arrest*, not
*re-offense*: crimes that never lead to an arrest are invisible, and arrest
intensity varies across neighborhoods and demographic groups. The data covers a
single county, a single two-year window, and only people who were arrested and
screened — it is **not representative of crime**, of the US, or even of Broward
County's general population. For comparison, African-Americans were roughly
{broward_black_share:.0%} of Broward County's population in 2013 but are
**{race_share['African-American']:.0%} of the defendants** in this dataset.

**Why it is used in AI-fairness research.** ProPublica's 2016 investigation
("Machine Bias", Angwin et al.) made it the canonical public benchmark: it
contains a deployed algorithm's actual scores, ground-truth follow-up, and
sensitive attributes, which is a rare combination that allows fairness metrics
to be computed openly.

## RQ2 — Does the dataset reflect historical and institutional inequalities?

The dataset is a record of **criminal justice decisions**, not of criminal
behavior. Each step of the funnel that produced a row involves discretionary
human decisions in which documented disparities exist:

1. **Policing bias** — who gets stopped, searched, and arrested. Arrest data
   over-represents heavily policed (disproportionately Black and low-income)
   neighborhoods, and this same mechanism generates the `two_year_recid` label.
2. **Charging and sentencing bias** — `c_charge_degree` and `priors_count`
   reflect prosecutorial and judicial choices, not just conduct.
3. **Socioeconomic inequality** — priors accumulate faster where people cannot
   afford bail, diversion programs, or private counsel; the count then feeds
   back into future risk scores.
4. **Feedback loops** — a high risk score can lead to detention, job loss and
   heavier surveillance, raising the probability of future *arrest* and
   apparently "confirming" the score.

Consequently, a model trained on this data learns the behavior of the Broward
County criminal justice system as much as the behavior of defendants.

## RQ3 — What demographic disparities exist in the dataset?

![Demographics](../figures/02_demographics.png)

- **Race:** African-American {race_counts['African-American']:,}
  ({race_share['African-American']:.1%}), Caucasian {race_counts['Caucasian']:,}
  ({race_share['Caucasian']:.1%}), Hispanic {race_counts['Hispanic']:,}
  ({race_share['Hispanic']:.1%}); Asian and Native American together are under 1%.
- **Sex:** {sex_share['Male']:.1%} male, {sex_share['Female']:.1%} female.
- **Age:** strongly right-skewed; median {df.age.median():.0f} years, with the
  bulk of defendants between 25 and 45.

### Risk score distribution

![Decile scores by race](../figures/02_decile_by_race.png)

![Risk categories by race](../figures/02_score_category_by_race.png)

Caucasian defendants' decile scores pile up at the low end (mean decile
{score_by_race['Caucasian']:.1f}), while African-American defendants' scores are
close to uniform across deciles (mean {score_by_race['African-American']:.1f}).
{(df[df.race == 'African-American'].score_text != 'Low').mean():.0%} of
African-American defendants are rated Medium or High risk versus
{(df[df.race == 'Caucasian'].score_text != 'Low').mean():.0%} of Caucasian
defendants. Observed two-year re-arrest rates differ too
({recid_by_race['African-American']:.0%} vs {recid_by_race['Caucasian']:.0%}),
but — as RQ2 argues — the label itself is generated by the same unequally
distributed enforcement.

### Error rates: the core ProPublica finding

![Error rates](../figures/02_error_rates.png)

Treating Medium/High as a positive prediction of recidivism:

| Group | False positive rate | False negative rate |
|-------|--------------------:|--------------------:|
| African-American | **{aa_fpr:.1%}** | {aa_fnr:.1%} |
| Caucasian | {c_fpr:.1%} | **{c_fnr:.1%}** |

African-American defendants who did **not** recidivate were flagged as risky
about **{aa_fpr / c_fpr:.1f}× as often** as Caucasian defendants; Caucasian
defendants who **did** recidivate were mislabelled low-risk about
{c_fnr / aa_fnr:.1f}× as often. The errors are asymmetric in direction: they
harm Black defendants (excess detention) and favor white defendants (excess
leniency). This asymmetry is the benchmark that the rest of the project tries
to detect, explain, and mitigate.
"""
    (REPORTS_DIR / "02_eda.md").write_text(report)
    print(f"Filtered cohort: {len(df):,} rows")
    print(f"AA FPR {aa_fpr:.3f} / FNR {aa_fnr:.3f}; Cauc FPR {c_fpr:.3f} / FNR {c_fnr:.3f}")
    print("Wrote reports/02_eda.md and 4 figures")


if __name__ == "__main__":
    main()
