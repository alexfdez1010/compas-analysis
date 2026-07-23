"""Shared data loading, filtering, and plotting style for the COMPAS analysis."""

from pathlib import Path

import matplotlib as mpl
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "compas-scores-two-years.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "figures"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

# Validated categorical palette (light mode), assigned in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
# Sequential blue ramp (ordinal use starts no lighter than step 250).
SEQUENTIAL = ["#86b6ef", "#3987e5", "#1c5cab"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Features used for modeling. `race` and `sex` are retained in the dataframe
# as sensitive attributes for auditing, and their use as model inputs is an
# explicit, discussed choice in each script.
FEATURES_NUMERIC = [
    "age",
    "priors_count",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
]
FEATURE_CHARGE = "c_charge_degree"  # F (felony) / M (misdemeanor)
SENSITIVE = ["race", "sex"]
TARGET = "two_year_recid"


def apply_plot_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.labelcolor": INK_SECONDARY,
            "text.color": INK,
            "axes.titlecolor": INK,
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 11,
            "figure.dpi": 150,
        }
    )


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)


def load_filtered() -> pd.DataFrame:
    """Apply ProPublica's cohort filter for the two-year recidivism analysis.

    - COMPAS screening within 30 days of arrest (otherwise the score may not
      relate to the charge in the record),
    - `is_recid != -1` (cases with no COMPAS record found),
    - ordinary traffic offenses (`c_charge_degree == 'O'`) excluded,
    - rows with a missing score removed.
    """
    df = load_raw()
    df = df[
        (df.days_b_screening_arrest <= 30)
        & (df.days_b_screening_arrest >= -30)
        & (df.is_recid != -1)
        & (df.c_charge_degree != "O")
        & (df.score_text != "N/A")
    ].copy()
    df["score_binary"] = (df.score_text != "Low").astype(int)  # Medium/High = 1
    return df.reset_index(drop=True)


RACE_DUMMIES = [
    "race_african_american",
    "race_caucasian",
    "race_hispanic",
    "race_other",
]


def build_features(df: pd.DataFrame, include_race: bool = True) -> pd.DataFrame:
    """Feature matrix for modeling.

    `include_race` is an explicit, documented choice: the baseline ("biased")
    model keeps race so that its influence can be exposed and measured; the
    de-biased pipeline handles it differently.
    """
    X = df[FEATURES_NUMERIC].copy()
    X["charge_felony"] = (df[FEATURE_CHARGE] == "F").astype(int)
    X["sex_male"] = (df.sex == "Male").astype(int)
    if include_race:
        X["race_african_american"] = (df.race == "African-American").astype(int)
        X["race_caucasian"] = (df.race == "Caucasian").astype(int)
        X["race_hispanic"] = (df.race == "Hispanic").astype(int)
        X["race_other"] = (~df.race.isin(
            ["African-American", "Caucasian", "Hispanic"]
        )).astype(int)
    return X


def train_test_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The persisted train/test split (rows of the filtered cohort)."""
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    return train, test
