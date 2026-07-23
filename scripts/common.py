"""Shared data loading, filtering, and plotting style for the COMPAS analysis."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

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


CONFUSION_LABELS = ["No recid", "Recid"]
_CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", [SURFACE, SEQUENTIAL[-1]])


def confusion_counts(y_true, y_pred) -> "np.ndarray":
    """Raw 2x2 count matrix ``[[TN, FP], [FN, TP]]`` (rows = true label)."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    cm = np.zeros((2, 2), dtype=int)
    for t in (0, 1):
        for p in (0, 1):
            cm[t, p] = int(np.sum((yt == t) & (yp == p)))
    return cm


def _draw_confusion(ax, cm, title=None, show_y=True) -> None:
    """Render one count matrix onto ``ax`` as a row-share-shaded heatmap.

    Cells are shaded by their share of the row (so the diagonal reads as
    per-class accuracy regardless of class imbalance) and annotated with the raw
    count and that row share.
    """
    row_share = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(row_share, cmap=_CONFUSION_CMAP, vmin=0, vmax=1, aspect="equal")
    for t in (0, 1):
        for p in (0, 1):
            ax.text(p, t - 0.10, f"{cm[t, p]:,}", ha="center", va="center",
                    fontsize=13, fontweight="bold",
                    color=INK if row_share[t, p] < 0.55 else SURFACE)
            ax.text(p, t + 0.18, f"{row_share[t, p]:.0%}", ha="center", va="center",
                    fontsize=9,
                    color=INK_SECONDARY if row_share[t, p] < 0.55 else SURFACE)
    ax.set_xticks([0, 1], CONFUSION_LABELS)
    ax.set_xlabel("Predicted")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    if show_y:
        ax.set_yticks([0, 1], CONFUSION_LABELS)
        ax.set_ylabel("Actual")
    else:
        ax.set_yticks([0, 1], ["", ""])
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title is not None:
        ax.set_title(title, pad=10)


def plot_confusion_by_group(y_true, y_pred, groups, order, suptitle,
                            path) -> dict:
    """Save a row of confusion matrices, one per group in ``order``.

    All panels share the row-share colour scale, so the darker false-positive
    cell for one group vs. another is directly comparable - this is where the
    racial error-rate asymmetry becomes visible. Returns ``{group: count matrix}``.
    """
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    grp = np.asarray(groups)
    mats = {g: confusion_counts(yt[grp == g], yp[grp == g]) for g in order}

    fig, axes = plt.subplots(1, len(order), figsize=(3.5 * len(order), 3.7))
    for i, (ax, g) in enumerate(zip(axes, order)):
        _draw_confusion(ax, mats[g], title=g, show_y=(i == 0))
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return mats


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
