"""Download the COMPAS dataset from ProPublica's compas-analysis repository.

Source: https://github.com/propublica/compas-analysis
The main file used throughout this project is `compas-scores-two-years.csv`,
the cohort ProPublica used for its two-year general recidivism analysis
(Broward County, Florida, 2013-2014).
"""

from pathlib import Path

import requests

BASE_URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master"
FILES = [
    "compas-scores-two-years.csv",
    "compas-scores-two-years-violent.csv",
]

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = DATA_DIR / name
        if dest.exists():
            print(f"[skip] {name} already exists")
            continue
        url = f"{BASE_URL}/{name}"
        print(f"[download] {url}")
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
        print(f"[saved] {dest} ({len(response.content):,} bytes)")


if __name__ == "__main__":
    main()
