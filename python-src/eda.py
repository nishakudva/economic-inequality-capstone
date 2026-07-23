"""
eda.py
======
Exploratory Data Analysis on the cleaned Adult Census Income data.

Explores >= 5 variables and their relationship to the income target:
    1. age
    2. education-num
    3. hours-per-week
    4. sex
    5. race
    6. marital-status
    7. capital-gain / capital-loss

Produces a plain-text summary to stdout and returns a findings dict that the
report generator reuses. Run standalone:  python src/eda.py
"""
from __future__ import annotations

import os
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CLEAN_PATH = os.path.join(DATA_DIR, "adult_clean.csv")


def load_clean(path: str = CLEAN_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run data_cleaning.py first."
        )
    return pd.read_csv(path)


def explore(df: pd.DataFrame) -> dict:
    """Return a structured dict of EDA findings across >= 5 variables."""
    out: dict = {}

    # Overall base rate for reference.
    out["overall_over50k_rate"] = round(df["income_binary"].mean(), 4)

    # 1. age - numeric summary + rate by decade band.
    out["age_summary"] = df["age"].describe().round(2).to_dict()
    age_bins = pd.cut(df["age"], [16, 25, 35, 45, 55, 65, 100],
                      labels=["17-25", "26-35", "36-45", "46-55", "56-65", "65+"])
    out["over50k_by_age_band"] = (
        df.groupby(age_bins, observed=True)["income_binary"].mean().round(3).to_dict()
    )

    # 2. education-num - correlation with target + rate by level.
    out["educnum_corr_income"] = round(
        df["education-num"].corr(df["income_binary"]), 3
    )
    out["over50k_by_education"] = (
        df.groupby("education")["income_binary"].mean().sort_values(ascending=False)
        .round(3).head(6).to_dict()
    )

    # 3. hours-per-week - summary + rate for full-time vs long hours.
    out["hours_summary"] = df["hours-per-week"].describe().round(2).to_dict()
    hb = pd.cut(df["hours-per-week"], [0, 34, 40, 60, 100],
                labels=["<35 (part-time)", "35-40", "41-60", "60+"])
    out["over50k_by_hours_band"] = (
        df.groupby(hb, observed=True)["income_binary"].mean().round(3).to_dict()
    )

    # 4. sex - the disparity headline.
    out["over50k_by_sex"] = (
        df.groupby("sex")["income_binary"].mean().round(3).to_dict()
    )

    # 5. race.
    out["over50k_by_race"] = (
        df.groupby("race")["income_binary"].mean().sort_values(ascending=False)
        .round(3).to_dict()
    )

    # 6. marital-status - strong single-variable signal.
    out["over50k_by_marital"] = (
        df.groupby("marital-status")["income_binary"].mean()
        .sort_values(ascending=False).round(3).to_dict()
    )

    # 7. capital-gain - share with any gain, and its association.
    out["pct_with_capital_gain"] = round((df["capital-gain"] > 0).mean() * 100, 1)
    out["over50k_when_has_capgain"] = round(
        df.loc[df["capital-gain"] > 0, "income_binary"].mean(), 3
    )

    return out


def _print_findings(f: dict) -> None:
    print("=== EDA Findings ===")
    print(f"Overall >50K rate: {f['overall_over50k_rate']:.1%}\n")
    print("Age band -> P(>50K):", f["over50k_by_age_band"])
    print("education-num vs income correlation:", f["educnum_corr_income"])
    print("Top education levels -> P(>50K):", f["over50k_by_education"])
    print("Hours band -> P(>50K):", f["over50k_by_hours_band"])
    print("Sex -> P(>50K):", f["over50k_by_sex"])
    print("Race -> P(>50K):", f["over50k_by_race"])
    print("Marital status -> P(>50K):", f["over50k_by_marital"])
    print(f"Share with capital-gain > 0: {f['pct_with_capital_gain']}% "
          f"(of those, {f['over50k_when_has_capgain']:.1%} earn >50K)")


def main() -> dict:
    df = load_clean()
    findings = explore(df)
    _print_findings(findings)
    return findings


if __name__ == "__main__":
    main()
