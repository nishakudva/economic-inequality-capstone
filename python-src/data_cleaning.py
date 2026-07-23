"""
data_cleaning.py
================
Step 1 (Assess) + Step 2 (Clean) of the Adult Census Income investigation.

Loads the raw UCI Adult data, audits it for quality problems, then applies a
documented cleaning pipeline. Every transformation is justified inline so the
pipeline is auditable end-to-end.

Cleaning operations applied to >= 5 variables:
    1. workclass       - '?' sentinel -> NaN -> rows dropped
    2. occupation      - '?' sentinel -> NaN -> rows dropped
    3. native-country  - '?' sentinel -> NaN -> rows dropped
    4. fnlwgt          - sampling-weight column dropped (not a personal attribute)
    5. income          - recoded into numeric binary target `income_binary`
    (+ whitespace stripped across all text columns, duplicate rows removed)

Run standalone:  python src/data_cleaning.py
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

# Data source: GitHub mirror of the UCI Adult training file (adult.data,
# 32,561 rows). Identical content to archive.ics.uci.edu; used because the
# UCI host is not always reachable. Swap RAW_URL for the UCI URL if preferred.
RAW_URL = (
    "https://raw.githubusercontent.com/"
    "saravrajavelu/Adult-Income-Analysis/master/adult_data.txt"
)

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week",
    "native-country", "income",
]

# Directory layout (resolved relative to this file so scripts run from anywhere)
_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def load_raw(url: str = RAW_URL) -> pd.DataFrame:
    """Load the raw dataset and strip whitespace from every text value."""
    df = pd.read_csv(url, header=None, names=COLUMNS, skipinitialspace=True)
    # Defensive strip: some mirrors retain leading spaces on text values.
    df = df.apply(lambda c: c.str.strip() if c.dtype == "object" else c)
    return df


def audit(df: pd.DataFrame) -> dict:
    """Step 1 - surface the quality problems isnull() alone would miss.

    Returns a dict of findings so callers (EDA, reports) can reuse them.
    """
    findings: dict = {}

    # Issue 1 - disguised missing values ('?' sentinel, invisible to isnull()).
    disguised = {}
    for col in df.select_dtypes(include="object").columns:
        n = (df[col].astype(str).str.strip() == "?").sum()
        if n > 0:
            disguised[col] = int(n)
    findings["disguised_missing"] = disguised
    findings["disguised_missing_total"] = int(sum(disguised.values()))

    # Issue 2 - duplicate rows.
    findings["duplicate_rows"] = int(df.duplicated().sum())

    # Issue 3 - fnlwgt is a sampling weight, not a personal attribute.
    findings["note_fnlwgt"] = (
        "fnlwgt is a sampling weight (how many people this row represents), "
        "not a fact about the individual - excluded from modelling."
    )

    # Issue 4 - heavy skew / outliers in capital-gain.
    cg = df["capital-gain"].describe()
    findings["capital_gain_75th"] = float(cg["75%"])
    findings["capital_gain_max"] = float(cg["max"])

    # Issue 5 - class imbalance in the target.
    dist = df["income"].str.strip().value_counts(normalize=True)
    findings["income_distribution"] = {k: round(float(v), 4) for k, v in dist.items()}
    findings["pct_over_50k"] = round(
        float(dist.get(">50K", 0.0)) * 100, 1
    )

    return findings


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2 - the documented cleaning pipeline."""
    df_clean = df.copy()

    # 2a: strip whitespace (export artifact: ' Male' != 'Male' to Python).
    for col in df_clean.select_dtypes(include="object").columns:
        df_clean[col] = df_clean[col].str.strip()

    # 2b: replace the '?' sentinel with real NaN so pandas can see the gaps.
    df_clean = df_clean.replace("?", np.nan)

    # 2c: drop rows missing workclass/occupation/native-country. These may be
    # missing not-at-random (unemployed / never-worked), so this is an
    # analytical choice; documented as a known limitation in the report.
    df_clean = df_clean.dropna(subset=["workclass", "occupation", "native-country"])

    # 2d: drop the sampling-weight column - not a personal attribute.
    df_clean = df_clean.drop(columns=["fnlwgt"])

    # 2e: remove duplicate rows so no observation is over-represented.
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # 2f: numeric binary target - models need numbers, not text.
    df_clean["income_binary"] = (
        df_clean["income"].str.strip() == ">50K"
    ).astype(int)

    return df_clean


def verify(df_clean: pd.DataFrame, n_raw: int) -> None:
    """The notebook's assertion block - fails loudly if cleaning slipped."""
    assert df_clean.isnull().sum().sum() == 0, "Missing values remain"
    assert df_clean.duplicated().sum() == 0, "Duplicate rows remain"
    assert "fnlwgt" not in df_clean.columns, "fnlwgt should be dropped"
    assert df_clean["income_binary"].isin([0, 1]).all(), "target must be 0/1"
    kept = len(df_clean)
    print("All checks passed. The dataset is clean.")
    print(f"Rows kept: {kept} ({kept / n_raw * 100:.1f}% of original)")


def main() -> pd.DataFrame:
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_raw()
    print(f"Raw shape: {df.shape[0]} rows x {df.shape[1]} columns")

    findings = audit(df)
    print("\n--- Data Audit ---")
    print("Disguised '?' missing values:", findings["disguised_missing"])
    print("Duplicate rows:", findings["duplicate_rows"])
    print("capital-gain 75th pct vs max:",
          findings["capital_gain_75th"], "->", findings["capital_gain_max"])
    print("Income distribution:", findings["income_distribution"])
    print(f"% earning >50K: {findings['pct_over_50k']}%")

    df_clean = clean(df)
    print(f"\nCleaned shape: {df_clean.shape}")
    verify(df_clean, len(df))

    raw_path = os.path.join(DATA_DIR, "adult_raw.csv")
    clean_path = os.path.join(DATA_DIR, "adult_clean.csv")
    df.to_csv(raw_path, index=False)
    df_clean.to_csv(clean_path, index=False)
    print(f"\nSaved: {raw_path}")
    print(f"Saved: {clean_path}")
    return df_clean


if __name__ == "__main__":
    main()
