"""
visualization.py
================
Step 3 (Visualize). Produces 6 publication-quality charts (>= 5 required),
each with labelled axes, a finding-based title, and saved as a PNG in figures/.

    1. Histogram        - distribution of hours worked per week
    2. Bar chart        - P(>50K) by education level
    3. Line chart       - P(>50K) across age
    4. Grouped bar      - P(>50K) by education, split by sex
    5. Correlation heatmap - numeric features vs income
    6. Box plot         - age distribution by income class

Run standalone:  python src/visualization.py
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")  # headless backend - no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from eda import load_clean

sns.set_palette("colorblind")
sns.set_style("whitegrid")

_HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_HERE)
FIG_DIR = os.path.join(PROJECT_ROOT, "figures")

DPI = 150


def _save(fig, name: str) -> str:
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")
    return path


def chart1_hours_histogram(df) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x="hours-per-week", bins=30, ax=ax)
    ax.axvline(40, color="crimson", ls="--", lw=1, label="40 h (full-time)")
    ax.set_xlabel("Hours worked per week")
    ax.set_ylabel("Number of people")
    ax.set_title("Most people cluster at exactly 40 hours, "
                 "but a long right tail works far more")
    ax.legend()
    return _save(fig, "chart1_hours_histogram.png")


def chart2_income_by_education(df) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    s = df.groupby("education")["income_binary"].mean().sort_values()
    sns.barplot(x=s.values, y=s.index, ax=ax)
    ax.set_xlabel("Proportion earning >$50K")
    ax.set_ylabel("Education level")
    ax.set_title("Earning >$50K rises sharply with education: "
                 "from near 0 to over 70% at doctorate level")
    return _save(fig, "chart2_income_by_education.png")


def chart3_income_by_age(df) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    s = df.groupby("age")["income_binary"].mean()
    sns.lineplot(x=s.index, y=s.values, ax=ax)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Proportion earning >$50K")
    ax.set_title("The chance of earning >$50K climbs until the "
                 "late 40s-50s, then declines toward retirement")
    return _save(fig, "chart3_income_by_age.png")


def chart4_income_edu_by_sex(df) -> str:
    fig, ax = plt.subplots(figsize=(11, 6))
    order = ["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate"]
    subset = df[df["education"].isin(order)]
    sns.barplot(data=subset, x="education", y="income_binary",
                hue="sex", order=order, ax=ax)
    ax.set_xlabel("Education level")
    ax.set_ylabel("Proportion earning >$50K")
    ax.set_title("At every education level, men earn >$50K at a "
                 "markedly higher rate than women")
    ax.legend(title="Sex")
    return _save(fig, "chart4_income_edu_by_sex.png")


def chart5_correlation_heatmap(df) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    num_cols = ["age", "education-num", "hours-per-week",
                "capital-gain", "capital-loss", "income_binary"]
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, cbar_kws={"label": "Pearson correlation"}, ax=ax)
    ax.set_title("education-num and hours-per-week are the strongest "
                 "numeric correlates of high income")
    return _save(fig, "chart5_correlation_heatmap.png")


def chart6_age_box_by_income(df) -> str:
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x="income", y="age", order=["<=50K", ">50K"], ax=ax)
    ax.set_xlabel("Income class")
    ax.set_ylabel("Age (years)")
    ax.set_title("People earning >$50K are older on average, "
                 "with a higher median and narrower young tail")
    return _save(fig, "chart6_age_box_by_income.png")


def generate_all(df=None) -> list[str]:
    if df is None:
        df = load_clean()
    return [
        chart1_hours_histogram(df),
        chart2_income_by_education(df),
        chart3_income_by_age(df),
        chart4_income_edu_by_sex(df),
        chart5_correlation_heatmap(df),
        chart6_age_box_by_income(df),
    ]


if __name__ == "__main__":
    generate_all()
