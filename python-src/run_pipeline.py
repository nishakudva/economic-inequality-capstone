"""
run_pipeline.py
===============
One-shot orchestrator: runs the full investigation end to end and writes every
artifact (cleaned data, figures, model results, fairness table) to disk.

Usage:
    python src/run_pipeline.py

Stages:
    1. Clean   -> data/adult_raw.csv, data/adult_clean.csv
    2. EDA     -> printed findings
    3. Visuals -> figures/chart1..6_*.png
    4. Models  -> reports/model_results.csv, figures/chart7_model_comparison.png
    5. Fairness-> reports/fairness_by_sex.csv
"""
from __future__ import annotations

import data_cleaning
import eda
import visualization
import modeling


def run() -> None:
    print("\n" + "=" * 60 + "\nSTAGE 1/4  DATA CLEANING\n" + "=" * 60)
    df_clean = data_cleaning.main()

    print("\n" + "=" * 60 + "\nSTAGE 2/4  EXPLORATORY DATA ANALYSIS\n" + "=" * 60)
    eda.explore(df_clean)
    eda.main()  # prints full findings

    print("\n" + "=" * 60 + "\nSTAGE 3/4  VISUALIZATION\n" + "=" * 60)
    visualization.generate_all(df_clean)

    print("\n" + "=" * 60 + "\nSTAGE 4/4  MODELLING + FAIRNESS\n" + "=" * 60)
    modeling.main()

    print("\nPipeline complete. See data/, figures/, and reports/.")


if __name__ == "__main__":
    run()
