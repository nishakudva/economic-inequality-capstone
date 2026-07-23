# Adult Census Income: End-to-End Data Investigation

Who earns above vs. below **$50,000/year**, and where that story gets complicated by
sex, education, and age. An end-to-end analysis of the **UCI Adult Census Income**
dataset (US Census Bureau, 1994; 32,561 records) covering the four capstone pillars:
**data cleaning, exploratory data analysis, visualization, and machine learning.**

> ⚠️ This is real 1994 US Census data. The patterns describe real inequities for real
> people, many of which persist today. Findings are historical and should motivate
> current measurement, not stand in for it.

---

## Headline findings

| Finding | Evidence |
|---|---|
| Only **24.1%** of people earn >$50K | 3-to-1 class imbalance — accuracy alone is misleading |
| Income rises steeply with **education** | ~0% (no diploma) → ~74% (doctorate/prof-school) |
| Income is a **life-cycle** effect | Peaks in the late 40s–50s, then declines |
| **Sex gap persists at every education level** | Men 32.3% vs women 12.1% earn >$50K overall |
| Best model is accurate but **not equal** | Gradient Boosting: 84.1% accuracy, yet recall on true high earners is 35.1% (women) vs 54.6% (men) |

---

## Repository structure

```
census-income-project/
├── README.md
├── data/
│   ├── adult_raw.csv          # raw pull, whitespace-stripped (32,561 rows)
│   └── adult_clean.csv        # cleaned, model-ready (26,904 rows)
├── python-src/
│   ├── data_cleaning.py       # Step 1 Assess + Step 2 Clean
│   ├── eda.py                 # Exploratory analysis over 7 variables
│   ├── visualization.py       # 6 publication-quality charts
│   ├── modeling.py            # 11 models + fairness check by sex
│   └── run_pipeline.py        # runs the whole thing end to end
├── graphs/
│   ├── chart1_hours_histogram.png
│   ├── chart2_income_by_education.png
│   ├── chart3_income_by_age.png
│   ├── chart4_income_edu_by_sex.png
│   ├── chart5_correlation_heatmap.png
│   ├── chart6_age_box_by_income.png
│   └── chart7_model_comparison.png
└── Nisha_CensusIncome_executed.ipynb      # Notebook updated with executed results from collab
└── Census_Income capstone project Q&A.html  # Q&N for the listed questions during the class
└── Main_Conclusions.html                 # Main overall conclusion from my project work.
```

---

## Responsible-use note

The best model reaches 84% accuracy but identifies high-earning women at a much lower
rate than men. **Do not deploy this model, or one trained on data like it, to screen
real people for loans, credit, insurance, or benefits.** It reflects a 1994 labour market
and encodes its inequities; averaged accuracy hides the disparity that a per-group recall
check reveals.

## License

Analysis code released for educational use only by Nisha Kudva. The UCI Adult dataset is distributed under the terms of the UCI Machine Learning Repository.
