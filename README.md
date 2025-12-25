# NHS Chronic ACSC Admission Rates Analysis (England, 2003/04–2023/24)

**Python • Time Series Analysis • Health Data • NHS Policy Research**

![NHS Logo](https://upload.wikimedia.org/wikipedia/commons/1/1b/NHS-Logo.svg)

This repository contains an analytical study of **unplanned hospital admission rates for chronic Ambulatory Care Sensitive Conditions (ACSCs) in England** over a 21-year period (2003/04–2023/24).

The project investigates **long-term trends, demographic inequalities, and short-term volatility**, and develops **forecasting models** to support NHS service planning and policy evaluation.

---

## Research Question

> **To what extent have unplanned hospital admission rates for chronic ambulatory care sensitive conditions in England changed between 2003/04 and 2023/24, and how do these trends vary by age, gender, and deprivation level?**

---

## Project Objectives

1. Measure long-term changes in chronic ACSC admission rates across England.  
2. Examine **age-specific trends**, with a focus on older populations.  
3. Assess **sex-based differences** in admission rates over time.  
4. Analyse **deprivation-related inequalities** in admissions.  
5. Develop and evaluate **time-series models** to project future admission trends.

---

## Tech Stack

- **Language:** Python  
- **Data Analysis:** pandas, numpy  
- **Visualisation:** matplotlib, seaborn, plotly  
- **Time Series Modelling:** statsmodels / Prophet  
- **Environment:** Jupyter Notebook  
---

## Dataset

**Source:**  
NHS Digital – *NHS Outcomes Framework Indicator 2.3.i: Unplanned hospitalisation for chronic ACSCs* (2025)
(https://digital.nhs.uk/data-and-information/publications/statistical/nhs-outcomes-framework/february-2025)

### Dataset Justification
- **Credibility:** Official NHS data with high policy relevance  
- **Temporal Coverage:** 21 consecutive financial years (2003/04–2023/24)  
- **Demographic Detail:** Age group, sex, deprivation decile, and region  
- **Standardised Metrics:** Admission rates per 100,000 population  
- **Statistical Reliability:** Includes confidence intervals for uncertainty assessment  

---

## Feature Engineering

To enhance interpretability and analytical depth, the following derived features were created:

- **`financial_year`**  
  Human-readable year labels (e.g., *2023/24*) derived from `year_start` for clearer time-series visualisation.

- **`ci_width`**  
  Difference between upper and lower confidence intervals, capturing statistical uncertainty.

- **`high_uncertainty`**  
  Binary flag identifying the top 10% of CI widths, highlighting less reliable estimates.

**Application in Analysis:**  
High-uncertainty observations are visually highlighted and interpreted cautiously when comparing trends across age, sex, and deprivation groups.

---

## Exploratory Data Analysis (EDA)

Exploratory data analysis was conducted using **Python (pandas, matplotlib, seaborn, plotly)** to examine temporal dynamics and demographic variation in chronic ACSC admission rates.

### Key Findings

- **Overall Trend:**  
  Admission rates exhibit a pronounced **“sawtooth” pattern**, with sharp annual rises and falls.

- **Long-Term Change:**  
  Despite a mild downward trend, **2023/24 admission rates returned to levels comparable to 2003/04**, indicating limited net improvement.

- **COVID-19 Impact:**  
  A substantial temporary reduction occurred in **2020/21**, reflecting reduced healthcare utilisation during the COVID-19 pandemic.

- **Demographic Inequalities:**  
  Persistently higher admission rates were observed among:
  - Older age groups (75+)  
  - More deprived populations  
  - Males (slightly higher than females over time)

---

## Year-on-Year Volatility Analysis

Year-to-year percentage change analysis revealed significant volatility:

- Annual changes frequently exceeding **±5%**  
- Increased volatility **post-2010** and during the **post-pandemic recovery**  
- The largest decline (≈ **–10%**) occurred in **2020–2021**  
- Pre-2010 years showed relatively stable fluctuations (–5% to +5%)

This approach:
- Quantifies short-term system shocks (e.g., COVID-19)
- Validates the observed “sawtooth” pattern beyond smoothed trends
- Highlights pressures masked by long-term averages

---

## Modelling & Forecasting

Time-series models were developed to project future chronic ACSC admission trends, incorporating:

- Temporal train/test splits  
- Performance evaluation using RMSE and MAE  
- Visualised confidence intervals  
- Comparison with baseline models  

These methods improve robustness and policy relevance.

---

