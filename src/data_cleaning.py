import pandas as pd
import numpy as np

# LOAD DATA
df = pd.read_excel(
    "../data/raw/NHSOF_2.3.i_I00708_D.xlsx",
    sheet_name="Indicator data",
    engine="openpyxl",
    skiprows=14,
)

# BASIC DATA OVERVIEW
print(df.info())
print(df.head())

# SAVING A COPY OF THE DATASET
df_before_cleaning = df.copy()

# TEXT STANDARDIZATION
df.columns = df.columns.str.strip()
object_cols = df.select_dtypes(include="object").columns
for col in object_cols:
    df[col] = df[col].astype(str).str.strip()

df.columns = df.columns.str.lower().str.replace(" ", "_")

# STANDARDIZE TEXT VALUES
text_columns = ["year", "breakdown", "level_description"]
for col in text_columns:
    if col in df.columns:
        df[col] = df[col].apply(lambda x: str(x).strip().lower() if pd.notna(x) else x)

# FEATURE ENGINEERING
df["year_start"] = df["year"].str.split("/").str[0]
df["year_start"] = pd.to_numeric(df["year_start"], errors="coerce")

# DATA QUALITY CHECKS & CLEANING
critical_cols = ["year", "breakdown", "level_description", "indicator_value"]
df_clean = df.dropna(subset=critical_cols)

numeric_columns = [
    "indicator_value",
    "lower_ci",
    "upper_ci",
    "standardised_ratio",
    "observed",
    "population",
    "percent_unclassified",
    "expected",
]

for col in numeric_columns:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(
            df_clean[col].replace("*", np.nan), errors="coerce"
        )

# Remove invalid confidence intervals
if all(col in df_clean.columns for col in ["lower_ci", "indicator_value", "upper_ci"]):
    invalid_ci = (df_clean["lower_ci"] > df_clean["indicator_value"]) | (
        df_clean["indicator_value"] > df_clean["upper_ci"]
    )
    df_clean = df_clean[~invalid_ci].copy()

# HANDLE MISSING VALUES
# 1. Drop rows with missing critical numeric columns
critical_numeric = ["indicator_value", "lower_ci", "upper_ci"]
df_clean = df_clean.dropna(
    subset=[col for col in critical_numeric if col in df_clean.columns]
)

# 2. Impute non-critical numeric columns by median per breakdown
for col in ["standardised_ratio", "observed"]:
    if col in df_clean.columns:
        df_clean[col] = df_clean.groupby("breakdown")[col].transform(
            lambda x: x.fillna(x.median())
        )

# 3. Fill percent_unclassified with 0 (or any other meaningful value)
if "percent_unclassified" in df_clean.columns:
    df_clean["percent_unclassified"] = df_clean["percent_unclassified"].fillna(0)

# COLUMN MANAGEMENT
columns_to_drop = [
    "period_of_coverage",
    "level",
    "quarter",
    "standardised_ratio_lower_ci",
    "standardised_ratio_upper_ci",
    "expected",
]
df_clean.drop(
    columns=[col for col in columns_to_drop if col in df_clean.columns], inplace=True
)

# DERIVED FEATURES
if "year_start" in df_clean.columns:
    df_clean["financial_year"] = (
        df_clean["year_start"].astype(int).astype(str)
        + "/"
        + (df_clean["year_start"].astype(int) + 1).astype(str).str[-2:]
    )

if all(col in df_clean.columns for col in ["lower_ci", "upper_ci"]):
    df_clean["ci_width"] = df_clean["upper_ci"] - df_clean["lower_ci"]

if "ci_width" in df_clean.columns:
    threshold = df_clean["ci_width"].quantile(0.9)
    df_clean["high_uncertainty"] = (df_clean["ci_width"] > threshold).astype(int)

# SORTING
sort_cols = ["year_start", "breakdown"]
if "level_description" in df_clean.columns:
    sort_cols.append("level_description")
df_clean = df_clean.sort_values(by=sort_cols).reset_index(drop=True)

print(df_clean.info())

# SAVE CLEANED DATA
df_before_cleaning.to_csv("../data/raw/before_cleaning.csv", index=False)
df_clean.to_csv("../data/processed/after_cleaning.csv", index=False)

# SAVE BREAKDOWN-SPECIFIC FILES
for breakdown in df_clean["breakdown"].unique():
    breakdown_df = df_clean[df_clean["breakdown"] == breakdown].copy()
    clean_name = breakdown.replace(" ", "_").replace("/", "_")
    file_path = f"../data/processed/{clean_name}.csv"
    breakdown_df.to_csv(file_path, index=False)
