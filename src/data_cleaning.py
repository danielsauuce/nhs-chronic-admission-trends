import pandas as pd
import numpy as np

# Loaded the dataset
df = pd.read_excel(
    "../data/raw/NHSOF_2.3.i_I00708_D.xlsx",
    sheet_name="Indicator data",
    engine="openpyxl",
    skiprows=14,
)

# Basic data overview
print(df.head())
print(df.info())
print(df.columns)
print(df.isna().sum())

# Created a copy of the raw dataset
df_before_cleaning = df.copy()

# Strip whitespaces from column names and values
df.columns = df.columns.str.strip()

# Strip whitespace from all object (string) columns
object_cols = df.select_dtypes(include="object").columns
for col in object_cols:
    df[col] = df[col].astype(str).str.strip()

# standardise columns name
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Standardise text columns
text_columns = ["Year", "Breakdown", "Level description"]

for col in text_columns:
    df[col] = df[col].apply(lambda x: str(x).strip().lower() if pd.notna(x) else x)


# Convert year column (e.g., 2023 from '2023/24')
df["Year"] = df["Year"].str.strip()
df["Year_Start"] = df["Year"].str.split("/").str[0]
df["Year_Start"] = pd.to_numeric(df["Year_Start"], errors="coerce")

# HANDLE MISSING VALUES
# Check missing values in critical columns
critical_cols = ["year", "breakdown", "level_description", "indicator_value"]
print("\nMissing values in critical columns:")
for col in critical_cols:
    if col in df.columns:
        missing = df[col].isna().sum()

# Remove rows with missing critical values
rows_before = len(df)
df_clean = df.dropna(subset=critical_cols)
rows_after = len(df_clean)
removed = rows_before - rows_after

# Validating confidence intervals
if all(col in df_clean.columns for col in ["lower_ci", "indicator_value", "upper_ci"]):
    invalid_ci = (df_clean["lower_ci"] > df_clean["indicator_value"]) | (
        df_clean["indicator_value"] > df_clean["upper_ci"]
    )
    invalid_count = invalid_ci.sum()

    if invalid_count > 0:
        df_clean = df_clean[~invalid_ci].copy()


# Convert numeric columns
numeric_columns = [
    "Indicator value",
    "Lower CI",
    "Upper CI",
    "Standardised ratio",
    "Observed",
    "Population",
    "Percent unclassified",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Replaced suppressed '*' with NaN
suppress_cols = [
    "Indicator value",
    "Lower CI",
    "Upper CI",
    "Standardised ratio",
    "Observed",
    "Expected",
    "Percent unclassified",
]
df[suppress_cols] = df[suppress_cols].replace("*", np.nan)

print(df.head())


# Dropping irrelevant or columns not needed for analysis
final_columns_to_drop = [
    "Period of coverage",
    "Level",
    "Quarter",
    "Standardised ratio lower CI",
    "Standardised ratio upper CI",
    "Expected",
]

df.drop(columns=final_columns_to_drop, inplace=True)


# Check for outliers (INFORMATIONAL ONLY)
if "indicator_value" in df_clean.columns:
    Q1 = df_clean["indicator_value"].quantile(0.25)
    Q3 = df_clean["indicator_value"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 3 * IQR
    upper_bound = Q3 + 3 * IQR

    outliers = (df_clean["indicator_value"] < lower_bound) | (
        df_clean["indicator_value"] > upper_bound
    )

# Sorting data for consistent ordering
sort_cols = ["Year", "Breakdown"]
if "Level description" in df_clean.columns:
    sort_cols.append("Level description")

df_clean = df_clean.sort_values(by=sort_cols).reset_index(drop=True)

# Created additional derived columns
# Financial year label (e.g., "2023/24")
if "Year" in df_clean.columns:
    df_clean["Financial_Year"] = (
        df_clean["Year"].astype(str) + "/" + (df_clean["Year"] + 1).astype(str).str[2:]
    )
    print("Created 'Financial_Year' column")

# CI width (measure of statistical uncertainty)
if all(col in df_clean.columns for col in ["Lower CI", "Upper CI"]):
    df_clean["CI_Width"] = df_clean["Upper CI"] - df_clean["Lower CI"]
    print("Created 'CI_Width' column (higher = more uncertainty)")

# Optional: Flag high-uncertainty rows (e.g., small populations)
if "Population" in df_clean.columns and "CI_Width" in df_clean.columns:
    df_clean["High_Uncertainty"] = (
        df_clean["CI_Width"] > df_clean["CI_Width"].quantile(0.9)
    ).astype(int)
    print("Created 'High_Uncertainty' flag (top 10% widest CIs)")

# Preserve cleaned view (after)
df_after_cleaning = df.copy()

# Save outputs
df_before_cleaning.to_csv("../data/raw/before_cleaning.csv", index=False)

df_after_cleaning.to_csv("../data/processed/after_cleaning.csv", index=False)


# Final check
print(df_after_cleaning.info())
print(df_after_cleaning.head())
