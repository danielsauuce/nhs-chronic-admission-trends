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

