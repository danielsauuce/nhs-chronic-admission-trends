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

