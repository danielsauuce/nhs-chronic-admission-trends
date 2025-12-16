import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")


# PLOTTING STYLE
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10


# LOAD DATA
df_before = pd.read_csv("../data/raw/before_cleaning.csv")
df_after = pd.read_csv("../data/processed/after_cleaning.csv")

england = pd.read_csv("../data/processed/england.csv")
age = pd.read_csv("../data/processed/age.csv")
gender = pd.read_csv("../data/processed/gender.csv")
deprivation = pd.read_csv("../data/processed/2015_deprivation_decile.csv")


# PLOT 0: MISSING DATA BEFORE VS AFTER CLEANING
missing_before_pct = (df_before.isnull().sum() / len(df_before)) * 100
missing_after_pct = (df_after.isnull().sum() / len(df_after)) * 100

missing_before_df = (
    missing_before_pct.sort_values(ascending=False)
    .head(10)
    .reset_index()
    .rename(columns={"index": "Column", 0: "Missing %"})
)
missing_after_df = (
    missing_after_pct.sort_values(ascending=False)
    .head(10)
    .reset_index()
    .rename(columns={"index": "Column", 0: "Missing %"})
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

axes[0].barh(
    missing_before_df["Column"],
    missing_before_df["Missing %"],
    color="#E74C3C",
    alpha=0.7,
)
axes[0].set_title("Before Cleaning: Missing Data (%)", fontweight="bold")
axes[0].invert_yaxis()
axes[0].set_xlabel("Missing Data (%)")

if missing_after_df["Missing %"].sum() > 0:
    axes[1].barh(
        missing_after_df["Column"],
        missing_after_df["Missing %"],
        color="#27AE60",
        alpha=0.7,
    )
    axes[1].invert_yaxis()
else:
    axes[1].text(
        0.5,
        0.5,
        "No Missing Data\n✓ 100% Complete",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#27AE60",
        transform=axes[1].transAxes,
    )
    axes[1].axis("off")

axes[1].set_title("After Cleaning: Missing Data (%)", fontweight="bold")
axes[1].set_xlabel("Missing Data (%)")

plt.tight_layout()
plt.savefig("../visualizations/plot0_missing_data_before_after.png", dpi=300)
plt.close()
