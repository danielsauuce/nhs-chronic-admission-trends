import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
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

# PLOT 1: ENGLAND OVERALL TREND
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(
    england["year_start"],
    england["indicator_value"],
    marker="o",
    linewidth=2.5,
    label="Observed Admission Rate",
)

ax.fill_between(
    england["year_start"],
    england["lower_ci"],
    england["upper_ci"],
    alpha=0.2,
    label="95% Confidence Interval",
)

z = np.polyfit(england["year_start"], england["indicator_value"], 1)
ax.plot(
    england["year_start"],
    np.poly1d(z)(england["year_start"]),
    linestyle="--",
    linewidth=2,
    color="red",
    label="Linear Trend",
)

ax.axvspan(2020, 2021, alpha=0.1, color="gray", label="COVID-19 Period")

ax.set_xlabel("Financial Year Start")
ax.set_ylabel("Admission Rate (per 100,000)")
ax.set_title("England: Chronic ACSC Admission Rates (2003/04–2023/24)")
ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig("../visualizations/plot1_england_trend.png", dpi=300)
plt.close()

# PLOT 3: YEAR-ON-YEAR % CHANGE
england = england.sort_values("year_start")
england["pct_change"] = england["indicator_value"].pct_change() * 100

fig, ax = plt.subplots(figsize=(14, 6))
colors = [
    "red" if x > 5 else "green" if x < -5 else "steelblue"
    for x in england["pct_change"][1:]
]

ax.bar(england["year_start"][1:], england["pct_change"][1:], color=colors, alpha=0.7)
ax.axhline(0, linewidth=0.8)

legend_elements = [
    Patch(facecolor="red", label="Increase > 5%"),
    Patch(facecolor="green", label="Decrease < -5%"),
    Patch(facecolor="steelblue", label="Change between -5% and 5%"),
]

ax.legend(handles=legend_elements, loc="upper right")

ax.set_xlabel("Financial Year Start")
ax.set_ylabel("% Change from Previous Year")
ax.set_title("Year-on-Year Percentage Change in Admission Rates")

plt.tight_layout()
plt.savefig("../visualizations/plot3_yoy_change.png", dpi=300)
plt.close()

# PLOT 4: ROLLING 3-YEAR CHANGE
england["rolling_change"] = england["pct_change"].rolling(3, center=True).mean()

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(
    england["year_start"],
    england["rolling_change"],
    marker="o",
    label="Rolling 3-Year Avg % Change",
)
ax.axhline(0, linewidth=0.8)
ax.legend()

ax.set_xlabel("Financial Year Start")
ax.set_ylabel("Rolling 3-Year Avg Change (%)")
ax.set_title("Acceleration / Deceleration of Admission Trends")

plt.tight_layout()
plt.savefig("../visualizations/plot4_rolling_change.png", dpi=300)
plt.close()

# PLOT 5: AGE GROUP TRENDS
fig, ax = plt.subplots(figsize=(14, 7))

for group in sorted(age["level_description"].unique()):
    subset = age[age["level_description"] == group]
    ax.plot(subset["year_start"], subset["indicator_value"], marker="o", label=group)

ax.set_xlabel("Financial Year Start")
ax.set_ylabel("Admission Rate (per 100,000)")
ax.set_title("Admission Rates by Age Group")
ax.legend(title="Age Group", ncol=2)

plt.tight_layout()
plt.savefig("../visualizations/plot5_age_trends.png", dpi=300)
plt.close()

# PLOT 6: AGE HEATMAP
age_pivot = age.pivot_table(
    index="level_description",
    columns="year_start",
    values="indicator_value",
    aggfunc="mean",
)

plt.figure(figsize=(16, 6))
ax = sns.heatmap(age_pivot, cmap="YlOrRd")
ax.collections[0].colorbar.set_label("Admission Rate (per 100,000)")

plt.title("Admission Rate Heatmap by Age Group and Year")
plt.tight_layout()
plt.savefig("../visualizations/plot6_age_heatmap.png", dpi=300)
plt.close()

# PLOT 7: AGE SLOPE CHART (START VS END)
age_start = age[age["year_start"] == age["year_start"].min()]
age_end = age[age["year_start"] == age["year_start"].max()]

comparison = age_start.merge(
    age_end, on="level_description", suffixes=("_start", "_end")
)

comparison["pct_change"] = (
    (comparison["indicator_value_end"] - comparison["indicator_value_start"])
    / comparison["indicator_value_start"]
    * 100
)

fig, ax = plt.subplots(figsize=(12, 6))

for _, row in comparison.iterrows():
    ax.plot(
        [0, 1],
        [row["indicator_value_start"], row["indicator_value_end"]],
        marker="o",
        color="#2E86AB",
    )
    ax.text(
        -0.05,
        row["indicator_value_start"],
        row["level_description"],
        ha="right",
        va="center",
    )
    ax.text(
        1.05,
        row["indicator_value_end"],
        f"{row['pct_change']:+.1f}%",
        ha="left",
        va="center",
    )

ax.set_xlim(-0.5, 1.5)
ax.set_xticks([0, 1])
ax.set_xticklabels([f"{age['year_start'].min()}", f"{age['year_start'].max()}"])
ax.set_ylabel("Admission Rate (per 100,000)")
ax.set_title("Age Group Admission Rates: Slope Chart (Start → End)")

ax.text(
    0.5,
    -0.15,
    "Each line represents an age group.\nLeft = earliest year, Right = latest year.",
    ha="center",
    va="top",
    transform=ax.transAxes,
    fontsize=9,
)

plt.tight_layout()
plt.savefig("../visualizations/plot7_age_slope_chart.png", dpi=300)
plt.close()

# PLOT 8: AGE % CHANGE RANKING
comparison_sorted = comparison.sort_values("pct_change")

fig, ax = plt.subplots(figsize=(10, 6))
colors = ["green" if x < 0 else "red" for x in comparison_sorted["pct_change"]]

ax.barh(
    comparison_sorted["level_description"],
    comparison_sorted["pct_change"],
    color=colors,
    alpha=0.7,
)
ax.axvline(0, color="black", linewidth=0.8)

ax.set_xlabel("% Change (Start → End)")
ax.set_title("Percentage Change in Admission Rates by Age Group")

plt.tight_layout()
plt.savefig("../visualizations/plot8_age_change_ranking.png", dpi=300)
plt.close()

# PLOT 9 & 10: GENDER
fig, ax = plt.subplots(figsize=(14, 7))

for g in gender["level_description"].unique():
    subset = gender[gender["level_description"] == g]
    ax.plot(
        subset["year_start"],
        subset["indicator_value"],
        marker="o",
        label=g.title(),
    )

ax.set_title("Admission Rates by Gender")
ax.legend(title="Gender")

plt.tight_layout()
plt.savefig("../visualizations/plot9_gender_trends.png", dpi=300)
plt.close()

male = gender[gender["level_description"] == "male"].sort_values("year_start")
female = gender[gender["level_description"] == "female"].sort_values("year_start")
diff = male["indicator_value"].values - female["indicator_value"].values

plt.figure(figsize=(14, 6))

plt.fill_between(
    male["year_start"],
    0,
    diff,
    where=(diff >= 0),
    color="#3498DB",
    alpha=0.5,
    label="Male > Female",
)
plt.fill_between(
    male["year_start"],
    0,
    diff,
    where=(diff < 0),
    color="#E74C3C",
    alpha=0.5,
    label="Female > Male",
)

plt.plot(
    male["year_start"],
    diff,
    color="black",
    linewidth=2,
    label="Male − Female Difference",
)

plt.axhline(0, linestyle="--", color="gray")
plt.xlabel("Financial Year Start")
plt.ylabel("Admission Rate Difference")
plt.title("Gender Gap in Admission Rates (Male − Female)")
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig("../visualizations/plot10_gender_difference.png", dpi=300)
plt.close()

# PLOT 16–18: DEPRIVATION
deprivation["decile"] = (
    deprivation["level_description"].str.extract(r"(\d+)").astype(int)
)

plt.figure(figsize=(14, 7))
sns.boxplot(data=deprivation, x="decile", y="indicator_value", palette="RdYlGn_r")
sns.regplot(
    x=deprivation["decile"],
    y=deprivation["indicator_value"],
    scatter=False,
    color="blue",
)

legend_elements = [
    Line2D([0], [0], color="black", lw=2, label="Distribution by Decile"),
    Line2D([0], [0], color="blue", lw=2, label="Linear Trend Across Deciles"),
]

plt.legend(handles=legend_elements)
plt.title("Admission Rates by Deprivation Decile with Trend")

plt.tight_layout()
plt.savefig("../visualizations/plot16_deprivation_boxplot.png", dpi=300)
plt.close()

plt.figure(figsize=(14, 7))
colors = sns.color_palette("RdYlGn", 10)

for d in range(1, 11):
    subset = deprivation[deprivation["decile"] == d]
    lw = 2.5 if d in [1, 5, 10] else 1.0
    plt.plot(
        subset["year_start"],
        subset["indicator_value"],
        marker="o",
        linewidth=lw,
        color=colors[d - 1],
        label=f"D{d}",
    )

plt.xlabel("Financial Year Start")
plt.ylabel("Admission Rate")
plt.title("Deprivation Trends Over Time (All Deciles)")
plt.legend(
    title="Deprivation Decile\n(1 = Most deprived, 10 = Least deprived)",
    ncol=2,
)

plt.tight_layout()
plt.savefig("../visualizations/plot17_deprivation_trends_all_deciles.png", dpi=300)
plt.close()

d1 = deprivation[deprivation["decile"] == 1].sort_values("year_start")
d10 = deprivation[deprivation["decile"] == 10].sort_values("year_start")
ratio = d1["indicator_value"].values / d10["indicator_value"].values

fig, ax1 = plt.subplots(figsize=(14, 6))

ax1.plot(
    d1["year_start"],
    d1["indicator_value"],
    marker="o",
    color="#D32F2F",
    label="Decile 1",
)
ax1.plot(
    d10["year_start"],
    d10["indicator_value"],
    marker="o",
    color="#388E3C",
    label="Decile 10",
)
ax1.set_ylabel("Admission Rate")

ax2 = ax1.twinx()
ax2.plot(
    d1["year_start"],
    ratio,
    marker="o",
    color="#8E24AA",
    label="Inequality Ratio (D1 / D10)",
)
ax2.axhline(1, linestyle="--", color="gray")
ax2.set_ylabel("Inequality Ratio")

ax1.set_xlabel("Financial Year Start")
fig.suptitle("Health Inequality: Absolute Rates and Relative Ratio")

ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("../visualizations/plot18_inequality_ratio_dual.png", dpi=300)
plt.close()
