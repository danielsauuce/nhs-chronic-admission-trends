import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# LOAD DATA
england = pd.read_csv("../data/processed/england.csv")
condition = pd.read_csv("../data/processed/condition.csv")

# DATA PREPARATION
england_model = england.dropna(subset=["year_start", "indicator_value"]).sort_values(
    "year_start"
)
X_eng = england_model[["year_start"]]
y_eng = england_model["indicator_value"]

# Split for train/test validation
X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y_eng, test_size=0.2, shuffle=False
)

# FIT LINEAR REGRESSION MODEL
model_eng = LinearRegression()
model_eng.fit(X_train, y_train)

# Predictions
y_train_pred = model_eng.predict(X_train)
y_test_pred = model_eng.predict(X_test)
england_model["predicted"] = model_eng.predict(X_eng)

# Forecast next 5 years
future_years_eng = pd.DataFrame(
    {"year_start": range(X_eng["year_start"].max() + 1, X_eng["year_start"].max() + 6)}
)
future_years_eng["forecast"] = model_eng.predict(future_years_eng)

# MODEL EVALUATION
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print(f"TRAIN RMSE: {rmse_train:.2f}, R²: {r2_train:.3f}")
print(f"TEST RMSE: {rmse_test:.2f}, R²: {r2_test:.3f}")

# PLOT 1: HISTORICAL + FORECAST
plt.figure(figsize=(14, 7))
plt.plot(
    england_model["year_start"],
    england_model["indicator_value"],
    marker="o",
    label="Observed",
)
plt.plot(
    england_model["year_start"],
    england_model["predicted"],
    linestyle="--",
    label="Fitted Trend",
)
plt.plot(
    future_years_eng["year_start"],
    future_years_eng["forecast"],
    linestyle="--",
    marker="o",
    label="Forecast",
)
plt.xlabel("Financial Year Start")
plt.ylabel("Admission Rate (per 100,000)")
plt.title("Forecast of Chronic ACSC Admission Rates in England")
plt.legend()
plt.tight_layout()
plt.savefig("../visualizations/Forecast/predictive_plot1_forecast_trend.png", dpi=300)
plt.close()

# PLOT 2: ACTUAL VS PREDICTED
plt.figure(figsize=(6, 6))
plt.scatter(y_eng, england_model["predicted"], alpha=0.7)
plt.plot(
    [y_eng.min(), y_eng.max()], [y_eng.min(), y_eng.max()], linestyle="--", color="red"
)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Admission Rates")
plt.tight_layout()
plt.savefig(
    "../visualizations/Forecast/predictive_plot2_actual_vs_predicted.png", dpi=300
)
plt.close()


