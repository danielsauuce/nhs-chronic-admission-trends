import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm  # for prediction intervals


# LOAD DATA
england = pd.read_csv("../data/processed/england.csv")
condition = pd.read_csv("../data/processed/condition.csv")


# DATA PREPARATION
england_model = england.dropna(subset=["year_start", "indicator_value"]).sort_values(
    "year_start"
)
england_model["financial_year"] = (
    england_model["year_start"].astype(str)
    + "/"
    + (england_model["year_start"] + 1).astype(str).str[-2:]
)
X_eng = england_model[["year_start"]]
y_eng = england_model["indicator_value"]

# Split data for train/test validation
X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y_eng, test_size=0.2, shuffle=False
)


# FIT LINEAR REGRESSION MODEL
model_eng = LinearRegression()
model_eng.fit(X_train, y_train)

# Predictions
england_model["predicted"] = model_eng.predict(X_eng)


# PREDICTION INTERVALS (95%)
X_sm = sm.add_constant(X_eng)
ols_model = sm.OLS(y_eng, X_sm).fit()
predictions = ols_model.get_prediction(X_sm)
pred_summary = predictions.summary_frame(alpha=0.05)  # 95% CI

england_model["pi_lower"] = pred_summary["obs_ci_lower"]
england_model["pi_upper"] = pred_summary["obs_ci_upper"]


# FORECAST NEXT 5 YEARS
future_years_eng = pd.DataFrame(
    {"year_start": range(X_eng["year_start"].max() + 1, X_eng["year_start"].max() + 6)}
)
future_years_eng["financial_year"] = (
    future_years_eng["year_start"].astype(str)
    + "/"
    + (future_years_eng["year_start"] + 1).astype(str).str[-2:]
)
future_years_eng["forecast"] = model_eng.predict(future_years_eng[["year_start"]])

# MODEL EVALUATION
y_train_pred = model_eng.predict(X_train)
y_test_pred = model_eng.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print(f"TRAIN RMSE: {rmse_train:.2f}, R²: {r2_train:.3f}")
print(f"TEST RMSE: {rmse_test:.2f}, R²: {r2_test:.3f}")


# PLOT 1: HISTORICAL + FORECAST + PREDICTION INTERVALS
plt.figure(figsize=(14, 7))

# Historical observed
plt.plot(
    england_model["financial_year"],
    england_model["indicator_value"],
    marker="o",
    label="Observed Admission Rate",
)

# Fitted trend
plt.plot(
    england_model["financial_year"],
    england_model["predicted"],
    linestyle="--",
    label="Fitted Trend",
)

# Prediction intervals
plt.fill_between(
    england_model["financial_year"],
    england_model["pi_lower"],
    england_model["pi_upper"],
    color="gray",
    alpha=0.2,
    label="95% Prediction Interval",
)

# Forecast
plt.plot(
    future_years_eng["financial_year"],
    future_years_eng["forecast"],
    linestyle="--",
    marker="o",
    label="Forecast Next 5 Years",
)

plt.xticks(rotation=45)
plt.xlabel("Financial Year")
plt.ylabel("Admission Rate (per 100,000)")
plt.title(
    "Forecast of Chronic ACSC Admission Rates in England with Prediction Interval"
)
plt.legend(loc="upper left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    "../visualizations/Forecast/predictive_plot1_forecast_trend_pi.png", dpi=300
)
plt.close()

# PLOT 2: ACTUAL VS PREDICTED
plt.figure(figsize=(6, 6))
plt.scatter(y_eng, england_model["predicted"], alpha=0.7)
plt.plot(
    [y_eng.min(), y_eng.max()], [y_eng.min(), y_eng.max()], linestyle="--", color="red"
)
plt.xlabel("Actual Admission Rate")
plt.ylabel("Predicted Admission Rate")
plt.title("Actual vs Predicted Admission Rates")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    "../visualizations/Forecast/predictive_plot2_actual_vs_predicted_pi.png", dpi=300
)
plt.close()

# PLOT 3: RESIDUALS
residuals = y_eng - england_model["predicted"]
plt.figure(figsize=(10, 6))
plt.scatter(england_model["predicted"], residuals, alpha=0.7)
plt.axhline(0, linestyle="--", color="red")
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot for Linear Regression Model")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("../visualizations/Forecast/predictive_plot3_residuals_pi.png", dpi=300)
plt.close()
