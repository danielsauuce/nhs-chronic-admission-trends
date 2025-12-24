import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import statsmodels.api as sm  # for prediction intervals
from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


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


# =============================================================================
# PROPHET TIME-SERIES MODEL (ADDED MODEL – DOES NOT MODIFY PREVIOUS CODE)
# =============================================================================

from prophet import Prophet
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# LOAD DATA
df_prophet = pd.read_csv("../data/processed/england.csv")
df_prophet = df_prophet[["year_start", "indicator_value"]].dropna()
df_prophet = df_prophet.rename(columns={"year_start": "ds", "indicator_value": "y"})
df_prophet["ds"] = pd.to_datetime(df_prophet["ds"], format="%Y")

# TEMPORAL TRAIN/TEST SPLIT
train_cutoff = pd.to_datetime("2021-01-01")
train = df_prophet[df_prophet["ds"] <= train_cutoff].copy()
test = df_prophet[df_prophet["ds"] > train_cutoff].copy()

# INITIALISE & FIT PROPHET
prophet_model = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    interval_width=0.95,
)
prophet_model.fit(train)

# PREDICT ON TEST
test_forecast = prophet_model.predict(test[["ds"]])
test_forecast = test_forecast.set_index("ds").reindex(test["ds"]).reset_index()  # ALIGN
y_true = test["y"].values
y_pred = test_forecast["yhat"].values

rmse_prophet = np.sqrt(mean_squared_error(y_true, y_pred))
r2_prophet = r2_score(y_true, y_pred)
mae_prophet = mean_absolute_error(y_true, y_pred)

# REFIT ON FULL DATA FOR 5-YEAR FORECAST
prophet_full = Prophet(
    yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,
    changepoint_prior_scale=0.05,
    interval_width=0.95,
)
prophet_full.fit(df_prophet)
future = prophet_full.make_future_dataframe(periods=5, freq="Y")
forecast = prophet_full.predict(future)

forecast_future = forecast[forecast["ds"] > df_prophet["ds"].max()]

# --- VISUALISATION 1: FULL FORECAST ---
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(df_prophet["ds"], df_prophet["y"], "o", color="black", label="Observed")
ax.plot(forecast["ds"], forecast["yhat"], color="blue", label="Fitted & Forecast")
ax.fill_between(
    forecast["ds"],
    forecast["yhat_lower"],
    forecast["yhat_upper"],
    alpha=0.25,
    color="gray",
    label="95% Prediction Interval",
)
ax.axvspan(
    pd.to_datetime("2020"),
    pd.to_datetime("2021"),
    alpha=0.15,
    color="orange",
    label="COVID-19 Period",
)
ax.set_xlabel("Year")
ax.set_ylabel("Admission Rate (per 100,000)")
ax.set_title("Prophet Forecast of Chronic ACSC Admission Rates (England)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    "../visualizations/Forecast/prophet_forecast_full.png", dpi=300, bbox_inches="tight"
)
plt.close()

# --- VISUALISATION 2: ACTUAL VS PREDICTED ---
# Align test y and yhat to ensure same length
test_aligned = test.copy()
test_aligned["yhat"] = y_pred
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(test_aligned["y"], test_aligned["yhat"], alpha=0.7, edgecolors="black")
ax.plot(
    [test_aligned["y"].min(), test_aligned["y"].max()],
    [test_aligned["y"].min(), test_aligned["y"].max()],
    "r--",
    label="Perfect Fit",
)
ax.set_xlabel("Actual Admission Rate")
ax.set_ylabel("Predicted Admission Rate")
ax.set_title("Prophet: Actual vs Predicted")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(
    "../visualizations/Forecast/prophet_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close()

# --- VISUALISATION 3: RESIDUALS ---
residuals = test_aligned["y"] - test_aligned["yhat"]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].scatter(test_aligned["yhat"], residuals, alpha=0.7)
axes[0].axhline(0, linestyle="--", color="red")
axes[0].set_xlabel("Fitted Values")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs Fitted")
axes[0].grid(alpha=0.3)
axes[1].hist(residuals, bins=10, edgecolor="black", alpha=0.7)
axes[1].axvline(0, linestyle="--", color="red")
axes[1].set_xlabel("Residuals")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution")
axes[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(
    "../visualizations/Forecast/prophet_residuals.png", dpi=300, bbox_inches="tight"
)
plt.close()

# --- VISUALISATION 4: COMPONENTS ---
fig = prophet_full.plot_components(forecast)
plt.savefig(
    "../visualizations/Forecast/prophet_components.png", dpi=300, bbox_inches="tight"
)
plt.close()
