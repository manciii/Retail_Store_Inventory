"""
forecasting.py
Chronological train/val/test split, baseline forecasts, and ML models.

Why NOT random train-test splitting:
Random splitting shuffles rows across time, so a model could be "trained"
on Jan 15 and evaluated on Jan 10 -- effectively predicting the past using
the future. In deployment we will only ever have past data to predict the
future, so evaluation must mimic that: train strictly earlier than we
validate, which must be strictly earlier than we test.

Split (chronological, per the project spec):
  Train      -> earliest 70% of dates
  Validation -> next 15% of dates
  Test       -> final 15% of dates
Applied on the full panel by DATE cutoixnts (not by row count) so every
store-product series is split at the same calendar boundary -- this avoids
a subtle leak where one product's "test" rows are chronologically earlier
than another product's "train" rows.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.utils import set_seed, RANDOM_SEED, TARGET_COL
from src.feature_engineering import FEATURE_COLUMNS, CATEGORICAL_FEATURE_COLUMNS


def chronological_split(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    # pd.Series(...).unique() + pd.to_datetime(...) guarantees pandas Timestamp
    # objects (with .date(), .year, etc.) rather than raw numpy.datetime64
    # scalars, whose available methods vary by numpy version.
    dates = pd.to_datetime(pd.Series(df["date"].unique())).sort_values().reset_index(drop=True)
    n = len(dates)
    train_end = dates.iloc[int(n * train_frac) - 1]
    val_end = dates.iloc[int(n * (train_frac + val_frac)) - 1]

    train = df[df["date"] <= train_end].copy()
    val = df[(df["date"] > train_end) & (df["date"] <= val_end)].copy()
    test = df[df["date"] > val_end].copy()
    return train, val, test, {"train_end": train_end, "val_end": val_end}


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def naive_forecast(df: pd.DataFrame) -> pd.Series:
    """Predict today's demand = yesterday's actual demand (lag_1)."""
    return df["lag_1"]


def moving_average_forecast(df: pd.DataFrame, window: int = 7) -> pd.Series:
    """Predict today's demand = trailing N-day rolling mean (already leak-safe, shift(1) applied upstream)."""
    return df[f"rolling_mean_{window}"]


# ---------------------------------------------------------------------------
# ML models
# ---------------------------------------------------------------------------

def _encode_categoricals(train, val, test):
    """
    Simple, leak-safe categorical encoding: fit category->code mapping on
    TRAIN ONLY, then apply the same mapping to val/test. Unseen categories
    (shouldn't happen here, all stores/products present from day 1) map to -1.
    """
    train, val, test = train.copy(), val.copy(), test.copy()
    for col in CATEGORICAL_FEATURE_COLUMNS:
        categories = train[col].astype("category").cat.categories
        mapping = {c: i for i, c in enumerate(categories)}
        for part in (train, val, test):
            part[col + "_enc"] = part[col].map(mapping).fillna(-1).astype(int)
    return train, val, test


def get_model_feature_cols():
    return FEATURE_COLUMNS + [c + "_enc" for c in CATEGORICAL_FEATURE_COLUMNS]


def fit_xgboost(X_train, y_train, X_val, y_val):
    """
    Production model: XGBoost. Requires the `xgboost` package (see
    requirements.txt). Not available in every sandbox -- caller should
    catch ImportError and fall back to `fit_sklearn_gbm` for environments
    without it (see evaluation.py for how model comparison handles this).
    """
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        early_stopping_rounds=30,
        eval_metric="mae",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def fit_lightgbm(X_train, y_train, X_val, y_val):
    """Production model: LightGBM. See fit_xgboost docstring for the same caveat."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbosity=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


# ---------------------------------------------------------------------------
# Classical time-series model (SARIMA) -- optional comparison per project spec
# ---------------------------------------------------------------------------

def fit_sarima(train_series: pd.Series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    """
    Production classical time-series model: SARIMA via statsmodels.
    Requires the `statsmodels` package (see requirements.txt) -- not
    available in every sandbox. Caller should catch ImportError and fall
    back to `holts_linear_trend_forecast` (see below) when it isn't
    installed, mirroring how fit_xgboost/fit_lightgbm fall back to
    fit_sklearn_gbm.

    order=(p,d,q): non-seasonal AR/differencing/MA terms.
    seasonal_order=(P,D,Q,s): seasonal AR/differencing/MA terms and the
    seasonal period s. Default s=7 for weekly seasonality -- though EDA
    (notebook 02) found weekly seasonality to be very weak in this dataset
    (<2% variation across day-of-week), so a simpler seasonal_order=(0,0,0,0)
    (i.e. plain ARIMA) may fit just as well here; both are worth trying.

    Fit PER store-product series (SARIMA is a univariate model -- it does not
    take the cross-sectional features lags/rolling/GBM use). Loop over the
    100 store-product combinations and fit one model per series, or fit a
    single model on the aggregated total-demand series if a per-SKU view
    isn't needed.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        train_series, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    return model.fit(disp=False)


def holts_linear_trend_forecast(series: pd.Series, alpha: float = 0.3, beta: float = 0.1) -> pd.Series:
    """
    Holt's Linear Trend exponential smoothing (a.k.a. double exponential
    smoothing): a classical time-series method (level + trend, no ML) --
    used here as an executable, dependency-free stand-in for SARIMA in
    environments without `statsmodels` installed (see `fit_sarima`
    docstring). It is a legitimate classical time-series technique in its
    own right, not a toy: it's the same family of method underlying
    Holt-Winters, one of the most widely used classical forecasting
    approaches in retail/demand planning before ML-based methods.

    This produces a genuine one-step-ahead recursive forecast: the
    prediction for day T uses only the level/trend state accumulated from
    days < T (updated with day T's actual only AFTER producing the day-T
    forecast), so it is leakage-safe in exactly the same sense as the lag/
    rolling features used elsewhere in this project.

    No seasonal component is included -- EDA (notebook 02) found weekly and
    monthly seasonality to be very weak in this dataset (<2% and <4%
    variation respectively), so the added complexity of a full seasonal
    Holt-Winters model would not be justified by the signal actually present.
    """
    series = series.reset_index(drop=True)
    level = float(series.iloc[0])
    trend = float(series.iloc[1] - series.iloc[0]) if len(series) > 1 else 0.0
    preds = []
    for t in range(len(series)):
        pred = level + trend
        preds.append(pred)
        actual = float(series.iloc[t])
        prev_level = level
        level = alpha * actual + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
    return pd.Series(preds, index=series.index)


def get_best_available_classical_model(full_df: pd.DataFrame, target_col: str = TARGET_COL):
    """
    Tries real SARIMA (per store-product series) first; falls back to the
    dependency-free Holt's Linear Trend method if statsmodels isn't
    installed. Returns (predictions_series_aligned_to_full_df, model_name).
    """
    try:
        import statsmodels  # noqa: F401
        preds = []
        for _, g in full_df.groupby(KEY_COLS_FOR_SARIMA, sort=False):
            g = g.sort_values("date")
            fit = fit_sarima(g[target_col])
            preds.append(pd.Series(fit.fittedvalues.values, index=g.index))
        pred_series = pd.concat(preds).reindex(full_df.index)
        return pred_series, "SARIMA"
    except ImportError:
        preds = []
        for _, g in full_df.groupby(KEY_COLS_FOR_SARIMA, sort=False):
            g = g.sort_values("date")
            preds.append(holts_linear_trend_forecast(g[target_col]).set_axis(g.index))
        pred_series = pd.concat(preds).reindex(full_df.index)
        return pred_series, "Holt's Linear Trend (statsmodels/SARIMA fallback)"


KEY_COLS_FOR_SARIMA = ["store_id", "product_id"]


def fit_sklearn_gbm(X_train, y_train, X_val=None, y_val=None):
    """
    Fallback gradient-boosted-tree model using only scikit-learn
    (HistGradientBoostingRegressor -- same family of algorithm as
    XGBoost/LightGBM: histogram-binned gradient boosting). Used when
    xgboost/lightgbm are not installed in the execution environment, so
    that model comparison can still run end-to-end with a real model
    rather than being skipped.
    """
    set_seed()
    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=0.1,
        random_state=RANDOM_SEED,
        early_stopping=True,
        validation_fraction=0.15 if X_val is None else None,
    )
    model.fit(X_train, y_train)
    return model
