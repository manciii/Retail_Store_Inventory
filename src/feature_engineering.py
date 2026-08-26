"""
feature_engineering.py
Leakage-safe feature engineering for demand forecasting.

Core rule enforced throughout this file: for a prediction made on day T for
store-product combo (s,p), a feature may only use information available
strictly BEFORE day T's Units Sold is known. Concretely that means:

  - Lag features use sales from T-1, T-7, T-14, T-28 (past days only).
  - Rolling means/stds are computed on a window ending at T-1 (shift(1)
    before rolling), NEVER including day T itself.
  - Calendar features (day_of_week, month, ...) are fine to use "as of T"
    because the calendar date is known in advance, not derived from sales.
  - price / promotion / inventory_level on day T are used as-is because in
    a real deployment these are known in advance (prices and promo
    calendars are set ahead of time; opening inventory for the day is a
    system record, not something derived from that day's sales).
  - legacy_demand_forecast is EXCLUDED as a feature (see data_processing.py
    docstring) to avoid using another (opaque, possibly leaky) forecast as
    an input to ours.

All groupby operations are scoped to (store_id, product_id) so lag/rolling
windows never leak across different store-product time series.
"""

import numpy as np
import pandas as pd

from src.utils import PROCESSED_DATA_PATH, FEATURES_DATA_PATH, TARGET_COL, KEY_COLS


LAGS = [1, 7, 14, 28]
ROLL_WINDOWS_MEAN = [7, 14, 28]
ROLL_WINDOWS_STD = [7, 28]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(KEY_COLS, observed=True)[TARGET_COL]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    # shift(1) FIRST so the rolling window for row T only sees T-1 and earlier.
    shifted = df.groupby(KEY_COLS, observed=True)[TARGET_COL].shift(1)
    for w in ROLL_WINDOWS_MEAN:
        df[f"rolling_mean_{w}"] = (
            shifted.groupby([df["store_id"], df["product_id"]])
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=[0, 1], drop=True)
        )
    for w in ROLL_WINDOWS_STD:
        df[f"rolling_std_{w}"] = (
            shifted.groupby([df["store_id"], df["product_id"]])
            .rolling(w, min_periods=2)
            .std()
            .reset_index(level=[0, 1], drop=True)
        )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["date"]
    df["day_of_week"] = dt.dt.dayofweek
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    # No explicit public-holiday calendar is provided in this dataset; the
    # `promotion` column doubles as "Holiday/Promotion" per the source data
    # dictionary, so it is used directly as the demand-shock indicator
    # instead of being re-derived (see business features below).
    return df


def add_business_features(df: pd.DataFrame) -> pd.DataFrame:
    # promotion, price, inventory_level are already present under those
    # names post-cleaning and are known ahead of day T -- kept as-is.
    df["price_vs_competitor"] = df["price"] - df["competitor_price"]
    df["price_vs_competitor_pct"] = df["price_vs_competitor"] / df["competitor_price"]
    return df


def add_target_encodings_leak_safe(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["category", "date"], observed=True)[TARGET_COL]
        .mean()
        .reset_index()
        .sort_values(["category", "date"])
    )
    daily["cat_expanding_mean"] = (
        daily.groupby("category", observed=True)[TARGET_COL]
        .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    df = df.merge(
        daily[["category", "date", "cat_expanding_mean"]],
        on=["category", "date"],
        how="left",
    )
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_business_features(df)
    df = add_target_encodings_leak_safe(df)

    # First 28 days of each store-product series have incomplete lag/rolling
    # history (NaNs) -- drop them rather than impute, since imputing invented
    # values for the coldest-start rows would inject noise. This costs
    # 28 * 100 combos = 2800 rows out of 73,100 (~3.8%), an acceptable trade.
    feature_cols = [f"lag_{l}" for l in LAGS] + [f"rolling_mean_{w}" for w in ROLL_WINDOWS_MEAN]
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    return df


FEATURE_COLUMNS = (
    [f"lag_{l}" for l in LAGS]
    + [f"rolling_mean_{w}" for w in ROLL_WINDOWS_MEAN]
    + [f"rolling_std_{w}" for w in ROLL_WINDOWS_STD]
    + ["day_of_week", "week_of_year", "month", "quarter", "year", "is_weekend"]
    + ["promotion", "price", "inventory_level", "discount_pct"]
    + ["price_vs_competitor", "price_vs_competitor_pct", "cat_expanding_mean"]
)
CATEGORICAL_FEATURE_COLUMNS = ["store_id", "product_id", "category", "region", "weather"]


def run(save: bool = True) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["date"])
    feat_df = build_features(df)
    if save:
        feat_df.to_csv(FEATURES_DATA_PATH, index=False)
    return feat_df


if __name__ == "__main__":
    feat_df = run()
    print(f"Feature dataframe shape: {feat_df.shape}")
    print(f"Feature columns ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")
    print(f"Any NaNs left in feature cols: {feat_df[FEATURE_COLUMNS].isnull().sum().sum()}")
    print(f"Saved to: {FEATURES_DATA_PATH}")
