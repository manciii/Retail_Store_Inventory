
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


#Classical time-series model (SARIMA) 


def fit_sarima(train_series: pd.Series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7)):
    
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        train_series, order=order, seasonal_order=seasonal_order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    return model.fit(disp=False)



def fit_sarima_all_series(full_df: pd.DataFrame, target_col: str = TARGET_COL):
    """
    Fits SARIMA per store-product series (SARIMA is univariate -- it only
    sees each series' own history, not the lag/rolling/GBM feature set) and
    returns one-step-ahead fitted predictions aligned to full_df's index.
    """
    preds = []
    for _, g in full_df.groupby(KEY_COLS_FOR_SARIMA, sort=False):
        g = g.sort_values("date")
        fit = fit_sarima(g[target_col])
        preds.append(pd.Series(fit.fittedvalues.values, index=g.index))
    return pd.concat(preds).reindex(full_df.index)


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
