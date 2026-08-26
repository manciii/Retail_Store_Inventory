

import numpy as np
import pandas as pd


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def wape(y_true, y_pred):
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Demand forecasts cannot be negative; clip predictions before scoring,
    # exactly as we would before showing a forecast to the business.
    y_pred = np.clip(y_pred, a_min=0, a_max=None)
    return {
        "MAE": round(mae(y_true, y_pred), 3),
        "RMSE": round(rmse(y_true, y_pred), 3),
        "MAPE_%": round(mape(y_true, y_pred), 3),
        "WAPE_%": round(wape(y_true, y_pred), 3),
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """results: {model_name: metrics_dict}"""
    return pd.DataFrame(results).T[["MAE", "RMSE", "MAPE_%", "WAPE_%"]]
