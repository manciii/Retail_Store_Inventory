

import numpy as np
import pandas as pd

from src.utils import BUSINESS_ASSUMPTIONS


def compute_demand_volatility(history_df: pd.DataFrame, target_col: str = "units_sold") -> pd.DataFrame:
    """
    Per store-product, standard deviation of daily demand computed from
    historical (train+val) actuals -- used as sigma_demand for safety stock.
    Using historical demand volatility (rather than in-sample forecast
    residuals, which would be optimistic) gives a more honest, slightly
    conservative estimate of real-world uncertainty.
    """
    return (
        history_df.groupby(["store_id", "product_id"], observed=True)[target_col]
        .std()
        .rename("sigma_demand")
        .reset_index()
    )


def build_recommendations(
    latest_snapshot: pd.DataFrame,
    volatility: pd.DataFrame,
    lead_time_days: int = None,
    service_level_z: float = None,
) -> pd.DataFrame:
    """
    latest_snapshot: one row per store-product with columns
        [store_id, product_id, category, current_inventory, forecast_demand]
        forecast_demand is the model's predicted DAILY demand for that combo.
    volatility: output of compute_demand_volatility.
    """
    lead_time_days = lead_time_days or BUSINESS_ASSUMPTIONS["lead_time_days"]
    service_level_z = service_level_z or BUSINESS_ASSUMPTIONS["service_level_z"]

    df = latest_snapshot.merge(volatility, on=["store_id", "product_id"], how="left")
    df["sigma_demand"] = df["sigma_demand"].fillna(df["sigma_demand"].median())

    df["safety_stock"] = service_level_z * df["sigma_demand"] * np.sqrt(lead_time_days)
    df["reorder_point"] = (df["forecast_demand"] * lead_time_days) + df["safety_stock"]
    df["recommended_order_qty"] = (df["reorder_point"] - df["current_inventory"]).clip(lower=0)

    # Business risk flags
    def classify(row):
        if row["current_inventory"] <= row["safety_stock"]:
            return " Stockout Risk"
        elif row["current_inventory"] <= row["reorder_point"]:
            return " Low Inventory"
        elif row["current_inventory"] <= row["reorder_point"] * 2.5:
            return " Healthy Inventory"
        else:
            return " Overstock"

    df["risk_flag"] = df.apply(classify, axis=1)

    df["safety_stock"] = df["safety_stock"].round(1)
    df["reorder_point"] = df["reorder_point"].round(1)
    df["recommended_order_qty"] = df["recommended_order_qty"].round(0)
    df["forecast_demand"] = df["forecast_demand"].round(1)

    cols = [
        "store_id", "product_id", "category", "current_inventory",
        "forecast_demand", "safety_stock", "reorder_point",
        "recommended_order_qty", "risk_flag",
    ]
    return df[cols].sort_values("risk_flag")
