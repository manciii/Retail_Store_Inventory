"""
utils.py
Shared configuration and helper utilities for the demand forecasting project.
Keeping config here (instead of hardcoding paths/constants in every script)
means every module and notebook stays in sync if a path or assumption changes.
"""

import os
import random
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Fix random seeds across libraries so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Paths (relative to project root -- no hardcoded absolute paths)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "retail_store_inventory.csv")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "clean_sales.csv")
FEATURES_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "features.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
SQL_DB_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "retail.db")

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------
DATE_COL = "date"
STORE_COL = "store_id"
PRODUCT_COL = "product_id"
CATEGORY_COL = "category"
REGION_COL = "region"
TARGET_COL = "units_sold"

KEY_COLS = [STORE_COL, PRODUCT_COL]

# Business assumptions used throughout the inventory optimization layer.
# Documented centrally so they can be defended/explained in one place.
BUSINESS_ASSUMPTIONS = {
    "lead_time_days": 7,       # days between placing an order and receiving stock
    "service_level_z": 1.65,   # z-score for ~95% service level (safety stock)
    "review_period_days": 7,   # how often reorder decisions are reviewed
}


def raw_to_clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename raw Kaggle columns to consistent snake_case names used everywhere else."""
    rename_map = {
        "Date": DATE_COL,
        "Store ID": STORE_COL,
        "Product ID": PRODUCT_COL,
        "Category": CATEGORY_COL,
        "Region": REGION_COL,
        "Inventory Level": "inventory_level",
        "Units Sold": TARGET_COL,
        "Units Ordered": "units_ordered",
        "Demand Forecast": "legacy_demand_forecast",
        "Price": "price",
        "Discount": "discount_pct",
        "Weather Condition": "weather",
        "Holiday/Promotion": "promotion",
        "Competitor Pricing": "competitor_price",
        "Seasonality": "seasonality_label",
    }
    return df.rename(columns=rename_map)
