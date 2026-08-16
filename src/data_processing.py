"""
data_processing.py
Data validation, quality checks, and cleaning for the retail inventory dataset.

Design principle: every cleaning decision is a function with a docstring
explaining WHY, not just what. Nothing is dropped silently.
"""

import numpy as np
import pandas as pd

from src.utils import raw_to_clean_columns, RAW_DATA_PATH, PROCESSED_DATA_PATH, TARGET_COL


def load_raw(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def validate_raw(df: pd.DataFrame) -> dict:
    """
    Run data-quality checks and return a report dict.
    We report issues rather than fixing them here, so validation and
    cleaning stay separate and auditable steps.
    """
    report = {}
    report["n_rows"] = len(df)
    report["n_duplicates"] = int(df.duplicated().sum())
    report["missing_by_col"] = df.isnull().sum().to_dict()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    negatives = {c: int((df[c] < 0).sum()) for c in numeric_cols}
    report["negative_value_counts"] = {k: v for k, v in negatives.items() if v > 0}

    # A panel dataset (store x product x date) should be a complete rectangle.
    # Gaps would silently corrupt lag/rolling features later, so we check now.
    combo_counts = df.groupby(["Store ID", "Product ID"]).size()
    report["panel_is_balanced"] = bool(combo_counts.nunique() == 1)
    report["rows_per_combo"] = int(combo_counts.iloc[0]) if report["panel_is_balanced"] else None

    # Units sold should never exceed opening inventory for that day+order.
    report["sold_exceeds_inventory"] = int((df["Units Sold"] > df["Inventory Level"]).sum())

    return report


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply cleaning decisions. Each step documents its reasoning:

    1. Rename columns to snake_case for consistent downstream use.
    2. Parse Date to datetime, sort by (store, product, date) -- required
       for any lag/rolling feature to be correct.
    3. Drop exact duplicate rows (none found in this dataset, but kept as
       a safeguard for future data refreshes).
    4. `legacy_demand_forecast` contains ~0.9% negative values (a noisy
       vendor-supplied forecast column, NOT our target). We do not use it
       as a model feature (using someone else's forecast as an input to
       our forecast would be circular / a leakage risk if it was ever
       computed using future info) -- we keep it only as a benchmark to
       compare our model against in evaluation.
    5. We do NOT clip or remove outliers in Units Sold / Inventory Level.
       Retail demand is naturally spiky (promotions, holidays) -- removing
       those points would strip out exactly the signal we want the model
       to learn. Outliers are instead handled implicitly by using
       tree-based models (robust to outliers) and by inspecting them in EDA.
    6. Categorical columns are cast to `category` dtype for memory
       efficiency and explicit, unambiguous category sets.
    """
    df = raw_to_clean_columns(df)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_id", "product_id", "date"]).reset_index(drop=True)

    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)

    cat_cols = ["store_id", "product_id", "category", "region", "weather", "seasonality_label"]
    for c in cat_cols:
        df[c] = df[c].astype("category")

    df["promotion"] = df["promotion"].astype(int)

    df.attrs["duplicates_dropped"] = dropped
    return df


def run(save: bool = True) -> pd.DataFrame:
    raw = load_raw()
    report = validate_raw(raw)
    clean_df = clean(raw)
    if save:
        clean_df.to_csv(PROCESSED_DATA_PATH, index=False)
    return clean_df, report


if __name__ == "__main__":
    clean_df, report = run()
    print("Validation report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"\nCleaned shape: {clean_df.shape}")
    print(f"Saved to: {PROCESSED_DATA_PATH}")
