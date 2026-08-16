"""
app.py
Streamlit dashboard: Overview, Forecasting, Inventory, Business Insights.

Run with:  streamlit run app/app.py   (from the project root)

Design: the app is read-only against pre-computed pipeline outputs
(data/processed/*.csv, models/*.csv). It does not retrain models on every
interaction -- retraining lives in the notebooks/src pipeline. This keeps
the dashboard fast and mirrors a realistic production split between an
offline training pipeline and an online serving/reporting layer.
"""

import os
import sys

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.utils import TARGET_COL  # noqa: E402

st.set_page_config(page_title="Demand Forecasting & Inventory Optimization", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


@st.cache_data
def load_data():
    clean = pd.read_csv(os.path.join(DATA_DIR, "clean_sales.csv"), parse_dates=["date"])
    test_pred = pd.read_csv(os.path.join(DATA_DIR, "test_predictions.csv"), parse_dates=["date"])
    try:
        rec = pd.read_csv(os.path.join(DATA_DIR, "inventory_recommendations.csv"))
    except FileNotFoundError:
        rec = pd.DataFrame()
    try:
        comparison = pd.read_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index_col=0)
    except FileNotFoundError:
        comparison = pd.DataFrame()
    return clean, test_pred, rec, comparison


clean, test_pred, rec, comparison = load_data()

st.title("📦 AI-Powered Demand Forecasting & Inventory Optimization")

tab_overview, tab_forecast, tab_inventory, tab_insights = st.tabs(
    ["Overview", "Forecasting", "Inventory", "Business Insights"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Units Sold (all-time)", f"{clean[TARGET_COL].sum():,.0f}")
    c2.metric("Total Inventory (latest snapshot)", f"{clean.groupby(['store_id','product_id']).tail(1)['inventory_level'].sum():,.0f}")
    c3.metric("Forecasted Demand (test period, total)", f"{test_pred['pred_demand'].sum():,.0f}")
    n_stockout = int((rec["risk_flag"] == "🔴 Stockout Risk").sum()) if not rec.empty else 0
    n_overstock = int((rec["risk_flag"] == "🔵 Overstock").sum()) if not rec.empty else 0
    c4.metric("Stockout-Risk SKUs", n_stockout)
    c5.metric("Overstock SKUs", n_overstock)

    st.subheader("Total Daily Demand Over Time")
    daily = clean.groupby("date")[TARGET_COL].sum()
    st.line_chart(daily)

    if not comparison.empty:
        st.subheader("Model Comparison")
        st.dataframe(comparison, use_container_width=True)

# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------
with tab_forecast:
    st.subheader("Filter")
    fc1, fc2, fc3, fc4 = st.columns(4)
    stores = sorted(clean["store_id"].unique())
    products = sorted(clean["product_id"].unique())
    categories = sorted(clean["category"].unique())
    regions = sorted(clean["region"].unique())

    sel_store = fc1.selectbox("Store", ["All"] + stores)
    sel_product = fc2.selectbox("Product", ["All"] + products)
    sel_category = fc3.selectbox("Category", ["All"] + categories)
    sel_region = fc4.selectbox("Region", ["All"] + regions)

    hist = clean.copy()
    pred = test_pred.copy()
    if sel_store != "All":
        hist, pred = hist[hist.store_id == sel_store], pred[pred.store_id == sel_store]
    if sel_product != "All":
        hist, pred = hist[hist.product_id == sel_product], pred[pred.product_id == sel_product]
    if sel_category != "All":
        hist, pred = hist[hist.category == sel_category], pred[pred.category == sel_category]
    if sel_region != "All":
        hist, pred = hist[hist.region == sel_region], pred[pred.region == sel_region]

    st.subheader("Historical Demand")
    hist_daily = hist.groupby("date")[TARGET_COL].sum()
    st.line_chart(hist_daily)

    st.subheader("Forecast vs Actual (Test Period)")
    if len(pred):
        pred_daily = pred.groupby("date").agg(actual=(TARGET_COL, "sum"), forecast=("pred_demand", "sum"))
        st.line_chart(pred_daily)
        mae = (pred_daily["actual"] - pred_daily["forecast"]).abs().mean()
        st.caption(f"Mean absolute error on this filtered slice (test period, daily-total basis): {mae:.1f} units/day")
    else:
        st.info("No test-period rows match this filter combination.")

# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
with tab_inventory:
    st.subheader("Inventory Recommendations")
    if rec.empty:
        st.warning("Run the inventory optimization step to populate this table.")
    else:
        risk_filter = st.multiselect(
            "Filter by risk", options=rec["risk_flag"].unique().tolist(),
            default=rec["risk_flag"].unique().tolist(),
        )
        st.dataframe(rec[rec["risk_flag"].isin(risk_filter)], use_container_width=True, height=500)

        st.subheader("Risk Distribution")
        st.bar_chart(rec["risk_flag"].value_counts())

# ---------------------------------------------------------------------------
# Business Insights
# ---------------------------------------------------------------------------
with tab_insights:
    st.subheader("Automatically Surfaced Insights")

    if not rec.empty:
        urgent = rec[rec["risk_flag"] == "🔴 Stockout Risk"].sort_values("recommended_order_qty", ascending=False)
        st.markdown(f"**{len(urgent)} SKUs require immediate reorder.** Top 10 by recommended order quantity:")
        st.dataframe(urgent.head(10), use_container_width=True)

        excess = rec[rec["risk_flag"] == "🔵 Overstock"]
        st.markdown(f"**{len(excess)} SKUs show excess inventory** relative to their reorder point.")
        if len(excess):
            st.dataframe(excess, use_container_width=True)

    st.subheader("Highest-Demand Products (Total Units Sold)")
    top_products = clean.groupby("product_id")[TARGET_COL].sum().sort_values(ascending=False).head(10)
    st.bar_chart(top_products)

    st.subheader("Store-Level Demand Patterns")
    store_avg = clean.groupby("store_id")[TARGET_COL].mean()
    st.bar_chart(store_avg)
    st.caption("Stores with unusually high/low average demand relative to the group are worth a closer look for local demand drivers.")
