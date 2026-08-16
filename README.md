# AI-Powered Demand Forecasting & Inventory Optimization

An end-to-end data science system that predicts daily product demand per
store and converts those forecasts into actionable inventory decisions
(reorder points, safety stock, order quantities, stockout/overstock risk).

![architecture](architecture_diagram.png)

---

## 1. Business problem

A retailer with 5 stores x 20 products needs to answer, every day, per SKU:

1. How much demand should we expect?
2. Which products are at risk of stockout or sitting overstocked?
3. When should we reorder, and how much?

This project builds that decision-support system end to end: SQL analysis
-> EDA -> leakage-safe feature engineering -> ML forecasting -> evaluation
-> explainability -> inventory optimization -> a dashboard and API.

---

## 2. Dataset

**Kaggle: Retail Store Inventory Forecasting Dataset**
`data/raw/retail_store_inventory.csv`

- 73,100 rows = 5 stores x 20 products x 731 days (Jan 2022 - Jan 2024)
- Perfectly balanced panel, no missing dates, no duplicate rows, no nulls
- Columns: `Date, Store ID, Product ID, Category, Region, Inventory Level,
  Units Sold, Units Ordered, Demand Forecast, Price, Discount, Weather
  Condition, Holiday/Promotion, Competitor Pricing, Seasonality`
- `Demand Forecast` is a legacy vendor forecast column (673 negative values
  found on inspection) -- excluded from modeling, kept only as a reference

---

## 3. Architecture

```
Raw Dataset -> Data Validation & Cleaning -> SQL ETL/Analysis -> EDA
  -> Time-Series Feature Engineering -> Baseline Models -> ML Models
  -> Time-Based Validation -> Model Comparison -> Explainability (SHAP)
  -> Inventory Optimization -> Business Recommendations
  -> Streamlit Dashboard / FastAPI -> Documentation
```

See `notebooks/01` through `06` for the full, runnable pipeline in sequence.

---

## 4. Data pipeline (`src/data_processing.py`)

Validation checks performed: nulls, duplicates, negative values, panel
completeness (every store-product combo has exactly 731 rows -- confirmed),
and `units_sold <= inventory_level` (confirmed, 0 violations).

Cleaning decisions (each documented with reasoning in the code):
renamed to snake_case, parsed and sorted by date, duplicates dropped (0
found), **outliers deliberately NOT removed** (retail demand is naturally
spiky; removing spikes would strip the exact signal the model needs to
learn), categoricals cast to `category` dtype.

---

## 5. SQL component (`sql/`)

Three scripts, 19 queries total, all executed against a SQLite copy of the
cleaned data (`data/processed/retail.db`) and verified to run without error:

- `sales_analysis.sql` -- daily/weekly/monthly aggregation, product/store/
  category performance (window functions + `RANK()`), top/bottom sellers,
  promotion-wise sales (`CASE WHEN`), revenue by region and month
- `inventory_analysis.sql` -- inventory turnover ratio, stockout-risk
  flagging (CTE + window function baseline), overstock detection
  (`HAVING` + correlated subquery), order-vs-sales ratio
- `vendor_analysis.sql` -- promotion lift per product (CTE + `JOIN`),
  competitor pricing position vs demand, discount depth vs demand

---

## 6. EDA findings (`notebooks/02_eda.ipynb`)

All numbers below are the actual, executed results on this dataset:

- **No meaningful weekly seasonality.** Average daily demand ranges only
  135.1-137.3 units across days of the week (<2% spread).
- **No meaningful monthly seasonality.** Monthly averages range only
  134.0-139.4 units.
- **Promotions show ~0% average lift** (136.51 units/day without promotion
  vs 136.42 with) -- a genuinely surprising, worth-flagging-to-the-business
  finding, not the expected "promotions boost sales" story.
- **Demand is evenly spread across categories** (1.96M-2.03M total units
  each, within 3% of one another).
- **Inventory level correlates with sales at r=0.59** -- the strongest
  simple relationship found in EDA, foreshadowing the SHAP/importance result.

---

## 7. Feature engineering (`src/feature_engineering.py`)

Leakage-safe by construction:

| Type | Features |
|---|---|
| Lag | `lag_1, lag_7, lag_14, lag_28` |
| Rolling (shift(1) applied before rolling) | `rolling_mean_{7,14,28}`, `rolling_std_{7,28}` |
| Calendar | `day_of_week, week_of_year, month, quarter, year, is_weekend` |
| Business | `promotion, price, inventory_level, discount_pct, price_vs_competitor(_pct)` |
| Leak-safe target encoding | `cat_expanding_mean` (expanding mean of each row's own past only) |

Verified in notebook 03 with a manual check: `lag_1` on day T exactly equals
`units_sold` on day T-1 for a sample series. First 28 days per store-product
combo are dropped (incomplete history) rather than imputed -- 2,800 of
73,100 rows (3.8%), final feature set: **70,300 rows x 22 model features**,
zero NaNs remaining.

---

## 8. Train/validation/test split (`src/forecasting.py`)

**Chronological split, not random** -- train on the earliest 70% of dates,
validate on the next 15%, test on the final 15%. Random splitting would let
a model "train" on a day chronologically after the day it's evaluated on,
which never happens at inference time; the split must mimic deployment.

- Train: 49,200 rows (through 2023-06-04)
- Validation: 10,500 rows (through 2023-09-17)
- Test: 10,600 rows (after 2023-09-17)

---

## 9. Models and results

| Model | MAE | RMSE | MAPE % | WAPE % |
|---|---:|---:|---:|---:|
| Naive (lag_1) | 119.59 | 152.94 | 361.92 | 87.61 |
| Moving Average (7d) | 93.32 | 115.52 | 347.68 | 68.36 |
| Holt's Linear Trend (classical, SARIMA fallback) | 96.59 | 120.52 | 347.38 | 70.79 |
| **Gradient-boosted trees** | **68.95** | **88.10** | **244.87** | **50.51** |

All numbers above were produced by actually running the pipeline on this
dataset (see `notebooks/04`, `05`, `06`) -- none are invented.

**Model selection: gradient-boosted trees.** Beats the moving-average
baseline by ~26% relative WAPE reduction and the naive baseline by ~42%.
MAPE stays high in absolute terms for every model -- this reflects the data
(EDA found very weak seasonality and many low-volume rows, which inflates
percentage error), not a modeling failure; WAPE is the fairer summary metric
for this reason.

**Classical time-series comparison, an honest finding:** Holt's Linear
Trend (level + trend exponential smoothing -- a real, executed classical
method, used here as a dependency-free stand-in for SARIMA, which needs
`statsmodels`, not installable in this offline sandbox) actually performs
slightly *worse* than the plain moving average (70.79% vs 68.36% WAPE).
This makes sense given EDA: with almost no real trend or seasonality in
this dataset, a method built to actively track trend ends up chasing noise
instead of signal, while a flat moving average doesn't have that failure
mode. It's a useful negative result that reinforces the case for the final
model: gradient-boosted trees beat every baseline, classical or otherwise,
by a clear margin. `src/forecasting.py::fit_sarima` targets real SARIMA via
`statsmodels` for you to run locally and compare directly.

> **Execution note:** this sandbox has no internet access, so the numbers
> above were generated with `sklearn.ensemble.HistGradientBoostingRegressor`
> (same family of algorithm as XGBoost/LightGBM: histogram-binned gradient
> boosting), used as a real, honestly-labeled substitute rather than
> fabricating XGBoost/LightGBM numbers. The production code in
> `src/forecasting.py` (`fit_xgboost`, `fit_lightgbm`) targets the real
> libraries listed in `requirements.txt` -- re-run `05_model_training.ipynb`
> in an environment with them installed (it tries XGBoost, then LightGBM,
> then falls back automatically) to get your final reportable numbers before
> using them in a CV or interview.

---

## 10. Explainability

Production pipeline: `shap.TreeExplainer` on the XGBoost/LightGBM model.
Not installable in this offline sandbox, so **permutation importance**
(`sklearn.inspection.permutation_importance`) was actually run instead --
same question ("how much does this feature matter to predictions?") at the
global level. Real, executed top features (increase in MAE when shuffled):

| Feature | Importance |
|---|---:|
| `inventory_level` | 31.81 |
| `week_of_year` | 0.099 |
| `weather` | 0.056 |
| `rolling_mean_14` | 0.038 |
| `day_of_week` | 0.020 |

**Business interpretation:**
- `inventory_level` dominates every other feature by ~2 orders of
  magnitude. Demand in this dataset is structurally **capped by stock
  availability** -- on low-inventory days you simply can't sell what isn't
  there. Direct implication: improving stock availability may itself lift
  realized sales, independent of forecasting accuracy.
- Recent demand history matters, but far less than inventory -- consistent
  with the weak seasonality found in EDA.
- **Promotion does not appear in the top 15 features** -- consistent with
  the ~0% average lift found in EDA. Worth investigating with the business
  (is the flag logged correctly? confounded with discount depth?) rather
  than assumed to be working.

---

## 11. Inventory optimization (`src/inventory_optimization.py`)

Standard inventory-theory formulas (not invented), applied to the model's
test-period forecasts:

```
Safety Stock  = Z * sigma_demand * sqrt(lead_time_days)
Reorder Point = (avg_daily_forecast_demand * lead_time_days) + Safety Stock
Order Qty     = max(0, Reorder Point - Current Inventory)
```

**Assumptions (stated explicitly, changeable in `src/utils.py`):**
`lead_time_days = 7`, `service_level_z = 1.65` (~95% service level).
`sigma_demand` is each store-product's historical demand standard
deviation, computed only on pre-test-period data (leak-safe).

**Real result on the 100 store-product combinations (test-period snapshot):**
93 flagged 🔴 Stockout Risk, 7 flagged 🟡 Low Inventory, 0 🟢/🔵.

This is a genuine, defensible finding, not a bug: with inventory caps
around 500 units and daily demand often 100-250 units, on-hand stock
represents only ~2-5 days of supply -- well short of what a 7-day lead
time requires. **Actionable recommendation:** either shorten supplier lead
time or raise target inventory levels; at the current combination, the
assortment is structurally under-stocked.

---

## 12. Business output

`data/processed/inventory_recommendations.csv` -- one row per store-product
with `current_inventory, forecast_demand, safety_stock, reorder_point,
recommended_order_qty`, and a risk flag (🔴 Stockout Risk / 🟡 Low
Inventory / 🟢 Healthy Inventory / 🔵 Overstock).

---

## 13. Dashboard & API (`app/`)

- `app/app.py` -- Streamlit dashboard: Overview / Forecasting (filterable by
  store, product, category, region) / Inventory / Business Insights tabs.
  Run: `streamlit run app/app.py`
- `app/api.py` -- FastAPI `POST /predict` endpoint (store, product, recent
  sales, inventory, promotion, price, date -> predicted demand, stockout
  risk, recommended order quantity). Run: `uvicorn app.api:app --reload`

---

## 14. Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, LightGBM, SHAP, SQLite/SQL,
matplotlib, seaborn, Streamlit, FastAPI.

---

## 15. How to run

```bash
pip install -r requirements.txt

python -m src.data_processing        # clean data
python -m src.feature_engineering    # build features
# then run notebooks 04-06 in order, or:
jupyter notebook notebooks/

streamlit run app/app.py             # dashboard
uvicorn app.api:app --reload         # API
```

---

## 16. Future improvements

- Rolling/expanding-window backtesting (multiple folds) instead of a single
  chronological split, for a more robust error estimate.
- Investigate the ~0% measured promotion lift with the business before
  trusting `promotion` as a demand driver.
- Add a real public-holiday calendar if one becomes available (this
  dataset's `Holiday/Promotion` column conflates the two).
- Quantile/probabilistic forecasts (e.g. LightGBM quantile loss) to
  replace the historical-std safety-stock approximation with true
  forecast-uncertainty-based safety stock.
