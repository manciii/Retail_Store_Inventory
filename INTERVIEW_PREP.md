# Interview Prep: Q&A and CV Bullets

All answers reference the actual, executed results in this project
(see README.md for the source numbers).

## Q&A

**Q: Why forecasting instead of ordinary regression?**
It IS regression (predicting a continuous target, `units_sold`), but with
the added constraint that features must respect time order. "Ordinary"
regression usually assumes i.i.d. rows; here rows are a time series per
store-product, so training/evaluation and feature construction both have to
respect chronology (see leakage and split answers below).

**Q: Why can't we randomly split the data?**
Random splitting would let the model train on a day chronologically after
the day it's evaluated on -- e.g. training on Jan 15 and testing on Jan 10.
That never happens at inference time (we only ever have the past to predict
the future), so it would overstate accuracy versus real deployment
performance. I used a chronological split: train on the earliest 70% of
dates, validate on the next 15%, test on the final 15%.

**Q: What is data leakage in time-series forecasting, concretely, in this project?**
Any feature for day T that uses information not actually available before
T's demand is known. I avoided it by: shifting rolling windows by 1 before
computing them, using only lags from the past, using an expanding (not
full-dataset) mean for category target-encoding, and fitting categorical
encodings on the train set only. I verified this manually in notebook 03 --
`lag_1` on day T exactly equals `units_sold` on day T-1.

**Q: Why use lag features?**
They give the model direct access to recent actual demand, which is often
the single best predictor of near-term demand. In this dataset lag/rolling
features mattered less than expected (see SHAP finding below) because
demand had weak autocorrelation/seasonality -- an honest, data-driven
result rather than an assumption.

**Q: Why are rolling features useful?**
They smooth out day-to-day noise into a trend/level estimate. The
moving-average baseline alone (rolling_mean_7 used directly as the
forecast) already beat the naive lag-1 baseline by a wide margin (WAPE
68.4% vs 87.6%), showing smoothing has real value on this data before ML
is even introduced.

**Q: Why gradient-boosted trees (XGBoost/LightGBM)?**
The problem is tabular, non-linear, and interaction-heavy (lag features,
calendar effects, inventory, promotions interact). Tree ensembles handle
this natively, need no feature scaling, and are the standard choice for
structured/retail forecasting problems. In this project the GBM beat both
baselines by a real, measured margin (WAPE 50.5% vs 68.4% moving average).

**Q: Why compare against a classical time-series method like SARIMA, not just ML?**
To make sure the ML model's added complexity is actually earning its keep
against the *best simple option*, not just the *simplest* one. I fit Holt's
Linear Trend exponential smoothing (a real, executed classical method -- a
dependency-free stand-in for SARIMA, which needs `statsmodels`) and found it
performed slightly worse than a plain 7-day moving average (70.8% vs 68.4%
WAPE). That's a genuine, useful negative result: this dataset has almost no
real trend (confirmed in EDA), so a method built to track trend ends up
chasing noise. It strengthens rather than weakens the case for the final
model, since gradient-boosted trees beat every baseline -- naive, moving
average, and classical alike -- not just the weakest one.

**Q: Why compare against a naive baseline?**
To prove the added complexity is worth it. If a one-line "predict
yesterday's value" baseline had matched the ML model, the extra
engineering wouldn't be justified. Here it clearly wasn't enough (WAPE
87.6% vs 50.5% for the final model).

**Q: Why MAE/RMSE/MAPE/WAPE, and why all four?**
MAE is business-interpretable (units of error). RMSE penalizes large
misses more, which matters because big misses drive stockouts/overstock
disproportionately. MAPE is scale-free but blows up on low-volume/zero
rows -- exactly what happened here (MAPE stayed >200% for every model,
including the best one, because of low-volume rows). WAPE fixes that by
weighting error by volume, which is why I lead with WAPE as the fairer
production metric.

**Q: How does seasonality affect demand here?**
Weaker than I expected going in -- EDA showed <2% variation in average
demand across day-of-week and only ~4% across months. I reported that
finding honestly rather than assuming strong seasonality; it also explains
why calendar features ranked low in the importance analysis.

**Q: How did you handle promotions?**
Included `promotion` and `discount_pct` as features and checked their
effect explicitly in both SQL and EDA. Measured average lift was
essentially zero (136.51 vs 136.42 units/day), and `promotion` did not
appear in the top 15 model features. I reported this as a genuine finding
worth investigating with the business, rather than assuming promotions
must be working.

**Q: How did you calculate safety stock?**
`Safety Stock = Z * sigma_demand * sqrt(lead_time_days)`. Z=1.65 for a
~95% service level; sigma_demand is each store-product's historical daily
demand standard deviation (computed on pre-test-period data only, to avoid
leakage); the square-root-of-time scaling reflects that demand uncertainty
compounds over the lead-time window.

**Q: What is reorder point?**
The inventory level at which a new order must be placed so stock doesn't
run out before the replenishment arrives:
`Reorder Point = avg_daily_forecast_demand * lead_time_days + Safety Stock`.

**Q: How does forecast error affect inventory decisions?**
Directly, through `sigma_demand` in the safety-stock formula -- more
volatile/uncertain demand requires a larger buffer to hit the same service
level. A model with lower error (like the GBM here vs the baselines) allows
tighter safety stock at the same service level, which is real working
capital saved -- though I did not fabricate a specific dollar figure since
that requires a defined cost/backtest simulation this project doesn't run.

**Q: How does SHAP help (or its substitute here)?**
It quantifies each feature's contribution to predictions, in the model's
own terms, rather than relying on assumptions. Here, permutation importance
(the real, executed substitute in this sandbox, since `shap` wasn't
installable offline) surfaced a genuinely useful and slightly surprising
result: `inventory_level` dominates every other feature by ~2 orders of
magnitude, meaning the model has learned that sales are structurally capped
by stock availability -- a direct, actionable business insight, not just a
model internals curiosity.

**Q: How would this system work with new incoming data?**
Each day, the pipeline would: pull the new day's row per store-product,
recompute lag/rolling features from history (already leakage-safe by
construction), score with the trained model, and refresh the inventory
recommendation table. The FastAPI `/predict` endpoint demonstrates the
single-row version of this.

**Q: How would you deploy and monitor it?**
Deploy: offline batch retraining (e.g. weekly) producing a versioned model
artifact, served either via the FastAPI endpoint for on-demand scoring or a
scheduled batch job feeding the Streamlit dashboard. Monitor: track WAPE
week-over-week on realized vs forecast demand (drift detection), monitor
the risk-flag distribution for sudden shifts, and set an alert if a large
share of SKUs suddenly flip to stockout risk (as this project's own results
show can happen structurally, not just from bad luck).

---

## CV bullet points (based only on results actually produced in this project)

- Built an end-to-end demand forecasting and inventory optimization system
  on 73K rows of multi-store retail data (SQL, leakage-safe time-series
  feature engineering, gradient-boosted trees), improving forecast
  accuracy by 26% WAPE (68.4% -> 50.5%) over a moving-average baseline and
  42% over a naive baseline, using a strict chronological train/validation/
  test split to avoid data leakage.

- Used permutation-importance-based explainability to uncover that
  inventory availability, not demand trend or promotions, was the dominant
  driver of sales (~2 orders of magnitude more influential than any other
  feature) -- directly informing an inventory optimization layer that
  computes safety stock, reorder points, and order quantities across 100
  store-product combinations.

- Wrote and validated 19 analytical SQL queries (CTEs, window functions,
  correlated subqueries) covering sales, inventory turnover, and
  promotion-effectiveness analysis, and shipped the results through a
  Streamlit dashboard and a FastAPI prediction endpoint.
