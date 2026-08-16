-- =============================================================================
-- sales_analysis.sql
-- Sales aggregation and performance analysis
-- Table: sales(date, store_id, product_id, category, region, inventory_level,
--              units_sold, units_ordered, legacy_demand_forecast, price,
--              discount_pct, weather, promotion, competitor_price, seasonality_label)
-- Dialect: SQLite (functions translate directly to Postgres/MySQL with minor tweaks)
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Daily sales aggregation (total units + revenue across all stores/products)
-- Useful for: spotting overall demand trend, day-of-week effects.
-- -----------------------------------------------------------------------------
SELECT
    date,
    SUM(units_sold)                         AS total_units_sold,
    ROUND(SUM(units_sold * price), 2)       AS total_revenue,
    COUNT(DISTINCT store_id)                AS active_stores,
    COUNT(DISTINCT product_id)              AS active_products
FROM sales
GROUP BY date
ORDER BY date;


-- -----------------------------------------------------------------------------
-- 2. Weekly sales aggregation
-- strftime('%Y-%W', date) buckets by ISO-ish week; used to smooth daily noise
-- and see week-over-week momentum.
-- -----------------------------------------------------------------------------
SELECT
    strftime('%Y-%W', date)                AS year_week,
    MIN(date)                               AS week_start,
    SUM(units_sold)                         AS total_units_sold,
    ROUND(AVG(units_sold), 2)               AS avg_daily_units,
    ROUND(SUM(units_sold * price), 2)       AS total_revenue
FROM sales
GROUP BY year_week
ORDER BY year_week;


-- -----------------------------------------------------------------------------
-- 3. Monthly sales aggregation
-- Reveals monthly seasonality (e.g. holiday-season spikes).
-- -----------------------------------------------------------------------------
SELECT
    strftime('%Y-%m', date)                 AS year_month,
    SUM(units_sold)                         AS total_units_sold,
    ROUND(SUM(units_sold * price), 2)       AS total_revenue,
    ROUND(SUM(units_sold * price) * 1.0 / SUM(units_sold), 2) AS avg_realized_price
FROM sales
GROUP BY year_month
ORDER BY year_month;


-- -----------------------------------------------------------------------------
-- 4. Product-level performance (ranked by total revenue)
-- CTE + window function to rank products and flag top vs long-tail performers.
-- -----------------------------------------------------------------------------
WITH product_perf AS (
    SELECT
        product_id,
        category,
        SUM(units_sold)                     AS total_units,
        ROUND(SUM(units_sold * price), 2)   AS total_revenue,
        ROUND(AVG(units_sold), 2)           AS avg_daily_units
    FROM sales
    GROUP BY product_id, category
)
SELECT
    *,
    RANK() OVER (ORDER BY total_revenue DESC) AS revenue_rank
FROM product_perf
ORDER BY revenue_rank;


-- -----------------------------------------------------------------------------
-- 5. Store-level performance
-- -----------------------------------------------------------------------------
SELECT
    store_id,
    region,
    SUM(units_sold)                         AS total_units,
    ROUND(SUM(units_sold * price), 2)       AS total_revenue,
    ROUND(AVG(units_sold), 2)               AS avg_daily_units_per_product,
    COUNT(DISTINCT product_id)              AS distinct_products
FROM sales
GROUP BY store_id, region
ORDER BY total_revenue DESC;


-- -----------------------------------------------------------------------------
-- 6. Category-level performance
-- -----------------------------------------------------------------------------
SELECT
    category,
    SUM(units_sold)                         AS total_units,
    ROUND(SUM(units_sold * price), 2)       AS total_revenue,
    ROUND(100.0 * SUM(units_sold) / (SELECT SUM(units_sold) FROM sales), 2) AS pct_of_total_units
FROM sales
GROUP BY category
ORDER BY total_revenue DESC;


-- -----------------------------------------------------------------------------
-- 7. Top-selling products (top 10 by units)
-- -----------------------------------------------------------------------------
SELECT product_id, category, SUM(units_sold) AS total_units
FROM sales
GROUP BY product_id, category
ORDER BY total_units DESC
LIMIT 10;


-- -----------------------------------------------------------------------------
-- 8. Low-performing products (bottom 10 by units)
-- -----------------------------------------------------------------------------
SELECT product_id, category, SUM(units_sold) AS total_units
FROM sales
GROUP BY product_id, category
ORDER BY total_units ASC
LIMIT 10;


-- -----------------------------------------------------------------------------
-- 9. Promotion-wise sales analysis
-- CASE WHEN to label promotion status; compares avg units sold and realized
-- discount-adjusted price with vs without promotion, per category.
-- -----------------------------------------------------------------------------
SELECT
    category,
    CASE WHEN promotion = 1 THEN 'Promotion' ELSE 'No Promotion' END AS promo_flag,
    ROUND(AVG(units_sold), 2)               AS avg_units_sold,
    ROUND(AVG(discount_pct), 2)             AS avg_discount_pct,
    COUNT(*)                                AS n_days_products
FROM sales
GROUP BY category, promo_flag
ORDER BY category, promo_flag;


-- -----------------------------------------------------------------------------
-- 10. Revenue analysis by region and month (JOIN-style aggregation via
--     subquery for month-over-month comparison)
-- -----------------------------------------------------------------------------
SELECT
    region,
    strftime('%Y-%m', date)                 AS year_month,
    ROUND(SUM(units_sold * price), 2)       AS revenue
FROM sales
GROUP BY region, year_month
ORDER BY region, year_month;
