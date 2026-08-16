-- =============================================================================
-- inventory_analysis.sql
-- Inventory turnover, stockout risk, and overstock analysis
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Inventory turnover proxy per product
-- Turnover = units sold / average inventory level. Higher = inventory moves
-- fast (good); very low = capital sitting idle in stock (overstock risk).
-- -----------------------------------------------------------------------------
SELECT
    product_id,
    category,
    ROUND(AVG(inventory_level), 2)          AS avg_inventory,
    SUM(units_sold)                         AS total_units_sold,
    ROUND(SUM(units_sold) * 1.0 / NULLIF(AVG(inventory_level), 0), 3) AS turnover_ratio
FROM sales
GROUP BY product_id, category
ORDER BY turnover_ratio ASC;


-- -----------------------------------------------------------------------------
-- 2. Stockout-risk analysis
-- A day is flagged "near-stockout" when closing inventory implied by
-- (inventory_level - units_sold) falls below 10% of that store-product's
-- typical inventory level. Window function computes the per-combo baseline.
-- -----------------------------------------------------------------------------
WITH combo_stats AS (
    SELECT
        store_id,
        product_id,
        AVG(inventory_level) OVER (PARTITION BY store_id, product_id) AS avg_inventory
    FROM sales
),
flagged AS (
    SELECT
        s.date,
        s.store_id,
        s.product_id,
        s.inventory_level,
        s.units_sold,
        (s.inventory_level - s.units_sold)                       AS implied_closing_inventory,
        c.avg_inventory,
        CASE
            WHEN (s.inventory_level - s.units_sold) < 0.10 * c.avg_inventory THEN 1
            ELSE 0
        END AS near_stockout_flag
    FROM sales s
    JOIN (SELECT DISTINCT store_id, product_id, avg_inventory FROM combo_stats) c
        ON s.store_id = c.store_id AND s.product_id = c.product_id
)
SELECT
    store_id,
    product_id,
    SUM(near_stockout_flag)                 AS near_stockout_days,
    COUNT(*)                                AS total_days,
    ROUND(100.0 * SUM(near_stockout_flag) / COUNT(*), 2) AS pct_days_near_stockout
FROM flagged
GROUP BY store_id, product_id
HAVING pct_days_near_stockout > 0
ORDER BY pct_days_near_stockout DESC
LIMIT 20;


-- -----------------------------------------------------------------------------
-- 3. Overstock analysis
-- Flags store-product combos where average inventory sits far above average
-- daily demand (i.e. many days' worth of stock idle on the shelf).
-- -----------------------------------------------------------------------------
SELECT
    store_id,
    product_id,
    category,
    ROUND(AVG(inventory_level), 2)          AS avg_inventory,
    ROUND(AVG(units_sold), 2)               AS avg_daily_sales,
    ROUND(AVG(inventory_level) / NULLIF(AVG(units_sold), 0), 1) AS days_of_supply
FROM sales
GROUP BY store_id, product_id, category
HAVING days_of_supply > (
    SELECT AVG(inventory_level) * 1.0 / NULLIF(AVG(units_sold), 0) * 1.5 FROM sales
)
ORDER BY days_of_supply DESC
LIMIT 20;


-- -----------------------------------------------------------------------------
-- 4. Inventory vs sales trend (monthly), per category
-- -----------------------------------------------------------------------------
SELECT
    category,
    strftime('%Y-%m', date)                 AS year_month,
    ROUND(AVG(inventory_level), 2)          AS avg_inventory,
    ROUND(AVG(units_sold), 2)               AS avg_units_sold
FROM sales
GROUP BY category, year_month
ORDER BY category, year_month;


-- -----------------------------------------------------------------------------
-- 5. Vendor / ordering pattern check
-- Compares units ordered vs units sold to see which products are
-- systematically over- or under-ordered relative to demand.
-- -----------------------------------------------------------------------------
SELECT
    product_id,
    SUM(units_ordered)                      AS total_ordered,
    SUM(units_sold)                         AS total_sold,
    ROUND(SUM(units_ordered) * 1.0 / NULLIF(SUM(units_sold), 0), 2) AS order_to_sales_ratio
FROM sales
GROUP BY product_id
ORDER BY order_to_sales_ratio DESC;
