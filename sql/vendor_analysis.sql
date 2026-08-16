-- =============================================================================
-- vendor_analysis.sql
-- Promotion effectiveness and competitor-pricing analysis
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Promotion lift: % increase in average units sold when promotion=1
-- vs promotion=0, per product. Subquery computes the baseline (no-promo)
-- rate so we can join it back and compute lift in one pass.
-- -----------------------------------------------------------------------------
WITH promo_avg AS (
    SELECT product_id, AVG(units_sold) AS avg_units_promo
    FROM sales WHERE promotion = 1
    GROUP BY product_id
),
no_promo_avg AS (
    SELECT product_id, AVG(units_sold) AS avg_units_no_promo
    FROM sales WHERE promotion = 0
    GROUP BY product_id
)
SELECT
    p.product_id,
    ROUND(p.avg_units_promo, 2)             AS avg_units_promo,
    ROUND(n.avg_units_no_promo, 2)          AS avg_units_no_promo,
    ROUND(100.0 * (p.avg_units_promo - n.avg_units_no_promo) / NULLIF(n.avg_units_no_promo, 0), 2) AS pct_lift
FROM promo_avg p
JOIN no_promo_avg n ON p.product_id = n.product_id
ORDER BY pct_lift DESC;


-- -----------------------------------------------------------------------------
-- 2. Competitor pricing gap and its relationship to units sold
-- Buckets each day into "we are cheaper" / "we are pricier" / "similar"
-- vs the competitor, then compares average demand across buckets.
-- -----------------------------------------------------------------------------
SELECT
    CASE
        WHEN price < competitor_price * 0.95 THEN 'We are cheaper'
        WHEN price > competitor_price * 1.05 THEN 'We are pricier'
        ELSE 'Price parity'
    END AS pricing_position,
    COUNT(*)                                AS n_rows,
    ROUND(AVG(units_sold), 2)               AS avg_units_sold
FROM sales
GROUP BY pricing_position
ORDER BY avg_units_sold DESC;


-- -----------------------------------------------------------------------------
-- 3. Discount depth vs demand response, per category
-- -----------------------------------------------------------------------------
SELECT
    category,
    discount_pct,
    ROUND(AVG(units_sold), 2)               AS avg_units_sold,
    COUNT(*)                                AS n_rows
FROM sales
GROUP BY category, discount_pct
ORDER BY category, discount_pct;


-- -----------------------------------------------------------------------------
-- 4. Weather and holiday/promotion combined effect on demand
-- -----------------------------------------------------------------------------
SELECT
    weather,
    CASE WHEN promotion = 1 THEN 'Promotion' ELSE 'No Promotion' END AS promo_flag,
    ROUND(AVG(units_sold), 2)               AS avg_units_sold
FROM sales
GROUP BY weather, promo_flag
ORDER BY weather, promo_flag;
