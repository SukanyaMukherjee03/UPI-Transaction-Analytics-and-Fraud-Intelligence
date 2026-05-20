-- ============================================================
-- UPI PULSE 2024 — SQL BUSINESS ANALYSIS
-- Database: PostgreSQL
-- Table: transactions
-- ============================================================


-- Q1. What is the overall scale of the UPI platform?

SELECT
    COUNT(*)                                          AS total_transactions,
    ROUND(SUM(amount_inr)/10000000.0, 2)             AS total_value_crore,
    ROUND(AVG(amount_inr), 0)                         AS avg_transaction_value,
    ROUND(100.0 * SUM(is_success)  / COUNT(*), 2)    AS success_rate,
    ROUND(100.0 * SUM(fraud_flag)  / COUNT(*), 3)    AS fraud_rate,
    COUNT(DISTINCT sender_state)                      AS states_covered,
    COUNT(DISTINCT sender_bank)                       AS banks_active
FROM transactions;


-- Q2. How does transaction volume and value vary across months?

SELECT
    month,
    month_num,
    COUNT(*)                              AS total_transactions,
    ROUND(SUM(amount_inr)/100000.0, 2)   AS total_value_lakhs,
    ROUND(AVG(amount_inr), 0)             AS avg_transaction_value,
    SUM(fraud_flag)                       AS fraud_cases
FROM transactions
GROUP BY month, month_num
ORDER BY month_num;


-- Q3. Which day of the week has the highest transaction activity?

SELECT
    day_of_week,
    COUNT(*)                              AS transaction_count,
    ROUND(AVG(amount_inr), 0)             AS avg_transaction_value
FROM transactions
GROUP BY day_of_week
ORDER BY transaction_count DESC;


-- Q4. At what hours of the day is UPI usage the highest?

SELECT
    hour_of_day,
    COUNT(*)                              AS transaction_count,
    ROUND(AVG(amount_inr), 0)             AS avg_transaction_value
FROM transactions
GROUP BY hour_of_day
ORDER BY hour_of_day;


-- Q5. How does transaction behavior differ between weekdays and weekends?

SELECT
    CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS day_type,
    COUNT(*)                              AS transaction_count,
    ROUND(AVG(amount_inr), 0)             AS avg_transaction_value,
    ROUND(100.0 * SUM(fraud_flag) / COUNT(*), 3) AS fraud_rate
FROM transactions
GROUP BY is_weekend;


-- Q6. Which transaction types dominate UPI usage?

SELECT
    transaction_type,
    COUNT(*)                                              AS transaction_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions), 2) AS share_percentage,
    ROUND(AVG(amount_inr), 0)                             AS avg_transaction_value
FROM transactions
GROUP BY transaction_type
ORDER BY transaction_count DESC;


-- Q7. Which merchant categories contribute the highest transaction value?

SELECT
    merchant_category,
    COUNT(*)                              AS transaction_count,
    ROUND(SUM(amount_inr)/100000.0, 2)   AS total_value_lakhs,
    ROUND(AVG(amount_inr), 0)             AS avg_transaction_value
FROM transactions
GROUP BY merchant_category
ORDER BY total_value_lakhs DESC;


-- Q8. Which sender age groups are the most active spenders?

SELECT
    sender_age_group,
    COUNT(*)                              AS transaction_count,
    ROUND(AVG(amount_inr), 0)             AS avg_spend
FROM transactions
GROUP BY sender_age_group
ORDER BY avg_spend DESC;


-- Q9. How do different age groups distribute spending across merchant categories?

SELECT
    sender_age_group,
    merchant_category,
    COUNT(*)                              AS transaction_count
FROM transactions
GROUP BY sender_age_group, merchant_category
ORDER BY transaction_count DESC;


-- Q10. Which states contribute the most to UPI transaction volume and value?

SELECT
    sender_state,
    COUNT(*)                              AS transaction_count,
    ROUND(SUM(amount_inr)/100000.0, 2)   AS total_value_lakhs
FROM transactions
GROUP BY sender_state
ORDER BY total_value_lakhs DESC;


-- Q11. Which banks dominate UPI usage?

SELECT
    sender_bank,
    COUNT(*)                                              AS transaction_count,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions), 2) AS market_share
FROM transactions
GROUP BY sender_bank
ORDER BY transaction_count DESC;


-- Q12. Which bank-to-bank routes are most frequently used?

SELECT
    CONCAT(sender_bank, ' to ', receiver_bank)   AS bank_flow,
    COUNT(*)                                      AS transaction_count,
    ROUND(SUM(amount_inr)/100000.0, 2)            AS total_value_lakhs
FROM transactions
GROUP BY sender_bank, receiver_bank
ORDER BY transaction_count DESC
LIMIT 10;


-- Q13. Which banks have the highest transaction failure rates?

SELECT
    sender_bank,
    COUNT(*)                                                            AS total_transactions,
    SUM(CASE WHEN transaction_status = 'FAILED' THEN 1 ELSE 0 END)    AS failed_transactions,
    ROUND(
        100.0 * SUM(CASE WHEN transaction_status = 'FAILED' THEN 1 ELSE 0 END)
        / COUNT(*), 2
    )                                                                   AS failure_rate
FROM transactions
GROUP BY sender_bank
ORDER BY failure_rate DESC;


-- Q14. Which device and network combinations are least reliable?

SELECT
    device_type,
    network_type,
    COUNT(*)                                                            AS transaction_count,
    ROUND(
        100.0 * SUM(CASE WHEN transaction_status = 'FAILED' THEN 1 ELSE 0 END)
        / COUNT(*), 3
    )                                                                   AS failure_rate
FROM transactions
GROUP BY device_type, network_type
ORDER BY failure_rate DESC;


-- Q15. Where is fraud most concentrated geographically?

SELECT
    sender_state,
    COUNT(*)                                          AS total_transactions,
    SUM(fraud_flag)                                   AS fraud_cases,
    ROUND(100.0 * SUM(fraud_flag) / COUNT(*), 3)     AS fraud_rate
FROM transactions
GROUP BY sender_state
ORDER BY fraud_cases DESC;


-- Q16. Which device and network combinations are most associated with fraud?

SELECT
    device_type,
    network_type,
    COUNT(*)                                          AS transactions,
    ROUND(100.0 * SUM(fraud_flag) / COUNT(*), 3)     AS fraud_rate
FROM transactions
GROUP BY device_type, network_type
ORDER BY fraud_rate DESC;


-- Q17. Are high-value transactions riskier than normal ones?

SELECT
    CASE
        WHEN amount_inr > 10000 THEN 'High Value (>10K)'
        ELSE 'Normal (<=10K)'
    END                                               AS transaction_segment,
    COUNT(*)                                          AS transaction_count,
    ROUND(100.0 * SUM(fraud_flag) / COUNT(*), 3)     AS fraud_rate
FROM transactions
GROUP BY 1;


-- Q18. What are the most fraud-prone combinations of state, type, and category?

SELECT
    sender_state,
    transaction_type,
    merchant_category,
    COUNT(*)                                          AS total_transactions,
    SUM(fraud_flag)                                   AS fraud_cases
FROM transactions
GROUP BY sender_state, transaction_type, merchant_category
ORDER BY fraud_cases DESC
LIMIT 10;


-- Q19. What is the distribution of transaction amounts?

SELECT
    PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY amount_inr) AS p25_value,
    PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY amount_inr) AS median_transaction,
    PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY amount_inr) AS p75_value,
    PERCENTILE_CONT(0.90) WITHIN GROUP (ORDER BY amount_inr) AS p90_value,
    PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY amount_inr) AS p99_value
FROM transactions;


-- Q20. Is the UPI platform dependent on a few dominant banks?

SELECT
    sender_bank,
    ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM transactions), 2) AS contribution_percentage
FROM transactions
GROUP BY sender_bank
ORDER BY contribution_percentage DESC
LIMIT 10;