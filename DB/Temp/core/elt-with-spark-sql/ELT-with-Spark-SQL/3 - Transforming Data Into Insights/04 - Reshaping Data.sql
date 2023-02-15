-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC 
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md # Aggregation
-- MAGIC Aggregate and reshape data with pivot tables, rollups, and cubes.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Run Setup
-- MAGIC 
-- MAGIC The setup script will create the data and declare necessary values for the rest of this notebook to execute.

-- COMMAND ----------

-- MAGIC %run ../Includes/setup-clean

-- COMMAND ----------

-- MAGIC %md We will work with this dataset on purchase events.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW purchase_events AS

SELECT
  device,
  user_id,
  event_timestamp,
  traffic_source,
  item.item_id,
  item.item_revenue_in_usd revenue,
  item.price_in_usd price,
  item.quantity
FROM 
  (SELECT *, explode(items) item FROM events_clean)
WHERE
  item.item_revenue_in_usd IS NOT NULL;

SELECT * FROM purchase_events

-- COMMAND ----------

-- MAGIC %md ## Pivot
-- MAGIC The PIVOT clause is used for data perspective. We can get the aggregated values based on specific column values, which will be turned to multiple columns used in SELECT clause. The PIVOT clause can be specified after the table name or subquery.
-- MAGIC 
-- MAGIC **`SELECT * FROM ()`**: The `SELECT` statement inside the parentheses is the input for this table.
-- MAGIC 
-- MAGIC **`PIVOT`**: The first argument in the clause is an aggregate function and the column to be aggregated. Then, we specify the pivot column in the `FOR` subclause. The `IN` operator contains the pivot column values. <br>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Example 1 - Total revenue by device and traffic source

-- COMMAND ----------

SELECT * FROM (
  SELECT 
    device,
    traffic_source,
    revenue
  FROM purchase_events)
PIVOT (
ROUND(sum(revenue), 2) AS total_revenue FOR traffic_source IN (
  'google',
  'email',
  'instagram',
  'direct',
  'youtube',
  'facebook'
))

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC Example 2 - Quantities of each item by user

-- COMMAND ----------

SELECT * FROM (
  SELECT
    item_id,
    quantity,
    user_id
  FROM
    purchase_events
) PIVOT (
  ROUND(sum(quantity), 2) AS total_quantity FOR item_id IN (
    "P_FOAM_S",
    "P_DOWN_S",
    "M_STAN_K",
    "M_STAN_Q",
    "M_PREM_Q",
    "M_PREM_K",
    "M_PREM_T",
    "M_STAN_T",
    "M_STAN_F",
    "P_FOAM_K",
    "P_DOWN_K",
    "M_PREM_F"
  )
)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Rollups
-- MAGIC 
-- MAGIC Rollups are operators used with the `GROUP BY` clause. They allow you to summarize data based on the columns passed to the `ROLLUP` operator.
-- MAGIC 
-- MAGIC In this example, we've calculated the total quantity of items purchased across all traffic sources and all devices, as well as the total quantity from each device.

-- COMMAND ----------

SELECT
  device,
  traffic_source,
  sum(quantity) quantity
FROM
  purchase_events
GROUP BY
  ROLLUP (device, traffic_source)
ORDER BY
  device,
  traffic_source

-- COMMAND ----------

-- MAGIC %md
-- MAGIC We can use the `COALESCE` function to make the output more readable.
-- MAGIC 
-- MAGIC Instead of showing the null value on the aggregated rows, we'll name them by the aggregates they represent: 
-- MAGIC 
-- MAGIC `"All devices"` and `"All traffic sources"`

-- COMMAND ----------

SELECT
  COALESCE(device, "All devices") device,
  COALESCE(traffic_source, "All traffic sources") traffic_source,
  sum(quantity) AS total_quantity
FROM
  purchase_events
GROUP BY
  ROLLUP (device, traffic_source)
ORDER BY
  device,
  traffic_source

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC Notice that the null values in the results table have been replaced with more descriptive names.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cube
-- MAGIC `CUBE` is also an operator used with the `GROUP BY` clause. Similar to `ROLLUP`, you can use `CUBE` to generate summary values for sub-elements grouped by column value.
-- MAGIC 
-- MAGIC `CUBE` is different than `ROLLUP` in that it will also generate subtotals for all combinations of grouping columns specified in the `GROUP BY` clause. 
-- MAGIC 
-- MAGIC Notice that the output for the example below shows some of additional values generated in this query. Data from `"All devices"` has been aggregated by traffic sources for all devices. 

-- COMMAND ----------

SELECT
  COALESCE(device, "All devices") device,
  COALESCE(traffic_source, "All traffic sources") traffic_source,
  sum(quantity) total_quantity
FROM
  purchase_events
GROUP BY
  CUBE (device, traffic_source)
ORDER BY
  device,
  traffic_source

-- COMMAND ----------

-- MAGIC %md We can see all the combinations of grouped data now.

-- COMMAND ----------

-- MAGIC %md # Case Study: Combine and Reshape Data

-- COMMAND ----------

-- MAGIC %md ## Create Transactions
-- MAGIC Create a new `transactions` table that flattens out the information contained in the `sales` table and joins this with `users` table. 
-- MAGIC 
-- MAGIC Join these tables on email address, but without propagating this email address forward (to avoid potential PII exposure in downstream tables).

-- COMMAND ----------

SELECT * FROM sales

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW transactions AS

SELECT * FROM (
  SELECT
    user_id,
    order_id,
    transaction_timestamp,
    total_item_quantity,
    purchase_revenue_in_usd,
    unique_items,
    a.items_exploded.item_id item_id,
    a.items_exploded.quantity quantity
  FROM
    ( SELECT *, explode(items) items_exploded FROM sales ) a
    INNER JOIN users b 
    ON a.email = b.email
) PIVOT (
  sum(quantity) FOR item_id in (
    'P_FOAM_K',
    'M_STAN_Q',
    'P_FOAM_S',
    'M_PREM_Q',
    'M_STAN_F',
    'M_STAN_T',
    'M_PREM_K',
    'M_PREM_F',
    'M_STAN_K',
    'M_PREM_T',
    'P_DOWN_S',
    'P_DOWN_K'
  )
)

-- COMMAND ----------

SELECT * FROM transactions

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
