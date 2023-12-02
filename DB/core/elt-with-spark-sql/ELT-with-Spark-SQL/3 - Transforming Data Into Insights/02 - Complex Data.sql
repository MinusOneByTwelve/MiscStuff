-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC 
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md # Transforming Complex Data
-- MAGIC 
-- MAGIC Apply built-in functions to transform a variety of complex data types to clean and enrich data.
-- MAGIC 
-- MAGIC ## Learning Objectives
-- MAGIC By the end of this lesson, you'll be able to:
-- MAGIC 1. Transform complex types using built-in functions
-- MAGIC 1. Convert and extract attributes from dates and timestamps
-- MAGIC 1. Manipulate text with string functions and regular expressions
-- MAGIC 1. Explode and flatten complex data using various array functions

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Run Setup
-- MAGIC 
-- MAGIC The setup script will create the data and declare necessary values for the rest of this notebook to execute.

-- COMMAND ----------

-- MAGIC %run ../Includes/setup-clean

-- COMMAND ----------

-- MAGIC %md ## Dates and Timestamps

-- COMMAND ----------

-- MAGIC %md ### Cast to timestamp

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW sale_timestamps AS
  SELECT 
    transaction_timestamp,
    CAST(transaction_timestamp / 1e6 AS timestamp) timestamp
  FROM sales;
  
SELECT * FROM sale_timestamps

-- COMMAND ----------

-- MAGIC %md ### Format datetimes

-- COMMAND ----------

SELECT *,
  date_format(timestamp, "MMMM dd, yyyy") date_string,
  date_format(timestamp, "HH:mm:ss.SSSSSS") time_string
FROM sale_timestamps

-- COMMAND ----------

-- MAGIC %md ### Extract from timestamp

-- COMMAND ----------

SELECT timestamp,
  year(timestamp) year,
  month(timestamp) month,
  dayofweek(timestamp) dayofweek,
  minute(timestamp) minute,
  second(timestamp) second
FROM sale_timestamps

-- COMMAND ----------

-- MAGIC %md ### Convert to date and manipulate datetimes

-- COMMAND ----------

SELECT *,
  to_date(timestamp) date,
  date_add(timestamp, 2) plus_two_days
FROM sale_timestamps

-- COMMAND ----------

-- MAGIC %md ## Complex Data

-- COMMAND ----------

SELECT items FROM sales

-- COMMAND ----------

-- MAGIC %md ### Explode arrays
-- MAGIC 
-- MAGIC Explode arrays in the item field of events
-- MAGIC 
-- MAGIC Array Functions: `explode`

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW events_exploded AS
  SELECT *, explode(items) item
  FROM events_clean;
  
SELECT * FROM events_exploded

-- COMMAND ----------

-- MAGIC %md ### Collect and flatten arrays
-- MAGIC Create cart item history for each user
-- MAGIC - Collect unique set of items in each user's cart history
-- MAGIC - Flatten the resulting arrays of arrays into a single array
-- MAGIC 
-- MAGIC Array Functions: `flatten`, `collect_set`

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW carts AS
SELECT user_id, 
  flatten(collect_set(items.item_id)) cart_items
FROM events_exploded
GROUP BY user_id;

SELECT * FROM carts

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Split strings into arrays
-- MAGIC Split item_name string to extract item details from purchases
-- MAGIC 
-- MAGIC String Functions: `split`

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW item_details AS
SELECT user_id, split(item.item_name, " ") details
FROM events_exploded;

SELECT * FROM item_details

-- COMMAND ----------

-- MAGIC %md ### Extract from arrays
-- MAGIC 
-- MAGIC Extract product and size options for standard quality items
-- MAGIC 
-- MAGIC Array Functions: `element_at`, `array_contains`, `collect_set`

-- COMMAND ----------

WITH mattress_purchase_details AS (
  SELECT element_at(details, 2) size,
    element_at(details, 1) quality,
    element_at(details, 3) product
  FROM item_details
  WHERE array_contains(details, "Standard")
)
SELECT product, collect_set(size) size_options
FROM mattress_purchase_details
GROUP BY product

-- COMMAND ----------

-- MAGIC %md ### Extract with Regex
-- MAGIC Use regex to extract domains from the email column
-- MAGIC 
-- MAGIC String Functions: `regexp_extract`

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW email_domains AS
  SELECT email, regexp_extract(email, "(?<=@)[^.]+(?=\.)", 0) domain
  FROM sales;
SELECT * FROM email_domains

-- COMMAND ----------

SELECT domain, count(*) AS count FROM email_domains
GROUP BY domain ORDER BY count DESC

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
