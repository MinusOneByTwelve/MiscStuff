-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC 
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md # Transformations
-- MAGIC 
-- MAGIC Inspect, dedupe, and validate datasets using query transformations and column expressions.
-- MAGIC 
-- MAGIC ## Learning Objectives
-- MAGIC By the end of this lesson, you'll be able to:
-- MAGIC 1. Construct queries and column expressions to transform data
-- MAGIC 1. Summarize datasets and describe null behaviors
-- MAGIC 1. Retrieve and remove duplicates based on select columns
-- MAGIC 1. Validate datasets for expected counts, missing values, and duplicate records

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Run Setup
-- MAGIC 
-- MAGIC The setup script will create the data and declare necessary values for the rest of this notebook to execute.

-- COMMAND ----------

-- MAGIC %run ../Includes/setup-load

-- COMMAND ----------

-- MAGIC %md ## Column Expressions
-- MAGIC 
-- MAGIC Spark SQL queries use `SELECT` statements to subset existing columns and construct column expressions from datasets.
-- MAGIC 
-- MAGIC The following examples subset and create columns from the `events_clean` dataset.

-- COMMAND ----------

SELECT * FROM events_clean

-- COMMAND ----------

-- MAGIC %md ### Subset columns
-- MAGIC 
-- MAGIC Use `SELECT` to select from existing columns.
-- MAGIC 1. Select existing columns
-- MAGIC 1. Select nested fields using `.` notation
-- MAGIC 1. Select all subfields in a column using `*`
-- MAGIC 1. Provide aliases to rename columns, optionally using `AS`

-- COMMAND ----------

SELECT 
  event_name event,
  device,
  geo.city city,
  geo.state state,
  ecommerce.*
FROM events_clean

-- COMMAND ----------

-- MAGIC %md ### Construct column expressions
-- MAGIC 
-- MAGIC Create column expressions from:
-- MAGIC 1. Existing columns: `field`, `field.subfield`
-- MAGIC 1. Operators: `>`, `IN`, `IS NOT NULL`
-- MAGIC 1. Built-in functions: `size()`, `round()`

-- COMMAND ----------

SELECT 
  event_name,
  device IN ("Android", "iOS") mobile_user,
  ecommerce.purchase_revenue_in_usd IS NOT NULL purchase_event
FROM events_clean

-- COMMAND ----------

-- MAGIC %md ## Query Expressions
-- MAGIC 
-- MAGIC Along with `SELECT`, many additional query clauses can be employed to express transformations in Spark SQL queries.
-- MAGIC 
-- MAGIC The following examples subset, sort, and aggregate records from the `events_clean` dataset.

-- COMMAND ----------

-- MAGIC %md ### Filter rows
-- MAGIC `WHERE` filters rows based on one or more condition expressions.
-- MAGIC 
-- MAGIC Filter records for purchase events from mobile devices, Android or iOS.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW mobile_purchase_events AS
SELECT *
FROM events_clean
WHERE device IN ("Android", "iOS")
  AND ecommerce.purchase_revenue_in_usd IS NOT NULL;

SELECT * FROM mobile_purchase_events

-- COMMAND ----------

-- MAGIC %md ### Deduplicate rows
-- MAGIC 
-- MAGIC `DISTINCT` returns rows with duplicates removed.

-- COMMAND ----------

SELECT DISTINCT * FROM mobile_purchase_events

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC `DISTINCT *` returns rows with duplicates removed. `DISTINCT col` returns unique values in column `col`.
-- MAGIC 
-- MAGIC Identify all distinct values of `device` in the events data.

-- COMMAND ----------

SELECT DISTINCT device FROM events_clean

-- COMMAND ----------

-- MAGIC %md ### Sort rows
-- MAGIC 
-- MAGIC `ORDER BY` returns rows sorted by the given columns or expressions. `DESC` specifies descending order.
-- MAGIC 
-- MAGIC Sort mobile purchase events by revenue in descending order.

-- COMMAND ----------

SELECT *
FROM mobile_purchase_events
ORDER BY ecommerce.purchase_revenue_in_usd DESC

-- COMMAND ----------

-- MAGIC %md ### Aggregate rows
-- MAGIC 
-- MAGIC `GROUP BY` groups rows based on the specified columns. Aggregate functions operate on grouped rows and return a single record for each group.
-- MAGIC 
-- MAGIC For each traffic source, compute:
-- MAGIC - Count of distinct users introduced by the traffic source
-- MAGIC - Average purchase revenue associated with the traffic source
-- MAGIC 
-- MAGIC Aggregate functions: `avg()`, `approx_count_distinct()`, `round()`

-- COMMAND ----------

SELECT traffic_source,
  approx_count_distinct(user_id) distinct_users,
  round(avg(ecommerce.purchase_revenue_in_usd), 2) avg_revenue
FROM mobile_purchase_events
GROUP BY traffic_source

-- COMMAND ----------

-- MAGIC %md Summarize columns by applying aggregate functions on all rows. Compute min, max, and mean for purchase revenue and item quantity.

-- COMMAND ----------

WITH purchase_events AS (
  SELECT ecommerce.purchase_revenue_in_usd purchase_revenue, ecommerce.total_item_quantity item_quantity
FROM events_clean)
SELECT 
  min(purchase_revenue), max(purchase_revenue), round(avg(purchase_revenue), 2) avg_purchase_revenue,
  min(item_quantity), max(item_quantity), round(avg(item_quantity), 2) avg_item_quantity
FROM purchase_events

-- COMMAND ----------

-- MAGIC %md # Case Study: Inspect and Clean Data

-- COMMAND ----------

-- MAGIC %md ### Inspect Data
-- MAGIC 
-- MAGIC Inspect new users records in `users_update` for missing values and duplicate records.

-- COMMAND ----------

SELECT * FROM users_update

-- COMMAND ----------

-- MAGIC %md #### Count column values
-- MAGIC 
-- MAGIC Count the number of non-null values in each column of `update_users`.
-- MAGIC 
-- MAGIC `count(col)` skips `NULL` values when counting specific columns or expressions.

-- COMMAND ----------

SELECT count(*) total_rows, 
  count(user_id) user_ids,
  count(user_first_touch_timestamp) user_timestamps,
  count(email) emails 
FROM users_update

-- COMMAND ----------

-- MAGIC %md #### Identify missing values
-- MAGIC 
-- MAGIC Count the number of null values, if any, in each column of `users_update`.
-- MAGIC 
-- MAGIC - `count(col)` counts the total number of **non-null** values for a specified column or expression
-- MAGIC - `count(*)` is a special case that counts the total number of rows without skipping `NULL` values
-- MAGIC 
-- MAGIC The difference between these values (`count(*)` - `count(col)`) for each column equals the number of null values in that column.

-- COMMAND ----------

SELECT
  count(*) - count(user_id) null_user_ids,
  count(*) - count(user_first_touch_timestamp) null_user_timestamps,
  count(*) - count(email) null_emails
FROM users_update

-- COMMAND ----------

-- MAGIC %md Count missing values with filters and aggregation.

-- COMMAND ----------

SELECT 
  sum(cast(user_id IS NULL AS INT)) null_user_ids, 
  sum(cast(user_first_touch_timestamp IS NULL AS INT)) null_user_timestamps, 
  sum(cast(email IS NULL AS INT)) null_emails
FROM users_update

-- COMMAND ----------

-- MAGIC %md #### Count distinct values
-- MAGIC 
-- MAGIC Count distinct and duplicate `user_id` values

-- COMMAND ----------

SELECT count(DISTINCT *) distinct_rows,
       count(DISTINCT user_id) distinct_users,       
       count(DISTINCT *) - count(DISTINCT user_id) duplicate_users
FROM users_update

-- COMMAND ----------

-- MAGIC %md Notice the count for distinct users is greater than the count for distinct rows. 
-- MAGIC 
-- MAGIC Rows containing `NULL` values were skipped from processing distinct row counts.
-- MAGIC 
-- MAGIC `count(*)` is the only case where `count()` includes records with `NULL` values.

-- COMMAND ----------

-- MAGIC %md ## Deduplicate Records
-- MAGIC 
-- MAGIC Find and remove duplicates based on select columns.

-- COMMAND ----------

-- MAGIC %md #### Identify duplicate values
-- MAGIC 
-- MAGIC Find records containing duplicate values for `user_id`.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW duplicate_users AS
SELECT user_id, count(user_id) user_count
FROM users_update
GROUP BY user_id
HAVING user_count > 1
ORDER BY user_count DESC;

SELECT * FROM duplicate_users

-- COMMAND ----------

-- MAGIC %md  #### Count duplicate records
-- MAGIC 
-- MAGIC Count the number of duplicate values for `user_id`.

-- COMMAND ----------

SELECT count(*) FROM duplicate_users

-- COMMAND ----------

-- MAGIC %md #### Deduplicate single column
-- MAGIC Deduplicate rows based on values in `user_id`. 

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW deduped_user_ids AS
SELECT user_id, max(email) email
FROM users_update
GROUP BY user_id;

SELECT * FROM deduped_user_ids

-- COMMAND ----------

SELECT count(*) FROM deduped_user_ids

-- COMMAND ----------

-- MAGIC %md
-- MAGIC #### Deduplicate based on multiple columns
-- MAGIC Deduplicate user records based on both `user_id` and `user_first_touch_timestamp`.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW deduped_users AS
SELECT user_id, user_first_touch_timestamp, max(email) email, max(updated) updated
FROM users_update
GROUP BY user_id, user_first_touch_timestamp;

SELECT * FROM deduped_users

-- COMMAND ----------

SELECT count(*) FROM deduped_users

-- COMMAND ----------

-- MAGIC %md Let's also deduplicate the new event records we loaded into the `events_raw` bronze table.
-- MAGIC 
-- MAGIC We can use the `json_payload` view we created from `events_raw` with a parsed  `json` column.

-- COMMAND ----------

SELECT count(*) FROM events_raw

-- COMMAND ----------

SELECT json FROM json_payload LIMIT 2

-- COMMAND ----------

-- MAGIC %md Count duplicate event records in `json_payload` based on `user_id` and `event_timestamp`.

-- COMMAND ----------

WITH event_duplicates AS (
  SELECT json.user_id, json.event_timestamp, count(*) num_rows
  FROM json_payload
  GROUP BY json.user_id, json.event_timestamp
  HAVING num_rows > 1)
SELECT count(*) FROM event_duplicates

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW deduped_events AS
SELECT json.* FROM (
SELECT max(json) json FROM json_payload
GROUP BY json.user_id, json.event_timestamp
)

-- COMMAND ----------

SELECT count(*) FROM deduped_events

-- COMMAND ----------

-- MAGIC %md ## Validate Datasets
-- MAGIC Validate datasets for expected counts, missing values, and duplicate records.

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ### Confirm deduplication
-- MAGIC Count null and non-null values in `email` before and after deduplicating `user_updates`.

-- COMMAND ----------

SELECT 
  sum(cast(email IS NULL AS int)) null_emails, 
  sum(cast(email IS NOT NULL AS int)) non_null_emails
FROM users_update

-- COMMAND ----------

SELECT 
  sum(cast(email IS NULL AS int)) null_emails, 
  sum(cast(email IS NOT NULL AS int)) non_null_emails
FROM deduped_users

-- COMMAND ----------

-- null count in users_update - null count in deduped_users = num removed duplicates
SELECT 16849 - 15710

-- COMMAND ----------

SELECT count(*) FROM duplicate_users

-- COMMAND ----------

-- MAGIC %md ### Confirm unique email per user
-- MAGIC Confirm that each email is associated with at most one `user_id`

-- COMMAND ----------

SELECT not max(user_count) > 1 at_most_one FROM (
  SELECT email, count(user_id) AS user_count
  FROM deduped_users
  WHERE email IS NOT NULL
  GROUP BY email
  )

-- COMMAND ----------

-- MAGIC %md ### Validate deduplicated events

-- COMMAND ----------

SELECT not max(num_rows) > 1 events_deduplicated FROM (
  SELECT user_id, event_timestamp, count(*) num_rows
  FROM deduped_events
  GROUP BY user_id, event_timestamp)

-- COMMAND ----------

-- MAGIC %md ### Validate deduplicated users

-- COMMAND ----------

SELECT not max(num_rows) > 1 users_deduplicated FROM (
  SELECT user_id, max(email), count(*) num_rows
  FROM deduped_users
  GROUP BY user_id)

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
