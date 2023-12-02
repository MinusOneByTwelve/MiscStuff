-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC 
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Combining Data
-- MAGIC 
-- MAGIC Combine datasets and lookup tables with various join types and strategies.
-- MAGIC 
-- MAGIC ### Learning Objectives
-- MAGIC By the end of this lesson, you'll be able to:
-- MAGIC - Combine datasets using different types of joins
-- MAGIC - Join records to a pre-existing lookup table
-- MAGIC - Examine and provide hints for join strategies

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Run Setup
-- MAGIC 
-- MAGIC The setup script will create the data and declare necessary values for the rest of this notebook to execute.

-- COMMAND ----------

-- MAGIC %run ../Includes/setup-complex

-- COMMAND ----------

-- MAGIC %md ## Join Types
-- MAGIC Apply different types of joins on `sales`, `users`, and `carts` to retrieve the emails of unconverted users with abandoned carts.
-- MAGIC 
-- MAGIC Use `sales` to create a view containing emails of converted users who have made purchases. Add a `converted` column to identify these users after joining the other datasets.

-- COMMAND ----------

CREATE OR REPLACE VIEW converted_users AS
SELECT DISTINCT email, True AS converted FROM sales;

SELECT * FROM converted_users

-- COMMAND ----------

-- MAGIC %md ### Left Join
-- MAGIC Returns all values from the left relation and the matched values from the right relation, or appends `NULL` if there is no match. Also referred to as a left outer join.
-- MAGIC 
-- MAGIC Perform an left join on the `users` dataset and the `converted_users` view created above. This will help identify any `user_id` values associated with each email, in addition to identifying users who have not yet made purchases.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW user_conversions_temp AS
SELECT user_id, a.email, converted
FROM users a
LEFT JOIN converted_users b
ON a.email = b.email
WHERE a.email IS NOT NULL;
  
SELECT * FROM user_conversions_temp

-- COMMAND ----------

-- MAGIC %md The left join will return all records from `users` and any matched records from `converted_users`. `user` records without any matches in the `converted_users` view will have `NULL` values for the `converted` field. 
-- MAGIC 
-- MAGIC Replace `NULL` values in the `converted` column with `False` to indicate unconverted users.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW user_conversions AS
SELECT user_id, email, ifnull(converted, False) converted
FROM user_conversions_temp;
  
SELECT * FROM user_conversions

-- COMMAND ----------

SELECT converted, count(*) num_users
FROM user_conversions
GROUP BY converted

-- COMMAND ----------

-- MAGIC %md ### Anti Join
-- MAGIC Returns values from the left relation that have no matches with the right. Also referred to as a left anti join.
-- MAGIC 
-- MAGIC Perform an anti join on the `users` dataset and `converted_users` view created above. This will return records for unconverted users who haven't yet made any purchases.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW unconverted_users AS
SELECT user_id, email FROM users a
ANTI JOIN converted_users b
ON a.email = b.email
WHERE a.email IS NOT NULL;
  
SELECT * FROM unconverted_users

-- COMMAND ----------

-- MAGIC %md For the next part, we'll use the cart item history view `carts` that we created in the last lesson.

-- COMMAND ----------

SELECT * FROM carts

-- COMMAND ----------

-- MAGIC %md ### Inner Join
-- MAGIC 
-- MAGIC Selects rows that have matching values in both relations. This is the default join in Spark SQL.
-- MAGIC 
-- MAGIC Perform an inner join on `unconverted_users` and `carts` to identify the emails and cart items of users who have added items to their carts but abandoned them without making any purchases.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW abandoned_carts AS
SELECT a.user_id, a.email, b.cart_items
FROM unconverted_users a
JOIN carts b 
ON a.user_id = b.user_id;
  
SELECT * FROM abandoned_carts

-- COMMAND ----------

-- MAGIC %md ## Join a Lookup Table
-- MAGIC 
-- MAGIC Lookup tables are normally small, historical tables used to enrich new data passing through an ETL pipeline.
-- MAGIC 
-- MAGIC In this example, we will use a small lookup table to get details for each item sold by this retailer.

-- COMMAND ----------

CREATE OR REPLACE TABLE item_lookup AS 
SELECT item_id, STRUCT(item_id, name, price) item
FROM parquet.`${c.source}/products/products.parquet`;

SELECT * FROM item_lookup

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW carts_exploded AS
SELECT *,  explode(cart_items) cart_item 
FROM carts;

SELECT * FROM carts_exploded

-- COMMAND ----------

-- MAGIC %md The items listed in `carts_exploded` only specify the ID of each item.
-- MAGIC 
-- MAGIC Perform an inner join on `carts_exploded` and the `item_lookup` table to add item details for each cart.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW cart_details AS
SELECT a.user_id, b.item
FROM carts_exploded a
INNER JOIN item_lookup b
ON a.cart_item = b.item_id;

SELECT * FROM cart_details

-- COMMAND ----------

SELECT user_id, collect_list(item) cart_items
FROM cart_details
GROUP BY user_id

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Examine Join Strategy
-- MAGIC Use the `EXPLAIN` command to view the physical plan used to execute the query.   
-- MAGIC Look for BroadcastHashJoin or BroadcastExchange.

-- COMMAND ----------

EXPLAIN FORMATTED 
SELECT * FROM cart_details

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC By default, Spark performed a broadcast join rather than a shuffle join. That is, it broadcasted `item_lookup` to the larger `carts_exploded`, replicating the smaller dataset on each node of our cluster. This avoided having to move the larger dataset across the cluster.

-- COMMAND ----------

-- MAGIC %md `autoBroadcastJoinThreshold`
-- MAGIC 
-- MAGIC We can access configuration settings to take a look at the broadcast join threshold. This specifies the maximum size in bytes for a table that broadcasts to worker nodes.

-- COMMAND ----------

SET spark.sql.autoBroadcastJoinThreshold

-- COMMAND ----------

-- MAGIC %md Re-examine physical plan when join is executed while broadcasting is disabled
-- MAGIC 1. Drop threshold to `-1` to disable broadcasting
-- MAGIC 1. Explain join
-- MAGIC 
-- MAGIC Now notice the lack of broadcast in the query physical plan.

-- COMMAND ----------

SET spark.sql.autoBroadcastJoinThreshold=-1

-- COMMAND ----------

-- MAGIC %md Notice a sort merge join is performed, rather than a broadcast join.

-- COMMAND ----------

EXPLAIN FORMATTED 
SELECT * FROM cart_details

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC ## Broadcast Join Hint
-- MAGIC Use a join hint to suggest broadcasting the lookup table for the join.
-- MAGIC 
-- MAGIC The join side with this hint will be broadcast regardless of `autoBroadcastJoinThreshold`.

-- COMMAND ----------

EXPLAIN FORMATTED
SELECT /*+ BROADCAST(b) */ a.user_id, b.item
FROM carts_exploded a
INNER JOIN item_lookup b
ON a.cart_item = b.item_id

-- COMMAND ----------

-- MAGIC %md Reset the original threshold.

-- COMMAND ----------

SET spark.sql.autoBroadcastJoinThreshold=10485760b

-- COMMAND ----------

-- MAGIC %md ## Adaptive Query Execution
-- MAGIC Adaptive Query Execution (AQE) is an optimization technique in Spark SQL that makes use of the runtime statistics to choose the most efficient query execution plan, which is enabled by default since Apache Spark 3.2.0. 
-- MAGIC 
-- MAGIC As of Spark 3.0, there are three major features in AQE including:
-- MAGIC - Coalescing post-shuffle partitions
-- MAGIC - Converting sort-merge join to broadcast join
-- MAGIC - Skew join optimization
-- MAGIC 
-- MAGIC Spark SQL can turn on and off AQE using `spark.sql.adaptive.enabled` as an umbrella configuration. 

-- COMMAND ----------

SET spark.sql.adaptive.enabled

-- COMMAND ----------

-- MAGIC %md #### Converting sort-merge join to broadcast join
-- MAGIC AQE converts sort-merge join to broadcast hash join when the runtime statistics of any join side is smaller than the adaptive broadcast hash join threshold.
-- MAGIC 
-- MAGIC `spark.sql.adaptive.autoBroadcastJoinThreshold` `(default = none)`
-- MAGIC 
-- MAGIC Configures the maximum size in bytes for a table that will be broadcast to all worker nodes when performing a join. By setting this value to -1 broadcasting can be disabled. The default value is same with `spark.sql.autoBroadcastJoinThreshold`. Note that, this config is used only in adaptive framework.

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
