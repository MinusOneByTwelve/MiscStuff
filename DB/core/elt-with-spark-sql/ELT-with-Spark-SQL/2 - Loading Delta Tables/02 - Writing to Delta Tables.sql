-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC 
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC # Writing to Delta Tables
-- MAGIC Use SQL DML statements to perform complete and incremental updates to existing Delta tables.
-- MAGIC 
-- MAGIC ## Learning Objectives
-- MAGIC By the end of this lesson, you'll be able to:
-- MAGIC - Overwrite data tables using `INSERT OVERWRITE`
-- MAGIC - Append to a table using `INSERT INTO`
-- MAGIC - Append, update, and delete from a table using `MERGE INTO`
-- MAGIC - Ingest data incrementally into tables using `COPY INTO`

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Run Setup
-- MAGIC 
-- MAGIC The setup script will create the data and declare necessary values for the rest of this notebook to execute.

-- COMMAND ----------

-- MAGIC %run ../Includes/setup-migrate

-- COMMAND ----------

-- MAGIC %md ## Complete Overwrites
-- MAGIC 
-- MAGIC We can use overwrites to atomically replace all of the data in a table. 
-- MAGIC 
-- MAGIC The following cells demonstrate two ways to overwrite data.
-- MAGIC 
-- MAGIC 1. Using the `CREATE OR REPLACE TABLE` statement
-- MAGIC 2. Using the `INSERT OVERWRITE` statement

-- COMMAND ----------

CREATE OR REPLACE TABLE events_clean
LOCATION "${c.events_clean_table_path}" AS
SELECT * FROM parquet.`${c.source}/events/events.parquet`

-- COMMAND ----------

INSERT OVERWRITE events_clean
SELECT * FROM parquet.`${c.source}/events/events.parquet`

-- COMMAND ----------

DESCRIBE EXTENDED events_clean

-- COMMAND ----------

-- MAGIC %md This keeps history of the previous table, but rewrites all data.

-- COMMAND ----------

DESCRIBE HISTORY events_clean

-- COMMAND ----------

-- MAGIC %md There are multiple benefits to overwriting tables instead of deleting and recreating tables:
-- MAGIC - Overwriting a table is much faster because it doesn’t need to list the directory recursively or delete any files.
-- MAGIC - The old version of the table still exists; can easily retrieve the old data using Time Travel.
-- MAGIC - It’s an atomic operation. Concurrent queries can still read the table while you are deleting the table.
-- MAGIC - Due to ACID transaction guarantees, if overwriting the table fails, the table will be in its previous state.

-- COMMAND ----------

-- MAGIC %md ## Append Rows
-- MAGIC 
-- MAGIC Use `INSERT INTO` to atomically append new rows to an existing Delta table.
-- MAGIC 
-- MAGIC We will append newly processed JSON records to our bronze events table created below.

-- COMMAND ----------

CREATE OR REPLACE TABLE events_raw
(key BINARY, offset BIGINT, partition BIGINT, timestamp BIGINT, topic STRING, value BINARY, date DATE)
LOCATION "${c.events_raw_table_path}";

-- COMMAND ----------

-- MAGIC %md #### Case Study: Ingest Raw Events
-- MAGIC The clickstream dataset represents the bulk of the data being processed. There is a planned migration for the on-prem Kafka service into the cloud. During the present POC phase, you have been given a sample of raw data written as JSON files. Each file contains all records consumed during a 5 second interval, stored with the full Kafka schema as a multiple-record JSON file.
-- MAGIC 
-- MAGIC Let's extract the raw data from JSON files to load into Delta.

-- COMMAND ----------

CREATE TABLE IF NOT EXISTS json_events_kafka
(key BINARY, offset INT, partition BIGINT, timestamp BIGINT, topic STRING, value BINARY)
USING JSON OPTIONS (path = "${c.source}/events/events-kafka.json");

-- COMMAND ----------

-- MAGIC %md Use `INSERT INTO` to add these new events to the `events_raw` bronze table.

-- COMMAND ----------

INSERT INTO events_raw
SELECT * FROM (
  SELECT *, to_date(cast(timestamp/1e3 as timestamp)) date
  FROM json_events_kafka)
WHERE date > '2020-07-03'

-- COMMAND ----------

-- MAGIC %md This allows for incremental updates to existing tables, which is much more efficient than overwriting each time.

-- COMMAND ----------

DESCRIBE HISTORY events_raw

-- COMMAND ----------

-- MAGIC %md ## Merge Updates
-- MAGIC 
-- MAGIC You can upsert data from a source table, view, or DataFrame into a target Delta table using the MERGE SQL operation. Delta Lake supports inserts, updates and deletes in MERGE, and supports extended syntax beyond the SQL standards to facilitate advanced use cases.

-- COMMAND ----------

-- MAGIC %md #### Update Users
-- MAGIC Our connected system sends a record every time we see a `user_id` for the first time, or receive an email for a particular `user_id` for the first time. Each `user_id` is associated with at most 1 email address.
-- MAGIC 
-- MAGIC Update historic records with updated emails and add new users. Let's create a new table and temporary view loaded with historical and new users.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW new_users AS
SELECT * FROM parquet.`${c.source}/users/users-30m.parquet`;

CREATE OR REPLACE TABLE historical_users AS
SELECT * FROM parquet.`${c.source}/users/users.parquet`

-- COMMAND ----------

MERGE INTO historical_users a
  USING new_users b
  ON a.user_id = b.user_id
  WHEN MATCHED THEN
    UPDATE SET a.email = b.email
  WHEN NOT MATCHED
    THEN INSERT *;

-- COMMAND ----------

-- MAGIC %md As we implemented the `users` table as a Type 1 SCD Delta table with an `updated` field, we can leverage this field while performing a merge operation.
-- MAGIC 
-- MAGIC Let's make sure that records that are updated OR inserted have the same timestamp. This operation will be completed as a single batch to avoid potentially leaving our table in a corrupt state.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW users_update AS
SELECT current_timestamp() updated, * 
FROM parquet.`${c.source}/users/users-30m.parquet`;

MERGE INTO users a
  USING users_update b
  ON a.user_id = b.user_id
  WHEN MATCHED AND a.email IS NULL AND b.email IS NOT NULL THEN
    UPDATE SET email = b.email, updated = b.updated
  WHEN NOT MATCHED THEN
    INSERT *

-- COMMAND ----------

DESCRIBE HISTORY users

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Insert-Only Merge for Deduplication
-- MAGIC 
-- MAGIC A common ETL use case is to collect logs into Delta table by appending them to a table. However, often the sources can generate duplicate log records and downstream deduplication steps are needed to take care of them. With merge, you can avoid inserting the duplicate records.
-- MAGIC 
-- MAGIC By default, the merge operation searches the entire Delta table to find matches in the source table.

-- COMMAND ----------

-- MAGIC %md #### Case Study: Merge Parsed Events
-- MAGIC 
-- MAGIC As you promote the events data from its raw form, confirm that an identical record isn't already in your cleaned table
-- MAGIC 
-- MAGIC Make sure that you match schemas with those records already in your `events_clean` table.

-- COMMAND ----------

CREATE OR REPLACE TEMP VIEW json_payload AS
SELECT from_json(cast(value as STRING),
("device STRING, ecommerce STRUCT< purchase_revenue_in_usd: DOUBLE, total_item_quantity: BIGINT, unique_items: BIGINT>, event_name STRING, event_previous_timestamp BIGINT, event_timestamp BIGINT, geo STRUCT< city: STRING, state: STRING>, items ARRAY< STRUCT< coupon: STRING, item_id: STRING, item_name: STRING, item_revenue_in_usd: DOUBLE, price_in_usd: DOUBLE, quantity: BIGINT>>, traffic_source STRING, user_first_touch_timestamp BIGINT, user_id STRING")) json
FROM events_raw;

CREATE OR REPLACE TEMP VIEW events_update AS 
SELECT json.* FROM json_payload;

-- COMMAND ----------

MERGE INTO events_clean a
USING events_update b
ON a.user_id = b.user_id AND a.event_timestamp = b.event_timestamp
WHEN NOT MATCHED THEN 
  INSERT *

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 
-- MAGIC 
-- MAGIC One way to speed up merge is to reduce the search space by adding known constraints in the match condition. 
-- MAGIC 
-- MAGIC For example, suppose you have a table that is partitioned by country and date and you want to use merge to update information for the last day and a specific country. Adding the condition below will make the query faster as it looks for matches only in the relevant partitions.
-- MAGIC ```
-- MAGIC events.country = 'USA' AND events.date = current_date() - INTERVAL 7 DAYS
-- MAGIC ```  
-- MAGIC It will also reduce the chances of conflicts with other concurrent operations.

-- COMMAND ----------

INSERT OVERWRITE events_clean
SELECT * FROM parquet.`${c.source}/events/events.parquet`;

MERGE INTO events_clean a
USING events_update b
ON a.user_id = b.user_id AND a.event_timestamp = b.event_timestamp
WHEN NOT MATCHED AND b.traffic_source = 'email' THEN 
  INSERT *

-- COMMAND ----------

-- MAGIC %md ## Load Incrementally
-- MAGIC Use `COPY INTO` to incrementally load data from external systems.
-- MAGIC - Data schema should be consistent
-- MAGIC - Duplicate records should try to be excluded or handled downstream
-- MAGIC - Potentially much cheaper than full table scan for data that grows predictably
-- MAGIC - Leveraged by many data ingestion partners

-- COMMAND ----------

-- MAGIC %md #### Case Study: Update Sales
-- MAGIC 
-- MAGIC Update the `sales` delta table by incrementally loading data from an external location where a number of new transactions arrive during a 30 minute window. Each sale should only be recorded once, at the time that the transaction is processed.
-- MAGIC 
-- MAGIC Make sure that the schemas match between your new records and your already processed data. Use a method that allows idempotent execution to avoid processing data multiple times.

-- COMMAND ----------

COPY INTO sales
FROM "${c.source}/sales/sales-30m.parquet"
FILEFORMAT = PARQUET

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
