-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC 
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Creating Delta Tables
-- MAGIC 
-- MAGIC After extracting data from external data sources, loading data into the Lakehouse is of the utmost importance to ensure that all of the benefits of the Databricks platform can be fully leveraged.
-- MAGIC 
-- MAGIC While different organizations may have varying policies for how data is initially loaded into Databricks, we typically recommend that early tables represent a mostly raw version of the data, and that validation and enrichment occur in later stages. This pattern ensures that even if data doesn't match expectations with regards to data types or column names, no data will be dropped, meaning that programmatic or manual intervention can still salvage data in a partially corrupted or invalid state.
-- MAGIC 
-- MAGIC This lesson will focus on the primary pattern used to create most tables, `CREATE TABLE _ AS SELECT` (CTAS) statements.
-- MAGIC 
-- MAGIC ## Learning Objectives
-- MAGIC By the end of this lesson, you'll be able to:
-- MAGIC - Use CTAS statements to create Delta Lake tables
-- MAGIC - Create new tables from existing views or tables
-- MAGIC - Enrich loaded data with additional metadata
-- MAGIC - Declare table schema with generated columns and descriptive comments
-- MAGIC - Set advanced options to control data location, quality enforcement, and partitioning

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Run Setup
-- MAGIC 
-- MAGIC The setup script will create the data and declare necessary values for the rest of this notebook to execute.

-- COMMAND ----------

-- MAGIC %run ../Includes/setup-extract

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Create an Empty Delta Table
-- MAGIC Use the `CREATE TABLE USING` statement to define an empty Delta table in the metastore.

-- COMMAND ----------

CREATE TABLE IF NOT EXISTS demo_users (user_id STRING, email STRING)

-- COMMAND ----------

-- MAGIC %md We won't need to explicitly state `USING DELTA`, as Delta is the default format.
-- MAGIC 
-- MAGIC Confirm this using `DESCRIBE EXTENDED`.

-- COMMAND ----------

DESCRIBE EXTENDED demo_users

-- COMMAND ----------

-- MAGIC %md ## Create Table From an Existing View
-- MAGIC 
-- MAGIC We can use the `CREATE TABLE AS SELECT` (CTAS) command to load data stored in different formats into Delta Lake tables. CTAS statements create and populate Delta tables using data retrieved from an input query.
-- MAGIC 
-- MAGIC Let's create a table using a CTAS statement from the view shown below.

-- COMMAND ----------

CREATE OR REPLACE VIEW view_csv
AS SELECT * FROM csv.`${c.source}/sales/sales.csv`;

SELECT * FROM view_csv

-- COMMAND ----------

-- MAGIC %md Without setting any options for reading CSV data, we end up with a single column of unparsed text.

-- COMMAND ----------

DESCRIBE EXTENDED view_csv

-- COMMAND ----------

-- MAGIC %md When creating a table from this view using a CTAS statement, we don't have the options needed to properly ingest CSV records. Because they inherit schemas from the query data, CTAS statements also do not support schema declarations.

-- COMMAND ----------

CREATE OR REPLACE TABLE sales_unparsed AS
SELECT * FROM view_csv;

SELECT * FROM sales_unparsed

-- COMMAND ----------

-- MAGIC %md The resulting table has the same schema as the source view.

-- COMMAND ----------

DESCRIBE EXTENDED sales_unparsed

-- COMMAND ----------

-- MAGIC %md ## Subset Columns From Existing Tables
-- MAGIC We can also use CTAS statements to load data from existing tables.
-- MAGIC 
-- MAGIC Let's reference the `csv_orders` table we created previously with the `CREATE TABLE USING` statement to process CSV data.

-- COMMAND ----------

SELECT * FROM csv_orders

-- COMMAND ----------

-- MAGIC %md The following CTAS statement creates a new table containing a subset of columns from this table.

-- COMMAND ----------

CREATE OR REPLACE TABLE purchases AS
SELECT order_id, transactions_timestamp, purchase_revenue_in_usd
FROM csv_orders;

SELECT * FROM purchases

-- COMMAND ----------

-- MAGIC %md CTAS statements can also reference files directly using the `format.filepath` syntax in the input query.

-- COMMAND ----------

CREATE OR REPLACE TABLE events AS
SELECT * FROM parquet.`${c.source}/events/events.parquet`

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Enrich Tables with Additional Metadata
-- MAGIC Create a managed Delta table enriched with additional metadata.
-- MAGIC 1. Add a table comment
-- MAGIC 1. Add an arbitrary key-value pair as a table property
-- MAGIC 1. Add columns to record current timestamp and filename
-- MAGIC 
-- MAGIC **NOTE**: A number of Delta Lake configurations are set using `TBLPROPERTIES`. When using this field as part of an organizational approach to data discovery and auditting, users should be made aware of which keys are leveraged for modifying default Delta Lake behaviors.

-- COMMAND ----------

CREATE OR REPLACE TABLE users_pii
COMMENT "Contains PII"
TBLPROPERTIES ('contains_pii' = True) AS
SELECT current_timestamp() updated, "users" dataset, *
FROM jdbc_users

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC All of the comments and properties for a given table can be reviewed using `DESCRIBE TABLE EXTENDED`.
-- MAGIC 
-- MAGIC **NOTE**: Delta Lake automatically adds several table properties on table creation.

-- COMMAND ----------

DESCRIBE EXTENDED users_pii

-- COMMAND ----------

-- MAGIC %md ## Declare Schema with Generated Columns
-- MAGIC Create a managed Delta table and declare a table schema to:
-- MAGIC 1. Define column names and types
-- MAGIC 1. Create generated columns
-- MAGIC 1. Add descriptive column comments
-- MAGIC 
-- MAGIC [Generated columns](https://docs.databricks.com/delta/delta-batch.html#deltausegeneratedcolumns) are a special type of column whose values are automatically generated based on a user-specified function over other columns in the Delta table.

-- COMMAND ----------

CREATE OR REPLACE TABLE sales_dates (
  order_id STRING, 
  transactions_timestamp STRING, 
  purchase_revenue_in_usd STRING,
  date DATE GENERATED ALWAYS AS (
    CAST(CAST(transactions_timestamp/1e6 AS timestamp) AS DATE))
    COMMENT "generated based on timestamp column")

-- COMMAND ----------

-- MAGIC %md If we write to `sales_dates` without providing values for the `date` column, Delta Lake automatically computes them.

-- COMMAND ----------

-- MAGIC %python 
-- MAGIC spark.table("purchases").write.mode("append").saveAsTable("sales_dates")

-- COMMAND ----------

SELECT * FROM sales_dates

-- COMMAND ----------

-- MAGIC %md The query automatically reads the most recent snapshot of the table for any query; you never need to run `REFRESH TABLE`.

-- COMMAND ----------

-- MAGIC %md ## Control Table Locations
-- MAGIC 
-- MAGIC We can create an external Delta table that is unmanaged by the metastore by specifying a `LOCATION` path for the table.

-- COMMAND ----------

CREATE OR REPLACE TABLE sales_external
LOCATION "${c.userhome}/tmp/sales_external"
AS SELECT * FROM sales_dates;

SELECT * FROM sales_external

-- COMMAND ----------

DESCRIBE EXTENDED sales_external

-- COMMAND ----------

-- MAGIC %md We can also create a Delta table using a `LOCATION` that *already* contains data stored using Delta. 
-- MAGIC 
-- MAGIC The resulting table will inherit configurations from the existing data, including the schema, partitioning, or table properties.

-- COMMAND ----------

CREATE TABLE IF NOT EXISTS sales_external_copy
LOCATION "${c.userhome}/tmp/sales_external";

SELECT * FROM sales_external_copy

-- COMMAND ----------

-- MAGIC %md **NOTE**: If we do specify any of these configurations, they must exactly match the configurations of the location data.
-- MAGIC If not, Delta Lake throws an exception that describes the discrepancy.

-- COMMAND ----------

-- CREATE TABLE IF NOT EXISTS sales_external_diff (
--   order_id STRING,
--   transactions_timestamp BIGINT,
--   purchase_revenue_in_usd STRING,
--   date date
-- )
-- LOCATION "${c.userhome}/tmp/sales_external";

-- SELECT * FROM sales_external_diff

-- COMMAND ----------

-- MAGIC %md ## Partition Tables
-- MAGIC You can partition a Delta table by columns to speed up queries or DML that have predicates involving the partition columns.
-- MAGIC 
-- MAGIC The cell below creates a table partitioned by the `date` column. Each unique value found in `date` will create a separate directory for data.
-- MAGIC 
-- MAGIC **NOTE:** Partitioning cannot be specified without defining the table schema.

-- COMMAND ----------

CREATE OR REPLACE TABLE sales_external (
  order_id STRING, 
  transactions_timestamp STRING, 
  purchase_revenue_in_usd STRING,
  date DATE GENERATED ALWAYS AS (
    CAST(CAST(transactions_timestamp/1e6 AS timestamp) AS DATE))
    COMMENT "generated based on timestamp column")
LOCATION "${c.userhome}/tmp/sales_external"
PARTITIONED BY (date)

-- COMMAND ----------

-- MAGIC %md Listing the location used for the table reveals that the unique values in the partition column are used to generate data directories. 

-- COMMAND ----------

-- MAGIC %python 
-- MAGIC spark.table("purchases").write.mode("overwrite").saveAsTable("sales_external")
-- MAGIC display(dbutils.fs.ls(f"{Paths.userhome}/tmp/sales_external"))

-- COMMAND ----------

-- MAGIC %md Note that the Parquet format used to store the data for Delta Lake leverages these partitions directly when determining column value (the column values for date are not stored redundantly within the data files).

-- COMMAND ----------

SELECT * FROM sales_external WHERE date = "2020-06-19"

-- COMMAND ----------

-- MAGIC %md Delta Lake automatically uses partitioning and statistics to read the minimum amount of data when there are applicable predicates in the query. 

-- COMMAND ----------

-- MAGIC %md 
-- MAGIC # Case Study: Migrate Datasets
-- MAGIC Migrate historical datasets from external sources into Delta Lake in the specified locations.
-- MAGIC 
-- MAGIC We will store our Delta Lake tables in a `delta_tables` directory in your userhome. For convenience, we declared a variable `base_path` with this file path, along with additional variables for the locations of the delta lake tables in this case study.

-- COMMAND ----------

SELECT "${c.base_path}"

-- COMMAND ----------

-- MAGIC %python 
-- MAGIC print(Paths.base_path)
-- MAGIC print(Paths.sales_table_path)
-- MAGIC print(Paths.users_table_path)
-- MAGIC print(Paths.events_clean_table_path)

-- COMMAND ----------

CREATE OR REPLACE TABLE sales
LOCATION "${c.sales_table_path}" AS
SELECT * FROM parquet.`${c.source}/sales/sales.parquet`

-- COMMAND ----------

CREATE OR REPLACE TABLE users
LOCATION "${c.users_table_path}" AS
SELECT current_timestamp() updated, *
FROM parquet.`${c.source}/users/users.parquet`    

-- COMMAND ----------

CREATE OR REPLACE TABLE events_clean
LOCATION "${c.events_clean_table_path}" AS
SELECT * FROM parquet.`${c.source}/events/events.parquet`

-- COMMAND ----------

-- MAGIC %python display(dbutils.fs.ls(Paths.base_path))

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
