-- Databricks notebook source
-- MAGIC %md
-- MAGIC ##### Query data via unity catalog using 3 level namespace

-- COMMAND ----------

SELECT * FROM demo_catalog.demo_schema.circuits;

-- COMMAND ----------

USE CATALOG demo_catalog;
USE SCHEMA demo_schema;
SELECT * FROM circuits;

-- COMMAND ----------

SELECT current_catalog()

-- COMMAND ----------

SHOW CATALOGS;

-- COMMAND ----------

SELECT current_schema();

-- COMMAND ----------

SHOW SCHEMAS;

-- COMMAND ----------

SHOW tables;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC display(spark.sql('SHOW tables'))

-- COMMAND ----------

-- MAGIC %python
-- MAGIC df = spark.table('demo_catalog.demo_schema.circuits')

-- COMMAND ----------

-- MAGIC %python 
-- MAGIC display(df)

-- COMMAND ----------

