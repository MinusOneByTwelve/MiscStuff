-- Databricks notebook source
-- MAGIC %md
-- MAGIC #### Create the external locations required for this project
-- MAGIC 1. Bronze
-- MAGIC 2. Silver
-- MAGIC 3. Gold

-- COMMAND ----------

CREATE EXTERNAL LOCATION IF NOT EXISTS databrickscourseucextdl_bronze
 URL "abfss://bronze@databrickscourseucextdl.dfs.core.windows.net/"
 WITH (STORAGE CREDENTIAL `databrickscourse-ext-storage-credential`);


-- COMMAND ----------

DESC EXTERNAL LOCATION databrickscourseucextdl_bronze;

-- COMMAND ----------

-- MAGIC %fs
-- MAGIC ls "abfss://bronze@databrickscourseucextdl.dfs.core.windows.net/"

-- COMMAND ----------

CREATE EXTERNAL LOCATION IF NOT EXISTS databrickscourseucextdl_silver
 URL "abfss://silver@databrickscourseucextdl.dfs.core.windows.net/"
 WITH (STORAGE CREDENTIAL `databrickscourse-ext-storage-credential`);

-- COMMAND ----------

CREATE EXTERNAL LOCATION IF NOT EXISTS databrickscourseucextdl_gold
 URL "abfss://gold@databrickscourseucextdl.dfs.core.windows.net/"
 WITH (STORAGE CREDENTIAL `databrickscourse-ext-storage-credential`);

-- COMMAND ----------

