# Databricks notebook source
# MAGIC %run ./Classroom-Setup

# COMMAND ----------

import re

course = "ncoaml"
username = spark.sql("SELECT current_user()").first()[0]
database = f"{re.sub('[^a-zA-Z0-9]', '_', username)}_{course}"

dbutils.widgets.text("mode", "setup")
mode = dbutils.widgets.get("mode")

if mode == "reset" or mode == "cleanup":
  spark.sql(f"DROP DATABASE IF EXISTS {database} CASCADE")

if mode != "cleanup":
  spark.sql(f"CREATE DATABASE IF NOT EXISTS {database}")
  spark.sql(f"USE {database}")

  spark.read.parquet(input_path).write.mode("overwrite").saveAsTable("sf_listings")

None
