# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)
DA.init()

DA.paths.datasets_path = DA.paths.datasets.replace("dbfs:/", "/dbfs/")
DA.paths.working_path = DA.paths.working_dir.replace("dbfs:/", "/dbfs/")

DA.conclude_setup()

