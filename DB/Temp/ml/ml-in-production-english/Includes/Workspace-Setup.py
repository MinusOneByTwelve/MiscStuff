# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

lesson_config.create_schema = False

DA = DBAcademyHelper(course_config, lesson_config)
DA.install_datasets(reinstall_datasets=False)

