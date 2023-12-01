# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)
DA.reset_lesson()
DA.init()

DA.paths.datasets_path = DA.paths.datasets.replace("dbfs:/", "/dbfs/")
DA.paths.working_path = DA.paths.working_dir.replace("dbfs:/", "/dbfs/")

if dbgems.get_notebook_path().endswith("/01-Experimentation/01-Feature-Store"):
    DA.paths.airbnb = f"{DA.paths.working_dir}/airbnb/airbnb.delta"

DA.init_mlflow_as_job()

DA.conclude_setup()

