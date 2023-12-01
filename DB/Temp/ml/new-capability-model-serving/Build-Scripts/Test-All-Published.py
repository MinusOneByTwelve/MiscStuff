# Databricks notebook source
# MAGIC %md # Scalable Machine Learning with Apache Spark
# MAGIC 
# MAGIC This test suite can be ran by simply hitting **Run All** however, it will only test the the single version specified in [Test-Config]($./Test-Config)
# MAGIC 
# MAGIC Multiple versions are tested via a parameter and generally initiated via a Spark Job
# MAGIC 
# MAGIC Any lingering state can be addressed by properly configuring [Reset]($../Source/Includes/Reset) to drop databases, tables or delete on-disk artifacts

# COMMAND ----------

# DBTITLE 1,Publish to Test
# MAGIC %run ./Publish-All $testing=true

# COMMAND ----------

# DBTITLE 1,Configure the Client
from dbacademy.dbtest import *
from dbacademy.dbrest import DBAcademyRestClient

client = DBAcademyRestClient()
home = dbgems.get_notebook_dir(offset=-2)
home = home.replace("-source", "-STUDENT-COPY")
print(f"Home: {home}")

# COMMAND ----------

# DBTITLE 1,Define Tests
from dbacademy import dbtest

# When True, successfull jobs will not be deleted and the results
# will include links to those successful jobs. Normally set to False
keep_success = False

def execute_tests():
  ignored_on_gcp = (dbgems.get_cloud() == "GCP")

  round_1 = dbtest.SuiteBuilder(client, course_name, test_type="Source")
  round_1.add(f"{home}/{dist_name}/Includes/Reset")
  round_1.add(f"{home}/{dist_name}/Version Info")

  round_2 = dbtest.SuiteBuilder(client, course_name, test_type="Source")
  round_2.add(f"{home}/{dist_name}/EC 01 - Your First Lesson")
  round_2.add(f"{home}/{dist_name}/EC 02 - Your Second Lesson")
  round_2.add(f"{home}/{dist_name}/EC 03 - Your Third Lesson")
  round_2.add(f"{home}/{dist_name}/EC 04 - You Fourth Lesson")

  round_2.add(f"{home}/{dist_name}/Solutions/Labs/EC 01L - Your First Lab")
  round_2.add(f"{home}/{dist_name}/Solutions/Labs/EC 02L - Your Second Lab")
  round_2.add(f"{home}/{dist_name}/Solutions/Labs/EC 03L - Your Third Lab")
  round_2.add(f"{home}/{dist_name}/Solutions/Labs/EC 04L - You Fourth Lab")
  
  all_jobs = {**round_1.jobs, **round_2.jobs}
  client.jobs().delete_by_name(all_jobs, success_only=False)
  
  # Rest the project before doing anything
  for job_name in round_1.jobs:
    dbtest.test_one_notebook(client, test_config, job_name, round_1.jobs[job_name])
  
  # Once reset, run all the tests
  dbtest.test_all_notebooks(client, round_2.jobs, test_config)  
  dbtest.wait_for_notebooks(client, test_config, round_2.jobs, fail_fast=False)

  client.jobs().delete_by_name(all_jobs, success_only = not keep_success)  

# COMMAND ----------

# DBTITLE 1,Execute Tests
execute_tests()

# COMMAND ----------

# DBTITLE 1,Test Resutls
df = spark.read.table(test_config.results_table).filter(f"suite_id = '{test_config.suite_id}'")
evaluator = dbtest.ResultsEvaluator(df)
displayHTML(evaluator.to_html())

# COMMAND ----------

assert evaluator.passed, "One or more failures detected"
