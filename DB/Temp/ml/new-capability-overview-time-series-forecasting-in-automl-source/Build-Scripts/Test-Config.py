# Databricks notebook source
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy --quiet --disable-pip-version-check

# COMMAND ----------

# DBTITLE 1,Configure the Test
import os
from dbacademy import dbgems
from dbacademy import dbrest
from dbacademy import dbtest
from dbacademy import dbpublish

client = dbrest.DBAcademyRestClient()

test_config = dbtest.TestConfig(
  name = "Example Course",                                     # The name of the course
  spark_version = dbgems.get_current_spark_version(),          # Current version
  workers = 0,                                                 # Test in local mode
  libraries = [],                                              # Libraries to attache to the cluster
  cloud = dbgems.get_cloud(),                                  # The cloud this test is running in
  instance_pool = dbgems.get_current_instance_pool_id(client), # AWS, GCP or MSA instance pool
)

print("Test Configuration")
test_config.print()
