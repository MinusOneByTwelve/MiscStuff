# Databricks notebook source
# MAGIC %md
# MAGIC # Validate & Update Repo

# COMMAND ----------

# DBTITLE 1,Common Configuration
# MAGIC %run ./Test-Config

# COMMAND ----------

# DBTITLE 1,Update the Source & Student Branches
source_home = dbgems.get_notebook_dir(offset=-2)
dbpublish.update_and_validate_git_branch(client, source_home)

print("-"*80)

student_home = source_home.replace("-source", "-STUDENT-COPY")
dbpublish.update_and_validate_git_branch(client, student_home)
