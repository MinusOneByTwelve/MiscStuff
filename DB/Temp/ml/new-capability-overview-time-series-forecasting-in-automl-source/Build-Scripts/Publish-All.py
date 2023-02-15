# Databricks notebook source
# DBTITLE 1,Publish All
# MAGIC %md
# MAGIC 0. Import the Test-Config
# MAGIC 0. Setup the version, dbr, distribution name, etc
# MAGIC 0. Determine if this is a publish-to-test or production-publish
# MAGIC 0. Define the artifacts to publish & publish

# COMMAND ----------

# DBTITLE 1,Import Test-Config
# MAGIC %run ./Test-Config

# COMMAND ----------

# DBTITLE 1,Configure
version = "1.0.0"
dbr = "DBR 9.1 Photon LTS"
dist_name = f"Example-Course"

from dbacademy import dbgems, dbpublish
from dbacademy.dbrest import DBAcademyRestClient

client = DBAcademyRestClient()

current_path = dbgems.get_notebook_dir()
published = current_path.startswith("/Repos/Published")
username = "Published" if published else dbgems.get_username()

source_repo = dbgems.get_notebook_dir(-2)
source_dir = f"{source_repo}/Source"

target_repo_name = "example-course-STUDENT-COPY"
target_dir = f"{dbgems.get_notebook_dir(-3)}/{target_repo_name}/{dist_name}"

print(f"Source Dir: {source_dir}")
print(f"Target Dir: {target_dir}")

# COMMAND ----------

# DBTITLE 1,Test or Publish
# Determine if we are in test mode or not.
try: testing = dbutils.widgets.get("testing").lower() == "true"
except: testing = False
print(f"Testing: {testing}")

# COMMAND ----------

# DBTITLE 1,Define Publishing Steps
publisher = dbpublish.Publisher(client, version, source_dir, target_dir, dbr=dbr, include_solutions=True)

publisher.add_path("Includes/Classroom-Setup")
publisher.add_path("Includes/Reset", include_solution=False)

publisher.add_path("Labs/EC 01L - Your First Lab")
publisher.add_path("Labs/EC 02L - Your Second Lab")
publisher.add_path("Labs/EC 03L - Your Third Lab")
publisher.add_path("Labs/EC 04L - You Fourth Lab")

publisher.add_path("EC 01 - Your First Lesson")
publisher.add_path("EC 02 - Your Second Lesson")
publisher.add_path("EC 03 - Your Third Lesson")
publisher.add_path("EC 04 - You Fourth Lesson")

publisher.add_path("Version Info", include_solution=False)

# COMMAND ----------

# DBTITLE 1,Publish
publisher.publish(testing, mode="DELETE")
