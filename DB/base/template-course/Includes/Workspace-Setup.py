# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Workspace Setup
# MAGIC Instructors should run this notebook to prepare the workspace for a class.
# MAGIC 
# MAGIC This creates or updates the following resources:
# MAGIC 
# MAGIC |Resource Type|Description|
# MAGIC |---|---|
# MAGIC |User Entitlements|User-specific grants to allow creating databases/schemas against the current catalog when they are not workspace-admins.|
# MAGIC |Instance Pool|`DBAcademy Pool` for use by students and the "student" and "jobs" policies|
# MAGIC |Cluster Policies| `DBAcademy All-Purpose Policy` for clusters running standard notebooks <br> `DBAcademy Jobs-Only Policy` for workflows/jobs <br> `DBAcademy DLT-Only Policy` for DLT piplines (automatically applied)|
# MAGIC |Shared SQL Warehouse|`Starter Warehouse` for Databricks SQL exercises|

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

setup_start = dbgems.clock_start()  # Start timer to benchmark execution duration

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get Class Config Parameters
# MAGIC Sets up the following widgets to collect parameters used to configure our environment as a means of controlling class cost.
# MAGIC 
# MAGIC - **Configure For** (required) - `All Users`, `Missing Users Only`, or `Current User Only`
# MAGIC - **Description** (optional) - a general purpose description of the class
# MAGIC - **Lab/Class ID** (optional) - `lab_id` is the name assigned to this event/class or alternatively its class number

# COMMAND ----------

from dbacademy.dbhelper import WorkspaceHelper
 
dbutils.widgets.dropdown(WorkspaceHelper.PARAM_CONFIGURE_FOR, "", 
                         WorkspaceHelper.CONFIGURE_FOR_OPTIONS, "Configure For (required)")

dbutils.widgets.text(WorkspaceHelper.PARAM_LAB_ID, "", "Lab/Class ID (optional)")
dbutils.widgets.text(WorkspaceHelper.PARAM_DESCRIPTION, "", "Description (optional)")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Run Init Script & Install Datasets
# MAGIC Main purpose of the next cell is to pre-install the datasets.
# MAGIC 
# MAGIC It has the side effect of create our DA object, which includes our REST client.

# COMMAND ----------

lesson_config.create_schema = False                 # We don't need a schema when configuring the workspace

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Create Class Instance Pools
# MAGIC The following cell configures the instance pool used for this class

# COMMAND ----------

instance_pool_id = DA.workspace.clusters.create_instance_pool()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Create The Three Class-Specific Cluster Policies
# MAGIC The following cells create the various cluster policies used by the class

# COMMAND ----------

DA.workspace.clusters.create_all_purpose_policy(instance_pool_id)
DA.workspace.clusters.create_jobs_policy(instance_pool_id)
DA.workspace.clusters.create_dlt_policy()
None

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Create Class-Shared Databricks SQL Warehouse/Endpoint
# MAGIC Creates a single wharehouse to be used by all students.
# MAGIC 
# MAGIC The configuration is derived from the number of students specified above.

# COMMAND ----------

DA.workspace.warehouses.create_shared_sql_warehouse(name="Starter Warehouse")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Configure User Entitlements
# MAGIC 
# MAGIC This task simply adds the "**databricks-sql-access**" entitlement to the "**users**" group ensuring that they can access the Databricks SQL view.

# COMMAND ----------

DA.workspace.add_entitlement_workspace_access()
DA.workspace.add_entitlement_databricks_sql_access()

# COMMAND ----------

print(f"Setup completed {dbgems.clock_stopped(setup_start)}")

