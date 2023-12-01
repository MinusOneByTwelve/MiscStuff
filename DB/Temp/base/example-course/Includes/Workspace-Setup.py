# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Workspace Setup
# MAGIC This notebook should be run by instructors to prepare the workspace for a class.
# MAGIC 
# MAGIC The key changes this notebook makes includes:
# MAGIC * Updating user-specific grants such that they can create databases/schemas against the current catalog when they are not workspace-admins.
# MAGIC * Configures three cluster policies:
# MAGIC     * **DBAcademy All-Purpose Policy** - which should be used on clusters running standard notebooks.
# MAGIC     * **DBAcademy Jobs-Only Policy** - which should be used on workflows/jobs
# MAGIC     * **DBAcademy DLT-Only Policy** - which should be used on DLT piplines (automatically applied)
# MAGIC * Create or update the shared **Starter Warehouse** for use in Databricks SQL exercises
# MAGIC * Create the Instance Pool **DBAcademy Pool** for use by students and the "student" and "jobs" policies.

# COMMAND ----------

# MAGIC %run ./_common

# COMMAND ----------

# Start a timer so we can benchmark execution duration.
setup_start = dbgems.clock_start()

# COMMAND ----------

# MAGIC %md
# MAGIC # Get Class Config
# MAGIC The three variables defined by these widgets are used to configure our environment as a means of controlling class cost.

# COMMAND ----------

from dbacademy.dbhelper import WorkspaceHelper

# Setup the widgets to collect required parameters.
dbutils.widgets.dropdown(WorkspaceHelper.PARAM_CONFIGURE_FOR, "", 
                         WorkspaceHelper.CONFIGURE_FOR_OPTIONS, "Configure For (required)")

# lab_id is the name assigned to this event/class or alternatively its class number
dbutils.widgets.text(WorkspaceHelper.PARAM_LAB_ID, "", "Lab/Class ID (optional)")

# a general purpose description of the class
dbutils.widgets.text(WorkspaceHelper.PARAM_DESCRIPTION, "", "Description (optional)")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Init Script & Install Datasets
# MAGIC The main affect of this call is to pre-install the datasets.
# MAGIC 
# MAGIC It has the side effect of create our DA object which includes our REST client.

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

from dbacademy.dbhelper import ClustersHelper

ClustersHelper.create_all_purpose_policy(client=DA.client, instance_pool_id=instance_pool_id)
ClustersHelper.create_jobs_policy(client=DA.client, instance_pool_id=instance_pool_id)
ClustersHelper.create_dlt_policy(client=DA.client, 
                                 instance_pool_id=None,
                                 lab_id=WorkspaceHelper.get_lab_id(), 
                                 workspace_description=WorkspaceHelper.get_workspace_description(),
                                 workspace_name=WorkspaceHelper.get_workspace_name(), 
                                 org_id=dbgems.get_org_id())

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Create Class-Shared Databricks SQL Warehouse/Endpoint
# MAGIC Creates a single wharehouse to be used by all students.
# MAGIC 
# MAGIC The configuration is derived from the number of students specified above.

# COMMAND ----------

from dbacademy.dbhelper.warehouses_helper_class import WarehousesHelper

DA.workspace.warehouses.create_shared_sql_warehouse(name=WarehousesHelper.WAREHOUSES_DEFAULT_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Configure User Entitlements
# MAGIC 
# MAGIC This task simply adds the "**databricks-sql-access**" entitlement to the "**users**" group ensuring that they can access the Databricks SQL view.

# COMMAND ----------

WorkspaceHelper.add_entitlement_workspace_access(client=DA.client)
WorkspaceHelper.add_entitlement_databricks_sql_access(client=DA.client)

# COMMAND ----------

print(f"Setup completed {dbgems.clock_stopped(setup_start)}")

