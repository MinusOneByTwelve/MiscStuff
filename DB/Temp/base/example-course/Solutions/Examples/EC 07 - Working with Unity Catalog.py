# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Working with Unity Catalog
# MAGIC 
# MAGIC Because multiple users can be running the same course in the same workspace, we have to enforce user-isolation to avoid conflicts between courseware consumers.
# MAGIC 
# MAGIC Historically this was done at the database level, creating a user-specific database.
# MAGIC 
# MAGIC With the introduction of UC, we can now provide that same level of user-level isolation at the course level with the following outcomes:
# MAGIC * We have to validate that the workspace actually supports UC.
# MAGIC * The catalog becomes user-specific.
# MAGIC * A schema within the catalog no longer needs to be user-specific.
# MAGIC * A course can now employ many schemas within the user-specific catalog.
# MAGIC * Collectively, this means content can now better align to real-world examples such as `customers_gold`, `customers_silver` and `customers_bronze`.

# COMMAND ----------

# MAGIC %md Just to demonstrate this behavior, let's take a look how initialization options affect our outcomes.
# MAGIC 
# MAGIC Normally we would express this in a Classroom-Setup, but for simplicity, we are limiting focus to this notebook.

# COMMAND ----------

# MAGIC %md Run the following two cells and note the lesson configuration, current catalog, and current schema.

# COMMAND ----------

# MAGIC %run ../Includes/_common

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)

print(f"Requires UC:           {lesson_config.requires_uc}")
print(f"Create User's Catalog: {lesson_config.create_catalog}")
print(f"Create User's Schema:  {lesson_config.create_schema}")
print("-"*80)
print(f"Current Catalog:       {DA.current_catalog}")
print(f"Current Schema:        {DA.current_schema}")

# COMMAND ----------

# MAGIC %md
# MAGIC Redefine the course's defaults to require UC and to create a user-specific catalog.
# MAGIC 
# MAGIC Note the combination of creating a catalog and creating a schema is not supported - specifically the directive to create a user-specific schema will be ignored.

# COMMAND ----------

# MAGIC %run ../Includes/_common

# COMMAND ----------

lesson_config.requires_uc = True                    # Overriding course defualts to require UC
lesson_config.create_catalog = True                 # Overriding course defaults to create a catalog
lesson_config.create_schema = False                 # Overriding course defaults to NOT create a schema

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

# COMMAND ----------

# MAGIC %md 
# MAGIC Run the following cell and note what DA thinks the catalog and schema are vs the actual current catalog and schema

# COMMAND ----------

print(f"Expected Catalog: {DA.catalog_name}")
print(f"Expected Schema:  {DA.schema_name}")
print("-"*80)
print(f"Current Catalog:  {DA.current_catalog}")
print(f"Current Schema:   {DA.current_schema}")

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md Now that we have called classroom-cleanup, review the current state one more time.

# COMMAND ----------

print(f"Expected Catalog: {DA.catalog_name}")
print(f"Expected Schema:  {DA.schema_name}")
print("-"*80)
print(f"Current Catalog:  {DA.current_catalog}")
print(f"Current Schema:   {DA.current_schema}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
