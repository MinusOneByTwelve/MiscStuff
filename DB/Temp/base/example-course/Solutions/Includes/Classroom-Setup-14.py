# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

# MAGIC %run ./_multi-task-jobs-config

# COMMAND ----------

lesson_config.enable_streaming_support = True       # Indicates that this lesson uses streaming (e.g. needs a checkpoint directory)

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs

# The location that the DLT databases should be written to
DA.paths.storage_location = f"{DA.paths.working_dir}/storage_location"

print()
DA.dlt_data_factory = DataFactory()                 # Create the DataFactory for pseudo streaming
DA.dlt_data_factory.load()                          # We "need" at least one batch to get started.
print()

DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

