# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

# Defined here because it is used only by this lesson
def create_magic_table(self):
    """
    This is a sample utility method that creates a table and insert some data. Because this is not user-facing, we do not monkey-patch it into DBAcademyHelper
    """
    DA.paths.magic_tbl = f"{DA.paths.working_dir}/magic"
    
    spark.sql(f"""
CREATE TABLE IF NOT EXISTS magic (some_number INT, some_string STRING)
USING DELTA
LOCATION '{DA.paths.magic_tbl}'
""")
    spark.sql("INSERT INTO magic VALUES (1, 'moo')")
    spark.sql("INSERT INTO magic VALUES (2, 'baa')")
    spark.sql("INSERT INTO magic VALUES (3, 'wolf')")
    
DBAcademyHelper.monkey_patch(create_magic_table)

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs

DA.create_magic_table()                             # Custom utility method to create the "magic" table

DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

