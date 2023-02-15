-- Databricks notebook source
-- MAGIC %md-sandbox
-- MAGIC 
-- MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
-- MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
-- MAGIC </div>

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Getting Started with this Course
-- MAGIC 
-- MAGIC This course is designed to teach ELT with Spark SQL using Databricks notebooks. The majority of the Spark SQL code featured will work identically in Databricks SQL queries; however, in order to make this course easy and safe to run in all Databricks environments, certain design decisions were made that include syntax and code that will only work in notebooks.
-- MAGIC 
-- MAGIC This notebook gives a brief overview of some of these elements that you'll be interacting with throughout the course.
-- MAGIC 
-- MAGIC ## Learning Objectives
-- MAGIC By the end of this lesson, you'll be able to:
-- MAGIC - Use the notebook magic `%run` to execute notebook-based setup scripts
-- MAGIC - Describe the differences between executing SQL in notebooks vs. SQL boxes
-- MAGIC - Reference Hive variables for string substitution in Spark SQL queries
-- MAGIC - Use `%python` cells to run arbitrary Python code

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Setup Notebook
-- MAGIC 
-- MAGIC Throughout this course, we use a setup notebook in order to configure certain databases and environmental variables to ensure that our lessons don't conflict with other users in a shared workspace.
-- MAGIC 
-- MAGIC The `%run` magic executes other notebooks in the same interactive session created by this notebook. You can click on the relative path to this notebook in the following cell to explore what is happening there.

-- COMMAND ----------

-- MAGIC %run ../Includes/setup $mode="reset"

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Cell-based Notebook Execution
-- MAGIC 
-- MAGIC While it's possible to just copy an entire SQL script into a notebook, we'll be taking advantage of the ordered, cell-based execution of notebooks.
-- MAGIC 
-- MAGIC Essentially, think of each cell as a query that can be executed separately.
-- MAGIC 
-- MAGIC Note that our setup script created a custom database based on your username and this class.

-- COMMAND ----------

SELECT current_database()

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Queries that return results will be displayed in a rendered tabular format, even if there's only a single result.
-- MAGIC 
-- MAGIC Queries that should never return a result will just print `OK` when they execute correctly. A `CREATE TABLE` statement is one such query.

-- COMMAND ----------

CREATE OR REPLACE TABLE test_table
(a INT, b STRING, c DOUBLE)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC When a query doesn't return a result, it tells us as much.

-- COMMAND ----------

SELECT * FROM test_table

-- COMMAND ----------

-- MAGIC %md
-- MAGIC When we update existing tables, metrics about the operation will be displayed.

-- COMMAND ----------

INSERT INTO test_table VALUES
  (1, "a", 1.1), 
  (2, "b", 2.2), 
  (3, "c", 3.3)

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Note that now we'll see those values that have been added to our table.

-- COMMAND ----------

SELECT * FROM test_table

-- COMMAND ----------

-- MAGIC %md
-- MAGIC 
-- MAGIC If we run the earlier cell containing this same query, we'll also see these results.
-- MAGIC 
-- MAGIC If we run the cell that inserts records again, we'll see that we now have duplicate records.
-- MAGIC 
-- MAGIC This is an important lesson: notebook cells can be executed manually in any order, and the order of execution will impact results of other cells executed. Generally, the safest way to develop notebooks toward production is to order queries in the same way you would if these were a SQL script.
-- MAGIC 
-- MAGIC The cell below combines our above statements into a single cell.

-- COMMAND ----------

CREATE OR REPLACE TABLE test_table
(a INT, b STRING, c DOUBLE);

INSERT INTO test_table VALUES
  (1, "a", 1.1), 
  (2, "b", 2.2), 
  (3, "c", 3.3);
  
SELECT * FROM test_table

-- COMMAND ----------

-- MAGIC %md
-- MAGIC Notebooks give us the flexibility to execute multiple statements or single statements within cells. Think of each cell as a saved query and a notebook as an ordered collection of queries.
-- MAGIC 
-- MAGIC We'll drop our `test_table` before continuing.

-- COMMAND ----------

DROP TABLE test_table

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Working with Hive Variables
-- MAGIC 
-- MAGIC For the purposes of this course, several Hive variables are declared in the setup script.
-- MAGIC 
-- MAGIC Using the `${}` syntax allows us to directly substitute these values into SQL queries. The example code uses this to avoid conflicting with other aspects of your workspace. 

-- COMMAND ----------

SELECT "${c.username}" as username, 
       "${c.userhome}" as userhome, 
       "${c.database_name}" as database_name, 
       "${c.source}" as source

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Default Database
-- MAGIC 
-- MAGIC The database created during setup specifies a location that should not conflict with any other users in your workspace.
-- MAGIC 
-- MAGIC All tables created during this course will store data here.
-- MAGIC 
-- MAGIC Generally, a workspace/database admin would configure databases for users ahead of time, making sure that groups of users have only the required permissions on both the databases and the underlying storage locations that are necessary for their roles.

-- COMMAND ----------

DESCRIBE DATABASE EXTENDED ${c.database_name}

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Using Python & Listing Files
-- MAGIC 
-- MAGIC Spark SQL does not have direct support for interacting with the Databricks File System (DBFS).
-- MAGIC 
-- MAGIC When necessary, these notebooks will use Python Databricks Utility calls to interact with the DBFS. Users who will be responsible for ingesting files from cloud-based object storage locations may benefit from learning these commands, though they will not be necessary for folks working exclusively with data sources already registered as tables or views.
-- MAGIC 
-- MAGIC The `%python` magic command indicates that a single notebook cell will execute on the Python kernel.

-- COMMAND ----------

-- MAGIC %python
-- MAGIC dbutils.fs.ls(Paths.source)

-- COMMAND ----------

-- MAGIC %md-sandbox
-- MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
-- MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
-- MAGIC <br/>
-- MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
