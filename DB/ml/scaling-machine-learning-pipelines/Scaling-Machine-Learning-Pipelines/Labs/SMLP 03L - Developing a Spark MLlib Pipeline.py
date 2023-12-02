# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab: Developing a Spark MLlib Pipeline
# MAGIC 
# MAGIC In this lab, you will have the opportunity to complete hands-on exercises in developing machine learning pipelines using Spark MLlib.
# MAGIC 
# MAGIC Specifically, you will:
# MAGIC 
# MAGIC 1. Prepare data preparation stages
# MAGIC 2. Define a distributed random forest regression model using Spark MLlib
# MAGIC 3. Define parameter grid and cross-validation objects using Spark MLlib
# MAGIC 4. Create and fit a Spark MLlib Pipeline with all data preparation and modeling stages
# MAGIC 5. Evaluate the performance of your model on the test set

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC 
# MAGIC Run the classroom-setup notebook to initialize all of our variables and load our course data.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> You will need to run this in every notebook of the course.

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will load in our split Tokyo listing data.

# COMMAND ----------

train_df = spark.read.format("delta").load(lab_3_train_path)
test_df = spark.read.format("delta").load(lab_3_test_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC 
# MAGIC ### Exercise 1: Data preparation stages
# MAGIC 
# MAGIC In this exercise, you will create data preparation stages for a Spark MLlib pipeline.
# MAGIC 
# MAGIC As a part of this, you will [use a custom data preparation transformer](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Transformer.html#pyspark.ml.Transformer) source from another notebook. Recall when we discussed imputing values, **we described the need for creating binary/dummy features indicating whether or not a value had been imputed**. However, since there's **no built-in Spark MLlib tool for this**, we'll need to define it ourselves.
# MAGIC 
# MAGIC Below, we're sourcing another notebook to create the transformer &mdash; if you're curious, you can check out that notebook [here]($./MissingValueIndicator).

# COMMAND ----------

# MAGIC %run "./MissingValueIndicator"

# COMMAND ----------

# MAGIC %md
# MAGIC Your task is to create the other stages of our pipeline:
# MAGIC 
# MAGIC * `Imputer`
# MAGIC * `StringIndexer`
# MAGIC * `VectorAssembler`
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** Refer back to the Preparing Data with Apache Spark demo notebook for an example.

# COMMAND ----------

# TODO
# Load necessary libraries
from pyspark.ml.feature import Imputer, StringIndexer, VectorAssembler
from pyspark.sql.functions import col, count, when
from pyspark.sql.types import DoubleType, StringType

# Identify the columns with missing values in the training set
double_cols = [column.name for column in train_df.schema.fields if column.dataType == DoubleType() and column.name != "price"]
missing_values_logic = [count(when(col(column).isNull(), column)).alias(column) for column in double_cols]
row_dict = train_df.select(missing_values_logic).first().asDict()
missing_cols = [column for column in row_dict if row_dict[column] > 0]
 
# Instantiate the MissingValueIndicator object
missing_output_cols = [column + "_na" for column in missing_cols]
missing_value_indicator = MissingValueIndicator(inputCols=missing_cols, outputCols=missing_output_cols)

# Instantiate the Imputer object
imputer = <FILL_IN>

# Create the StringIndexer object
categorical_cols = [column.name for column in train_df.schema.fields if column.dataType == StringType()]
indexed_cols = [column + "_index" for column in categorical_cols]
string_indexer = <FILL_IN>

# Create the VectorAssembler object
feature_cols = double_cols + missing_output_cols + indexed_cols
vector_assembler = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 2: Distributed random forest regression model
# MAGIC 
# MAGIC In this exercise, you will instantiate a distributed random forest regression model.
# MAGIC 
# MAGIC Spark MLlib has its own implementation of random forest: [**`RandomForestRegressor`** for regression](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.RandomForestRegressor.html#pyspark.ml.regression.RandomForestRegressor) and [**`RandomForestClassifier`** for classification](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.RandomForestClassifier.html#pyspark.ml.classification.RandomForestClassifier). This implementation uses the **distributed decision tree** implementation in Spark MLlib to distribute the development of each tree in the forest.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> We're programmatically identifying the needed value for `maxBins`!

# COMMAND ----------

# TODO
# Load necessary libraries
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import countDistinct
from pyspark.sql.types import StringType

# Identify maximum categorical cardinality
cardinality_logic = [countDistinct(column).alias(column) for column in categorical_cols] 
row_dict = train_df.select(cardinality_logic).first().asDict()
max_cardinality = max(row_dict.values())

# Instantiate RandomForestRegressor
rfr = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 3: Parameter grid and cross-validation
# MAGIC 
# MAGIC In this exercise, you will define a `ParamGrid` [using **`ParamGridBuilder`**](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html#pyspark.ml.tuning.ParamGridBuilder) and [a **`CrossValidator`**](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html#pyspark.ml.tuning.CrossValidator) for tuning hyperparameters using Spark MLlib.
# MAGIC 
# MAGIC Use the following parameter grid specification:
# MAGIC 
# MAGIC * `maxDepth` (max depth of each decision tree): 4, 6
# MAGIC * `numTrees` (the number of trees in the forest): 10, 20
# MAGIC 
# MAGIC For the `CrossValidator`, use the following:
# MAGIC 
# MAGIC * The `RandomForestRegressor` as the estimator
# MAGIC * [The **`RegressionEvaluator`**](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.RegressionEvaluator.html#pyspark.ml.evaluation.RegressionEvaluator) with `"price"` as the label column and `"rmse"` as the evaluation metric
# MAGIC * 3 folds
# MAGIC * Seed of 42
# MAGIC * Parallelism of 4

# COMMAND ----------

# TODO
# Load the necessary libraries
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Create the ParamGrid
param_grid = <FILL_IN>

# Create the RegressionEvaluator
evaluator = <FILL_IN>

# Create the CrossValidator
cv = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC **Question:** Will increasing the parallelism result in the hyperparameter optimization process finishing earlier?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 4: Pipeline
# MAGIC 
# MAGIC In this exercise, you will define and fit a [Spark MLlib Pipeline](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Pipeline.html#pyspark.ml.Pipeline) using the estimators and transformers defined above.
# MAGIC 
# MAGIC Use the `MissingValueIndicator`, `Imputer`, `StringIndexer`, `VectorAssembler`, and `CrossValidator` as stages.

# COMMAND ----------

# TODO
# Load the necessary libraries
from pyspark.ml import Pipeline

# Instantiate the Pipeline
stages_list = <FILL_IN>
pipeline = Pipeline(stages=stages_list)

# Fit the Pipeline
pipeline_model = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 5: Evaluate on the test set
# MAGIC 
# MAGIC In this exercise, you will evaluate your fitted PipelineModel on the test set.
# MAGIC 
# MAGIC Use the `RegressionEvaluator` defined in the earlier exercise to compute the RMSE of the training and test set.
# MAGIC 
# MAGIC How do the training RMSE and the test RMSE compare?

# COMMAND ----------

# TODO
# Get train predictions
train_preds = <FILL_IN>

# Evaluate the train RMSE
train_rmse = <FILL_IN>

# Evaluate the test RMSE
test_preds = <FILL_IN>

# Evaluate the train RMSE
test_rmse = <FILL_IN>

# Print results
print(f"Training RMSE = {train_rmse}")
print(f"Test RMSE = {test_rmse}")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
