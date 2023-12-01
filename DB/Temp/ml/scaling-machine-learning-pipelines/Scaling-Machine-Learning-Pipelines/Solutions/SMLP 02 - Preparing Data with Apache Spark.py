# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Preparing Data with Apache Spark
# MAGIC 
# MAGIC In this notebook, we'll demonstrate how to prepare data for machine learning at scale using Apache Spark.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC 
# MAGIC Run the classroom-setup notebook to initialize all of our variables and load our course data.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> You will need to run this in every notebook of the course.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we will load our course data.

# COMMAND ----------

london_listings_df = spark.read.format("delta").load(lesson_2_path).drop("id", "host_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train-test split
# MAGIC 
# MAGIC First, we'll split our Spark DataFrame into a training set and a test set.
# MAGIC 
# MAGIC #### Train-test split reproducibility
# MAGIC 
# MAGIC Recall that in order to ensure that our split is reproducible, we would need to:
# MAGIC 
# MAGIC 1. Set a seed
# MAGIC 2. Maintain data partitioning
# MAGIC 3. Maintain cluster configuration
# MAGIC 
# MAGIC Unfortunately, we don't always have control of these things.
# MAGIC 
# MAGIC #### Recommended best practice
# MAGIC 
# MAGIC As a result, we recommend the following workflow to avoid the need to reproduce the split as best we can.
# MAGIC 
# MAGIC We recommend first splitting the data using the [**`randomSplit`** method](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.randomSplit.html#pyspark.sql.DataFrame.randomSplit), and then writing out each split DataFrame using Delta.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> When using Databricks Runtime for Machine Learning 8.0 or greater, Spark DataFrames will write using Delta by default.

# COMMAND ----------

train_df, test_df = london_listings_df.randomSplit([.8, .2], seed=42)

train_df.write.format("delta").mode("overwrite").option("overwriteSchema", True).save(lesson_2_train_path)
test_df.write.format("delta").mode("overwrite").option("overwriteSchema", True).save(lesson_2_test_path)

# COMMAND ----------

# MAGIC %md
# MAGIC When we need to access this data later in our pipeline, we can reload the already-split DataFrames rather than trying to reproduce our split.

# COMMAND ----------

train_df = spark.read.format("delta").load(lesson_2_train_path)
test_df = spark.read.format("delta").load(lesson_2_test_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Imputing missing values
# MAGIC 
# MAGIC Next, we'll demonstrate how to impute missing values using Spark MLlib.
# MAGIC 
# MAGIC #### Convert integers to doubles
# MAGIC 
# MAGIC Spark MLlib requires that imputed columns be of type `double`, so we'll need to start by identifying integer columns and casting them to the correct type.

# COMMAND ----------

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

# Get a list of integer columns
integer_cols = [column.name for column in train_df.schema.fields if column.dataType == IntegerType()]

# Loop through integer columns to cast each one to double
doubles_train_df = train_df
for column in integer_cols:
    doubles_train_df = doubles_train_df.withColumn(column, col(column).cast("double"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Create dummy variables
# MAGIC 
# MAGIC To create dummy variables, we need to get a list of all columns with missing values. Next, we'll again loop through each of these columns to create a new dummy variable indicating whether each respective column's values are missing.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** Be sure to exclude the target variable from this step!

# COMMAND ----------

from pyspark.sql.functions import count, isnan, when
from pyspark.sql.types import DoubleType

# Get a list of numeric columns – don't include the target
double_cols = [column.name for column in doubles_train_df.schema.fields if column.dataType == DoubleType() and column.name != "price"]

# Get a list of numeric columns with missing values
missing_values_logic = [count(when(col(column).isNull(), column)).alias(column) for column in double_cols]
row_dict = doubles_train_df.select(missing_values_logic).first().asDict()
missing_cols = [column for column in row_dict if row_dict[column] > 0]

# Loop through missing columns to create a new dummy column for missing values
dummy_train_df = doubles_train_df
for column in missing_cols:
    dummy_train_df = dummy_train_df.withColumn(column + "_na", when(col(column).isNull(), 1.0).otherwise(0.0))
    
display(dummy_train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### The `Imputer` class
# MAGIC 
# MAGIC Finally, we'll impute the missing values. We'll use the [**`Imputer`** class](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Imputer.html#pyspark.ml.feature.Imputer) from Spark MLlib.
# MAGIC 
# MAGIC The `Imputer` class is an **estimator**.
# MAGIC 
# MAGIC #### Estimators and transformers
# MAGIC 
# MAGIC **Estimators** are algorithms (modeling or data manipulation) that learn on a DataFrame. As a result, estimators in Spark MLlib have a `fit` method that takes a DataFrame as an argument. Estimators return transformers.
# MAGIC 
# MAGIC **Transformers** are the objects that actually transform DataFrames. They take a DataFrame as input, and they return a new, transformed DataFrame.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> You can read more about [estimators](https://spark.apache.org/docs/latest/ml-pipeline.html#estimators) and [transformers](https://spark.apache.org/docs/latest/ml-pipeline.html#transformers) in the Spark MLlib documentation.
# MAGIC 
# MAGIC #### Fit the `Imputer` estimator on the DataFrame
# MAGIC 
# MAGIC Now, we'll perform the imputation.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** Be sure to import from `pyspark.ml` for the DataFrame-based Spark MLlib API.
# MAGIC 
# MAGIC Because we need to calculate the mean, median, or mode value for each column when imputing missing values, the `Imputer` class is an estimator &mdash; recall that this means it needs to be fit on the DataFrame.

# COMMAND ----------

# Import the Imputer class
from pyspark.ml.feature import Imputer

# Instantiate the imputer with a list of input and output columns
imputer = Imputer(strategy="median", inputCols=missing_cols, outputCols=missing_cols)

# Fit the imputer on the dataframe to create and imputer model
imputer_model = imputer.fit(dummy_train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Use the `ImputerModel` to fill in the missing values
# MAGIC 
# MAGIC Now our fit [imputer model](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.ImputerModel.html#pyspark.ml.feature.ImputerModel) &mdash; a transformer &mdash; is ready to actually fill in the missing values. 

# COMMAND ----------

# Transform the DataFrame to impute its missing values
imputed_train_df = imputer_model.transform(dummy_train_df)
display(imputed_train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Assemble the feature vector
# MAGIC 
# MAGIC The last thing we'll cover in this demo is how to assemble our feature vector. This is a necessary step for Spark MLlib models.
# MAGIC 
# MAGIC #### Import the `VectorAssembler`
# MAGIC 
# MAGIC Spark MLlib has a [class called **`VectorAssembler`**](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.VectorAssembler.html#pyspark.ml.feature.VectorAssembler) that we'll use to create our feature vector. We first need to import the class.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# COMMAND ----------

# MAGIC %md
# MAGIC #### Instantiate the `VectorAssembler`
# MAGIC 
# MAGIC Now, we'll define our `VectorAssembler` with the appropriate input columns.

# COMMAND ----------

# Get a list of all columns to be used as features
feature_columns = [column for column in imputed_train_df.columns if column != "price"]

vector_assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Transform the training DataFrame
# MAGIC 
# MAGIC Because `VectorAssembler` is a transformer, we don't need to call a `fit` method.
# MAGIC 
# MAGIC Instead, we use the `transform` method to create the feature vector.

# COMMAND ----------

feature_vector_train_df = vector_assembler.transform(imputed_train_df).select("price", "features")
display(feature_vector_train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build a model
# MAGIC We aren't teaching how to build Spark MLlib models in this lesson (we'll do it next!), but we'll do a quick demo just to show that the `VectorAssembler` is necessary!
# MAGIC 
# MAGIC In the below cell, we load and build a linear regression model. Notice the argument to the [**`featureCol`** parameter](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.LinearRegression.html#pyspark.ml.regression.LinearRegression.featuresCol) is a single column of vector type.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> If we tried to pass in a list of column names, we'd get an error when working with most model types in Spark MLlib.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(labelCol="price", featuresCol="features")
lr.fit(feature_vector_train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Saving the training data
# MAGIC 
# MAGIC It's a good idea to save both the feature-vector DataFrame **and** the non-assembled DataFrame.
# MAGIC 
# MAGIC This ensures that you're able to use a variety of machine learning libraries downstream.

# COMMAND ----------

imputed_train_df.write.format("delta").mode("overwrite").save(lesson_2_train_prepared_path)
feature_vector_train_df.write.format("delta").mode("overwrite").save(lesson_2_train_feature_vector_path)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
