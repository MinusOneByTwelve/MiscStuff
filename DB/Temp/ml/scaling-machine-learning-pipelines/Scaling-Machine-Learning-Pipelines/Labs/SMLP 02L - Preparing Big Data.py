# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab: Preparing Big Data
# MAGIC 
# MAGIC In this lab, you will have the opportunity to complete hands-on exercises in preparing big data using Apache Spark.
# MAGIC 
# MAGIC Specifically, you will:
# MAGIC 
# MAGIC 1. Perform the best-practice workflow for a train-test split using the Spark DataFrame API and Delta Lake
# MAGIC 2. One-hot encode categorical features using Spark MLlib's `OneHotEncoder` estimator
# MAGIC 3. Assemble a feature vector using Spark MLlib's `VectorAssembler`

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
# MAGIC Next, we will load our course data.

# COMMAND ----------

tokyo_listings_df = spark.read.format("delta").load(lab_2_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC 
# MAGIC ### Exercise 1: Train-test split
# MAGIC 
# MAGIC In this exercise, you will perform the best-practice workflow for a train-test split using the Spark DataFrame API and Delta Lake.
# MAGIC 
# MAGIC Recall that due to things like changing cluster configurations and data partitioning, it can be difficult to ensure a reproducible train-test split. As a result, we recommend:
# MAGIC 
# MAGIC 1. Split the data using a random seed
# MAGIC 2. Write out the train and test DataFrames
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** Look at the code comments and refer back to the Preparing Data with Apache Spark demonstration notebook for guidance. You can also check out the [**`randomSplit`** documentation](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.sql.DataFrame.randomSplit.html#pyspark.sql.DataFrame.randomSplit).

# COMMAND ----------

# TODO
# Split listings_df with 85 percent of the data in train_df and 15 percent of the data in test_df
<FILL_IN>

# Write train_df to train_exercise_path using Delta
<FILL_IN>

# Write test_df to test_exercise_path using Delta
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 2: Imputing missing values
# MAGIC 
# MAGIC In this exercise, you will impute the missing values of the numeric columns.
# MAGIC 
# MAGIC Recall our best practice workflow:
# MAGIC 
# MAGIC 1. Convert integer columns to double columns
# MAGIC 2. Create dummy features for double columns with missing values
# MAGIC 3. Impute the missing values in the original double columns
# MAGIC 
# MAGIC In this exercise, imputing the missing values with the **mean** rather than the **median**.
# MAGIC 
# MAGIC Because this is a lot of code (which we hope you reuse!), you'll just need to fill in a few key areas toward the bottom of the cell.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** Refer back to the previous lesson for help! Or you can check out the [**`Imputer`** documentation](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.Imputer.html#pyspark.ml.feature.Imputer).

# COMMAND ----------

# TODO
from pyspark.ml.feature import Imputer
from pyspark.sql.functions import col, count, isnan, when
from pyspark.sql.types import DoubleType, IntegerType

# Get a list of integer columns
integer_cols = [column.name for column in train_df.schema.fields if column.dataType == IntegerType()]

# Loop through integer columns to cast each one to double
doubles_train_df = train_df
for column in integer_cols:
    doubles_train_df = doubles_train_df.withColumn(column, col(column).cast("double"))

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
   
# Instantiate the imputer with a list of input and output columns
imputer = <FILL_IN>

# Fit the imputer on the dataframe to create and imputer model
imputer_model = <FILL_IN>

# Transform the DataFrame to impute its missing values
imputed_train_df = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 3: One-hot encoding
# MAGIC 
# MAGIC In this exercise, you will one-hot encode categorical features using Spark MLlib's `OneHotEncoder` estimator.
# MAGIC 
# MAGIC If you are unfamiliar with one-hot encoding, there's a description below. If you're already familiar, you can skip ahead to the **One-hot encoding in Spark MLlib** section toward the bottom of the cell.
# MAGIC 
# MAGIC #### Categorical features in machine learning
# MAGIC 
# MAGIC Many machine learning algorithms are not able to accept categorical features as inputs. As a result, data scientists and machine learning engineers need to determine how to handle them. 
# MAGIC 
# MAGIC An easy solution would be remove the categorical features from the feature set. While this is quick, **you are removing potentially predictive information** &mdash; so this usually isn't the best strategy.
# MAGIC 
# MAGIC Other options include ways to represent categorical features as numeric features. A few common options are:
# MAGIC 
# MAGIC 1. **One-hot encoding**: create dummy/binary variables for each category
# MAGIC 2. **Target/label encoding**: replace each category value with a value that represents the target variable (e.g. replace a specific category value with the mean of the target variable for rows with that category value)
# MAGIC 3. **Embeddings**: use/create a vector-representation of meaningful words in each category's value
# MAGIC 
# MAGIC Each of these options can be really useful in different scenarios. We're going to focus on one-hot encoding here.
# MAGIC 
# MAGIC #### One-hot encoding basics
# MAGIC 
# MAGIC One-hot encoding creates a binary/dummy feature for each category in each categorical feature.
# MAGIC 
# MAGIC In the example below, the feature **Animal** is split into three binary features &mdash; one for each value in **Animal**. Each binary feature's value is equal to 1 if its respective category value is present in **Animal** for each row. If its category value is not present in the row, the binary feature's value will be 0.
# MAGIC 
# MAGIC ![One-hot encoding image](https://s3-us-west-2.amazonaws.com/files.training.databricks.com/images/mlewd/Scaling-Machine-Learning-Pipelines/one-hot-encoding.png)
# MAGIC 
# MAGIC #### One-hot encoding in Spark MLlib
# MAGIC 
# MAGIC Even if you understand one-hot encoding, it's important to learn how to perform it using Spark MLlib.
# MAGIC 
# MAGIC To one-hot encode categorical features in Spark MLlib, we are going to use two classes: [the **`StringIndexer`** class](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.StringIndexer.html#pyspark.ml.feature.StringIndexer) and [the **`OneHotEncoder`** class](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.OneHotEncoder.html#pyspark.ml.feature.OneHotEncoder).
# MAGIC 
# MAGIC * The `StringIndexer` class indexes string-type columns to a numerical index. Each unique value in the string-type column is mapped to a unique integer.
# MAGIC * The `OneHotEncoder` class accepts indexed columns and converts them to a one-hot encoded vector-type feature.
# MAGIC 
# MAGIC #### Applying the `StringIndexer` -> `OneHotEncoder` workflow
# MAGIC 
# MAGIC First, we'll need to index the categorical features of the DataFrame. `StringIndexer` takes a few arguments:
# MAGIC 
# MAGIC 1. A list of categorical columns to index.
# MAGIC 2. A list names for the indexed columns being created.
# MAGIC 3. Directions for how to handle new categories when transforming data.
# MAGIC 
# MAGIC Because `StringIndexer` has to learn which categories are present before indexing, it's an **estimator** &mdash; remember that means we need to call its `fit` method. Its result can then be used to transform our data.

# COMMAND ----------

# Import StringIndexer from Spark MLlib's feature module
from pyspark.ml.feature import StringIndexer

# Create a list of categorical features and a list of indexed feature names
from pyspark.sql.types import StringType
categorical_cols = [column.name for column in imputed_train_df.schema.fields if column.dataType == StringType()]
index_cols = [column + "_index" for column in categorical_cols]

# Instantiate the StringIndexer with the column lists
string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_cols, handleInvalid="skip")

# Fit the StringIndexer on the training data
string_indexer_model = string_indexer.fit(imputed_train_df)

# Transform train_df using the string_indexer_model
indexed_df = string_indexer_model.transform(imputed_train_df)
display(indexed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Once our data has been indexed, we are ready to use the `OneHotEncoder` estimator.
# MAGIC 
# MAGIC Since this is a new exercise, try to fill in a few key areas to complete the task.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** Look at the [`OneHotEncoder` documentation](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.feature.OneHotEncoder.html#pyspark.ml.feature.OneHotEncoder) and our previous Spark MLlib workflows that use estimators for guidance.

# COMMAND ----------

# TODO
# Import OneHotEncoder from Spark MLlib's feature module
from pyspark.ml.feature import OneHotEncoder

# Create a list of one-hot encoded feature names
ohe_cols = [column + "_ohe" for column in categorical_cols]

# Instantiate the OneHotEncoder with the column lists
ohe = <FILL_IN>(inputCols=<FILL_IN>, outputCols=<FILL_IN>, handleInvalid="keep")

# Fit the OneHotEncoder on the indexed data
ohe_model = ohe.<FILL_IN>(indexed_df)

# Transform indexed_df using the ohe_model
ohe_df = ohe_model.<FILL_IN>(indexed_df)
display(ohe_df)

# COMMAND ----------

# MAGIC %md
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Although we're covering one-hot encoding here, it's **not always the best choice for preparing your categorical features**. Consider the other options listed above, and check out this [blog post](https://towardsdatascience.com/one-hot-encoding-is-making-your-tree-based-ensembles-worse-heres-why-d64b282b5769) detailing why one-hot encoding is an **inefficient** and potentially **problematic** strategy for tree-based models.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Exercise 4: Assembling a feature vector
# MAGIC 
# MAGIC In this final exercise, you will assemble a feature vector using Spark MLlib's `VectorAssembler`.
# MAGIC 
# MAGIC Recall that Spark uses the `VectorAssembler` to create efficient representations of its features. 
# MAGIC 
# MAGIC In the below code cell, you will:
# MAGIC 
# MAGIC 1. Import `VectorAssembler` from Spark MLlib
# MAGIC 2. Create a list of columns that are to be used as features (remember to exclude the label column and index columns!)
# MAGIC 3. Instantiate the VectorAssembler
# MAGIC 4. Use the VectorAssembler to transform the data
# MAGIC 5. Write out both the feature vector data and the non-assembled data
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** Look at the code comments and refer back to the Preparing Data with Apache Spark demonstration notebook for guidance. 

# COMMAND ----------

# TODO
# Import VectorAssembler from Spark MLlib
from pyspark.ml.feature import VectorAssembler

# Create a list of all columns that are to be used as feature
from pyspark.sql.types import DoubleType
numeric_cols = [
    column.name for column in imputed_train_df.schema.fields if column.dataType == DoubleType() and column.name != "price"
]
feature_cols = <FILL_IN> + <FILL_IN>

# Instantiate the VectorAssembler
<FILL_IN>

# Transform ohe_df using vector_assembler
<FILL_IN>

# Save the vectorized and non-assembled training data
<FILL_IN>
<FILL_IN>

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
