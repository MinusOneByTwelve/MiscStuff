# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab: Parallelizing Single-node Training and Inference
# MAGIC 
# MAGIC In this lab, you will have the opportunity use what you've learned from the previous lesson to parallelize single-node training and inference.
# MAGIC 
# MAGIC Specifically, you will:
# MAGIC 
# MAGIC 1. Determine whether to use a Pandas UDF workflow or a Pandas Function API workflow
# MAGIC 2. Train neighborhood-specific models in parallel for each neighborhood in Tokyo
# MAGIC 3. Perform inference to predict the price of each Airbnb rental in Tokyo using the appropriate neighborhood-specific model

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
# MAGIC Next, we will load our course data and split it into a training set and an inference set.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> We are loading in both cleaned Tokyo-specific data. Notice we aren't dropping the unique `id` column for the data &mdash; we'll need it later.

# COMMAND ----------

model_df = spark.read.format("delta").load(lab_5_model_path)
inference_df = spark.read.format("delta").load(lab_5_inference_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC 
# MAGIC ### Exercise 1: Pandas UDF vs. Pandas Function API
# MAGIC 
# MAGIC In this first exercise, you will determine whether to use a [**Pandas UDF** workflow](https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html) or a [**Pandas Function API** workflow](https://docs.databricks.com/spark/latest/spark-sql/pandas-function-apis.html) to complete this lab.
# MAGIC 
# MAGIC In order you do this, you'll need to understand three things:
# MAGIC 
# MAGIC 1. The task you're being asked to complete in this lab
# MAGIC 2. When to use a Pandas UDF
# MAGIC 3. When to use the Pandas Function APIs
# MAGIC 
# MAGIC These are detailed below.
# MAGIC 
# MAGIC #### 1. The task
# MAGIC 
# MAGIC In this lab, you're being asked to develop and apply **group-specific** machine learning models to predict the price of of Airbnb rentals. You should train one model for each neighborhood in Tokyo on a training set, and you should compute predictions for each row in the inference set using the model trained on rental's in each respective row's neighborhood.
# MAGIC 
# MAGIC #### 2. When to use a Pandas UDF
# MAGIC 
# MAGIC From the [Databricks documentation](https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html):
# MAGIC 
# MAGIC > A pandas user-defined function (UDF)—also known as vectorized UDF—is a user-defined function that uses Apache Arrow to transfer data and pandas to work with the data. pandas UDFs allow vectorized operations that can increase performance up to 100x compared to row-at-a-time Python UDFs.
# MAGIC 
# MAGIC A key requirement of Pandas UDFs is that they output the same number of records as they have as input.
# MAGIC 
# MAGIC #### 3. When to use the Pandas Function APIs
# MAGIC 
# MAGIC For the Pandas Function APIs, the [Databricks documentation](https://docs.databricks.com/spark/latest/spark-sql/pandas-function-apis.html) says:
# MAGIC 
# MAGIC > pandas function APIs enable you to directly apply a Python native function, which takes and outputs pandas instances, to a PySpark DataFrame. Similar to pandas user-defined functions, function APIs also use Apache Arrow to transfer data and pandas to work with the data; however, Python type hints are optional in pandas function APIs.
# MAGIC 
# MAGIC And in the specific Grouped Map part of the [documentation](https://docs.databricks.com/spark/latest/spark-sql/pandas-function-apis.html#grouped-map), it says:
# MAGIC 
# MAGIC > You transform your grouped data via groupBy().applyInPandas() to implement the “split-apply-combine” pattern.
# MAGIC 
# MAGIC 
# MAGIC **Question:** Which toolset-based workflow should be used to complete the specific task?

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 2: Parallelizing grouped training
# MAGIC 
# MAGIC Next, you'll need to train a regression model for each neighborhood in Tokyo to predict the price of a rental.
# MAGIC 
# MAGIC Fill in the blanks below to complete the exercise.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** Refer back to the previous lesson's notebook for guidance.

# COMMAND ----------

# TODO
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from pyspark.sql.types import FloatType, IntegerType, StringType, StructField, StructType
from pyspark.sql.functions import lit

# Define the train_model function
def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:

    # Pull metadata
    neighbourhood = <FILL_IN>
    n_used = df_pandas.shape[0]
    run_id = <FILL_IN>
    experiment_id = <FILL_IN>

    # Train the model
    X = df_pandas.drop(["price", "neighbourhood_cleansed", "run_id", "experiment_id", "id"], axis=1)
    y = df_pandas["price"]
    rfr = RandomForestRegressor(n_estimators=10, max_depth=8)
    rfr.fit(X, y)

    # Evaluate the model
    predictions = rfr.predict(X)
    rmse = mean_squared_error(y, predictions, squared=False)

    # Resume the top-level training
    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id):
        
        # Create a nested run for the specific neighbourhood
        with mlflow.start_run(run_name=neighbourhood, experiment_id=experiment_id, nested=True) as run:
            mlflow.sklearn.log_model(rfr, neighbourhood)
            mlflow.log_metric("rmse", rmse)

            artifact_uri = <FILL_IN>
            
            # Create a return pandas DataFrame that matches the schema above
            return_df = <FILL_IN>

    return return_df 

# Define the return_schema
return_schema = StructType([
    <FILL_IN>,
    StructField("n_used", IntegerType()),
    StructField("model_path", StringType()),
    StructField("rmse", FloatType())
])

# Run the model for each neighborhood
mlflow.set_experiment("/Users/" + username + "/SMLP-Lab-5")
with mlflow.start_run(run_name="Training session for all neighborhoods") as run:
    
    # Get run_id and experiment_id
    run_id = run.info.run_id
    experiment_id = mlflow.get_experiment_by_name("/Users/" + username + "/SMLP-Lab-5").experiment_id

    # Apply function to each group
    train_output_df = <FILL_IN>
    
display(train_output_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 3: Parallelizing grouped inference
# MAGIC 
# MAGIC Finally, you'll need to apply each respective regression model using an `apply_model` function.
# MAGIC 
# MAGIC Fill in the blanks below to complete the exercise.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** Refer back to the previous lesson's notebook for guidance.

# COMMAND ----------

# TODO
# Combine the data
combined_df = <FILL_IN>
 
# Define the apply_model function
def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:
 
    # Get model path from metadata
    <FILL_IN>
 
    # Subset inference set to features
    X = df_pandas.drop(["price", "neighbourhood_cleansed", "id", "n_used", "rmse", "model_path"], axis=1)
 
    # Load and apply model to inference set
    model = mlflow.sklearn.load_model(model_path)
    prediction = model.predict(X)
 
    # Create return DataFrame
    return_df = <FILL_IN>
    return return_df
 
# Define the return_schema
return_schema = StructType([
    StructField("id", StringType()),
    StructField("prediction", FloatType())
])
 
# Apply the correct respective model to each row
prediction_df = <FILL_IN>
display(prediction_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
