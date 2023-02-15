# Databricks notebook source
# MAGIC %md
# MAGIC # Creating an AutoML Experiment - Python API
# MAGIC AutoML can be used both via the user interface and via a Python-based API.
# MAGIC 
# MAGIC In this demonstration, we're going to develop a baseline model using the Python API.
# MAGIC 
# MAGIC ##### Objectives
# MAGIC 1. Set up and run an experiment
# MAGIC 1. Evaluate the results
# MAGIC 1. Register the best model
# MAGIC 1. Use the model to perform batch inference

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC 
# MAGIC First, we'll run the `Classroom-Setup` notebook to set up our environment.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set up and run the experiment
# MAGIC 
# MAGIC We will set up a regression experiment with our AirBnB dataset, with `price` as the target column and RMSE as the metric.

# COMMAND ----------

airbnb_data = spark.read.parquet(input_path)
trainDF, testDF = airbnb_data.randomSplit([.8, .2], seed=42)

# COMMAND ----------

import databricks.automl as automl

model = automl.regress(
    dataset = trainDF, 
    target_col = "price",
    primary_metric = "rmse",
    timeout_minutes = 5,
    max_trials = 10
) 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register a model
# MAGIC 
# MAGIC Once the AutoML experiment is done, we can identify the best model from the experiment and register that model to the MLflow Model Registry.

# COMMAND ----------

import mlflow

client = mlflow.tracking.MlflowClient()

run_id = model.best_trial.mlflow_run_id
model_name = "airbnb-price"
model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(model_uri, model_name)

# COMMAND ----------

print(model_details)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use the model to perform batch inference

# COMMAND ----------

predict = mlflow.pyfunc.spark_udf(spark, model_uri)
predDF = testDF.withColumn("prediction", predict(*testDF.drop("price").columns))
display(predDF)
