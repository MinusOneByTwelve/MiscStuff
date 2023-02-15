# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC # MLflow Model Life Cycle
# MAGIC 
# MAGIC In this notebook, we will: 
# MAGIC - Import and prepare a training dataset 
# MAGIC - Train a model using MLflow to track metrics and more
# MAGIC - Register the model with MLflow

# COMMAND ----------

import pandas as pd
data_path = "https://raw.githubusercontent.com/mlflow/mlflow/master/examples/sklearn_elasticnet_wine/wine-quality.csv"
data = pd.read_csv(data_path)
display(data)

# COMMAND ----------

from sklearn.model_selection import train_test_split

# The "quality" column is a scalar between 3 and 9. We're trying to predict a wine's quality given some chemical makeup.
X = data.drop("quality", axis=1)
y = data["quality"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# COMMAND ----------

# MAGIC %md Now onto our model with MLflow. We will import and set our logging parameters. Don't forget to change your run name!

# COMMAND ----------

import numpy
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model_name = "sk-learn-random-forest-reg-model"

with mlflow.start_run(run_name="ms course") as run:
    params = {"n_estimators": 5, "random_state": 42}
    rand_forst_reg = RandomForestRegressor(**params)

    # Log parameters and metrics using the MLflow Autolog APIs
    mlflow.autolog()
    
    # Fit the model 
    rand_forst_reg.fit(X_train, y_train)
    
    # Logging the Model's RMSE
    predictions = rand_forst_reg.predict(X_test)
    rmse = numpy.sqrt(mean_squared_error(y_test, predictions))
    mlflow.log_metric("rmse", rmse)
    
    # Log the sklearn model and register as version 1
    mlflow.sklearn.log_model(
        sk_model=rand_forst_reg,
        artifact_path="sklearn-model",
        registered_model_name=model_name
    )

# COMMAND ----------

# MAGIC %md
# MAGIC Now, navigate to Experiments UI to view the registered model.

# COMMAND ----------

X_train.head(1).values

# COMMAND ----------


