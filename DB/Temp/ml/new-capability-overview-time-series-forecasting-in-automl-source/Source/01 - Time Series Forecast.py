# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Time Series Forecasting with AutoML
# MAGIC 
# MAGIC In this notebook, we're going to go over creating a time series forecast using AutoML. By the end of thise notebook, you should be able to describe the process of creating a Time Series forecast using the AutoML API.
# MAGIC 
# MAGIC We're going to:
# MAGIC - Import a dataset
# MAGIC - Regsiter a model
# MAGIC - Bring a model back into a notebook
# MAGIC - Use that model to forecast
# MAGIC - Create two visualizations about predictions

# COMMAND ----------

import pyspark.pandas as ps
df = ps.read_csv("/databricks-datasets/COVID/covid-19-data")
df["date"] = ps.to_datetime(df['date'], errors='coerce')
df["cases"] = df["cases"].astype(int)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## AutoML training
# MAGIC The following command starts an AutoML run. You must provide the column that the model should predict in the target_col argument and the time column. When the run completes, you can follow the link to the best trial notebook to examine the training code.
# MAGIC 
# MAGIC This example also specifies:
# MAGIC 
# MAGIC - horizon=30, to specify that AutoML should forecast 30 days into the future.
# MAGIC - frequency="d" to specify that a forecast should be provided for each day.
# MAGIC - primary_metric="mdape" to specify the metric to optimize for during training.

# COMMAND ----------

import databricks.automl
import logging
 
# Disable informational messages from fbprophet
logging.getLogger("py4j").setLevel(logging.WARNING)
 
summary = databricks.automl.forecast(df, target_col="cases", time_col="date", horizon=30, frequency="d",  primary_metric="mdape")b

# COMMAND ----------

# MAGIC %md ## Load the model with MLflow
# MAGIC 
# MAGIC MLFlow allows you to easily import models back into Python by using the AutoML trial_id .

# COMMAND ----------

import mlflow.pyfunc
from mlflow.tracking import MlflowClient
 
run_id = MlflowClient()
trial_id = summary.best_trial.mlflow_run_id
 
model_uri = "runs:/{run_id}/model".format(run_id=trial_id)
pyfunc_model = mlflow.pyfunc.load_model(model_uri)

# COMMAND ----------

model = pyfunc_model._model_impl.python_model.model()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Use the model to make a forecast
# MAGIC 
# MAGIC Call the predict_timeseries model method to generate a forecast. See the Prophet documentation for more details.

# COMMAND ----------

forecast = pyfunc_model._model_impl.python_model.predict_timeseries()
forecast.head()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Plot the forecast with change points and trend
# MAGIC In the plots below, the thick black line shows the time series dataset, and the red line is the forecast created by the AutoML Prophet model.

# COMMAND ----------

from prophet.plot import plot_plotly, plot_components_plotly
 
plot_plotly(model, forecast, changepoints=True, trend=True)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Plot the forecast components
# MAGIC 
# MAGIC The forecast components are the different signals the model uses to make its forecast. Examples of forecast components are the trend, seasonality, and holidays.

# COMMAND ----------

plot_components_plotly(model, forecast)

# COMMAND ----------


