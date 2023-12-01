# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Parallelizing Grouped Model Training and Inference
# MAGIC 
# MAGIC In this notebook, we'll demonstrate how to apply grouped machine learning model training and inference using Pandas UDFs and the Pandas Function APIs.

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
# MAGIC ## Inference with Pandas UDFs
# MAGIC 
# MAGIC In this part of the noteboook, we'll demonstrate how to parallelize single-node model inference using vectorized [Pandas UDFs](https://spark.apache.org/docs/3.0.0/sql-pyspark-pandas-with-arrow.html).
# MAGIC 
# MAGIC #### Split data
# MAGIC 
# MAGIC First, we're going to load two different sets:
# MAGIC 
# MAGIC 1. A **modeling set** to facilitate modeling &mdash; this will then get split into a training and a test set
# MAGIC 2. An **inference set** to facilitate the demonstration of parallelizing inference

# COMMAND ----------

model_df = spark.read.format("delta").load(lesson_5_model_path)
model_pdf = model_df.toPandas()

inference_df = spark.read.format("delta").load(lesson_5_inference_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build and log model
# MAGIC 
# MAGIC In order to demonstrate this process, we'll need to build and log a simple single-node model.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> It's important that we log this model so we can easily load it back in during inference time.

# COMMAND ----------

import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

mlflow.set_experiment("/Users/" + username + "/SMLP-Lesson-5-SN")
with mlflow.start_run(run_name="sklearn-random-forest") as run:
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        model_pdf.drop(["price", "neighbourhood_cleansed", "id"], axis=1), 
        model_pdf[["price"]].values.ravel(),
        test_size=0.2,
        random_state=42
    )

    # Create model
    rfr = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rfr.fit(X_train, y_train)

    # Log model
    mlflow.sklearn.log_model(rfr, "random-forest-model")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define Pandas UDF for prediction
# MAGIC 
# MAGIC Next, we need to define the Pandas UDF we'll use for prediction.
# MAGIC 
# MAGIC Our first step is to decorate the function using the `@pandas_udf` decorator.
# MAGIC 
# MAGIC Next, we create a function with the following signature:
# MAGIC 
# MAGIC * **Input**: An iterator of DataFrames
# MAGIC * **Output**: An iterator of series
# MAGIC 
# MAGIC This will look similar to the iterator of series -> iterator of series [workflow used here](https://docs.databricks.com/spark/latest/spark-sql/udf-python-pandas.html#iterator-of-series-to-iterator-of-series-udf).
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> When we use an iterator as input/output, we only need to load the model once per executor rather than with each batch/partition &mdash; this reduces overhead to help us scale!

# COMMAND ----------

from typing import Iterator
from pyspark.sql.functions import pandas_udf

# Use the pandas_udf decorator and indicate that a double-type value is being returned
@pandas_udf("double")
def udf_predict(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.Series]:
    
    # Load the model for the entire iterator
    model_path = f"runs:/{run.info.run_id}/random-forest-model" 
    model = mlflow.sklearn.load_model(model_path)
    
    # Iterate through and get a prediction for each batch (partition)
    for features in iterator:
        pdf = pd.concat(features, axis=1)
        yield pd.Series(model.predict(pdf))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Apply Pandas UDF to Spark DataFrame
# MAGIC 
# MAGIC And finally, we can apply our single-node model to our Spark DataFrame in parallel.
# MAGIC 
# MAGIC Even though the API is relatively simple, recall everything happening here:
# MAGIC 
# MAGIC 1. Each partition of `inference_df` is being converted to a Pandas DataFrame using Apache Arrow
# MAGIC 2. The model is being loaded on each executor
# MAGIC 3. An iterator of Pandas DataFrames is passed to each executor
# MAGIC 4. A Pandas Series is returned from each executor and converted back to Spark using Apace Arrow
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Because we need to broadcast the model onto each executor, this approach could become inefficient if your model is extremely large in memory.

# COMMAND ----------

prediction_df = inference_df.withColumn(
    "prediction", 
    udf_predict(*inference_df.drop("price", "neighbourhood_cleansed", "id").columns)
)
display(prediction_df.select("price", "prediction"))

# COMMAND ----------

# MAGIC %md
# MAGIC In addition to using the Pandas UDF `predict` to apply the model to a Spark DataFrame, we can also use MLflow.

# COMMAND ----------

mlflow_predict = mlflow.pyfunc.spark_udf(spark, run.info.artifact_uri + "/random-forest-model")
prediction_df = inference_df.withColumn(
    "prediction", 
    mlflow_predict(*inference_df.drop("price", "neighbourhood_cleansed", "id").columns)
)
display(prediction_df.select("price", "prediction"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parallelizing Grouped Training with the Pandas Function API
# MAGIC 
# MAGIC In this part of the notebook, we'll demo how to parallelize the training of group-specific single-node models with [the Pandas Function API](https://docs.databricks.com/spark/latest/spark-sql/pandas-function-apis.html).
# MAGIC 
# MAGIC #### Define `train_model` function
# MAGIC 
# MAGIC First, we need to create our `train_model` function using nested MLflow runs.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> We are returning metadata from this function. This will be helpful when we perform grouped model inference later in this notebook.

# COMMAND ----------

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

def train_model(df_pandas: pd.DataFrame) -> pd.DataFrame:

    # Pull metadata
    neighbourhood = df_pandas["neighbourhood_cleansed"].iloc[0] # This works because df_pandas is neighborhood-specific
    n_used = df_pandas.shape[0]
    run_id = df_pandas["run_id"].iloc[0]                 # Pulls run ID to do a nested run
    experiment_id = df_pandas["experiment_id"].iloc[0]   # Pulls experiment ID for Jobs

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
        
        # Create a nested run for the specific nieghbourhood
        with mlflow.start_run(run_name=neighbourhood, experiment_id=experiment_id, nested=True) as run:
            mlflow.sklearn.log_model(rfr, neighbourhood)
            mlflow.log_metric("rmse", rmse)

            artifact_uri = f"runs:/{run.info.run_id}/{neighbourhood}"
            
            # Create a return pandas DataFrame that matches the schema above
            return_df = pd.DataFrame(
                [[neighbourhood, n_used, artifact_uri, rmse]], 
                columns=["neighbourhood_cleansed", "n_used", "model_path", "rmse"]
            )

    return return_df 


# COMMAND ----------

# MAGIC %md
# MAGIC #### Define return schema
# MAGIC 
# MAGIC We need to define the schema of the DataFrame being returned from our `train_model` function.

# COMMAND ----------

from pyspark.sql.types import FloatType, IntegerType, StringType, StructField, StructType

return_schema = StructType([
    StructField("neighbourhood_cleansed", StringType()),  # unique nieghbourhood name
    StructField("n_used", IntegerType()),                 # number of records used in training
    StructField("model_path", StringType()),              # path to the model for a given neighbourhood
    StructField("rmse", FloatType())                      # metric for model performance
])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Apply `train_model` function to each group
# MAGIC 
# MAGIC Now we apply the `train_model` function to each group.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Notice how we are passing the 

# COMMAND ----------

from pyspark.sql.functions import lit

# Start parent run
mlflow.set_experiment("/Users/" + username + "/SMLP-Lesson-5-Grouped")
with mlflow.start_run(run_name="Training session for all neighborhood") as run:
    
    # Get run_id and experiment_id
    run_id = run.info.run_id
    experiment_id = mlflow.get_experiment_by_name("/Users/" + username + "/SMLP-Lesson-5-Grouped").experiment_id

    # Apply function to each group
    train_output_df = (model_df
        .withColumn("run_id", lit(run_id))                     # Add run_id to pass into function
        .withColumn("experiment_id", lit(experiment_id))       # Add experiment_id to pass into function
        .groupby("neighbourhood_cleansed")                     # Group by neighbourhood
        .applyInPandas(train_model, schema=return_schema)      # Apply train_model function for each neighbourhood
    )

display(train_output_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Parallelizing Grouped Inference with the Pandas Function API
# MAGIC 
# MAGIC In this part of the notebook, we'll demo how to parallelize the training of group-specific single-node models.
# MAGIC 
# MAGIC #### Combine inference data and metadata
# MAGIC 
# MAGIC We need to start by combining the `inference_df` and the `train_output_df` into a single `combined_df`. This provides an easy way to pass all of the information into the `apply_model` function defined below.

# COMMAND ----------

combined_df = inference_df.join(train_output_df, "neighbourhood_cleansed")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define `apply_model` function
# MAGIC 
# MAGIC From here on, this process looks very similar to how group-specific training was parallelized above:
# MAGIC 
# MAGIC 1. Define an `apply_model` function to apply
# MAGIC 2. Define a `return_schema` for the return DataFrame of `apply_model`
# MAGIC 3. Split the data by group, apply the function, and return the combined results
# MAGIC 
# MAGIC We'll start by defining the `apply_model` function.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Be sure to drop the metadata columns we passed in with the DataFrame!

# COMMAND ----------

def apply_model(df_pandas: pd.DataFrame) -> pd.DataFrame:

    # Get model path from metadata
    model_path = df_pandas["model_path"].iloc[0]

    # Subset inference set to features
    X = df_pandas.drop(["price", "neighbourhood_cleansed", "id", "n_used", "rmse", "model_path"], axis=1)

    # Load and apply model to inference set
    model = mlflow.sklearn.load_model(model_path)
    prediction = model.predict(X)

    # Create return DataFrame
    return_df = pd.DataFrame({
        "id": df_pandas["id"],      # A unique identifier is key to linking predictions back to the DF later
        "prediction": prediction
    })
    return return_df

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define return schema
# MAGIC 
# MAGIC And again, we define a return schema.

# COMMAND ----------

return_schema = StructType([
    StructField("id", StringType()),
    StructField("prediction", FloatType())
])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Apply `apply_model` function to each group
# MAGIC 
# MAGIC And now we follow the split-apply-combine workflow using the `apply_model` function.
# MAGIC 
# MAGIC The resulting predictions were each computed using each group's respective model! 

# COMMAND ----------

prediction_df = combined_df.groupby("neighbourhood_cleansed").applyInPandas(apply_model, schema=return_schema)
display(prediction_df)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
