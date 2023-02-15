# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC # Hyperopt Lab
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Use Hyperopt to find the best hyperparameters for the wine quality dataset!

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

train_df = spark.read.format("delta").load(f"{DA.paths.datasets}/wine-quality/train")
X_train = train_df.toPandas()
y_train = X_train.pop("quality")

val_df = spark.read.format("delta").load(f"{DA.paths.datasets}/wine-quality/val")
X_val = val_df.toPandas()
y_val = X_val.pop("quality")

test_df = spark.read.format("delta").load(f"{DA.paths.datasets}/wine-quality/test")
X_test = test_df.toPandas()
y_test = X_test.pop("quality")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Keras Model
# MAGIC 
# MAGIC We will define our Neural Network in Keras and use the hyperparameters given by Hyperopt.

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.layers import Dense, Normalization
from tensorflow.keras.models import Sequential
tf.random.set_seed(42)

def create_model(hpo):
    normalize_layer = Normalization()
    normalize_layer.adapt(X_train)
    
    model = Sequential()
    model.add(normalize_layer)
    model.add(Dense(int(hpo["dense_l1"]), input_dim=11, activation="relu")) # You can change the activation functions too!
    model.add(Dense(int(hpo["dense_l2"]), activation="relu"))
    model.add(Dense(1, activation="linear"))
    return model

# COMMAND ----------

from hyperopt import fmin, hp, tpe, SparkTrials

def run_nn(hpo):
    model = create_model(hpo)

    # Select Optimizer
    optimizer_call = getattr(tf.keras.optimizers, hpo["optimizer"])
    optimizer = optimizer_call(hpo["learning_rate"])

    # Compile model
    model.compile(loss="mse",
                  optimizer=optimizer,
                  metrics=["mse"])

    history = model.fit(X_train, y_train, validation_data=[X_val,y_val], batch_size=32, epochs=10, verbose=2)

    # Evaluate our model
    obj_metric = history.history["val_loss"][-1] 
    return obj_metric

# COMMAND ----------

# MAGIC %md Now try experimenting with different hyperparameters + values!

# COMMAND ----------

# TODO
import numpy as np

space = {"dense_l1": hp.quniform("dense_l1", 10, 30, 1),
         "dense_l2": <FILL_IN>,
         <FILL_IN>: <FILL_IN>,
         <FILL_IN>: <FILL_IN>,
        }

spark_trials = SparkTrials(parallelism=<FILL_IN>)

best_hyperparams = fmin(run_nn, space, algo=tpe.suggest, max_evals=30, trials=spark_trials, rstate=np.random.default_rng(42))
best_hyperparams

# COMMAND ----------

# MAGIC %md You can continue to tweak the search space to identify candidate models, as well as track and compare them with MLflow!

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
