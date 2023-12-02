# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md 
# MAGIC # Hyperopt
# MAGIC 
# MAGIC The <a href="https://github.com/hyperopt/hyperopt" target="_blank">Hyperopt library</a> allows for parallel hyperparameter tuning using either random search or [Tree of Parzen Estimators (TPE)](https://optunity.readthedocs.io/en/latest/user/solvers/TPE.html). With MLflow, we can record the hyperparameters and corresponding metrics for each hyperparameter combination. You can read more on <a href="https://github.com/hyperopt/hyperopt/blob/master/docs/templates/scaleout/spark.md" target="_blank">SparkTrials w/ Hyperopt</a>.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:
# MAGIC  - Use Hyperopt to train and optimize a feed-forward neural net

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

train_df = spark.read.format("delta").load(f"{DA.paths.datasets}/california-housing/train")
X_train = train_df.toPandas()
y_train = X_train.pop("label")

val_df = spark.read.format("delta").load(f"{DA.paths.datasets}/california-housing/val")
X_val = val_df.toPandas()
y_val = X_val.pop("label")

test_df = spark.read.format("delta").load(f"{DA.paths.datasets}/california-housing/test")
X_test = test_df.toPandas()
y_test = X_test.pop("label")

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
    model.add(Dense(int(hpo["dense_l1"]), input_dim=8, activation="relu"))
    model.add(Dense(int(hpo["dense_l2"]), activation="relu"))
    model.add(Dense(1, activation="linear"))
    return model

# COMMAND ----------

from hyperopt import fmin, hp, tpe, SparkTrials

def run_nn(hpo):
    model = create_model(hpo)

    # Select Optimizer
    optimizer_call = getattr(tf.keras.optimizers, hpo["optimizer"])
    optimizer = optimizer_call(learning_rate=hpo["learning_rate"])

    # Compile model
    model.compile(loss="mse",
                  optimizer=optimizer,
                  metrics=["mse"])

    history = model.fit(X_train, y_train, validation_data=[X_val,y_val], batch_size=64, epochs=10, verbose=2)

    # Evaluate our model
    obj_metric = history.history["val_loss"][-1]
    return obj_metric

# COMMAND ----------

# MAGIC %md ### Setup hyperparameter space and training
# MAGIC 
# MAGIC We need to create a search space for Hyperopt and set up SparkTrials to allow Hyperopt to run in parallel using Spark worker nodes. MLflow will automatically track the results of Hyperopt's tuning trials.

# COMMAND ----------

space = {
    "dense_l1": hp.quniform("dense_l1", 10, 30, 1),
    "dense_l2": hp.quniform("dense_l2", 10, 30, 1),
    "learning_rate": hp.loguniform("learning_rate", -5, 0),
    "optimizer": hp.choice("optimizer", ["Adadelta", "Adam"])
 }

spark_trials = SparkTrials(parallelism=4)

best_hyperparam = fmin(fn=run_nn, 
                       space=space, 
                       algo=tpe.suggest, 
                       max_evals=16, 
                       trials=spark_trials,
                       rstate=np.random.default_rng(42))

best_hyperparam

# COMMAND ----------

# MAGIC %md
# MAGIC To view the MLflow experiment associated with the notebook, click the MLflow API icon on the top context menu, next to the "Run all" button. A new menu will open on the right where you can click on each run, or open the experiment UI.
# MAGIC 
# MAGIC To understand the effect of tuning a hyperparameter:
# MAGIC 1. Select the resulting runs and click Compare.
# MAGIC 2. In the Scatter Plot, select a hyperparameter for the X-axis and loss for the Y-axis.
# MAGIC 
# MAGIC With these new/best parameters, you have the best parameters within this search space, and can build, track, and log a new MLflow model with these values!

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Classroom Cleanup
# MAGIC 
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson:

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
