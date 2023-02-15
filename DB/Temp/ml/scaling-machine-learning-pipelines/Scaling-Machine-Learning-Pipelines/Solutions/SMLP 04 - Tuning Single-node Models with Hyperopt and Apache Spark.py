# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Tune Single-node Models with Hyperopt and Apache Spark
# MAGIC 
# MAGIC In this notebook, we'll demonstrate how tune single-node machine learning models using Hyperopt and Apache Spark.

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
# MAGIC ## Data Preparation
# MAGIC 
# MAGIC First, we will load our data for this lesson. It's London listing data with only numeric features.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> We'll be working with Scikit-learn in this lesson, so we're going to work with a pandas DataFrame.

# COMMAND ----------

train_pdf = spark.read.format("delta").load(lesson_4_train_path).toPandas()
test_pdf = spark.read.format("delta").load(lesson_4_test_path).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we'll assign common data variables for use with Scikit-learn.

# COMMAND ----------

X_train = train_pdf.drop(["price"], axis=1)
X_test = test_pdf.drop(["price"], axis=1)

y_train = train_pdf["price"]
y_test = test_pdf["price"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hyperopt Workflow
# MAGIC 
# MAGIC Next, we will create the different pieces needed for parallelizing hyperparameter tuning with [Hyperopt](http://hyperopt.github.io/hyperopt/) and Apache Spark.
# MAGIC 
# MAGIC ### Create objective function
# MAGIC 
# MAGIC First, we need to [create an **objective function**](http://hyperopt.github.io/hyperopt/getting-started/minimizing_functions/). This is the function that Hyperopt will call for each set of inputs.
# MAGIC 
# MAGIC The basic requirements are:
# MAGIC 
# MAGIC 1. An **input** `params` including hyperparameter values to use when training the model
# MAGIC 2. An **output** containing a loss metric on which to optimize
# MAGIC 
# MAGIC In this case, we are specifying values of `max_depth` and `n_estimators` and returning the RMSE as our loss metric.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Notice that we are cross-validating within this function with `cross_val_score`! Remember that this drastically increases the number of models we need to compute. If you're really crunched for time, use a train-validation split instead.

# COMMAND ----------

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from numpy import mean
from hyperopt import STATUS_OK
  
def objective_function(params):

    # Set the hyperparameters that we want to tune:
    max_depth = int(params["max_depth"])
    n_estimators = int(params["n_estimators"])

    regressor = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators, random_state=42)

    # Compute the average cross-validation metric
    mse_scorer = make_scorer(mean_squared_error, squared=False)
    cv_rmse = mean(cross_val_score(regressor, X_train, y_train, scoring=mse_scorer, cv=3))
    
    return {"loss": cv_rmse, "status": STATUS_OK}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the search space
# MAGIC 
# MAGIC Next, we need to [define the **search space**](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/).
# MAGIC 
# MAGIC To do this, we need to import Hyperopt and use its `quniform` function to specify the range for each hyperparameter.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Remember that we aren't defining the actual values like grid search. Hyperopt's TPE algorithm will intelligently suggest hyperparameter values from within this range.

# COMMAND ----------

from hyperopt import hp

search_space = {
  "max_depth": hp.quniform("max_depth", 1, 10, 1),
  "n_estimators": hp.quniform("n_estimators", 5, 50, 1)
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Call the `fmin` operation
# MAGIC 
# MAGIC The `fmin` function is where we put Hyperopt to work.
# MAGIC 
# MAGIC To make this work, we need:
# MAGIC 
# MAGIC 1. The `objective_function`
# MAGIC 2. The `search_space`
# MAGIC 3. The `tpe.suggest` optimization algorithm
# MAGIC 4. A `SparkTrials` object to distribute the trials across a cluster using Spark
# MAGIC 5. The maximum number of evaluations or trials denoted by `max_evals`
# MAGIC 
# MAGIC In this case, we'll be computing up to 20 trials with 4 trials being run concurrently.
# MAGIC 
# MAGIC When the optimization process is finished, we train a final model using those hyperparameter values on the entire training/cross-validation dataset.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> While Hyperopt automatically logs its trials to MLflow under a single parent run, we are going manually specifying that parent run to log our final trained model details.

# COMMAND ----------

# Import the necessary libraries
from hyperopt import fmin, tpe, STATUS_OK, SparkTrials
import mlflow

# Start a parent MLflow run
mlflow.set_experiment("/Users/" + username + "/SMLP-Lesson-4")
with mlflow.start_run():
    # The number of models we want to evaluate
    num_evals = 20

    # Set the number of models to be trained concurrently
    spark_trials = SparkTrials(parallelism=4)

    # Run the optimization process
    best_hyperparam = fmin(
        fn=objective_function, 
        space=search_space,
        algo=tpe.suggest, 
        trials=spark_trials,
        max_evals=num_evals
    )

    # Get optimal hyperparameter values
    best_max_depth = int(best_hyperparam["max_depth"])
    best_n_estimators = int(best_hyperparam["n_estimators"])

    # Train model on entire training data
    regressor = RandomForestRegressor(max_depth=best_max_depth, n_estimators=best_n_estimators, random_state=42)
    regressor.fit(X_train, y_train)

    # Evaluator on train and test set
    train_rmse = mean_squared_error(y_train, regressor.predict(X_train), squared=False)
    test_rmse = mean_squared_error(y_test, regressor.predict(X_test), squared=False)
    
    mlflow.log_param("max_depth", best_max_depth)
    mlflow.log_param("n_estimators", best_n_estimators)
    mlflow.log_metric("loss", test_rmse)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
