# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Lab: Tuning an XGBoost Model with Hyperopt and Apache Spark
# MAGIC 
# MAGIC In this lab, you will have the opportunity to complete hands-on exercises in tuning single-node models using Hyperopt and Apache Spark.
# MAGIC 
# MAGIC Specifically, you will:
# MAGIC 
# MAGIC 1. Create an objective function to build an XGBoost regression model
# MAGIC 2. Define a search space for the number of sequential trees, the learning rate, and the maximum depth of trees in each model.
# MAGIC 3. Run the hyperparameter tuning process using the `fmin` operation with early stopping, all within a parent MLflow run
# MAGIC 4. Review the MLflow experiment to determine which hyperparameter values were found to be optimal, the training RMSE, and the test RMSE

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
# MAGIC Next, we will load our data for the lesson.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> We are loading in a version of our data that only includes Airbnb listings from Tokyo.

# COMMAND ----------

train_pdf = spark.read.format("delta").load(lab_4_train_path).toPandas()
test_pdf = spark.read.format("delta").load(lab_4_test_path).toPandas()

X_train = train_pdf.drop(["price"], axis=1)
X_test = test_pdf.drop(["price"], axis=1)

y_train = train_pdf["price"]
y_test = test_pdf["price"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exercises
# MAGIC 
# MAGIC ### Exercise 1: Objective function
# MAGIC 
# MAGIC In this exercise, you will create an objective function to build an [XGBoost regression model](https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn) (`XGBRegressor`).
# MAGIC 
# MAGIC Recall that there are two basic requirements for an objective function:
# MAGIC 
# MAGIC 1. An **input** `params` including hyperparameter values to use when training the model
# MAGIC 2. An **output** containing a loss metric on which to optimize
# MAGIC 
# MAGIC In this case, we'll be using the following inputs:
# MAGIC 
# MAGIC * `n_estimators` (integer): The number of gradient boosted trees
# MAGIC * `max_depth` (integer): The maximum tree depth for gradient boosted trees
# MAGIC * `learning_rate` (float): The learning rate
# MAGIC 
# MAGIC Create an objective function that accepts these parameters, builds an `XGBRegressor`, and returns the cross-validated RMSE as output.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** Look at the example in the lesson to learn what to return from the objective function!

# COMMAND ----------

# TODO
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from numpy import mean
from hyperopt import STATUS_OK

def objective_function(<FILL_IN>):

    # Set the hyperparameters that we want to tune:
    <FILL_IN>

    regressor = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)

    # Compute the average cross-validation metric
    mse_scorer = make_scorer(mean_squared_error, squared=False)
    cv_rmse = mean(cross_val_score(regressor, X_train, y_train, scoring=mse_scorer, cv=3))
    
    return <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 2: Search space
# MAGIC 
# MAGIC In this exercise, you will define a search space for the number of sequential trees, the learning rate, and the maximum depth of trees in each model.
# MAGIC 
# MAGIC In the example in the lesson demo, we used `n_estimators` and `max_depth` – **these are both integers**, so we used Hyperopt's `quniform` operation to create their search spaces.
# MAGIC 
# MAGIC But `learning_rate` is a **floating point** between 0.0 and 1.0, so we need to use a **different Hyperopt function** to specify its space.
# MAGIC 
# MAGIC Refer to the below table to determine which Hyperopt function should be used to specify the space for `learning_rate`:
# MAGIC 
# MAGIC 
# MAGIC Hyperparameter Type | Suggested Hyperopt range
# MAGIC -- | --
# MAGIC Maximum depth, number of trees, max 'bins' in Spark ML decision trees | hp.quniform with min >= 1
# MAGIC Learning rate | hp.loguniform with max = 0 (because exp(0) = 1.0)
# MAGIC Regularization strength | hp.uniform with min = 0 or hp.loguniform
# MAGIC Ratios or fractions, like Elastic net ratio | hp.uniform with min = 0, max = 1
# MAGIC Shrinkage factors like eta in xgboost | hp.uniform with min = 0, max = 1
# MAGIC Loss criterion in decision trees (ex: gini vs entropy) | hp.choice
# MAGIC Activation function (e.g. ReLU vs leaky ReLU) | hp.choice
# MAGIC Optimizer (e.g. Adam vs SGD) | hp.choice
# MAGIC Neural net layer width, embedding size | hp.quniform with min >>= 1
# MAGIC 
# MAGIC Once you've determined which function to use, create a search space with the following ranges for each hyperparameter:
# MAGIC 
# MAGIC * `n_estimators`: [5, 25)
# MAGIC * `max_depth`: [2, 10)
# MAGIC * `learning_rate`: [0.01, 1.0]
# MAGIC   
# MAGIC <img src="https://files.training.databricks.com/images/icon_hint_24.png"/>&nbsp;**Hint:** Be careful with setting the minimum and maximum for `learning_rate`! Check out the <a href="https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions" target="_blank" rel="noopener noreferrer">Hyperopt documentation</a> to determine how to set the minimum and maximum values!

# COMMAND ----------

# TODO
from hyperopt import hp
from numpy import log

learning_rate_min = log(<FILL_IN>)
learning_rate_max = log(<FILL_IN>)

search_space = {
    "max_depth": <FILL_IN>,
    "n_estimators": <FILL_IN>,
    "learning_rate": <FILL_IN>
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 3: `fmin` operation
# MAGIC 
# MAGIC In this exercise, you will run the hyperparameter tuning process using the `fmin` operation with early stopping, all within a parent MLflow run.
# MAGIC 
# MAGIC Pass the following arguments to the `fmin` operation to control the process:
# MAGIC 
# MAGIC 1. The `objective_function`
# MAGIC 2. The `search_space`
# MAGIC 3. The `tpe.suggest` optimization algorithm
# MAGIC 4. A `SparkTrials` object with a parallelism of 4
# MAGIC 5. A `max_evals` of 32

# COMMAND ----------

# TODO
# Import the necessary libraries
from hyperopt import fmin, tpe, STATUS_OK, SparkTrials
import mlflow
 
# Start a parent MLflow run
mlflow.set_experiment("/Users/" + username + "/SMLP-Lab-4")
with mlflow.start_run():
    # The number of models we want to evaluate
    num_evals = <FILL_IN>
 
    # Set the number of models to be trained concurrently
    spark_trials = <FILL_IN>
 
    # Run the optimization process
    best_hyperparam = fmin(
        fn=<FILL_IN>, 
        space=<FILL_IN>,
        algo=<FILL_IN>, 
        trials=<FILL_IN>,
        max_evals=<FILL_IN>
    )
 
    # Get optimal hyperparameter values
    best_max_depth = int(<FILL_IN>)
    best_n_estimators = int(<FILL_IN>)
    best_learning_rate = <FILL_IN>
 
    # Train model on entire training data
    regressor = xgb.XGBRegressor(
        max_depth=best_max_depth, 
        n_estimators=best_n_estimators, 
        learning_rate=best_learning_rate
    )
    regressor.fit(X_train, y_train)
 
    # Evaluator on train and test set
    train_rmse = mean_squared_error(y_train, regressor.predict(X_train), squared=False)
    test_rmse = mean_squared_error(y_test, regressor.predict(X_test), squared=False)

    mlflow.log_param("max_depth", best_max_depth)
    mlflow.log_param("n_estimators", best_n_estimators)
    mlflow.log_param("learning_rate", best_learning_rate)
    mlflow.log_metric("loss", test_rmse)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exercise 4: Viewing run results
# MAGIC 
# MAGIC In this exercise, you will review the MLflow experiment to determine which hyperparameter values were found to be optimal, the training RMSE, and the test RMSE.
# MAGIC 
# MAGIC **Question**: What was the optimal value for each hyperparameter?
# MAGIC 
# MAGIC **Question**: What was the average cross-validated RMSE for the models trained with the optimal hyperparameters?
# MAGIC 
# MAGIC **Question**: What was the test RMSE of the final model?

# COMMAND ----------

# MAGIC %md
# MAGIC If you'd like to learn more about Hyperopt, there's a great [Databricks blogpost](https://databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html) on how to optimally use Hyperopt with Spark and MLflow.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
