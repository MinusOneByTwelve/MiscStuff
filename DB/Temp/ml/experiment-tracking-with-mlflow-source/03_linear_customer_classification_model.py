# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Building a Linear Customer Classfication Model

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------
# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %run ./includes/main/python/preprocessing

# COMMAND ----------

# MAGIC %run ./includes/main/python/experiment

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

# COMMAND ----------

data = (X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Grid-Searched Model Fitting
# MAGIC
# MAGIC The following models were fit using a grid-searched, cross validation with
# MAGIC the respective parameter dictionaries:
# MAGIC
# MAGIC  - Ridge,
# MAGIC    - `{'alpha' : logspace(-5,5,11)}`
# MAGIC  - Lasso,
# MAGIC    - `{'alpha' : logspace(-5,5,11)}`
# MAGIC  - Elastic Net,
# MAGIC    - `{'alpha' : logspace(-5,5,11), 'l1_ratio' : linspace(0,1,11)}`

# COMMAND ----------

estimator = LogisticRegression(max_iter=10000)
param_grid = {
  'C' : np.logspace(-5,5,11),
  "penalty" : ['l2']
}
mlflow_run(experiment_id, estimator, param_grid, data)

# COMMAND ----------

estimator = LogisticRegression(max_iter=10000)
param_grid = {
  'C' : np.logspace(-5,5,11),
  "penalty" : ['l1'], "solver" : ['saga']
}
mlflow_run(experiment_id, estimator, param_grid, data)

# COMMAND ----------

estimator = LogisticRegression(max_iter=10000)
param_grid = {
  'C' : np.logspace(-5,5,11),
  "penalty" : ['elasticnet'],
  'l1_ratio' : np.linspace(0,1,11),
  "solver" : ['saga']
}
mlflow_run(experiment_id, estimator, param_grid, data)

# COMMAND ----------

# MAGIC %md ### Display Experiment Results

# COMMAND ----------

# MAGIC %md ### Display Coefficients Associated with Each Classifier

# COMMAND ----------

prepare_results(experiment_id)

# COMMAND ----------

# MAGIC %md ### Display Coefficients Associated with Each Classifier

# COMMAND ----------

prepare_coefs(experiment_id, le.classes_, X_train.columns)
