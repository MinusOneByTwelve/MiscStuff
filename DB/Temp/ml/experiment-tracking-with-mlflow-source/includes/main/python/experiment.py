# Databricks notebook source
import os
import pandas as pd
import numpy as np
import mlflow

from sklearn.model_selection import GridSearchCV

# COMMAND ----------

# TODO
# def mlflow_run(experiment_id, estimator, param_grid, data):
#     (X_train, X_test, y_train, y_test) = data
#
#     with mlflow.start_run(experiment_id=experiment_id) as run:
#         gs = GridSearchCV(estimator, param_grid)
#         # fit the Grid Search Model on the training data
#         FILL_THIS_IN
#
#         # score the Grid Search Model on the TRAINING data
#         train_acc = FILL_THIS_IN
#         # score the Grid Search Model on the TESTING data
#         test_acc = FILL_THIS_IN
#         mlflow.log_param("model",
#                          (str(estimator.__class__)
#                           .split('.')[-1].replace("'>","")))
#
#         mlflow.sklearn.log_model(gs.best_estimator_, "model")
#
#         for param, value in gs.best_params_.items():
#             mlflow.log_param(param, value)
#         # log the TRAINING accuracy
#         mlflow.log_metric("train acc", FILL_THIS_IN)
#         # log the TESTING accuracy
#         mlflow.log_metric("test acc", FILL_THIS_IN)

# COMMAND ----------

# ANSWER
def mlflow_run(experiment_id, estimator, param_grid, data):
    (X_train, X_test, y_train, y_test) = data

    with mlflow.start_run(experiment_id=experiment_id) as run:
        gs = GridSearchCV(estimator, param_grid)
        gs.fit(X_train, y_train)

        train_acc = gs.score(X_train, y_train)
        test_acc = gs.score(X_test, y_test)
        mlflow.log_param("model",
                         (str(estimator.__class__)
                          .split('.')[-1].replace("'>","")))

        mlflow.sklearn.log_model(gs.best_estimator_, "model")

        for param, value in gs.best_params_.items():
            mlflow.log_param(param, value)
        mlflow.log_metric("train acc", train_acc)
        mlflow.log_metric("test acc", test_acc)

# COMMAND ----------

def prepare_results(experiment_id):
    results = mlflow.search_runs(experiment_id)
    columns = [
      col for col in results.columns
      if any([
        'metric' in col,
        'param' in col,
        'artifact' in col
      ])
    ]
    return results[columns]

# COMMAND ----------

def prepare_coefs(experiment_id, lifestyles, feature_columns):

    runs = mlflow.search_runs(experiment_id)
    runs = runs[runs.status == "FINISHED"]
    models = runs.artifact_uri.apply(lambda uri: mlflow.sklearn.load_model(uri + "/model"))

    models = [
      {**model.get_params(),
        "coefs" : model.coef_
      } for model in models.values
    ]
    coefs = pd.DataFrame(models)
    coefs = coefs[["C", "l1_ratio", "penalty", "coefs"]]
    coefs["coefs"] = (
      coefs["coefs"]
      .apply(
        lambda artifact: [
          (lifestyle, coefs)
          for lifestyle, coefs
          in zip(lifestyles, artifact)
        ]
      )
    )
    coefs = coefs.explode("coefs")
    coefs["lifestyle"] = coefs["coefs"].apply(lambda artifact: artifact[0])
    coefs["coefs"] = coefs["coefs"].apply(lambda artifact: artifact[1])
    coefs.set_index(["C", "l1_ratio", "penalty", "lifestyle"], inplace=True)
    coefs = coefs["coefs"].apply(pd.Series)
    coefs.columns = feature_columns
    ax = coefs.T.plot(figsize=(20,7))
    ax.set_xticks(range(len(coefs.columns)));
    ax.set_xticklabels(coefs.columns.tolist(), rotation=45)
    return coefs, ax
