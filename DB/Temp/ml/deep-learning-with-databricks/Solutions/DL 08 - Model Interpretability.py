# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Interpretability
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Use [SHAP](https://shap.readthedocs.io/en/latest) to understand which features are most important in the model's prediction for wine quality

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

import pandas as pd
from sklearn.preprocessing import StandardScaler

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

# MAGIC %md Build Model

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization, InputLayer
tf.random.set_seed(42)

normalize_layer = Normalization()
normalize_layer.adapt(X_train)

def build_model():
    return Sequential([
        InputLayer(input_shape=(11,)), # For SHAP, it needs to know the input dimensions, so putting input layer first
        normalize_layer,
        Dense(50, input_dim=11, activation="relu"),
        Dense(20, activation="relu"),
        Dense(1, activation="linear")])

# COMMAND ----------

model = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mse", metrics=["mse"])
model.summary()

# COMMAND ----------

history = model.fit(X_train, y_train, validation_data=[X_val, y_val], epochs=30, batch_size=32, verbose=2)

# COMMAND ----------

# MAGIC %md ## SHAP
# MAGIC 
# MAGIC SHAP <a href="https://github.com/slundberg/shap" target="_blank">SHapley Additive exPlanations</a> is another approach to explain the output of a machine learning model. See the <a href="http://papers.nips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions" target="_blank">SHAP NIPS</a> paper for details, and Christoph Molnar's book chapter on <a href="https://christophm.github.io/interpretable-ml-book/shapley.html" target="_blank">Shapley Values</a>.
# MAGIC 
# MAGIC ![](https://raw.githubusercontent.com/slundberg/shap/master/docs/artwork/shap_diagram.png)
# MAGIC 
# MAGIC Great <a href="https://blog.dominodatalab.com/shap-lime-python-libraries-part-1-great-explainers-pros-cons/" target="_blank">blog post</a> comparing LIME (another model explainability method) to SHAP. SHAP provides greater theoretical guarantees than LIME, but at the cost of additional compute. 

# COMMAND ----------

import shap

help(shap.DeepExplainer)

# COMMAND ----------

model.summary()

# COMMAND ----------

import numpy as np
shap.initjs()
shap_explainer = shap.DeepExplainer(model, X_train[:200])
base_value = model.predict(X_train).mean() # base value = average prediction
shap_values = shap_explainer.shap_values(X_test[0:1].to_numpy())
y_pred = model.predict(X_test[0:1])
print(f"Actual rating: {y_test.values[0]}, Predicted rating: {y_pred[0][0]}")
                   
# Saving to File b/c can't display IFrame directly in Databricks: https://github.com/slundberg/shap/issues/101
file_path = "/tmp/shap.html"
shap.save_html(file_path, shap.force_plot(base_value, 
                                          shap_values[0], 
                                          features=X_test[0:1],
                                          feature_names=X_test.columns, 
                                          show=False))

# COMMAND ----------

# MAGIC %md ## Visualize
# MAGIC 
# MAGIC * Red pixels increase the model's output while blue pixels decrease the output.
# MAGIC 
# MAGIC Here's a great <a href="https://christophm.github.io/interpretable-ml-book/shapley.html" target="_blank">article</a> discussing how SHAP works under the hood.
# MAGIC 
# MAGIC From the <a href="https://proceedings.neurips.cc/paper/2017/file/8a20a8621978632d76c43dfd28b67767-Paper.pdf" target="_blank">original SHAP paper</a>:
# MAGIC > Base value is the value that would be predicted if we did not know any features for the current output.
# MAGIC 
# MAGIC In other words, it is the mean prediction. 
# MAGIC 
# MAGIC Red/Blue: Features that push the prediction higher (to the right) are shown in red, and those pushing the prediction lower are shown in blue.

# COMMAND ----------

import codecs

f = codecs.open(file_path, "r")
displayHTML(f.read())

# COMMAND ----------

# MAGIC %md The values on the bottom show the true values of **`X_test[0]`**.

# COMMAND ----------

import pandas as pd

pd.DataFrame(X_test.to_numpy()[0], X_test.columns, ["features"])

# COMMAND ----------

# MAGIC %md
# MAGIC Let's see what the corresponding SHAP values are to each feature.

# COMMAND ----------

shap_features = pd.DataFrame(shap_values[0][0], X_test.columns, ["features"])
shap_features.sort_values("features")

# COMMAND ----------

# MAGIC %md
# MAGIC Visualize feature importance summary

# COMMAND ----------

shap.summary_plot(shap_values[0], X_test, plot_type="bar", feature_names=X_test.columns)

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
