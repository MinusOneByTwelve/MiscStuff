# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Preprocessing with tf.keras layers
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC - Learn to use normalization/standardization for better model convergence

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md Let's set up our modules and load in the dataset from Delta.

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization

tf.random.set_seed(42)

train_df = spark.read.format("delta").load(f"{DA.paths.datasets}/california-housing/train")
X_train = train_df.toPandas()
y_train = X_train.pop("label")

test_df = spark.read.format("delta").load(f"{DA.paths.datasets}/california-housing/test")
X_test = test_df.toPandas()
y_test = X_test.pop("label")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the distribution of our features.

# COMMAND ----------

import pandas as pd

X_train.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalization
# MAGIC 
# MAGIC Because our features are all on different scales, it's going to be more difficult for our neural network during training. Let's do feature-wise standardization.
# MAGIC 
# MAGIC We are going to use the [Normalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization) from TensorFlow, which will remove the mean (zero-mean) and scale to unit variance. Even though tf.keras calls its `Normalization` layer "normalization", it implements the standardization formula under the hood (and not normalization - confusing, we know!). 
# MAGIC 
# MAGIC $$x' = \frac{x - \bar{x}}{\sigma}$$
# MAGIC 
# MAGIC In the next command, the `normalize_layer` will compute the feature mean & standard deviation that it learns from `X_train`, but `X_train` itself is unmodified.

# COMMAND ----------

normalize_layer = Normalization()
normalize_layer.adapt(X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalization inside model

# COMMAND ----------

def build_model():
    return Sequential([normalize_layer,
                       Dense(20, input_dim=8, activation="relu"),
                       Dense(20, activation="relu"),
                       Dense(1, activation="linear")]) # Keep the output layer as linear because this is a regression problem

# COMMAND ----------

# MAGIC %md
# MAGIC Any statistics stored in Normalization layer is not trainable. It is calculated once when the X_train data is read. 

# COMMAND ----------

model = build_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
model.summary()

# COMMAND ----------

history = model.fit(X_train,
                    y_train,
                    epochs=3,
                    batch_size=32
                   )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate test data
# MAGIC 
# MAGIC Since we included the Normalization layer inside the model, we can now deploy this model without having to worry about normalization again. The model will handle preprocessing on the test data. This eliminates the risk of preprocessing mismatch. 

# COMMAND ----------

model.evaluate(X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Normalization outside model

# COMMAND ----------

# MAGIC %md
# MAGIC As you can see, including the preprocessing layer inside the model is pretty straightforward for inference. Some argue that this will slow down training since preprocessing happens once per epoch. However, the slow down due to `tf.keras.Normalization` is quite negligible since it is a simple operation, according to the author of [Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow](https://learning.oreilly.com/library/view/hands-on-machine-learning/9781098125967/).
# MAGIC 
# MAGIC However, if you wish to do `tf.keras.Normalization` outside the model, you can do it as well, similar to [sklearn.preprocessing.StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).

# COMMAND ----------

normalize_layer = tf.keras.layers.Normalization()
normalize_layer.adapt(X_train)
X_train_scaled = normalize_layer(X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC Train a model on scaled data now, without Normalization layer

# COMMAND ----------

outside_model = Sequential([Dense(20, input_dim=8, activation="relu"),
                            Dense(20, activation="relu"),
                            Dense(1, activation="linear")])

outside_model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
outside_model.summary()

# COMMAND ----------

outside_model.fit(X_train_scaled, y_train, epochs=3, batch_size=32)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate test data
# MAGIC 
# MAGIC Moving the normalization outside the model can speed up training a little bit by not redoing the same operation each epoch. 
# MAGIC 
# MAGIC However, for inference, the model will not preprocess its input automatically. To combat this, we need to create a new model that wraps the normalization layer and model together.

# COMMAND ----------

inference_model = tf.keras.Sequential([normalize_layer, outside_model])
y_pred = inference_model(X_test)
y_pred

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
