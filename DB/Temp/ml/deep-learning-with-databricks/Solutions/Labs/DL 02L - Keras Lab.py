# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Keras Lab: Wine Quality Dataset
# MAGIC Let's build a Keras model to predict the quality rating on the <a href="https://archive.ics.uci.edu/ml/datasets/wine+quality" target="_blank">Red Wine Quality Dataset</a>.
# MAGIC 
# MAGIC This dataset contains features based on the physicochemical tests of a wine, and the label you will try to predict is the **`quality`**, the rating of the wine out of 10. 
# MAGIC 
# MAGIC ![](https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/Wine_grapes03.jpg/340px-Wine_grapes03.jpg)
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Build and evaluate your first Keras model!
# MAGIC   

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

train_df = spark.read.format("delta").load(f"{DA.paths.datasets}/wine-quality/train")
X_train = train_df.toPandas()
y_train = X_train.pop("quality")
                                 
test_df = spark.read.format("delta").load(f"{DA.paths.datasets}/wine-quality/test")
X_test = test_df.toPandas()
y_test = X_test.pop("quality")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at the distribution of our features.

# COMMAND ----------

X_train.describe()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Keras Neural Network Life Cycle Model
# MAGIC 
# MAGIC <img style="width:20%" src="https://files.training.databricks.com/images/5_cycle.jpg" >

# COMMAND ----------

# MAGIC %md # 1. Define a Network
# MAGIC 
# MAGIC We need to specify our <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense" target="_blank">dense layers</a>.
# MAGIC 
# MAGIC The first layer should have 50 units, the second layer, 20 units, and the last layer 1 unit. For all of the layers, make the activation function **`relu`** except for the last layer, as that activation function should be **`linear`**.

# COMMAND ----------

# ANSWER
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
tf.random.set_seed(42)

model = Sequential()
model.add(Dense(50, input_dim=11, activation="relu"))
model.add(Dense(20, activation="relu"))
model.add(Dense(1))
model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Question
# MAGIC 
# MAGIC If you did the previous cell correctly, you should see there are 600 parameters for the first layer. Why are there 600?
# MAGIC 
# MAGIC **HINT**: Add in **`use_bias=False`** in the Dense layer, and you should see a difference in the number of parameters (don't forget to set this back to **`True`** before moving on)

# COMMAND ----------

# ANSWER
# There are 600 parameters because 11*50 = 550, and for each unit in the first hidden layer, there is an associated bias unit, for a total of 600 learnable parameters.

# COMMAND ----------

# MAGIC %md # 2. Compile a Network
# MAGIC 
# MAGIC To <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile" target="_blank">compile</a> the network, we need to specify the loss function, which optimizer to use, and a metric to evaluate how well the model is performing.
# MAGIC 
# MAGIC Use **`mse`** as the loss function, **`adam`** as the optimizer, and **`mse`** as the evaluation metric.

# COMMAND ----------

# ANSWER
model.compile(optimizer="adam", loss="mse", metrics=["mse"])

# COMMAND ----------

# MAGIC %md # 3. Fit a Network
# MAGIC 
# MAGIC Now we are going to <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit" target="_blank">fit</a> our model to our training data. Set **`epochs`** to 30 and **`batch_size`** to 32, **`verbose`** to 2.

# COMMAND ----------

# ANSWER
history = model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=2)

# COMMAND ----------

# MAGIC %md
# MAGIC Visualize model loss

# COMMAND ----------

import matplotlib.pyplot as plt

def view_model_loss():
    plt.clf()
    plt.plot(history.history["loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()
view_model_loss()

# COMMAND ----------

# MAGIC %md # 4. Evaluate Network

# COMMAND ----------

# ANSWER
model.evaluate(X_test, y_test)

# COMMAND ----------

# MAGIC %md # 5. Make Predictions

# COMMAND ----------

# ANSWER
model.predict(X_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Wahoo! You successfully just built your first neural network in Keras! 

# COMMAND ----------

# MAGIC %md ## BONUS:
# MAGIC Try around with changing some hyperparameters. See what happens if you increase the number of layers, or change the optimizer, etc. What about standardizing the data??
# MAGIC 
# MAGIC If you have time, how about building a baseline model to compare against?

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
