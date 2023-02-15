# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Linear Regression
# MAGIC 
# MAGIC Before you attempt to throw a neural network at a problem, you want to establish a __baseline model__. Often, this will be a simple model, such as linear regression. Once we establish a baseline, then we can get started with Deep Learning.
# MAGIC 
# MAGIC The slides for the course can be found <a href="https://brookewenig.github.io/DeepLearning.html" target="_blank">here</a>.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Build a linear regression model using scikit-learn and reimplement it in Keras 
# MAGIC  - Modify the # of epochs
# MAGIC  - Visualize the loss during training

# COMMAND ----------

# MAGIC %run ./Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC Let's start by making a simple array of features, and the label we are trying to predict is y = 2*X + 1.

# COMMAND ----------

import numpy as np
np.set_printoptions(suppress=True)

X = np.arange(-10, 11).reshape((21,1))
y = 2*X + 1

list(zip(X, y))

# COMMAND ----------

import matplotlib.pyplot as plt

plt.plot(X, y, "ro", label="True y")

plt.title("X vs. y")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Use sklearn to establish our baseline (for simplicity, we are using the same dataset for training and testing in this toy example, but we will change that later).
# MAGIC 
# MAGIC NOTE: We are using the [Databricks ML Runtime](https://docs.databricks.com/runtime/mlruntime.html) which comes with [MLflow](https://mlflow.org/) pre-installed, as well as enabled MLflow autologging to automatically track our machine learning experiments. We'll do a deep dive into MLflow later, but you will see the output of the MLflow Run logged below.

# COMMAND ----------

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC To get a sense of how well this model performed, let's import a metric of error estimation, the [mean_squared_error](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error) in this case.

# COMMAND ----------

from sklearn.metrics import mean_squared_error

y_pred = lr.predict(X)
mse = mean_squared_error(y, y_pred)
print(mse)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize Predictions
# MAGIC 
# MAGIC The mean squared error term above is one way to understand how well this model performs. But we can also use the model to predict values and see how well it matches the truth with a graph.

# COMMAND ----------

plt.plot(X, y, "ro", label="True y")
plt.plot(X, y_pred, label="Pred. y")

plt.title("X vs. True y and Pred. y (Linear Regression)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()

plt.show()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Keras
# MAGIC 
# MAGIC Now that we have established a baseline model, let's see if we can build a fully-connected neural network that can meet or exceed our linear regression model. A fully-connected neural network is simply a set of matrix multiplications followed by some non-linear function (to be discussed later). 
# MAGIC 
# MAGIC <a href="https://www.tensorflow.org/guide/keras" target="_blank">Keras</a> is a high-level API to build neural networks and was released by François Chollet in 2015. It is now the official high-level API of TensorFlow. 
# MAGIC 
# MAGIC ##### Steps to build a Keras model
# MAGIC <img style="width:20%" src="https://files.training.databricks.com/images/5_cycle.jpg" >

# COMMAND ----------

# MAGIC %md # 1. Define an n-Layer Neural Network
# MAGIC 
# MAGIC Here, we need to specify the dimensions of our input and output layers. When we say something is an n-layer neural network, we count all of the layers except the input layer. 
# MAGIC 
# MAGIC A special case of neural network with no hidden layers and no non-linearities is actually just linear regression :).
# MAGIC 
# MAGIC For the next few labs, we will use the <a href="https://www.tensorflow.org/api_docs/python/tf/keras" target="_blank">Sequential model</a> from Keras.

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
tf.random.set_seed(42)

# The Sequential model is a linear stack of layers.
model = Sequential()

model.add(Dense(units=1, input_dim=1, activation="linear"))

# COMMAND ----------

# MAGIC %md
# MAGIC We can check the model definition by calling **`.summary()`**. Note the two parameters - any thoughts on why there are TWO?

# COMMAND ----------

model.summary()

# COMMAND ----------

# MAGIC %md # 2. Compile a Network
# MAGIC 
# MAGIC To <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#compile" target="_blank">compile</a> the network, we need to specify the loss function and which optimizer to use. We'll talk more about optimizers and loss metrics in the next lab.
# MAGIC 
# MAGIC For right now, we will use **`mse`** (mean squared error) for our loss function, and the **`adam`** optimizer.

# COMMAND ----------

model.compile(loss="mse", optimizer="adam")

# COMMAND ----------

# MAGIC %md # 3. Fit a Network
# MAGIC 
# MAGIC Let's <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit" target="_blank">fit</a> our model on X and y, this will use the optimizer and loss function we set previously to train on the data.

# COMMAND ----------

model.fit(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC And take a look at the predictions.

# COMMAND ----------

keras_pred = model.predict(X)
keras_pred

# COMMAND ----------

# MAGIC %md
# MAGIC We can also see this model's **`mse`** using the same function as before from sklearn. Take note of how this compares to our **baseline model**.

# COMMAND ----------

mse = mean_squared_error(y, keras_pred)
print(mse) 

# COMMAND ----------

# MAGIC %md
# MAGIC As before, we can use a plot to get a visual representation of how well this new model fits to the truth. 

# COMMAND ----------

def keras_pred_plot(keras_pred):
    plt.clf()
    plt.plot(X, y, "ro", label="True y")
    plt.plot(X, keras_pred, label="Pred y")

    plt.title("X vs. True y and Pred. y (Keras)")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend(numpoints=1)
    plt.show()

keras_pred_plot(keras_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC What went wrong?? Turns out there a few more hyperparameters we need to set. Let's take a look at <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit" target="_blank">Keras documentation</a>.
# MAGIC 
# MAGIC The parameter **`epochs`** specifies how many passes you want over your entire dataset. Let's increase the number of epochs, and look at how the **`mse`** decreases.
# MAGIC 
# MAGIC Here we are capturing the output of model.fit() as it returns a History object, which keeps a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).

# COMMAND ----------

history = model.fit(X, y, epochs=20) 

# COMMAND ----------

def view_model_loss():
    plt.clf()
    plt.plot(history.history["loss"])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.show()

view_model_loss()

# COMMAND ----------

# MAGIC %md
# MAGIC This value for *`mse`* is still far higher than for our **baseline model**, so let's try increasing the epochs even more.

# COMMAND ----------

history = model.fit(X, y, epochs=4000)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's inspect how our model decreased the loss (**`mse`**) as the number of epochs increases.

# COMMAND ----------

view_model_loss()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's extract model weights and compare them to our equation for 2X+1.

# COMMAND ----------

print(model.get_weights())
predicted_w = model.get_weights()[0][0][0]
predicted_b = model.get_weights()[1][0]

print(f"predicted_w: {predicted_w}")
print(f"predicted_b: {predicted_b}")

# COMMAND ----------

# MAGIC %md
# MAGIC Wahoo! We were able to approximate y=2*X + 1 quite well! If we trained for some more epochs, we should be able to approximate this function exactly (at risk of overfitting of course). 
# MAGIC 
# MAGIC As a final note, let's see how the graph looks on this new set of weights.

# COMMAND ----------

# MAGIC %md # 4. Evaluate Network
# MAGIC 
# MAGIC As mentioned previously, we want to make sure our neural network can beat our benchmark. 

# COMMAND ----------

model.evaluate(X, y) # Prints loss value & metrics values for the model in test mode (both are MSE here)

# COMMAND ----------

# MAGIC %md # 5. Make Predictions

# COMMAND ----------

keras_pred = model.predict(X)
keras_pred

# COMMAND ----------

keras_pred_plot(keras_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC Alright, this was a very simple, contrived example. Let's go ahead and make this a bit more interesting in the next lab!

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
