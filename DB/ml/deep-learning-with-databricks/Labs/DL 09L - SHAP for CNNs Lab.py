# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # SHAP for CNNs
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Use SHAP to generate explanation behind a model's predictions

# COMMAND ----------

# MAGIC %md
# MAGIC Here, we will be using the <a href="https://github.com/zalandoresearch/fashion-mnist" target="_blank">Fashion MNIST</a> dataset that contains 70,000 grayscale images. <br> 
# MAGIC <br>
# MAGIC There are 10 categories altogether: T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, and ankle boots.
# MAGIC 
# MAGIC <img src="https://tensorflow.org/images/fashion-mnist-sprite.png" width=500>

# COMMAND ----------

import tensorflow as tf
### split data into training and testing sets
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's take a look at a sample image.

# COMMAND ----------

import matplotlib.pyplot as plt

class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

plt.imshow(X_train[0])
plt.colorbar()
plt.xlabel(class_names[y_train[0]])

# COMMAND ----------

# MAGIC %md
# MAGIC Let's also look at the images we will use to generate predictions and explanations.

# COMMAND ----------

for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_test[i])
    plt.xlabel(class_names[y_test[i]])

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's do some preprocessing on the images

# COMMAND ----------

from tensorflow.keras import layers

num_classes = 10

### Input image dimensions
### Each image has 28 x 28 pixels
img_rows, img_cols = 28, 28

X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

### Scale the images in both the training and testing sets to a range of 0 to 1. 
X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255
print("X_train shape: ", X_train.shape)
print(X_train.shape[0], "train images")
print(X_test.shape[0], "test images")

### Convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's construct a simple CNN model and compile the model.

# COMMAND ----------

model = tf.keras.models.Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# COMMAND ----------

# MAGIC %md
# MAGIC Start the training process.

# COMMAND ----------

EPOCHS = 3
BATCH_SIZE = 128

### Restrict the training set to only 5000 images to reduce training time
model.fit(X_train[:5000], y_train[:5000],
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(X_test[:1000], y_test[:1000]))

score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC Now, we are onto the fun part of retrieving the model's predictions!

# COMMAND ----------

import numpy as np
np.random.seed(1234)

N_test = 9
# sample_index = np.random.choice(X_test.shape[0], N_test, replace=False) #optional if you want to shuffle test example
sample_index = np.arange(N_test)
X_test_sample = X_test[sample_index]
y_test_sample = y_test[sample_index]
predictions = model.predict(X_test_sample)
prob_array = [max(predictions[i]) for i in range(N_test)]
class_array = [class_names[np.argmax(predictions[i])] for i in range(N_test)]
print(list(zip(class_array, prob_array)))

# COMMAND ----------

# MAGIC %md
# MAGIC Let's compare the predictions against the ground truth labels.

# COMMAND ----------

import pandas as pd

true_class_labels = [class_names[np.argmax(y_test_sample[i])] for i in range(N_test)]
result_data = {"True Label": true_class_labels, "Predicted Label": class_array, "Probability": prob_array}
df = pd.DataFrame(data=result_data)
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using SHAP to retrieve explanation behind predictions
# MAGIC 
# MAGIC We can see that the model predicted the first 9 test images correctly. But why? Can we use SHAP to help us understand why the error occurred?
# MAGIC 
# MAGIC We are going to use **`shap.GradientExplainer`** to explain pixel attributions to the predictions. From the <a href="https://shap-lrjball.readthedocs.io/en/latest/generated/shap.GradientExplainer.html#shap.GradientExplainer" target="_blank">SHAP documentation</a>:
# MAGIC > GradientExplainer Explains a model using expected gradients. Expected gradients an extension of the integrated gradients method (Sundararajan et al. 2017), a feature attribution method designed for differentiable models based on an extension of Shapley values to infinite player games (Aumann-Shapley values).
# MAGIC 
# MAGIC To read more about <a href="https://christophm.github.io/interpretable-ml-book/pixel-attribution.html" target="_blank">pixel attribution here</a>

# COMMAND ----------

# TODO
import shap

## Select a set of background examples to take an expectation over
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

e = shap.GradientExplainer(<FILL_IN>)
shap_values = e.shap_values(X_test[:9])

# COMMAND ----------

# MAGIC %md
# MAGIC The plot below explains the output prediction for the images. 
# MAGIC 
# MAGIC According to the documentation linked above:
# MAGIC > Red pixels increase the model's output while blue pixels decrease the output. The input images are shown on the left, and as nearly transparent grayscale backings behind each of the explanations. The sum of the SHAP values equals the difference between the expected model output (averaged over the background dataset) and the current model output.

# COMMAND ----------

# TODO
## Plot the pixel attributions
## get class name in the plot, index_names need to have dimension [N_sample, N_output]
index_names = np.array([class_names]*N_test)
### Note that the negative sign in front of X_test is to remove the image background
shap.image_plot(<FILL_IN>, -X_test_sample, index_names)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Note that for the first (row 1) and the last picture (row 9), mostly "blank" outer background is important for the model to generate a "ankle boot" and "sandal" prediction respectively. 
# MAGIC 
# MAGIC What else do you see? How can you explain the misclassification of the 7th picture (row 7) using this SHAP-generated image?

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
