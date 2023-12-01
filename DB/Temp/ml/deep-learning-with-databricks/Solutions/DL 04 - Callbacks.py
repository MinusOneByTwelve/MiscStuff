# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Callbacks
# MAGIC 
# MAGIC Congrats on building your first neural network! In this notebook, we will cover even more topics to improve your model building. After you learn the concepts here, you will apply them to the neural network you just created.
# MAGIC 
# MAGIC We will use the California Housing Dataset.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Add validation data
# MAGIC  - Generate model checkpointing/callbacks
# MAGIC  - Use TensorBoard

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

train_df = spark.read.format("delta").load(f"{DA.paths.datasets}/california-housing/train")
X_train = train_df.toPandas()
y_train = X_train.pop("label")

test_df = spark.read.format("delta").load(f"{DA.paths.datasets}/california-housing/test")
X_test = test_df.toPandas()
y_test = X_test.pop("label")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## 1. Define Model

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization  
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(42)

normalize_layer = Normalization()
normalize_layer.adapt(X_train)

model = Sequential([
    normalize_layer,
    Dense(20, input_dim=8, activation="relu"),
    Dense(20, activation="relu"),
    Dense(1, activation="linear")
])

model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Validation Data
# MAGIC 
# MAGIC Let's take a look at the <a href="https://www.tensorflow.org/api_docs/python/tf/keras/Sequential#fit" target="_blank">.fit()</a> method in the docs to see all of the options we have available! 
# MAGIC 
# MAGIC We can either explicitly specify a validation dataset, or we can specify a fraction of our training data to be used as our validation dataset.
# MAGIC 
# MAGIC Typically, a validation set will be a curated part of the training dataset that covers a decent spread of the statistics that the larger training dataset will contain. A validation dataset can be pulled out of the training dataset, or a curated validation dataset can be created and kept seperately. We will show firstly how to create a validation dataset from the training data, and then refer to a pre-curated validaiton dataset for the rest of the course (where available).
# MAGIC 
# MAGIC The reason why we need a validation dataset is to evaluate how well we are performing on unseen data (neural networks will overfit if you train them for too long!).
# MAGIC 
# MAGIC We can specify **`validation_split`** to be any value between 0.0 and 1.0 (defaults to 0.0).

# COMMAND ----------

history = model.fit(X_train, y_train, validation_split=.2, epochs=10, batch_size=64, verbose=2)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Checkpointing
# MAGIC 
# MAGIC After each epoch, we want to save the model. However, we will pass in the flag **`save_best_only=True`**, which will only save the model if the validation loss decreased. This way, if our machine crashes or we start to overfit, we can always go back to the "good" state of the model.
# MAGIC 
# MAGIC To accomplish this, we will use the ModelCheckpoint <a href="https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint" target="_blank">callback</a>. History is an example of a callback that is automatically applied to every Keras model.

# COMMAND ----------

from tensorflow.keras.callbacks import ModelCheckpoint

filepath = f"{DA.paths.working_path}/keras_checkpoint_weights.ckpt"

model_checkpoint = ModelCheckpoint(filepath=filepath, verbose=1, save_best_only=True)

# COMMAND ----------

# MAGIC %md ## 4. Tensorboard
# MAGIC 
# MAGIC Tensorboard provides a nice UI to visualize the training process of your neural network and can help with debugging! We can define it as a callback.
# MAGIC 
# MAGIC Here are links to Tensorboard resources:
# MAGIC * <a href="https://www.tensorflow.org/tensorboard/get_started" target="_blank">Getting Started with Tensorboard</a>
# MAGIC * <a href="https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras" target="_blank">Profiling with Tensorboard</a>
# MAGIC * <a href="https://www.datacamp.com/community/tutorials/tensorboard-tutorial" target="_blank">Effects of Weight Initialization</a>
# MAGIC 
# MAGIC 
# MAGIC Here is a <a href="https://databricks.com/blog/2020/08/25/tensorboard-a-new-way-to-use-tensorboard-on-databricks.html" target="_blank">Databricks blog post</a> that contains an end-to-end example of using Tensorboard on Databricks. 

# COMMAND ----------

# MAGIC %load_ext tensorboard

# COMMAND ----------

log_dir = f"/tmp/{DA.username}"

# COMMAND ----------

# MAGIC %md
# MAGIC We just cleared out the log directory above in case you re-run this notebook multiple times.

# COMMAND ----------

# MAGIC %tensorboard --logdir $log_dir

# COMMAND ----------

# MAGIC %md Now let's add in our model checkpoint and Tensorboard callbacks to our **`.fit()`** command.
# MAGIC 
# MAGIC Click the refresh button on Tensorboard to view the Tensorboard output when the training has completed.

# COMMAND ----------

from tensorflow.keras.callbacks import TensorBoard

### Here, we set histogram_freq=1 so that we can visualize the distribution of a Tensor over time. 
### It can be helpful to visualize weights and biases and verify that they are changing in an expected way. 
### Refer to the Tensorboard documentation linked above.

tensorboard = TensorBoard(log_dir, histogram_freq=1)
history = model.fit(X_train, y_train, validation_split=.2, epochs=10, batch_size=64, verbose=2, callbacks=[model_checkpoint, tensorboard])

# COMMAND ----------

model.evaluate(X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Go back and click to refresh the Tensorboard! Note that under the **`histograms`** tab, you will see histograms of **`kernel`** in each layer; **`kernel`** represents the weights of the neural network.
# MAGIC 
# MAGIC If you are curious about how different initial weight initialization methods affect the training of neural networks, you can change the default weight initialization within the first Dense layer of your neural network. This <a href="https://www.tensorflow.org/api_docs/python/tf/keras/initializers" target="_blank">documentation</a> lists all the types of weight initialization methods that are supported by Tensorflow.
# MAGIC 
# MAGIC <pre><code><strong>model = Sequential([
# MAGIC     Dense(20, input_dim=8, activation="relu", kernel_initializer="<insert_different_weight_initialization_methods_here>"),
# MAGIC     Dense(20, activation="relu"),
# MAGIC     Dense(1, activation="linear")
# MAGIC ])
# MAGIC </strong></code></pre>
# MAGIC 
# MAGIC If you would like to share your Tensorboard result with your peer, you can check out <a href="https://tensorboard.dev/" target="_blank">TensorBoard.dev</a> (currently in preview) that allows you to share the dashboard. All you need to do is to upload your Tensorboard logs. 

# COMMAND ----------

# MAGIC %md
# MAGIC Now, head over to the lab to apply the techniques you have learned to the Wine Quality dataset! 

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
