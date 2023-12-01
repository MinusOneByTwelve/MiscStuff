# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Convolutional Neural Networks (CNNs) for NLP
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson, you learn: 
# MAGIC - How to apply 1D convolutions to classify text sentiment

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Data
# MAGIC 
# MAGIC The following few cells follow the same text preprocessing steps as the previous notebook when we built bi-directional LSTMs. CNN for NLP pipeline is only different in the model building part! 

# COMMAND ----------

### We are reusing the same configurations used in bi-directional LSTMs 
vocab_size = 10000
max_length = 400

# COMMAND ----------

from pyspark.sql.functions import col, when
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

text_df = (spark.read.parquet(f"{DA.paths.datasets}/reviews/reviews_cleaned.parquet")
           .select("Text", "Score")
           .limit(5000) ### limit to only 5000 rows to reduce training time
           .withColumn("sentiment", when(col("Score") > 3, 1).otherwise(0))
          )

### Splitting data into train/test
train_df, test_df = text_df.randomSplit([0.8, 0.2])
train_pdf = train_df.toPandas()
X_train = train_pdf["Text"].values
y_train = train_pdf["sentiment"].values

### Tokenization
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)
### Convert the texts to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_seq_padded = pad_sequences(X_train_seq, maxlen=max_length, padding="post")

### Follow the same process for test_df
test_pdf = test_df.toPandas()
X_test = test_pdf["Text"].values
y_test = test_pdf["sentiment"].values
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_seq_padded = pad_sequences(X_test_seq, maxlen=max_length, padding="post")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Let's build a Convolutional Neural Network (CNN) for sentiment analysis!
# MAGIC 
# MAGIC Notice that we keep the hyperparameter values the same as the previous LSTM notebook. But we also have two new hyperparameters here: **`filters`** and **`kernel_size`** unique to CNNs.

# COMMAND ----------

batch_size = 32
embedding_dim = 300 
hidden_dim = 250
epochs = 1

### Only for CNN
filters = 250
kernel_size = 3

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's define our architecture. There is a new component we haven't learned, which is **dropout**. 
# MAGIC 
# MAGIC **Dropout** is a regularization method that reduces overfitting by randomly and temporarily removing nodes during training. 
# MAGIC 
# MAGIC It works like this: <br>
# MAGIC 
# MAGIC * Apply to most type of layers (e.g. fully connected, convolutional, recurrent) and larger networks
# MAGIC * Temporarily and randomly remove nodes and their connections during each training cycle
# MAGIC 
# MAGIC ![](https://files.training.databricks.com/images/nn_dropout.png)
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> See the original paper here: <a href="http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf" target="_blank">Dropout: A Simple Way to Prevent Neural Networks from Overfitting</a>

# COMMAND ----------

import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalMaxPool1D, Dense, Dropout, Embedding

model = Sequential([
  Embedding(vocab_size, embedding_dim, input_length=max_length),
  Conv1D(filters, kernel_size, strides=1, padding="valid", activation="relu", input_shape=(max_length, embedding_dim)),
  GlobalMaxPool1D(),
  Dense(hidden_dim, activation="relu"),
  Dropout(0.1),
  Dense(1, activation="sigmoid")
])

# COMMAND ----------

# MAGIC %md
# MAGIC What is Global Max Pooling? 
# MAGIC 
# MAGIC - We set the pool size to be equal to the input size, so the max of the entire input is the global max pooling output value. 
# MAGIC - It further reduces the dimensionality.
# MAGIC - <a href="https://github.com/christianversloot/machine-learning-articles/blob/main/what-are-max-pooling-average-pooling-global-max-pooling-and-global-average-pooling.md#global-max-pooling" target="_blank">Click here to read more.</a>
# MAGIC - Or <a href="https://github.com/keras-team/keras/blob/3d176e926f848c5aacd036d6095ab015a2f8cc83/keras/layers/pooling.py#L433" target="_blank">click here to look at the Keras source code</a>
# MAGIC - Example papers that use global max pooling:
# MAGIC   - <a href="https://arxiv.org/pdf/1604.00187.pdf" target="_blank">A Deep CNN for Word Spotting in Handwritten Documents, 2017</a>
# MAGIC   - <a href="https://hal.inria.fr/hal-01015140/file/Oquab15.pdf" target="_blank">Is object localization for free? 2015</a>
# MAGIC 
# MAGIC <img src="https://github.com/christianversloot/machine-learning-articles/raw/main/images/Global-Max-Pooling-1.png">

# COMMAND ----------

model.summary()

# COMMAND ----------

from tensorflow.keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.001), loss="binary_crossentropy", metrics=["accuracy", "AUC"])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train CNN and log using MLflow

# COMMAND ----------

import mlflow

mlflow.autolog()

with mlflow.start_run() as run:
  history = model.fit(X_train_seq_padded, 
                      y_train, 
                      batch_size=batch_size, 
                      epochs=epochs, 
                      validation_split=0.1, 
                      verbose=1)

history

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate the model

# COMMAND ----------

test_loss, test_accuracy, test_auc = model.evaluate(X_test_seq_padded, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply distributed inference
# MAGIC 
# MAGIC The code below is also the same as in the previous LSTM notebook.

# COMMAND ----------

import pandas as pd

logged_model = f"runs:/{run.info.run_id}/model"

### Load model as a Spark UDF
predict = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

df = spark.createDataFrame(pd.concat([pd.DataFrame(data=y_test, columns=["label"]), 
                                      pd.DataFrame(X_test_seq_padded), 
                                      pd.DataFrame(data=X_test, columns=["text"])], axis=1))

pred_df = (df
           .withColumn("predictions", predict(*df.drop("text", "label").columns))
           .select("text", "label", "predictions")
           .withColumn("predicted_label", when(col("predictions") > 0.5, 1).otherwise(0)))

display(pred_df)

# COMMAND ----------

# MAGIC %md
# MAGIC What did you notice about the differences between using CNN and LSTM for sentiment analysis? 

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
