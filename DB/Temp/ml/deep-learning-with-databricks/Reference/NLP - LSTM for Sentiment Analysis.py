# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Sentiment Analysis with LSTM and MLflow
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC - Build a bi-directional Long Short Term Memory (LSTM) model using <a href="https://www.tensorflow.org/api_docs/python/tf/keras" target="_blank">tensorflow.keras</a> to classify the sentiment of text reviews
# MAGIC - Log model inputs and outputs using <a href="https://www.mlflow.org/docs/latest/index.html" target="_blank">MLflow</a>

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

from pyspark.sql.functions import col, when
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import mlflow
import mlflow.tensorflow

# COMMAND ----------

text_df = (spark.read.parquet(f"{DA.paths.datasets}/reviews/reviews_cleaned.parquet")
           .select("Text", "Score")
           .limit(5000) ### limit to only 5000 rows to reduce training time
          )

# COMMAND ----------

### Ensure that there are no missing values
text_df.filter(col("Score").isNull()).count()

# COMMAND ----------

text_df = text_df.withColumn("sentiment", when(col("Score") > 3, 1).otherwise(0))
display(text_df)

# COMMAND ----------

positive_review_percent = text_df.filter(col("sentiment") == 1).count() / text_df.count() * 100
print(f"{positive_review_percent}% of reviews are positive")

# COMMAND ----------

train_df, test_df = text_df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC It's a practice to check that the training data and the testing data follows the same label distribution as the original data before splitting. Indeed, we see about 76% of positive reviews across the train, test, and the original data. 

# COMMAND ----------

train_positive_review_percent = train_df.filter(col("sentiment") == 1).count() / train_df.count() * 100
test_positive_review_percent = test_df.filter(col("sentiment") == 1).count() / test_df.count() * 100
print(f"{train_positive_review_percent:.1f}% of reviews in the train_df are positive")
print(f"{test_positive_review_percent:.1f}% of reviews in the test_df are positive")

# COMMAND ----------

train_pdf = train_df.toPandas()
X_train = train_pdf["Text"].values
y_train = train_pdf["sentiment"].values

# COMMAND ----------

# MAGIC %md
# MAGIC ### Tokenization
# MAGIC 
# MAGIC The first step in NLP is to break up the text into smaller units, words. Instead of treating the entire text as 1 string, we want to break it up into a list of strings, where each string is a single word, for further processing. This process is called **tokenizing**. This is a crucial first step because we are able to match/compare and featurize specific words better than we can a single string of words.

# COMMAND ----------

vocab_size = 10000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)
### convert the texts to sequences
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_seq

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's compute some basic statistics to understand our training data more!

# COMMAND ----------

l = [len(i) for i in X_train_seq]
l = np.array(l)
print(f"minimum number of words: {l.min()}")
print(f"median number of words: {np.median(l)}")
print(f"average number of words: {l.mean()}")
print(f"maximum number of words: {l.max()}")

# COMMAND ----------

print(X_train[0])
print("\n")
### The text gets converted to a list of integers
print(X_train_seq[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Padding
# MAGIC 
# MAGIC Padding is a common NLP preprocessing step to ensure that each sentence has the same number of tokens, regardless of the number of words present in each sentence.

# COMMAND ----------

max_length = 400
X_train_seq_padded = pad_sequences(X_train_seq, maxlen=max_length)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Repeat the process of tokenization and padding for **`test_df`**
# MAGIC 
# MAGIC The same processing should also be applied to the **`test_df`** as well.

# COMMAND ----------

test_pdf = test_df.toPandas()
X_test = test_pdf["Text"].values
y_test = test_pdf["sentiment"].values
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_seq_padded = pad_sequences(X_test_seq, maxlen=max_length)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define bi-directional LSTM Architecture
# MAGIC 
# MAGIC A bi-directional LSTM architecture is largely the same as the base LSTM architecture. But it has additional capacity to understand text as it can scan the text from right to left, in addition from left to right. The bi-directional architecture mimics how humans read text. We often read text to its left and right to figure out the context or to guess the meaning of an unknown word. 
# MAGIC 
# MAGIC There are a couple hyperparameters within the LSTM architecture itself that can be tuned:
# MAGIC 
# MAGIC - **`embedding_dim`** : The embedding layer encodes the input sequence into a sequence of dense vectors of dimension **`embedding_dim`**.
# MAGIC - **`lstm_out`** : The LSTM transforms the vector sequence into a single vector of size **`lstm_out`**, containing information about the entire sequence.
# MAGIC 
# MAGIC <img src="https://www.researchgate.net/profile/Latifa-Nabila-Harfiya/publication/344751031/figure/fig2/AS:948365760155651@1603119425682/The-unfolded-architecture-of-Bidirectional-LSTM-BiLSTM-with-three-consecutive-steps.png" width=500>

# COMMAND ----------

embedding_dim = 128
lstm_out = 64

### Input for variable-length sequences of integers
inputs = keras.Input(shape=(None,), dtype="int32")

### Embed each integer (i.e. each word) in a 128-dimensional word vectors
x = layers.Embedding(vocab_size, embedding_dim)(inputs)

### Add 2 bidirectional LSTMs
x = layers.Bidirectional(layers.LSTM(lstm_out, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(lstm_out))(x)

### Add a classifier
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train LSTM and log using MLflow
# MAGIC 
# MAGIC Notice that the compilation of the neural network here is the same as we have previously done in other notebooks. We can control the learning rate, batch size, number of epoch, and also specify the proportion of validation data! 

# COMMAND ----------

mlflow.tensorflow.autolog()

with mlflow.start_run() as run:
  
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), 
                  loss="binary_crossentropy", 
                  metrics=["accuracy", "AUC"])

    model.fit(X_train_seq_padded, 
              y_train, 
              batch_size=32, 
              epochs=1, 
              validation_split=0.1)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply inference at scale using **`mlflow.pyfunc.spark_udf`**
# MAGIC 
# MAGIC You can read more about Spark UDFs with MLflow <a href="https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#mlflow.pyfunc.spark_udf" target="_blank">here</a>.

# COMMAND ----------

logged_model = f"runs:/{run.info.run_id}/model"

### Load model as a Spark UDF
predict = mlflow.pyfunc.spark_udf(spark, model_uri=logged_model)

# COMMAND ----------

df = spark.createDataFrame(pd.concat([pd.DataFrame(data=y_test, columns=["label"]), 
                                      pd.DataFrame(X_test_seq_padded), 
                                      pd.DataFrame(data=X_test, columns=["text"])], axis=1))

# COMMAND ----------

pred_df = (df
           .withColumn("predictions", predict(*df.drop("text", "label").columns))
           .select("text", "label", "predictions")
           .withColumn("predicted_label", when(col("predictions") > 0.5, 1).otherwise(0)))

display(pred_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Yay! We just built a bi-directional LSTM to classify sentiments! 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate on test_data

# COMMAND ----------

test_loss, test_accuracy, test_auc = model.evaluate(X_test_seq_padded, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Even though our model only has two bi-directional LSTM layers and we trained for only one epoch, the model still performed pretty well! To improve performance, we could try increasing the number of epochs as a next step.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
