# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Distributed Inference with Convolutional Neural Networks
# MAGIC 
# MAGIC We will use pre-trained Convolutional Neural Networks (CNNs), trained with the image dataset from <a href="http://www.image-net.org/" target="_blank">ImageNet</a>, to make scalable predictions with Pandas Scalar Iterator UDFs.
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Analyze popular CNN architectures
# MAGIC  - Apply pre-trained CNNs to images using Pandas Scalar Iterator UDF

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## VGG16
# MAGIC ![vgg16](https://miro.medium.com/max/940/1*3-TqqkRQ4rWLOMX-gvkYwA.png)
# MAGIC 
# MAGIC We are going to start with the VGG16 model, which was introduced by Simonyan and Zisserman in their 2014 paper <a href="https://arxiv.org/abs/1409.1556" target="_blank">Very Deep Convolutional Networks for Large Scale Image Recognition</a>.
# MAGIC 
# MAGIC Let's start by downloading VGG's weights and model architecture.

# COMMAND ----------

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
import numpy as np

vgg16_model = VGG16(weights="imagenet")

# COMMAND ----------

# MAGIC %md
# MAGIC We can look at the model summary. Look at how many parameters there are! Imagine if you had to train all 138,357,544 parameters from scratch! This is one motivation for re-using existing model weights.
# MAGIC 
# MAGIC **RECAP**: What is a convolution? Max pooling?

# COMMAND ----------

# MAGIC %md
# MAGIC **Question**: What do the input and output shapes represent?

# COMMAND ----------

vgg16_model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inception-V3 + Batch Normalization
# MAGIC 
# MAGIC In 2016, developers from <a href="https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf" target="_blank">Google published a paper</a> updating their Inception architecture with a number of optimizations.  This included a technique known as batch normalization.
# MAGIC 
# MAGIC 
# MAGIC **Batch normalization is a technique that applies to very deep neural networks (especially CNNs) that standardizes the inputs to a layer for each mini-batch.** Generally speaking, this reduces the number of training epochs needed by stabilizing the learning process.
# MAGIC 
# MAGIC There are two main hypotheses for why this works:
# MAGIC 
# MAGIC - Each layer in a deep neural network (with 10+ layers, for instance) expects the inputs from the previous layer to come from the same distribution.  However, in practice each layer is being updated, changing the distribution of its output to the next layer.  This is called "internal covariate shift" and can result in an unstable learning process since each layer is effectively learning a moving target. 
# MAGIC - This technique smoothes objective function and thereby improves the learning process.
# MAGIC 
# MAGIC Batch normalization should generally not be used with dropout, another regularization technique (discussed in the GANs notebook).  While there's some contention over which is a more effective method <a href="https://link.springer.com/article/10.1007/s11042-019-08453-9" target="_blank">see this paper for details</a>, batch normalization is generally preferred over dropout for deep neural networks.
# MAGIC 
# MAGIC Let's load the Inception V3 model to compare architectures with VGG16.

# COMMAND ----------

from tensorflow.keras.applications.inception_v3 import InceptionV3

inception_model = InceptionV3()

# COMMAND ----------

# MAGIC %md Take a look at the architecture noted where batch normalization is performed.

# COMMAND ----------

inception_model.summary()

# COMMAND ----------

# MAGIC %md Looking for more reference architectures?  Check out <a href="https://www.tensorflow.org/api_docs/python/tf/keras/applications" target="_blank">**`tf.keras.applications`** for what's available out of the box.</a>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply pre-trained model
# MAGIC 
# MAGIC We are going to make a helper method to resize our images to be 224 x 224, and output the top 3 classes for a given image. This is the expected input shape for VGG16.
# MAGIC 
# MAGIC In TensorFlow, it represents the images in a channels-last manner: (samples, height, width, color_depth)

# COMMAND ----------

def predict_images(images, model):
    for i in images:
        print(f"Processing image: {i}")
        img = image.load_img(i, target_size=(224, 224))
        # Convert to numpy array for Keras image format processing
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        preds = model.predict(x)
        # Decode the results into a list of tuples (class, description, probability)
        print(f"Predicted: {decode_predictions(preds, top=3)[0]}\n")

# COMMAND ----------

# MAGIC %md-sandbox ## Images
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://files.training.databricks.com/images/pug.jpg" height="150" width="150" alt="Databricks Nerds!" style=>
# MAGIC   <img src="https://files.training.databricks.com/images/strawberries.jpg" height="150" width="150" alt="Databricks Nerds!" style=>
# MAGIC   <img src="https://files.training.databricks.com/images/rose.jpg" height="150" width="150" alt="Databricks Nerds!" style=>
# MAGIC   
# MAGIC </div>
# MAGIC 
# MAGIC Let's make sure the datasets are already mounted.

# COMMAND ----------

img_paths = [
    f"{DA.paths.datasets}/img/pug.jpg".replace("dbfs:/", "/dbfs/"),
    f"{DA.paths.datasets}/img/strawberries.jpg".replace("dbfs:/", "/dbfs/"),
    f"{DA.paths.datasets}/img/rose.jpg".replace("dbfs:/", "/dbfs/")
]

predict_images(img_paths, vgg16_model)

# COMMAND ----------

# MAGIC %md
# MAGIC The network did so well with the pug and strawberry! What happened with the rose? Well, it turns out that **`rose`** was not one of the 1000 categories that VGG16 had to predict. But it is quite interesting it predicted **`sea_anemone`** and **`vase`**.

# COMMAND ----------

# MAGIC %md
# MAGIC You can play around with this with your own images by doing:
# MAGIC 
# MAGIC **`%sh wget <image_url>/<image_name.jpg> -P /dbfs/tmp/`**

# COMMAND ----------

# MAGIC %md-sandbox ## Classify Co-Founders of Databricks
# MAGIC <div style="text-align: left; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://files.training.databricks.com/images/Ali-Ghodsi-4.jpg" height="150" width="150" alt="Databricks Nerds!" style=>
# MAGIC   <img src="https://files.training.databricks.com/images/andy-konwinski-1.jpg" height="150" width="150" alt="Databricks Nerds!" style=>
# MAGIC   <img src="https://files.training.databricks.com/images/ionS.jpg" height="150" width="150" alt="Databricks Nerds!" style=>
# MAGIC   <img src="https://files.training.databricks.com/images/MateiZ.jpg" height="200" width="150" alt="Databricks Nerds!" style=>
# MAGIC   <img src="https://files.training.databricks.com/images/patrickW.jpg" height="150" width="150" alt="Databricks Nerds!" style=>
# MAGIC   <img src="https://files.training.databricks.com/images/Reynold-Xin.jpg" height="150" width="150" alt="Databricks Nerds!" style=>
# MAGIC </div>

# COMMAND ----------

# MAGIC %md Load these images into a DataFrame.

# COMMAND ----------

df = spark.read.format("binaryFile").load(f"{DA.paths.datasets}/img/founders/")
display(df)

# COMMAND ----------

# MAGIC %md Let's wrap the prediction code inside a UDF so we can apply this model in parallel on each row of the DataFrame.

# COMMAND ----------

from pyspark.sql.types import StringType, ArrayType

@udf(ArrayType(StringType()))
def vgg16_predict_udf(path):
    img = image.load_img(path.replace("dbfs:/", "/dbfs/"), target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    model = VGG16(weights="imagenet")
    preds = model.predict(x)

    # Decode the results into a list of strings (class, description, probability)  
    return [f"{label}: {prob:.3f}" for _, label, prob in decode_predictions(preds, top=3)[0]]

results_df = df.withColumn("predictions", vgg16_predict_udf("path"))
display(results_df)

# COMMAND ----------

# MAGIC %md ### Pandas/Vectorized UDF
# MAGIC 
# MAGIC Pandas/Vectorized UDFs are available in Python to help speed up the computation by leveraging Apache Arrow. <a href="https://arrow.apache.org/" target="_blank">Apache Arrow</a> is an in-memory columnar data format that is used in Spark to efficiently transfer data between JVM and Python processes with near-zero (de)serialization cost. See more <a href="https://spark.apache.org/docs/latest/api/python/user_guide/sql/arrow_pandas.html" target="_blank">here</a>.
# MAGIC 
# MAGIC * <a href="https://databricks.com/blog/2017/10/30/introducing-vectorized-udfs-for-pyspark.html" target="_blank">Blog post</a>
# MAGIC * <a href="https://spark.apache.org/docs/latest/sql-programming-guide.html#pyspark-usage-guide-for-pandas-with-apache-arrow" target="_blank">Documentation</a>
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2017/10/image1-4.png" alt="Benchmark" width ="500" height="1500">

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Pandas Scalar Iterator UDF
# MAGIC 
# MAGIC If you define your own UDF to apply a model to each record of your DataFrame in Python, opt for pandas/vectorized UDFs for optimized serialization and deserialization. However, if your model is very large, then there is high overhead for the pandas UDF to repeatedly load the same model for every batch in the same Python worker process. In Spark 3.0, pandas UDFs can accept an iterator of pandas.Series or pandas.DataFrame so that you can load the model only once instead of loading it for every series in the iterator.
# MAGIC 
# MAGIC This way the cost of any set-up needed (like loading the VGG16 model in our case) will be incurred fewer times. When the number of images you’re working with is greater than **`spark.conf.get("spark.sql.execution.arrow.maxRecordsPerBatch")`**, which is 10,000 by default, you'll see significant speed ups over a pandas scalar UDF because it iterates through batches of pd.Series.
# MAGIC 
# MAGIC It has the general syntax of: 
# MAGIC **`@pandas_udf(...)
# MAGIC def predict(iterator):
# MAGIC     model = ... # load model
# MAGIC     for features in iterator:
# MAGIC         yield model.predict(features)`**
# MAGIC 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> If the workers cached the model weights after loading it for the first time, subsequent calls of the same UDF with the same model loading will become significantly faster. 

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
from typing import Iterator

def preprocess(image_path):
    path = image_path.replace("dbfs:/", "/dbfs/")
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = preprocess_input(x)
    return x

@pandas_udf(ArrayType(StringType()))
def vgg16_predict_pandas_udf(image_data_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    # Load model outside of for loop
    model = VGG16(weights="imagenet") 
    for image_data_series in image_data_iter:
        # Apply functions to entire series at once
        x = image_data_series.map(preprocess) 
        x = np.stack(list(x.values))
        preds = model.predict(x)
        top_3s = decode_predictions(preds, top=3)

        yield pd.Series([[f"{label}: {prob:.3f}" for _, label, prob in top_3] for top_3 in top_3s])

display(df.withColumn("predictions", vgg16_predict_pandas_udf("path")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pandas Function API
# MAGIC 
# MAGIC Instead of using a Pandas UDF, we can use a Pandas Function API. This new category in Apache Spark 3.0 enables you to directly apply a Python native function, which takes and outputs Pandas instances against a PySpark DataFrame. Pandas Functions APIs supported in Apache Spark 3.0 are: grouped map, map, and co-grouped map.
# MAGIC 
# MAGIC **`mapInPandas()`** takes an iterator of pandas.DataFrame as input, and outputs another iterator of pandas.DataFrame. It's flexible and easy to use if your model requires all of your columns as input, but it requires serialization/deserialization of the whole DataFrame (as it is passed to its input). You can control the size of each pandas.DataFrame with the **`spark.sql.execution.arrow.maxRecordsPerBatch`** config.
# MAGIC 
# MAGIC Because mapInPandas requires deserializing all of your columns, we will only be selecting the **`path`** column prior to applying the model.

# COMMAND ----------

def map_pandas_predict(image_data_iter: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
    model = VGG16(weights="imagenet") 
    for image_data_series in image_data_iter:
        image_path_series = image_data_series["path"]
        x = image_path_series.map(preprocess) 
        x = np.stack(list(x.values))
        preds = model.predict(x)
        top_3s = decode_predictions(preds, top=3)

        results = [[f"{label}: {prob:.3f}" for _, label, prob in top_3] for top_3 in top_3s]
        yield pd.concat([image_path_series, pd.Series(results, name="prediction")], axis=1)

display(df.select("path").mapInPandas(map_pandas_predict, schema="path:STRING, prediction:ARRAY<STRING>"))

# COMMAND ----------

# MAGIC %md
# MAGIC In the next lab, we will cover how to utilize existing components of the VGG16 architecture, and how to retrain the final classifier.

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
