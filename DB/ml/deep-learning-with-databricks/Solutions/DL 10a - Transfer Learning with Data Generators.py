# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Transfer Learning with Data Generators
# MAGIC 
# MAGIC The idea behind transfer learning is to take knowledge from one model doing some task, and transfer it to build another model doing a similar task.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Motivate why transfer learning is a promising frontier for deep learning 
# MAGIC  - Compare transfer learning approaches
# MAGIC  - Perform transfer learning to create an cat vs dog classifier

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md ### Why Transfer Learning
# MAGIC 
# MAGIC In 2016, Andrew Ng claimed that transfer learning will be the next driver of commercial machine learning success after supervised learning.  Why?<br><br>
# MAGIC 
# MAGIC - A fundamental assumption of most machine learning approaches is that you train a model from scratch on a new dataset
# MAGIC - Transfer learning stores knowledge gained from solving one problem on a different, related problem
# MAGIC - More closely resembles human learning
# MAGIC 
# MAGIC What types of features could be transferred from one task to the next in the following cases?<br><br>
# MAGIC 
# MAGIC - Image recognition
# MAGIC - Natural language processing
# MAGIC - Speech recognition
# MAGIC - Time series

# COMMAND ----------

# MAGIC %md ### Common Pre-Trained Models
# MAGIC 
# MAGIC Keras exposes a number of deep learning models (architectures) along with pre-trained weights.  They are available in the **`tensorflow.keras.applications`** package and <a href="https://www.tensorflow.org/api_docs/python/tf/keras/applications" target="_blank">the full list is available here.</a>  
# MAGIC 
# MAGIC Transfer learning...<br><br>
# MAGIC 
# MAGIC - Saves a lot of time and resources over retraining models from scratch
# MAGIC - Are often pre-trained using the ImageNet dataset
# MAGIC - Repurposes earlier layers that encode higher level features (e.g. edges in images)
# MAGIC - Uses custom final layers specific to the new task
# MAGIC 
# MAGIC Below is a comparison of common reference architectures and pre-trained weights used in transfer learning:
# MAGIC 
# MAGIC | Network | Year | Top-5 ImageNet Accuracy | # of Params | 
# MAGIC |---------|------|-------------------|-------------|
# MAGIC | <a href="https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf" target="_blank">AlexNet</a> | 2012 | 84.7% | 62M |
# MAGIC | <a href="https://arxiv.org/abs/1409.1556" target="_blank">VGGNet</a> | 2014 | 92.3% | 138M |
# MAGIC | <a href="https://arxiv.org/pdf/1409.4842.pdf" target="_blank">Inception v1</a> | 2014 | 93.3% | 6.4M |
# MAGIC | <a href="https://arxiv.org/abs/1512.03385" target="_blank">ResNet-152</a> | 2015 | 95.5% | 60.3M | 
# MAGIC | <a href="https://arxiv.org/abs/1512.00567" target="_blank">Inception v3</a> | 2015 | 94.4% | 23.8M | 
# MAGIC | <a href="https://arxiv.org/abs/1610.02357" target="_blank">XCeption</a> | 2016 | 94.5% | 22.8M | 
# MAGIC | <a href="https://arxiv.org/pdf/1707.07012.pdf" target="_blank">NasNet</a> | 2017 | 95.3% | 22.6M | 
# MAGIC | <a href="https://arxiv.org/pdf/1704.04861.pdf" target="_blank">MobileNet</a> | 2017 | 89.5% | 4.24M |
# MAGIC | <a href="https://arxiv.org/pdf/1905.11946.pdf" target="_blank">EfficientNet B5</a> | 2019 | 96.7% | 30M | 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> See this <a href="https://paperswithcode.com/sota/image-classification-on-imagenet" target="_blank">website</a> that compiles metrics on a variety of deep learning architectures.

# COMMAND ----------

# MAGIC %md ### The Implementation Details
# MAGIC 
# MAGIC We want to make a classifier that distinguishes between cats and dogs. To do this, we'll use VGG16, but instead of predicting 1000 classes, we will predict 2 classes (cat or dog).  We have 3 options:<br><br>
# MAGIC 
# MAGIC 1. Use **only the architecture from VGG16**, initialize the weights at random. This is a computationally expensive approach and requires a lot of data as you are training hundreds of millions of weights from scratch.
# MAGIC 2. Use **both the architecture from VGG16 and weights** pre-trained on ImageNet.  Leave earlier layers frozen and retrain the later layers.  This is less computationally expensive and still requires a good amount of data.
# MAGIC 3. Use both the architecture from VGG16 and the weights, but **freeze the entire network, and add an additional layer.**  In this case, we would only train the final classification layer specific to our problem.  This is fast and works with small amounts of data.
# MAGIC 
# MAGIC Since our dataset is small and similar to the task VGG16 was trained on, we'll choose option 3.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_best_24.png"/> Would you want to use a high or low learning rate for transfer learning? Or different learning rates for different layers?

# COMMAND ----------

# MAGIC %md
# MAGIC Notice how few images we have in our training data. Will our neural network be able to distinguish between the animals?

# COMMAND ----------

from pyspark.sql.functions import lit

df_cats = spark.read.format("binaryFile").load(f"{DA.paths.datasets}/img/cats/*.jpg").withColumn("label", lit("cat"))
df_dogs = spark.read.format("binaryFile").load(f"{DA.paths.datasets}/img/dogs/*.jpg").withColumn("label", lit("dog"))

display(df_cats)

# COMMAND ----------

display(df_dogs)

# COMMAND ----------

# MAGIC %md Do a train/test split.

# COMMAND ----------

cat_data = df_cats.toPandas()
dog_data = df_dogs.toPandas()

train_data = cat_data.iloc[:32].append(dog_data.iloc[:32])
train_data["path"] = train_data["path"].apply(lambda x: x.replace("dbfs:/", "/dbfs/"))

test_data = cat_data.iloc[32:].append(dog_data.iloc[32:])
test_data["path"] = test_data["path"].apply(lambda x: x.replace("dbfs:/", "/dbfs/"))

print(f"Train data samples: {len(train_data)} \tTest data samples: {len(test_data)}")

# COMMAND ----------

# MAGIC %md
# MAGIC Here we use Keras functional API. The Keras functional API is a way to create models that is more flexible than the tf.keras.Sequential API. The functional API can handle models with non-linear topology, models with shared layers, and models with multiple inputs or outputs. The main idea that a deep learning model is usually a directed acyclic graph (DAG) of layers. So the functional API is a way to build graphs of layers.

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Input
import pandas as pd
tf.random.set_seed(42)

# Load original model with pretrained weights from imagenet
# We do not want to use the ImageNet classifier at the top since it has many irrelevant categories
base_model = applications.VGG16(weights="imagenet", include_top=False)

# Freeze base model
base_model.trainable = False

# Create new model on top
img_height = 224
img_width = 224
inputs = Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
outputs = Dense(1, activation="sigmoid")(x) # we want to output probabilities for both classes
model = Model(inputs, outputs)
model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC To train the model, we use <a href="https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator" target="_blank">ImageDataGenerator</a> class. Generators are useful when your data is very large, as you only need to load one batch of data into memory at a time. In general, ImageDataGenerator is used to configure random transformations and normalization operations to be done on your image data during training, as well as instantiate generators of augmented image batches (and their labels). 
# MAGIC 
# MAGIC These generators can be used with Keras model methods that accept data generators as inputs. <a href="https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_dataframe" target="_blank">flow_from_dataframe()</a> takes the dataframe and the path to a directory to generates batches. 

# COMMAND ----------

import mlflow.tensorflow
from tensorflow.keras.optimizers import Adam

# Check out the MLflow UI as this runs
mlflow.tensorflow.autolog(every_n_iter=2)

model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.005), metrics=["accuracy"]) 

# Loading training data
batch_size = 8
train_datagen = ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input)
train_generator = train_datagen.flow_from_dataframe(dataframe=train_data, 
                                                    directory=None, 
                                                    x_col="path", 
                                                    y_col="label", 
                                                    class_mode="binary", 
                                                    target_size=(img_height, img_width), 
                                                    batch_size=batch_size)

print(f"Class labels: {train_generator.class_indices}")

step_size = train_generator.n//train_generator.batch_size

# Train the model
# You might want to increase the # of epochs, but it will take longer to train
model.fit(train_generator, epochs=3, steps_per_epoch=step_size, verbose=2)

# COMMAND ----------

# MAGIC %md ## Evaluate the Accuracy

# COMMAND ----------

# Evaluate model on test set
test_datagen = ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input)

# Small dataset so we can evaluate it in one batch
batch_size = test_data.count()[0]

test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_data, 
    directory=None, 
    x_col="path", 
    y_col="label", 
    class_mode="binary", 
    target_size=(img_height, img_width),
    shuffle=False,
    batch_size=batch_size
)

step_size = test_generator.n//test_generator.batch_size

eval_results = model.evaluate(test_generator, steps=step_size)
print(f"Loss: {eval_results[0]}. Accuracy: {eval_results[1]}")

# COMMAND ----------

# MAGIC %md ## Visualize the Results
# MAGIC 
# MAGIC Since we used sigmoid + binary crossentropy, it computes the probability of class 0 (which is cats) being **`True`**, by analzying the single probability output. 

# COMMAND ----------

predictions = pd.DataFrame({
    "Prediction": ((model.predict(test_generator) >= 0.5)+0).ravel(),
    "True": test_generator.classes,
    "Path": test_data["path"].apply(lambda x: x.replace("/dbfs", "dbfs:"))
}).replace({v: k for k, v in train_generator.class_indices.items()})

all_images_df = df_cats.union(df_dogs).drop("label")
predictions_df = spark.createDataFrame(predictions)

display(all_images_df.join(predictions_df, predictions_df.Path==all_images_df.path).select("content", "Prediction", "True"))

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
