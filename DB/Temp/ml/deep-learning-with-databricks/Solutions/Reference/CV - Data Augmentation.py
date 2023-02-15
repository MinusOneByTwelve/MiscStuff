# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Augmentation
# MAGIC 
# MAGIC Data augmentation generates new training examples from existing training examples via several random transformations. By augmenting data, the model should never see the exact same picture more than once during the model training process, and helps add additional labeled data points if your data is small. Hence, this step can help prevent model overfitting and helps model to generalize to new data. 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC - Configure random transformation to augment data via **`ImageDataGenerator`**
# MAGIC - Perform data augmentation

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup

# COMMAND ----------

from pyspark.sql.functions import lit

df_cats = spark.read.format("binaryFile").load(f"{DA.paths.datasets}/img/cats/*.jpg").withColumn("label", lit("cat"))
df_dogs = spark.read.format("binaryFile").load(f"{DA.paths.datasets}/img/dogs/*.jpg").withColumn("label", lit("dog"))

cat_data = df_cats.toPandas()
dog_data = df_dogs.toPandas()

display(df_cats)

# COMMAND ----------

# MAGIC %md ### Data Augmentation
# MAGIC 
# MAGIC There are many techniques to augment your data. Please reference this <a href="https://towardsdatascience.com/exploring-image-data-augmentation-with-keras-and-tensorflow-a8162d89b844" target="_blank">blog post</a> for visualizations on individual transformations. Below, we will highlight the commonly used options. <br>
# MAGIC <br>
# MAGIC - Rotation (**`rotation_range`**)
# MAGIC   - Rotates the image in certain angles
# MAGIC - Width (**`width_shift_range`**)
# MAGIC   - Shifts the image to the left or right
# MAGIC - Height (**`height_shift_range`**)
# MAGIC   - Shift the image to the top or bottom
# MAGIC - Shearing (**`shear_range`**)
# MAGIC    - Slants the shape of the image. This creates a sort of ‘stretch’ in the image, which is not seen in rotation. 
# MAGIC - Zoom (**`zoom_range`)
# MAGIC    - Magnifies or zooms out the image
# MAGIC - Flipping (**`horizontal_flip`**, **`vertical_flip`**)
# MAGIC    - Horizontally or vertically flips the image
# MAGIC - Rescale
# MAGIC    - A value by which we will multiply the data before any other processing. Our original images consist in RGB coefficients in the 0-255, but such values would be too high for our models to process (given a typical learning rate), so we target values between 0 and 1 instead by scaling with a 1/255 factor.
# MAGIC 
# MAGIC Reference:
# MAGIC - Options for **`ImageDataGenerator`** class in the <a href="https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator" target="_blank">docs</a>
# MAGIC - Other data augmentation libraries, such as <a href="https://github.com/albumentations-team/albumentations#list-of-augmentations" target="_blank">Albumentations</a>

# COMMAND ----------

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1/255,
                             rotation_range=40,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             shear_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             vertical_flip=False, # No upside down cats or dogs!
                             fill_mode="nearest")

# COMMAND ----------

# MAGIC %md
# MAGIC Let's examine what these augmentations will do to one of our cat images. 

# COMMAND ----------

from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

img_path_1 = cat_data["path"][0].replace("dbfs:/", "/dbfs/")
img = image.load_img(img_path_1, target_size = (224, 224))

x = image.img_to_array(img)   # this is a Numpy array with shape (3, 224, 224)
x = x.reshape((1,) + x.shape) # this is a Numpy array with shape (1, 3, 224, 224)

plt.figure(figsize=(10, 10))
for i, batch in enumerate(datagen.flow(x, batch_size=1, seed=42)):
    ax = plt.subplot(3, 3, i + 1)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    if i == 8:
        break # loops indefinitely, need to break the loop at some point

# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's augment the entire train data and retrain the model. 
# MAGIC 
# MAGIC First, create stratified train and test data. 

# COMMAND ----------

train_data = cat_data.iloc[:32].append(dog_data.iloc[:32])
train_data["path"] = train_data["path"].apply(lambda x: x.replace("dbfs:/", "/dbfs/"))

test_data = cat_data.iloc[32:].append(dog_data.iloc[32:])
test_data["path"] = test_data["path"].apply(lambda x: x.replace("dbfs:/", "/dbfs/"))

# COMMAND ----------

# MAGIC %md
# MAGIC This time let's use ResNet-50 instead of VGG16.

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras import applications
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Input
import mlflow.tensorflow
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(42)

img_height = 224
img_width = 224

# Load original model with pretrained weights from imagenet
base_model = applications.ResNet50(weights="imagenet")
base_model.trainable = False

# Create new model on top
inputs = Input(shape=(img_height, img_width, 3))
x = base_model(inputs, training=False)
outputs = Dense(1, activation="sigmoid")(x) # we want to output probabilities for both classes
model = Model(inputs, outputs)
model.summary()

# Check out the MLflow UI as this runs
mlflow.tensorflow.autolog(every_n_iter=2)

model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])

# COMMAND ----------

# MAGIC %md
# MAGIC We are going to train the model in the cell below.

# COMMAND ----------

batch_size = 8

# Note that we are NOT passing in the rescaling argument here, 
# since we are using resnet's pretrained weights, we need to use resnet's preprocess_input function
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False, # No upside down cats or dogs!
    fill_mode="nearest",
    preprocessing_function=applications.resnet.preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=None,
    x_col="path",
    y_col="label",
    target_size=(img_height, img_width),
    color_mode="rgb",
    class_mode="binary",
    batch_size=batch_size,
    seed=42 # random seed for shuffling and transformations.
)

print(f"Class labels: {train_generator.class_indices}")
step_size_train = train_generator.n//train_generator.batch_size

model.fit(train_generator, steps_per_epoch=step_size_train, epochs=10, verbose=1)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's check the model accuracy on the test set. Note that we are not performing data augmentation on the test dataset. 

# COMMAND ----------

test_datagen = ImageDataGenerator(preprocessing_function=applications.resnet.preprocess_input)

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
    batch_size=batch_size,
    seed=42
)

step_size_test = test_generator.n//test_generator.batch_size

eval_results = model.evaluate(test_generator, steps=step_size_test)
print(f"Loss: {eval_results[0]}. Accuracy: {eval_results[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Visualize predictions

# COMMAND ----------

import pandas as pd

predictions = pd.DataFrame({
    "Prediction": (model.predict(test_generator, steps=step_size_test) > .5).astype(int).flatten(),
    "True": test_generator.classes,
    "Path": test_data["path"].apply(lambda x: x.replace("/dbfs", "dbfs:"))
}).replace({v: k for k, v in train_generator.class_indices.items()})

all_images_df = df_cats.union(df_dogs).drop("label")
predictions_df = spark.createDataFrame(predictions)

display(all_images_df.join(predictions_df, predictions_df.Path==all_images_df.path))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Additional note
# MAGIC 
# MAGIC If you were to plan on tuning the entire model, you need to change **`base_model.trainable`** from **`False`** to **`True`** to fine tune the entire model. Additionally, you should use a lower learning rate. 
# MAGIC However, note that fine-tuning the entire model will take a significantly long time to train compared to before due to the large number of trainable parameters.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
