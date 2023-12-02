# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # X-Ray Vision with Transfer Learning
# MAGIC 
# MAGIC In this notebook, we will use Transfer Learning on images of chest x-rays to predict if a patient has pneumonia (bacterial or viral) or is normal. 
# MAGIC 
# MAGIC This data set was obtained from <a href="http://dx.doi.org/10.17632/rscbjbr9sj.2" target="_blank">http://dx.doi.org/10.17632/rscbjbr9sj.2</a>. The <a href="http://dx.doi.org/10.17632/rscbjbr9sj.2" target="_blank">source</a> of the data is:
# MAGIC Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), “Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”, Mendeley Data, v2
# MAGIC 
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC  - Build a model to predict if a patient has pneumonia or not using transfer learning

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md ## Visualize train and test data
# MAGIC 
# MAGIC The datasets are placed into 4 different folders (2 folders for train and 2 folders for test). "normal" class is placed into "normal" folder. "pneumonia" class dataset is placed in "pneumonia" folder. There are two types of pneumonia in the dataset: virus and bacteria, but for our problem today, we will just predict normal/pneumonia.
# MAGIC 
# MAGIC Let's visually take a look at images from each of the classes.

# COMMAND ----------

from pyspark.sql.functions import lit

train_sample_normal = spark.read.format("binaryFile").load(f"{DA.paths.datasets}/chest-xray/train/normal/IM-0115-0001.jpeg").withColumn("label", lit(0))
train_sample_pneumonia = (spark.read.format("binaryFile")
                          .load([f"{DA.paths.datasets}/chest-xray/train/pneumonia/person1000_bacteria_2931.jpeg",
                                 f"{DA.paths.datasets}/chest-xray/train/pneumonia/person1000_virus_1681.jpeg"])
                          .withColumn("label", lit(1)))
combined_sample = train_sample_normal.union(train_sample_pneumonia)
display(combined_sample)

# COMMAND ----------

# MAGIC %md ## Build and Fine-tune the Model
# MAGIC 
# MAGIC We are going to use a pretrained VGG16 model. We will apply all layers of the neural network, but remove the last layer which outputted 1000 categories. Instead, we will replace it with a new dense layer which outputs the class probabilities.

# COMMAND ----------

from tensorflow.keras import applications
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
import tensorflow as tf
tf.random.set_seed(42)

vgg16_model = applications.VGG16(weights="imagenet")
model = Sequential()

for layer in vgg16_model.layers[:-1]: # Exclude last layer from copying
    layer.trainable = False  # only last layer is fine-tuned
    model.add(layer)

model.add(Dense(1, activation="sigmoid"))
model.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC Let's compile and train the model. In order to do that:
# MAGIC 0. Compile the model
# MAGIC 0. Create a generator to read data in **`batch_size=16`** from the sub-directories.
# MAGIC 0. Train the model with **`steps_per_epoch=5`** and **`epochs=5`**

# COMMAND ----------

# ANSWER
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Compile the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

batch_size = 16
img_width = 224
img_height = 224
# Loading training data
datagen = ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input)
train_generator = datagen.flow_from_directory(directory=f"{DA.paths.datasets}/chest-xray/train/".replace("dbfs:/", "/dbfs/"), 
                                              class_mode="binary", 
                                              classes=["normal", "pneumonia"],
                                              batch_size=batch_size,
                                              target_size=(img_height, img_width))

print(f"Class labels: {train_generator.class_indices}")

# Find class weight since the classes are imbalanced
class_weight = class_weight.compute_class_weight("balanced",
                                                 np.unique(train_generator.classes), 
                                                 train_generator.classes)
class_weight_dic = {0: class_weight[0], 
                    1: class_weight[1]}

# To save time, we only use 5 steps in each epoch i.e. 80 images per epoch.
steps_per_epoch = 5
epochs = 5
# Train the model 
model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, class_weight=class_weight_dic) 

# COMMAND ----------

# MAGIC %md ## Evaluate the model
# MAGIC 
# MAGIC In the code above, we saw our accuracy on our training dataset. Let's see how well it performs on our test dataset.

# COMMAND ----------

# ANSWER
test_datagen = ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input)

test_generator = datagen.flow_from_directory(directory=f"{DA.paths.datasets}/chest-xray/test/".replace("dbfs:/", "/dbfs/"), 
                                             class_mode="binary", 
                                             classes=["normal", "pneumonia"],
                                             batch_size=batch_size,
                                             target_size=(img_height, img_width))

eval_results = model.evaluate(test_generator)
accuracy = round(eval_results[1]*100, 2)
print(f"Predicted accuracy: {accuracy}%")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
