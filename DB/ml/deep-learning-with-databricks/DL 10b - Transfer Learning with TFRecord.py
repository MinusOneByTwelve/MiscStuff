# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Transfer Learning with TFRecord
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC - Learn TFRecord & how to use TFRecord for model training

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## What is TFRecord?
# MAGIC <br>
# MAGIC 
# MAGIC - The default data format for TF and is optimized using Google’s Protocol Buffers (aka `protobuf`). 
# MAGIC - A simple binary format that contains a sequence of binary records, enabling very efficient reading and writing.
# MAGIC - By default, `TFRecordDataset` reads files sequentially, so you will see later we need to use `num_parallel_calls` to allow reading multiple files in parallel.
# MAGIC - It is a series of `tf.Examples`, where each `Example` is a key-value pair. 
# MAGIC 
# MAGIC [Reference documentation](https://www.tensorflow.org/tutorials/load_data/tfrecord)

# COMMAND ----------

display(dbutils.fs.ls(f"{DA.paths.datasets}/img/cats"))

# COMMAND ----------

# MAGIC %md
# MAGIC Even though storing data in a directory-like structure like above is simple, it has major drawbacks:
# MAGIC - Slows down loading process
# MAGIC   - Opening a file is a time-consuming operation, so having to open a larger number of files adds significant overhead to the training time. 

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Train test split
# MAGIC 
# MAGIC Here, we simply identify the paths that contain the images, rather than using Spark to read as dataframes.

# COMMAND ----------

import glob

cats_dir = f"{DA.paths.datasets.replace('dbfs:/', '/dbfs/')}/img/cats/cats*.jpg"
dogs_dir = f"{DA.paths.datasets.replace('dbfs:/', '/dbfs/')}/img/dogs/dog*.jpg"

train_cat_list = list(glob.glob(cats_dir)[:32])
train_dog_list = list(glob.glob(dogs_dir)[:32])

test_cat_list = list(glob.glob(cats_dir)[32:])
test_dog_list = list(glob.glob(dogs_dir)[32:])

train_data_paths = train_cat_list + train_dog_list 
test_data_paths = test_cat_list + test_dog_list

print(len(train_data_paths), len(test_data_paths))

# COMMAND ----------

from dataclasses import dataclass

@dataclass
class TrainConfig:

    img_height: int = 224
    img_width: int = 224
    img_channels: int = 3
    
    batch_size: int = 64
    epochs: int = 10    
    learning_rate: float = 0.0125
    verbose: int = 1

# COMMAND ----------

# MAGIC %md
# MAGIC ## Writing as TFRecord

# COMMAND ----------

from pathlib import Path
import os
import tensorflow as tf

def write_to_tfrecords(target_dir, data_paths, prefix):
    tfr_path = f"{target_dir}/tf_records/"
    dbutils.fs.mkdirs(tfr_path)
    tfr_file_path = os.path.join(target_dir, f"cats_dogs_{prefix}.tfrecords").replace("dbfs:/", "/dbfs/")
    writer = tf.io.TFRecordWriter(tfr_file_path)
    
    # the number of classes of images
    classes = {"0" : "cats", 
               "1" : "dogs"}

    # Loop to convert each image array to raw bytes one at a time
    for img_path in data_paths:
      
        # Get the integer value for the associated label 
        label = Path(img_path).parent.name
        label_int = list(classes.values()).index(label)

        img_raw = open(img_path, "rb").read()
        img_shape = tf.io.decode_jpeg(img_raw).shape
        # To store data, we need to add feature to it
        # Each feature has a key value pair, where the key holds the string name and the value holds the data
        my_features = {"height": tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[0]])),
                       "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[1]])),
                       "depth": tf.train.Feature(int64_list=tf.train.Int64List(value=[img_shape[2]])),
                       "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                       # tf.train.Feature must be in either bytes, float, or integer 
                       "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(label_int)]))}
            
        example = tf.train.Example(features=tf.train.Features(feature=my_features)) 
        writer.write(example.SerializeToString())
    writer.close()
    
    print(f"Written to {tfr_file_path}")
    return tfr_file_path

# COMMAND ----------

tfr_train_file_path =  write_to_tfrecords(DA.paths.working_dir, train_data_paths, prefix="train")

# COMMAND ----------

tfr_test_file_path = write_to_tfrecords(DA.paths.working_dir, test_data_paths, prefix="test")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading TFRecord
# MAGIC 
# MAGIC To load TFRecord, we need to parse each Example using `tf.io.parse_single_example` and provide the instructions (feature descriptions) on how to serialize the features. We can then apply the parsing function to all the data by using `map` function later. 

# COMMAND ----------

from dataclasses import dataclass

@dataclass
class DataIngest:
    img_height: int
    img_width: int
    img_channels: int

    def read_from_tfrecords(self, serialized_example):
        """Read TFRecords"""
        image_feature_description = {
            "img_raw": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64),
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
            "depth": tf.io.FixedLenFeature([], tf.int64),
        }
        # Parse the input tf.train.Example proto using the dictionary above.
        parsed = tf.io.parse_single_example(serialized_example, image_feature_description)
        # parse the data to get the original image
        img = tf.image.decode_jpeg(parsed["img_raw"], channels=self.img_channels)
        label = parsed["label"]

        return img, label   
    
    def preprocess_image(self, img):
        """Resizes and pads image to target width and height"""
        return tf.image.resize_with_pad(img, self.img_height, self.img_width) 

# COMMAND ----------

# MAGIC %md
# MAGIC If you need to debug, you can take one image from the file path, for example:
# MAGIC ```
# MAGIC raw_image_dataset = tf.data.TFRecordDataset(tfr_train_file_path)
# MAGIC raw = next(iter(raw_image_dataset))
# MAGIC parsed = tf.io.parse_single_example(raw, image_feature_description)
# MAGIC ```
# MAGIC 
# MAGIC There are a couple of important things to notice below:
# MAGIC - `num_parallel_calls`, `prefetch`, and `map()` are important for us to exploit multiple cores. 
# MAGIC - `shuffle`: This allows the training process to benefit from a non-deterministic order of reading data, improving model quality.
# MAGIC 
# MAGIC You might also read that some pipelines have `cache()` after `map()` – but before shuffling, prefetching, and batching – to cache the content to RAM. However, this is not recommended if your data is large. 

# COMMAND ----------

AUTOTUNE = tf.data.AUTOTUNE 

def parse_tfrecords(tfrecords_path: str, 
                    img_height: int, 
                    img_width: int,
                    img_channels: int, 
                    batch_size: int) -> tf.data.Dataset:

    data_ingest = DataIngest(img_height, img_width, img_channels)
    ds = tf.data.TFRecordDataset(tfrecords_path)
    
    def _preprocess_img_label(img, label):
        return data_ingest.preprocess_image(img), label
    
    parsed_image_dataset = (ds
                            .map(data_ingest.read_from_tfrecords, num_parallel_calls=AUTOTUNE)
                            .map(_preprocess_img_label, num_parallel_calls=AUTOTUNE)
                            .cache() ## NOT RECOMMENDED for large datasets
                            .shuffle(32)
                            .prefetch(AUTOTUNE)
                            .batch(batch_size))

    return parsed_image_dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Model

# COMMAND ----------

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
tf.random.set_seed(42)

def build_compile_model(cfg: TrainConfig):

    model = tf.keras.models.Sequential([
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", 
                      input_shape=[cfg.img_height, cfg.img_width, cfg.img_channels]),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=cfg.learning_rate), metrics=["accuracy"]) 
    
    return model 

# COMMAND ----------

train_cfg = TrainConfig(img_height=224,
                        img_width=224,
                        img_channels=3,
                        batch_size=8,
                        epochs=4,
                        learning_rate=0.0125
                       )

# COMMAND ----------

import mlflow

train_ds = parse_tfrecords(tfr_train_file_path, 
                           train_cfg.img_height, 
                           train_cfg.img_width, 
                           train_cfg.img_channels, 
                           train_cfg.batch_size)
model = build_compile_model(train_cfg)
history = model.fit(train_ds, epochs=train_cfg.epochs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transfer Learning with VGG16
# MAGIC 
# MAGIC Below, we are using tf.keras's Functional API. However, you could also use the sequential API as well: 
# MAGIC 
# MAGIC ```
# MAGIC base_model = applications.VGG16(weights="imagenet")
# MAGIC 
# MAGIC model = Sequential([base_model, 
# MAGIC                     Dense(1, activation="sigmoid")
# MAGIC                     ])
# MAGIC ```
# MAGIC 
# MAGIC Note:
# MAGIC - If you are using the base model's layer directly, setting `base_model.trainable = False` would have no effect. Then you should do:
# MAGIC 
# MAGIC ```
# MAGIC base_model = applications.VGG16(weights="imagenet")
# MAGIC model = Model(inputs=base_model.input, outputs=ouputs)
# MAGIC 
# MAGIC ### Use for loop to turn off trainable
# MAGIC for layer in base_model.layers:
# MAGIC   layer.trainable = False
# MAGIC ```
# MAGIC 
# MAGIC Refer to the [documentation](https://www.tensorflow.org/guide/keras/transfer_learning) for more details.

# COMMAND ----------

from tensorflow.keras import applications
from tensorflow.keras.models import Model 

def build_compile_vgg16_model(cfg: TrainConfig):

    # Load original model with pretrained weights from imagenet
    # We do not want to use the ImageNet classifier at the top since it has many irrelevant categories
    base_model = applications.VGG16(weights="imagenet", include_top=False)

    # Freeze base model
    base_model.trainable = False

    # Create new model on top
    inputs = layers.Input(shape=(cfg.img_height, cfg.img_width, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x) # we want to output probabilities for both classes
    model = Model(inputs, outputs)
    
    model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=cfg.learning_rate), metrics=["accuracy"]) 
    
    return model 

# COMMAND ----------

vgg16_model = build_compile_vgg16_model(train_cfg)
vgg16_history = vgg16_model.fit(train_ds, epochs=train_cfg.epochs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation

# COMMAND ----------

test_ds = parse_tfrecords(tfr_test_file_path, 
                          train_cfg.img_height, 
                          train_cfg.img_width, 
                          train_cfg.img_channels, 
                          train_cfg.batch_size)
eval_results = model.evaluate(test_ds, steps=2)

# COMMAND ----------

eval_results_vgg16 = vgg16_model.evaluate(test_ds, steps=2)

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
