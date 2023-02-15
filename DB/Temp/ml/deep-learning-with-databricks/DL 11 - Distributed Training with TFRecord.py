# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Distributed Training with TFRecord
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you:<br>
# MAGIC - Conduct distributed training with TFRecord using `spark-tensorflow-distributor`.
# MAGIC 
# MAGIC As you saw earlier, Horovod is a candidate for distributed training and works well with tf.keras, PyTorch. Here, we introduce another framework specific to TFRecord. This library provides a Spark wrapper around tf.keras's existing distributed training strategy. Spark TensorFlow Distributor makes use of the barrier execution mode, allowing workers to share data between each other during execution. 
# MAGIC 
# MAGIC References:
# MAGIC - [spark-tensorflow-distributor](https://github.com/tensorflow/ecosystem/tree/master/spark/spark-tensorflow-distributor)
# MAGIC - Another [notebook example](https://docs.databricks.com/_static/notebooks/deep-learning/spark-tensorflow-distributor.html) using the package
# MAGIC - TensorFlow's distributed training [documentation](https://www.tensorflow.org/guide/distributed_training?hl=en)

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Import functions for reading and parsing TFRecords
# MAGIC 
# MAGIC All these functions are from the previous notebook

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
        img = tf.image.decode_jpeg(parsed["img_raw"], channels=self.img_channels)
        label = parsed["label"]

        return img, label   
    
    def preprocess_image(self, img):
        """Resizes and pads image to target width and height"""
        return tf.image.resize_with_pad(img, self.img_height, self.img_width) 

# COMMAND ----------

import tensorflow as tf
AUTOTUNE = tf.data.AUTOTUNE 

def parse_tfrecords(tfrecords_path: str, 
                    img_height: int, 
                    img_width: int,
                    img_channels: int,
                    batch_size: int):
  
    data_ingest = DataIngest(img_height, img_width, img_channels)
    ds = tf.data.TFRecordDataset(tfrecords_path)
    
    def _preprocess_img_label(img, label):
        return data_ingest.preprocess_image(img), label
    
    parsed_image_dataset = (ds
                            .map(data_ingest.read_from_tfrecords, num_parallel_calls=AUTOTUNE)
                            .map(_preprocess_img_label, num_parallel_calls=AUTOTUNE)
                            .cache()
                            .shuffle(32)
                            .prefetch(AUTOTUNE)
                            .batch(batch_size))

    return parsed_image_dataset

# COMMAND ----------

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
# MAGIC ## Set up training configurations

# COMMAND ----------

train_cfg = TrainConfig(img_height=224,
                        img_width=224,
                        img_channels=3,
                        batch_size=8,
                        epochs=4,
                        learning_rate=0.0125
                       )

# COMMAND ----------

# MAGIC %md ## Create TFRecord dataset
# MAGIC As we did in the previous notebook

# COMMAND ----------

from pathlib import Path
import os
import tensorflow as tf
import glob

cats_dir = f"{DA.paths.datasets.replace('dbfs:/', '/dbfs/')}/img/cats/cats*.jpg"
dogs_dir = f"{DA.paths.datasets.replace('dbfs:/', '/dbfs/')}/img/dogs/dog*.jpg"

train_cat_list = list(glob.glob(cats_dir)[:32])
train_dog_list = list(glob.glob(dogs_dir)[:32])

test_cat_list = list(glob.glob(cats_dir)[32:])
test_dog_list = list(glob.glob(dogs_dir)[32:])

train_data_paths = train_cat_list + train_dog_list 
test_data_paths = test_cat_list + test_dog_list


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

tfr_train_file_path =  write_to_tfrecords(DA.paths.working_dir, train_data_paths, prefix="train")
tfr_test_file_path = write_to_tfrecords(DA.paths.working_dir, test_data_paths, prefix="test")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC `spark-tensorflow-distributor` requires all training code to be wrapped within a single function. Notice the code is highly similar to the single-node implementation, with the exception of the data sharding option. 
# MAGIC 
# MAGIC `tf.data.experimental.AutoShardPolicy` dictates how the inputs will be sharded. When it's set to `DATA`, the data will be sharded (or shared) by each worker or process. Based on the [documentation](https://www.tensorflow.org/api_docs/python/tf/data/experimental/AutoShardPolicy):
# MAGIC > each worker will process the whole dataset and discard the portion that is not for itself.
# MAGIC 
# MAGIC It is recommended to make the imports inside the function so that the training function can be serialized. 

# COMMAND ----------

def train():
    
    from dataclasses import dataclass
    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import Adam
    tf.random.set_seed(42)
        
    def build_compile_model(cfg):

        model = tf.keras.models.Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=[cfg.img_height, cfg.img_width, cfg.img_channels]),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid")
        ])

        model.compile(loss="binary_crossentropy", optimizer=Adam(learning_rate=cfg.learning_rate), metrics=["accuracy"]) 

        return model 

    model = build_compile_model(train_cfg)
    train_ds = parse_tfrecords(tfr_train_file_path, 
                                   train_cfg.img_height, 
                                   train_cfg.img_width, 
                                   train_cfg.img_channels, 
                                   train_cfg.batch_size)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_ds = train_ds.with_options(options)

    history = model.fit(train_ds, epochs=1, steps_per_epoch=8)

    test_ds = parse_tfrecords(tfr_test_file_path, 
                              train_cfg.img_height, 
                              train_cfg.img_width, 
                              train_cfg.img_channels, 
                              train_cfg.batch_size)
    eval_results = model.evaluate(test_ds, steps=2)

    return model , eval_results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train on driver node
# MAGIC 
# MAGIC Under the hood, tf.keras is distribution-aware, so you can easily use `tf.distribute.Stategy` to enable distributed training. `MirroredStrategyRunner` provides a Spark wrapper around `tf.distribute.MirroredStrategy`.
# MAGIC 
# MAGIC Why is it called "mirrored"? 
# MAGIC - This strategy supports synchronous distributed training on multiple CPUs/GPUs on one machine. 
# MAGIC - A model replica is created per GPU device; hence, the model variables are also mirrored across all replica. 
# MAGIC - The model variables are in sync since model updates apply to all of them
# MAGIC 
# MAGIC Best practices
# MAGIC - Since `.fit()` automatically splits each training batch across all replicas, it's best that the batch size is divisible by the number of replicas (# of CPUs/GPUs). This is to ensure that each replica gets batches of the same size. 
# MAGIC 
# MAGIC Refer to [documentation here](https://www.tensorflow.org/guide/distributed_training?hl=en#mirroredstrategy).

# COMMAND ----------

from spark_tensorflow_distributor import MirroredStrategyRunner
import mlflow
import mlflow.keras

with mlflow.start_run() as local_run:
    runner = MirroredStrategyRunner(num_slots=1, local_mode=True, use_gpu=False)
    local_model, local_eval_results = runner.run(train)
    mlflow.keras.log_model(local_model, artifact_path="local_model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load model using MLflow and evaluate on test set

# COMMAND ----------

loaded_model = mlflow.keras.load_model(model_uri=f"runs:/{local_run.info.run_id}/local_model")

test_ds = parse_tfrecords(tfr_test_file_path, 
                           train_cfg.img_height, 
                           train_cfg.img_width, 
                           train_cfg.img_channels, 
                           train_cfg.batch_size)

loaded_model.evaluate(test_ds)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train on workers
# MAGIC 
# MAGIC Now that we are certain that the code works, we can leverage all the slots available. On GPU clusters, the number of slots is equal to the number of available GPUs. On CPU clusters, pick a number that's reasonable. You also need to toggle `local_mode` to `False` to set the training to happen on workers. 
# MAGIC 
# MAGIC If you want to use a custom strategy for distributed training, you can create your own `tf.distribute.Strategy` and turn `use_custom_strategy` to `True`. Refer to:
# MAGIC - spark-tensorflow-distributor [notebook by Databricks](https://docs.databricks.com/_static/notebooks/deep-learning/spark-tensorflow-distributor.html)
# MAGIC - Tensorflow documentation [here](https://www.tensorflow.org/tutorials/distribute/custom_training) and [here](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy#in_short_2)
# MAGIC 
# MAGIC Since we are not using a cluster with workers, if we use `use_custom_strategy=False`, `MirroredStrategyRunner` will construct a `MultiWorkerMirroredStrategy` for the user automatically, which will result in failure. Hence, we override the custom strategy below to use the non-worker strategy. Refer to the [source code](https://github.com/tensorflow/ecosystem/blob/fd74f96d3e3cdf40d8b2714b2ce36749920fe093/spark/spark-tensorflow-distributor/spark_tensorflow_distributor/mirrored_strategy_runner.py#L273) here if you are interested in the implementation detail.

# COMMAND ----------

with mlflow.start_run() as dist_run:
    dist_runner = MirroredStrategyRunner(num_slots=4, local_mode=False, use_custom_strategy=True, use_gpu=False)
    dist_model, dist_eval_results = dist_runner.run(train)
    mlflow.keras.log_model(dist_model, "dist_model")

# COMMAND ----------

loaded_dist_model = mlflow.keras.load_model(model_uri=f"runs:/{dist_run.info.run_id}/dist_model")

loaded_dist_model.evaluate(test_ds)

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
