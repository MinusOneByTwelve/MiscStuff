# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Load data with Petastorm 
# MAGIC 
# MAGIC <a href="https://github.com/uber/petastorm" target="_blank">Petastorm</a> enables single machine or distributed training and evaluation of deep learning models from datasets in Apache Parquet format and datasets that are already loaded as Spark DataFrames. It supports ML frameworks such as TensorFlow, PyTorch, and PySpark and can be used from pure Python code.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC - Perform data preparation using Petastorm, and comparing two approaches to do so: Upstream & Downstream Vectorization

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

train_df = spark.read.format("delta").load(f"{DA.paths.datasets}/california-housing/train")

val_df = spark.read.format("delta").load(f"{DA.paths.datasets}/california-housing/val")
X_val = val_df.toPandas()
y_val = X_val.pop("label")

test_df = spark.read.format("delta").load(f"{DA.paths.datasets}/california-housing/test")

# COMMAND ----------

# MAGIC %md
# MAGIC Below, we define training configurations.
# MAGIC 
# MAGIC **Note**: We define a path for Petastorm to store cached data. Petastorm takes data from a Spark DataFrame, checks if it's already cached and persisted in the file system, and caches it if not. Here, we define where the cache path is to provide a directory for the intermediate files later. 
# MAGIC 
# MAGIC <img src="https://blog.uber-cdn.com/cdn-cgi/image/width=2160,quality=80,onerror=redirect,format=auto/wp-content/uploads/2022/08/Petastorm_Figure_02.png" width=600>

# COMMAND ----------

# MAGIC %md
# MAGIC What is `dataclass`? It is a new feature since Python 3.7 that allows class definition with less boilerplate code, reducing code verbosity. If you are curious to read more about this, go to this [blog post](https://towardsdatascience.com/9-reasons-why-you-should-start-using-python-dataclasses-98271adadc66) or [Python's documentation](https://docs.python.org/3/library/dataclasses.html).

# COMMAND ----------

from dataclasses import dataclass

@dataclass
class TrainConfig:
    
    batch_size: int = 64
    epochs: int = 10 
    learning_rate: float = 0.001
    verbose: int = 1
    prefetch: int = 1  # We will look at this later
    validation_data: tuple = (X_val, y_val)
    
    # Define directory the underlying files are copied to
    # Leverages Network File System (NFS) location for better performance if using a single node cluster
    petastorm_cache: str = f"file:/{DA.paths.working_dir}/petastorm"
    
    # uncomment the line below if working with a multi node cluster (can't use NFS location)
    # petastorm_cache: str = f"file:///{DA.paths.working_dir}/petastorm".replace("///dbfs:/", "/dbfs/")

    dbutils.fs.rm(petastorm_cache, recurse=True)
    dbutils.fs.mkdirs(petastorm_cache)
    petastorm_workers_count: int = spark.sparkContext.defaultParallelism

# COMMAND ----------

target_col = "label"
feature_cols = train_df.columns
feature_cols.remove(target_col)
feature_cols

# COMMAND ----------

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Normalization

tf.random.set_seed(42)


def build_compile_model(normalize_layer, cfg):
    model = Sequential([normalize_layer,
                        Dense(20, input_dim=len(feature_cols), activation="relu"),
                        Dense(20, activation="relu"),
                        Dense(1, activation="linear")]
                        ) 
    optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    
    return model

# COMMAND ----------

# MAGIC %md
# MAGIC ### Preprocessing Data for Petastorm 
# MAGIC Delta tables inherently store data in Parquet columnar format. Petastorm requires row format for transformation or filtering. You can read [Uber's blog post](https://www.uber.com/blog/petastorm/) on how Petastorm is implemented. There are two approaches to carrying out transformations: 
# MAGIC * Upstream Vectorization
# MAGIC * Downstream Vectorization 
# MAGIC 
# MAGIC Both approaches require us to use Petastorm TransformSpec function, which we will show later.

# COMMAND ----------

# MAGIC %md
# MAGIC ####  Approach 1: Upstream Vectorization
# MAGIC 
# MAGIC This approach handles as much transformation as possible prior to passing data to Petastorm and TensorFlow. 
# MAGIC 
# MAGIC Benefits of `SparkDatasetConverter` and `make_spark_converter`:
# MAGIC 1. Caches the data if it has not been cached
# MAGIC 2. Automatically converts SparkML vector into 1D arrays. 
# MAGIC     - This removes the need to use SparkML for training ML models, allowing us to use a variety of deep learning frameworks to train models. 

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline
from petastorm.spark import SparkDatasetConverter, make_spark_converter

def create_petastorm_converters_vec(train_df, cfg, feature_cols, target_col="label"):
    # Set a cache directory for intermediate data storage 
    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, cfg.petastorm_cache)
  
    vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    transformed_df = vector_assembler.transform(train_df).select("features", target_col)
    
    # Below, we repartition the dataframe for efficiency: we'd like the dataframe to be evenly distributed across workers 
    converter_train = make_spark_converter(transformed_df.repartition(cfg.petastorm_workers_count))
    
    return converter_train

# COMMAND ----------

cfg = TrainConfig()

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that Petastorm caches the DataFrame in the path with this directory format: `{datetime}-appid-{spark.sparkContext.applicationId}-{uuid4}`. Date time string format is `%Y%m%d%H%M%S` that represents the time of dataframe materialization. 

# COMMAND ----------

converter_train_vec = create_petastorm_converters_vec(train_df, cfg, feature_cols, target_col)

# COMMAND ----------

# MAGIC %md
# MAGIC At this stage, the dataframe has not yet been materialized. It is only materialized when we call `converter_train.make_tf_dataset` later in order to actively feed data to TensorFlow training framework. We can now further define additional preprocessing logic using `TransformSpec`. All the transformations will apply to each row on the Spark thread, which is why we needed to specify `petastorm_workers_count` in our training configurations.
# MAGIC 
# MAGIC About `TransformSpec`:
# MAGIC 1. The output shape of `TransformSpec` is not automatically known by Petastorm. Therefore, we need to supply the length of our feature vector in `edit_fields`. 
# MAGIC 2. `selected_fields` indicates the fields to be selected, and the post-transform schema will respect the field order in `selected_fields`. 
# MAGIC 
# MAGIC Refer to [documentation](https://petastorm.readthedocs.io/en/latest/api.html?highlight=transformspec#module-petastorm.transform).

# COMMAND ----------

from petastorm import TransformSpec

tf_spec_vec = TransformSpec(
    edit_fields=[("features", np.float32, (len(feature_cols),), False)], 
    selected_fields=["features", target_col]
)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Gotchas about `epoch`:
# MAGIC - When using `tf.keras.layers.Normalization` (or other preprocessing layers), remember that `.adapt()` is not epoch-aware. This means that we need to have two separate calls of `converter_train.make_tf_dataset` to
# MAGIC   - normalize our data in one go: `num_epochs=1`
# MAGIC   - train the DL model with infinite batches of data: `num_epochs=None`
# MAGIC 
# MAGIC - The epoch we are referring to is separate from training epochs, which dictates how many times TF sees the entire dataset. Petastorm epoch is the number of times for Petastorm to pass over all rows. 
# MAGIC 
# MAGIC Why `prefetch`?
# MAGIC - It is important for model performance. We are creating a dataset that is always `n` batches ahead, while our model is training on another batch.
# MAGIC - Refer to [tf.keras's documentation here](https://www.tensorflow.org/guide/data_performance#prefetching).
# MAGIC   - When the data is large, we usually prefer to leave to `tf.data.AUTOTUNE` to decide how many batches to prefetch, rather than using trials and errors to set it manually.

# COMMAND ----------

### This `with` statement here creates train_ds in TF dataset format to feed into a TF algorithm. This means that what we supply to TransformSpec will be performed on each batch of data.
with converter_train_vec.make_tf_dataset(transform_spec=tf_spec_vec,
                                        workers_count=cfg.petastorm_workers_count, 
                                        batch_size=cfg.batch_size,
                                        prefetch=cfg.prefetch,
                                        num_epochs=1 # adapt() is not epoch aware; setting num_epochs=None creates an infinite dataset. Hence an infinite loop.
                                        ) as train_ds:
    # Number of steps required to go through one epoch
    steps_per_epoch = len(converter_train_vec) // cfg.batch_size
    normalizer_vec = Normalization(axis=-1)
    normalizer_vec.adapt(train_ds.map(lambda x: x.features))

# COMMAND ----------

# MAGIC %md
# MAGIC Again, note that we use **`num_epochs=None`** to generate infinite batches of data to avoid handling the last incomplete batch. This is particularly useful in the distributed training scenario, where we need to guarantee that the numbers of data records seen on all workers are identical. Given that the length of each data shard may not be identical, setting **`num_epochs`** to any specific number would fail to meet the guarantee.

# COMMAND ----------

# reset the Petastorm converter to create an 'infinite' dataset
with converter_train_vec.make_tf_dataset(workers_count=cfg.petastorm_workers_count, 
                                     batch_size=cfg.batch_size,
                                     prefetch=cfg.prefetch,
                                     num_epochs=None,
                                    ) as train_ds:
    
    dataset = train_ds.map(lambda x: (x.features, x.label)) # Not strictly necessary as part of our TransformSpec
    
    
    model_vec = build_compile_model(normalizer_vec, cfg)
    history_vec = model_vec.fit(dataset, 
                        # Do not specify the batch_size if your data is in the form of a dataset (since they generate batches)
                        # Instead, use steps_per_epoch
                        steps_per_epoch=steps_per_epoch, 
                        epochs=cfg.epochs,
                        validation_data=cfg.validation_data
                       )

# COMMAND ----------

# MAGIC %md
# MAGIC #### Approach 2: Downstream Vectorization
# MAGIC 
# MAGIC Here, we use the approach of passing the conversion of feature vector to 1D arrays to Petastorm. While this approach works, we recommend pushing as many data transformations upstream to allow higher efficiency when training. 

# COMMAND ----------

def create_petastorm_converters(train_df, cfg):
    """Create Petastorm converter objects from train and val Spark DataFrames"""    
    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, cfg.petastorm_cache)
    # Same as before, we repartition the dataframe for efficiency: we'd like the dataframe to be evenly distributed across workers
    converter_train = make_spark_converter(train_df.repartition(cfg.petastorm_workers_count))
    
    return converter_train

# COMMAND ----------

cfg = TrainConfig()
converter_train = create_petastorm_converters(train_df, cfg)

# COMMAND ----------

# MAGIC %md
# MAGIC In addition to specifying the output shape and schema, we instruct Petastorm how to preprocess our data. Here we need to prepare the data to be a 1D array from a vector. Hence, we define `make_transform_row` to perform the transformations.
# MAGIC 
# MAGIC About `make_transform_row`:
# MAGIC 1. It operates in Spark threads
# MAGIC 2. The input and output of this dataframe has to be a dictionary, which pandas.DataFrame is. 
# MAGIC 3. We take each feature and transform it into a 1D numpy or a tensor. 

# COMMAND ----------

import pandas as pd
import numpy as np

def make_transform_row(feature_cols, label_col):
    def apply(pdf):
        res = pd.DataFrame()
        res["features"] = pdf[feature_cols].apply(lambda x: np.array(x.tolist()), axis=1)
        res[label_col] = pdf[label_col]
        return res
    return apply

tf_spec = TransformSpec(
    make_transform_row(feature_cols, target_col),
    edit_fields=[("features", np.float32, (len(feature_cols),), False) # Tensorflow expects a list of tuples
                ], 
    selected_fields=["features", "label"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC Train the normalizer

# COMMAND ----------

with converter_train.make_tf_dataset(transform_spec=tf_spec,
                                     workers_count=cfg.petastorm_workers_count, 
                                     batch_size=cfg.batch_size,
                                     prefetch=2,
                                     num_epochs=1 
                                    ) as train_ds:
    
    steps_per_epoch = len(converter_train) // cfg.batch_size
    normalizer = Normalization(axis=-1)
    normalizer.adapt(train_ds.map(lambda x: x.features))

# COMMAND ----------

# reset the Petastorm converter to create an 'infinite' dataset
with converter_train.make_tf_dataset(transform_spec=tf_spec,
                                      workers_count=cfg.petastorm_workers_count, 
                                      batch_size=cfg.batch_size,
                                      prefetch=2,
                                      num_epochs=None
                                     ) as train_ds:
    model = build_compile_model(normalizer, cfg)
    history = model.fit(train_ds, 
                          steps_per_epoch=steps_per_epoch,
                          epochs=cfg.epochs,
                          validation_data=cfg.validation_data
                         )

# COMMAND ----------

# MAGIC %md
# MAGIC Delete the cached files 

# COMMAND ----------

converter_train_vec.delete()
converter_train.delete()

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
