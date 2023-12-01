# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md # Horovod with Petastorm Lab
# MAGIC 
# MAGIC In this lab we are going to build upon our previous lab model trained on the Wine Quality dataset and distribute the deep learning training process using both HorovodRunner and Petastorm.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Prepare your data for use with Horovod
# MAGIC  - Distribute the training of our model using HorovodRunner
# MAGIC  - Use Parquet files as input data for our distributed deep learning model with Petastorm + Horovod

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md ## 1. Load and process data

# COMMAND ----------

train_df = spark.read.format("delta").load(f"{DA.paths.datasets}/wine-quality/train")

val_df = spark.read.format("delta").load(f"{DA.paths.datasets}/wine-quality/val")
X_val = val_df.toPandas()
y_val = X_val.pop("quality")

test_df = spark.read.format("delta").load(f"{DA.paths.datasets}/wine-quality/test")
X_test = test_df.toPandas()
y_test = X_test.pop("quality")

# COMMAND ----------

# MAGIC %md Let's set up our target column, as well as our feature columns.

# COMMAND ----------

target_col = "quality"
feature_cols = train_df.columns
feature_cols.remove(target_col)
feature_cols

# COMMAND ----------

# MAGIC %md Let's use the new [dataclass](https://docs.python.org/3/library/dataclasses.html) workflow to create our training configuration, TrainConfig:

# COMMAND ----------

# TODO

from dataclasses import dataclass

@dataclass
class TrainConfig:
    
    batch_size: FILL_IN
    epochs: FILL_IN
    learning_rate: FILL_IN
    verbose: FILL_IN
    prefetch: FILL_IN
    validation_data = FILL_IN
    
    # Define directory the underlying files are copied to
    # Leverages Network File System (NFS) location for better performance if using a single node cluster
    petastorm_cache: str = f"file:///{DA.paths.working_dir}/petastorm"
    
    # uncomment the line below if working with a multi node cluster (can't use NFS location)
    # petastorm_cache: str = f"file:///{DA.paths.working_dir}/petastorm".replace("///dbfs:/", "/dbfs/")

    dbutils.fs.rm(petastorm_cache, recurse=True)
    dbutils.fs.mkdirs(petastorm_cache)
    petastorm_workers_count: FILL_IN

# COMMAND ----------

# MAGIC %md ## 2. Vectorize Dataset

# COMMAND ----------

# TODO

from petastorm.spark import SparkDatasetConverter, make_spark_converter

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline

def create_petastorm_converters_vec(FILL_IN, FILL_IN, FILL_IN, FILL_IN):
    # Set a cache directory for intermediate data storage 
    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, cfg.petastorm_cache)
    
    vector_assembler = VectorAssembler(FILL_IN, FILL_IN)
    
    t_df = vector_assembler.transform(FILL_IN).select(FILL_IN, FILL_IN)
    
    converter_train = make_spark_converter(t_df.repartition(FILL_IN))
    
    return converter_train

# COMMAND ----------

# TODO

cfg = FILL_IN

# COMMAND ----------

# TODO
converter_train = create_petastorm_converters_vec(FILL_IN, FILL_IN, FILL_IN, FILL_IN)

# COMMAND ----------

# MAGIC %md ## 3. Horovod
# MAGIC 
# MAGIC In order to distribute the training of our Keras model with Horovod, we must define our **`run_training_horovod`** training function

# COMMAND ----------

import horovod.tensorflow.keras as hvd
import mlflow
import horovod
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from petastorm import TransformSpec

tf.random.set_seed(42)
## get databricks credentials for mlflow tracking
databricks_host = mlflow.utils.databricks_utils.get_webapp_url()
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

checkpoint_dir = f"{DA.paths.working_dir}/petastorm_checkpoint_weights.ckpt"
dbutils.fs.rm(checkpoint_dir, True)
#checkpoint_dir = checkpoint_dir.replace("dbfs:/", "/dbfs/")

# COMMAND ----------

# TODO

def run_training_horovod():
    # Horovod: initialize Horovod.
    FILL_IN
    # If using GPU, see example in docs pin GPU to be used to process local rank (one GPU per process)
    # These steps are skipped on a CPU cluster
    # https://horovod.readthedocs.io/en/stable/tensorflow.html?#horovod-with-tensorflow
    
    # Including MLflow
    import mlflow
    import os

    # Configure Databricks MLflow environment
    mlflow.set_tracking_uri("databricks")
    os.environ["DATABRICKS_HOST"] = FILL_IN
    os.environ["DATABRICKS_TOKEN"] = FILL_IN

    tf_spec = TransformSpec(
        edit_fields=[(FILL_IN, FILL_IN, FILL_IN, FILL_IN)], 
        selected_fields=[FILL_IN, FILL_IN]
    )

 ## Create tf_dataset to build Normalization Layer with just one epoch. 

    with converter_train.make_tf_dataset(transform_spec=FILL_IN,
                                     workers_count=FILL_IN, 
                                     batch_size=FILL_IN,
                                     prefetch=FILL_IN,
                                     num_epochs=FILL_IN 
                                    ) as train_ds:
        # Number of steps required to go through one epoch
        steps_per_epoch = len(converter_train) // cfg.batch_size
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(train_ds.map(lambda x: FILL_IN))
    
    with converter_train.make_tf_dataset(workers_count=FILL_IN, 
                                         batch_size=FILL_IN,
                                         prefetch=FILL_IN,
                                         num_epochs=FILL_IN,
                                         cur_shard=FILL_IN, 
                                         shard_count=FILL_IN
                                        ) as train_ds:
        
        dataset = train_ds.map(lambda x: (FILL_IN, FILL_IN)) #<--- target col needs to be a non-whitespace name x.['...'] does not work 
        steps_per_epoch = FILL_IN

        model = Sequential([normalizer, 
                            Dense(20, input_dim=len(feature_cols), activation="relu"),
                            Dense(20, activation="relu"),
                            Dense(1, activation="linear")]
                            ) 
        
        # Horovod: adjust learning rate based on number of GPUs/CPUs
        optimizer = FILL_IN
        
        #############################################################################################
        
        # Adding in Distributed Optimizer
        optimizer_dist = hvd.DistributedOptimizer(FILL_IN)
        model.compile(FILL_IN, FILL_IN, FILL_IN)
        
        # Adding in callbacks
        callbacks = [
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            FILL_IN,
            
            # Horovod: average metrics among workers at the end of every epoch.
            # Note: This callback must be in the list before the ReduceLROnPlateau,
            # TensorBoard or other metrics-based callbacks.
            FILL_IN,
            
            # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
            # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
            # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
            FILL_IN,
            
            # Reduce the learning rate if training plateaus.
            FILL_IN
        ]
        
        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == FILL_IN:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_dir.replace("dbfs:/", "/dbfs/"), 
                                                                monitor="loss", 
                                                                save_best_only=True))

        history = model.fit(FILL_IN, 
                            steps_per_epoch=FILL_IN,
                            epochs=FILL_IN,
                            callbacks=FILL_IN,
                            verbose=FILL_IN,
                            validation_data=FILL_IN
                           )
        
        # MLflow Tracking (Log only from Worker 0)
        if hvd.rank() == FILL_IN:    

            # Log events to MLflow
            with mlflow.start_run(run_id = active_run_id) as run:
                # Log MLflow Parameters
                mlflow.log_param("num_layers", FILL_IN)
                mlflow.log_param("optimizer_name", FILL_IN)
                mlflow.log_param("learning_rate", FILL_IN)
                mlflow.log_param("batch_size", FILL_IN)
                mlflow.log_param("hvd_np", FILL_IN)

                # Log MLflow Metrics
                mlflow.log_metric("train loss", FILL_IN)

                # Log Model
                mlflow.keras.log_model(FILL_IN, FILL_IN)

# COMMAND ----------

# MAGIC %md Let's now run our model on all workers.

# COMMAND ----------

# TODO

from sparkdl import HorovodRunner

## OPTIONAL: You can enable Horovod Timeline as follows, but can incur slow down from frequent writes, and have to export out of Databricks to upload to chrome://tracing
# import os
# os.environ["HOROVOD_TIMELINE"] = f"{DA.paths.working_dir}/_timeline.json"

with mlflow.start_run() as run:  
    # Get active run_uuid
    active_run_id = FILL_IN
    hvd_np = FILL_IN
    hr = HorovodRunner(FILL_IN, FILL_IN)
    hr.run(FILL_IN)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## 4. Loading Model and Evaluation
# MAGIC 
# MAGIC Since we included the Normalization layer inside the model, we can now use this model in production without having to worry about normalization again. 
# MAGIC Let's load the model a number of ways: from the Keras API, from the MLflow run, and as a spark_ufd

# COMMAND ----------

# MAGIC %md ### Loading from the Keras API

# COMMAND ----------

# TODO

from tensorflow.keras.models import load_model

Load the model and print model summary
trained_model = FILL_IN
print(trained_model.summary())

Evaluate the model on the test data
FILL_IN

# COMMAND ----------

# MAGIC %md ### Loading from the MLflow run

# COMMAND ----------

# TODO

from sklearn.metrics import mean_squared_error

model_uri = FILL_IN
mlflow_model = FILL_IN
y_pred = FILL_IN
mean_squared_error(FILL_IN,FILL_IN)

# COMMAND ----------

# MAGIC %md ### Leveraging Spark for scalable inference using MLflow's Spark UDF

# COMMAND ----------

# TODO

load the model using spark_udf predict on test data and use `display()` to print the results
predict = mlflow.pyfunc.spark_udf(spark, model_uri)
display(test_df.withColumn("quality_prediction", predict(*feature_cols)))

# COMMAND ----------

# MAGIC %md ## 5. Delete the `converter_train` from cache

# COMMAND ----------

converter_train.delete()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
