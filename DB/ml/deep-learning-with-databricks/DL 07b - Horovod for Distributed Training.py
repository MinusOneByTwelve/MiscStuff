# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Horovod
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC  - Use Horovod to train a distributed neural network
# MAGIC  
# MAGIC Similar to Petastorm, Horovod is open-source developed by Uber. From [Uber's blog post](https://www.uber.com/blog/horovod/), 
# MAGIC 
# MAGIC > [Horovod is] named after a traditional Russian folk dance in which performers dance with linked arms in a circle, much like how distributed TensorFlow processes use Horovod to communicate with each other.
# MAGIC 
# MAGIC The goal is to allow single node training to multiple nodes. Horovod uses a data parallelization strategy by distributing the training to multiple nodes in parallel. 
# MAGIC <br>
# MAGIC <br>
# MAGIC - Each node receives a copy of the model and a batch of the dataset 
# MAGIC - The node/worker computes the gradients on the assigned batch of data
# MAGIC - Horovod uses `ring allreduce` algorithm to synchronize and average the gradients across nodes
# MAGIC - All nodes/workers update their models' weights
# MAGIC 
# MAGIC <img src="http://eng.uber.com/wp-content/uploads/2017/10/image2-1.png" width=500>
# MAGIC <br>
# MAGIC We will use `HorovodRunner` to wrap the single Python function that includes the training procedure. Spark implements barrier execution mode that allows multiple operations to be coordinated; for example, in neural networks, the training process needs to coordinate backpropagation and forward pass. HorovodRunner integrates with Spark's barrier execution/scheduling mode. If you are interested in more implementation details, refer to this <a href="https://docs.microsoft.com/en-us/azure/databricks/applications/deep-learning/distributed-training/horovod-runner" target="_blank">documentation link</a>.
# MAGIC 
# MAGIC <!-- ![](https://files.training.databricks.com/images/horovod-runner.png) --> 
# MAGIC 
# MAGIC For additional resources, see:
# MAGIC * [Paper released by Uber](https://arxiv.org/abs/1802.05799)

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

train_df = spark.read.format("delta").load(f"{DA.paths.datasets}/california-housing/train")

val_df = spark.read.format("delta").load(f"{DA.paths.datasets}/california-housing/val")
X_val = val_df.toPandas()
y_val = X_val.pop("label")

test_df = spark.read.format("delta").load(f"{DA.paths.datasets}/california-housing/test")
X_test = test_df.toPandas()
y_test = X_test.pop("label")

# COMMAND ----------

target_col = "label"
feature_cols = train_df.columns
feature_cols.remove(target_col)
feature_cols

# COMMAND ----------

# MAGIC %md
# MAGIC The preprocessing code is the same from the previous notebook.

# COMMAND ----------

from dataclasses import dataclass

@dataclass
class TrainConfig:
    
    batch_size: int = 64
    epochs: int = 10 
    learning_rate: float = 0.001
    verbose: int = 1
    prefetch: int = 2 
    validation_data = [X_val, y_val]
    
    # Define directory the underlying files are copied to
    # Leverages Network File System (NFS) location for better performance if using a single node cluster
    petastorm_cache: str = f"file:///{DA.paths.working_dir}/petastorm"
    
    # uncomment the line below if working with a multi node cluster (can't use NFS location)
    # petastorm_cache: str = f"file:///{DA.paths.working_dir}/petastorm".replace("///dbfs:/", "/dbfs/")

    dbutils.fs.rm(petastorm_cache, recurse=True)
    dbutils.fs.mkdirs(petastorm_cache)
    petastorm_workers_count: int = spark.sparkContext.defaultParallelism

# COMMAND ----------

from petastorm.spark import SparkDatasetConverter, make_spark_converter

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml import Pipeline

def create_petastorm_converters_vec(train_df, cfg, feature_cols, target_col="label"):
    # Set a cache directory for intermediate data storage 
    spark.conf.set(SparkDatasetConverter.PARENT_CACHE_DIR_URL_CONF, cfg.petastorm_cache)
    
    vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    
    t_df = vector_assembler.transform(train_df).select("features", target_col)
    
    converter_train = make_spark_converter(t_df.repartition(cfg.petastorm_workers_count))
    
    return converter_train

# COMMAND ----------

cfg = TrainConfig()

# COMMAND ----------

converter_train = create_petastorm_converters_vec(train_df, cfg, feature_cols, target_col)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Communication 
# MAGIC 
# MAGIC Notice that in Petastorm's `converter_train.make_tf_dataset` call below, we have `cur_shard=hvd.rank()` and `shard_count=hvd.size()`. This is because Horovod uses Message Passing Interface (MPI) implementation to set up the distributed infrastructure for the nodes to communicate with each other. 
# MAGIC 
# MAGIC Say we launch our training step on 4 VMs, each having 4 GPUs. If we launched one copy of the script per GPU:
# MAGIC 
# MAGIC * Size would be the number of processes, in this case, 16.
# MAGIC 
# MAGIC * Rank would be the unique process ID from 0 to 15 (size - 1).
# MAGIC 
# MAGIC * Local rank would be the unique process ID within the VM from 0 to 3.
# MAGIC 
# MAGIC Reference: 
# MAGIC * [Horovod docs](https://github.com/horovod/horovod/blob/master/docs/concepts.rst)

# COMMAND ----------

# MAGIC %md ## Defining Horovod Training
# MAGIC 
# MAGIC Let's build up the function that will train our model in a distrubted manner

# COMMAND ----------

# Firstly we need to get our modules and set up some logistics
import horovod.tensorflow.keras as hvd
import mlflow
import horovod

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from petastorm import TransformSpec


# As we will be sending the function to run on multiple machines, we need to be able to send data to our driver node for mlflow, this means we need to get our hostname and a token
tf.random.set_seed(42)
## get databricks credentials for mlflow tracking
databricks_host = mlflow.utils.databricks_utils.get_webapp_url()
databricks_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# Finally, we need to checkpoint our data incase of a failure or stoppage.
checkpoint_dir = f"{DA.paths.working_dir}/petastorm_checkpoint_weights.ckpt"
dbutils.fs.rm(checkpoint_dir, True)
#checkpoint_dir = checkpoint_dir.replace("dbfs:/", "/dbfs/")


# COMMAND ----------

def run_training_horovod():
    # Horovod: initialize Horovod.
    hvd.init()
    # If using GPU, see example in docs pin GPU to be used to process local rank (one GPU per process)
    # These steps are skipped on a CPU cluster
    # https://horovod.readthedocs.io/en/stable/tensorflow.html?#horovod-with-tensorflow
    
    # While we already included mlflow in our notebook, this function will be sent to new nodes so we will need it included again
    import mlflow
    import os

    # Configure Databricks MLflow environment
    mlflow.set_tracking_uri("databricks")
    os.environ["DATABRICKS_HOST"] = databricks_host
    os.environ["DATABRICKS_TOKEN"] = databricks_token
    
    ########################################################################################################################################################
    ############################## Like the previous lesson, we need to create our dataset with the vectorization ##########################################
    tf_spec = TransformSpec(
        edit_fields=[("features", np.float32, (len(feature_cols),), False)], 
        selected_fields=["features", target_col]
    )
    with converter_train.make_tf_dataset(transform_spec=tf_spec,
                                     workers_count=cfg.petastorm_workers_count, 
                                     batch_size=cfg.batch_size,
                                     prefetch=cfg.prefetch,
                                     num_epochs=1 
                                    ) as train_ds:
        # Number of steps required to go through one epoch
        steps_per_epoch = len(converter_train) // cfg.batch_size
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(train_ds.map(lambda x: x.features))
    
    with converter_train.make_tf_dataset(workers_count=cfg.petastorm_workers_count, 
                                         batch_size=cfg.batch_size,
                                         prefetch=cfg.prefetch,
                                         num_epochs=None,
                                         cur_shard=hvd.rank(), 
                                         shard_count=hvd.size()
                                        ) as train_ds:
        
        dataset = train_ds.map(lambda x: (x.features, x.label))
        steps_per_epoch = len(converter_train) // (cfg.batch_size*hvd.size())

        model = Sequential([normalizer, 
                            Dense(20, input_dim=len(feature_cols), activation="relu"),
                            Dense(20, activation="relu"),
                            Dense(1, activation="linear")]
                            ) 
    ### We need to go further now, with the horovod optimizer, callbacks, and training with this converter_train call

      
        # Horovod: adjust learning rate based on number of GPUs/CPUs
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
        
        
        # Adding in Distributed Optimizer
        optimizer = hvd.DistributedOptimizer(optimizer)
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        
        # Adding in callbacks
        callbacks = [
            # Horovod: broadcast initial variable states from rank 0 to all other processes.
            # This is necessary to ensure consistent initialization of all workers when
            # training is started with random weights or restored from a checkpoint.
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            
            # Horovod: average metrics among workers at the end of every epoch.
            # Note: This callback must be in the list before the ReduceLROnPlateau,
            # TensorBoard or other metrics-based callbacks.
            hvd.callbacks.MetricAverageCallback(),
            
            # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
            # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
            # the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
            hvd.callbacks.LearningRateWarmupCallback(initial_lr=cfg.learning_rate*hvd.size(), warmup_epochs=5, verbose=cfg.verbose),
            
            # Reduce the learning rate if training plateaus.
            tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", patience=10, verbose=2)
        ]
        
        # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
        if hvd.rank() == 0:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoint_dir.replace("dbfs:/", "/dbfs/"), 
                                                                monitor="loss", 
                                                                save_best_only=True))
            
        # Here we will do our actual training of the model on each node
        history = model.fit(train_ds, 
                            steps_per_epoch=steps_per_epoch,
                            epochs=cfg.epochs,
                            callbacks=callbacks,
                            verbose=cfg.verbose,
                            validation_data=cfg.validation_data
                           )
        
        # MLflow Tracking (Log only from Worker 0)
        if hvd.rank() == 0:    

            # Log events to MLflow
            with mlflow.start_run(run_id = active_run_id) as run:
                # Log MLflow Parameters
                mlflow.log_param("num_layers", len(model.layers))
                mlflow.log_param("optimizer_name", "Adam")
                mlflow.log_param("learning_rate", cfg.learning_rate)
                mlflow.log_param("batch_size", cfg.batch_size)
                mlflow.log_param("hvd_np", hvd_np)

                # Log MLflow Metrics
                mlflow.log_metric("train loss", history.history["loss"][-1])

                # Log Model
                mlflow.keras.log_model(model, "model")

# COMMAND ----------

# MAGIC %md Test it out on just the driver (negative sign indicates running on the driver).

# COMMAND ----------

from sparkdl import HorovodRunner

with mlflow.start_run() as run:  
    # Get active run_uuid
    active_run_id = mlflow.active_run().info.run_id
    hvd_np = -1
    hr = HorovodRunner(np=hvd_np, driver_log_verbosity="all")
    hr.run(run_training_horovod)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run on all workers

# COMMAND ----------

## OPTIONAL: You can enable Horovod Timeline as follows, but can incur slow down from frequent writes, and have to export out of Databricks to upload to chrome://tracing
# import os
# os.environ["HOROVOD_TIMELINE"] = f"{DA.paths.working_dir}/_timeline.json"

with mlflow.start_run() as run:  
    
    # Get active run_uuid
    active_run_id = mlflow.active_run().info.run_id
    hvd_np = spark.sparkContext.defaultParallelism
    hr = HorovodRunner(np=hvd_np, driver_log_verbosity="all")
    hr.run(run_training_horovod)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Loading Model and Evaluation
# MAGIC 
# MAGIC Since we included the Normalization layer inside the model, we can now use this model in production without having to worry about normalization again. 

# COMMAND ----------

from tensorflow.keras.models import load_model

trained_model = load_model(checkpoint_dir.replace("dbfs:/", "/dbfs/"))
print(trained_model.summary())

trained_model.evaluate(X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC Load model from MLflow run

# COMMAND ----------

from sklearn.metrics import mean_squared_error

model_uri = f"runs:/{run.info.run_id}/model"
mlflow_model = mlflow.pyfunc.load_model(model_uri)
y_pred = mlflow_model.predict(X_test)
mean_squared_error(y_test, y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Leveraging Spark for scalable inference using MLflow's Spark UDF

# COMMAND ----------

predict = mlflow.pyfunc.spark_udf(spark, model_uri)
display(test_df.withColumn("prediction", predict(*feature_cols)))

# COMMAND ----------

# MAGIC %md Let's <a href="https://petastorm.readthedocs.io/en/latest/api.html#petastorm.spark.spark_dataset_converter.SparkDatasetConverter.delete" target="_blank">delete</a> the cached files

# COMMAND ----------

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
