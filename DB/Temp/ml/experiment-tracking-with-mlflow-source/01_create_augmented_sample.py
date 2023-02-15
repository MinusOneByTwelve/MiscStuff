# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Create Augmented Sample

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %md ## Define Spark References to Data
# MAGIC
# MAGIC In the next cell, we use Apache Spark to define a reference to the data
# MAGIC we will be working with.
# MAGIC
# MAGIC We need to create references to the following Delta tables:
# MAGIC
# MAGIC - `user_profile_data`
# MAGIC - `health_profile_data`
# MAGIC

# COMMAND ----------
# TODO
# # Use spark.read to create references to the two tables as dataframes

# user_profile_df = spark.read FILL_THIS_IN
# health_profile_df = spark.read FILL_THIS_IN

# COMMAND ----------
# ANSWER
# Use spark.read to create references to the two tables as dataframes

user_profile_df = spark.read.table("user_profile_data")
health_profile_df = spark.read.table("health_profile_data")

# COMMAND ----------

# SOURCE_ONLY
user_profile_df = spark.read.format("delta").load(dimUserPath)
health_profile_df = spark.read.format("delta").load(silverDailyPath)

# COMMAND ----------

# MAGIC %md ### Generate a Sample of Users

# COMMAND ----------

user_profile_sample_df = user_profile_df.sample(0.1)

display(user_profile_sample_df.groupby("lifestyle").count())

# COMMAND ----------

# MAGIC %md ### Join the User Profile Data to the Health Profile Data
# MAGIC
# MAGIC 1. Join the two dataframes, `user_profile_sample_df` and `health_profile_df`
# MAGIC 1. Perform the join using the `_id` column.
# MAGIC
# MAGIC If successful, You should have 365 times as many rows as are in the user sample.

# COMMAND ----------

# TODO
# health_profile_sample_df = (
#   FILL_THIS_IN
# )
#
# assert 365*user_profile_sample_df.count() == health_profile_sample_df.count()

# COMMAND ----------

# ANSWER
health_profile_sample_df = (
  user_profile_sample_df
  .join(health_profile_df, "_id")
)

assert 365*user_profile_sample_df.count() == health_profile_sample_df.count()

# COMMAND ----------

# MAGIC %md ## Aggregate the Data to Generate Numerical Features
# MAGIC
# MAGIC You should perform the following aggregations:
# MAGIC
# MAGIC - mean `BMI` aliased to `mean_BMI`
# MAGIC - mean `active_heartrate` aliased to `mean_active_heartrate`
# MAGIC - mean `resting_heartrate` aliased to `mean_resting_heartrate`
# MAGIC - mean `VO2_max` aliased to `mean_VO2_max`
# MAGIC - mean `workout_minutes` aliased to `mean_workout_minutes`

# COMMAND ----------

# TODO
# from pyspark.sql.functions import mean, col
#
# health_tracker_sample_agg_df = (
#     health_profile_sample_df.groupBy("_id")
#     .agg(
#         FILL_THIS_IN
#     )
# )

# COMMAND ----------

# ANSWER
from pyspark.sql.functions import mean, col

health_tracker_sample_agg_df = (
    health_profile_sample_df.groupBy("_id")
    .agg(
        mean("BMI").alias("mean_BMI"),
        mean("active_heartrate").alias("mean_active_heartrate"),
        mean("resting_heartrate").alias("mean_resting_heartrate"),
        mean("VO2_max").alias("mean_VO2_max"),
        mean("workout_minutes").alias("mean_workout_minutes")
    )
)

# COMMAND ----------

# MAGIC %md ### Join the Aggregate Data to User Data to Augment with Categorical Features
# MAGIC 1. Join the two dataframes, `health_tracker_sample_agg_df` and `user_profile_df`
# MAGIC 1. Perform the join using the `_id` column.

# COMMAND ----------

# TODO
# health_tracker_augmented_df = FILL_THIS_IN

# COMMAND ----------

# ANSWER
health_tracker_augmented_df = (
  health_tracker_sample_agg_df
  .join(user_profile_df, "_id")
)

# COMMAND ----------

# MAGIC %md ### Select only the following features, in this order:
# MAGIC
# MAGIC - `mean_BMI`
# MAGIC - `mean_active_heartrate`
# MAGIC - `mean_resting_heartrate`
# MAGIC - `mean_VO2_max`
# MAGIC - `mean_workout_minutes`
# MAGIC - `female`
# MAGIC - `country`
# MAGIC - `occupation`
# MAGIC - `lifestyle`

# COMMAND ----------

# TODO
# health_tracker_augmented_df = (
#   health_tracker_augmented_df
#     FILL_THIS_IN
# )

# COMMAND ----------

# ANSWER
health_tracker_augmented_df = (
  health_tracker_augmented_df
  .select(
    "mean_BMI",
    "mean_active_heartrate",
    "mean_resting_heartrate",
    "mean_VO2_max",
    "mean_workout_minutes",
    "female",
    "country",
    "occupation",
    "lifestyle"
  )
)

# COMMAND ----------

# MAGIC %md ##### Run this Assertion to Verify The Schema

# COMMAND ----------

from pyspark.sql.types import _parse_datatype_string

augmented_schema = """
  mean_BMI double,
  mean_active_heartrate double,
  mean_resting_heartrate double,
  mean_VO2_max double,
  mean_workout_minutes double,
  female boolean,
  country string,
  occupation string,
  lifestyle string
"""

assert health_tracker_augmented_df.schema == _parse_datatype_string(augmented_schema)

# COMMAND ----------

# MAGIC %md ### Write the Augmented Dataframe to a Delta Table
# MAGIC
# MAGIC Use the following path: `goldPath + "health_tracker_augmented"`

# COMMAND ----------

# TODO
# (
#   health_tracker_augmented_df.write
#   .format("delta")
#   .mode("overwrite")
#   FILL_THIS_IN
# )

# COMMAND ----------

# ANSWER
(
  health_tracker_augmented_df.write
  .format("delta")
  .mode("overwrite")
  .save(goldPath + "health_tracker_augmented")
)
