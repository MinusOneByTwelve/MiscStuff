# Databricks notebook source
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from pyspark.sql.functions import mean, col

# COMMAND ----------

# You can add %run ../../configuration here and run it if you would like to
# execute this notebook while you develop.

# Be sure to remove this line when you are done.

# COMMAND ----------

# TODO
# # instatiate a standard scaler and a one-hot encoder
# # the standard scaler should have no arguments
# # the one-hot encoder should have "sparse" set to False,
# # "drop" set to None, and "handle_unknown" set to "ignore"
#
# ss = FILL_THIS_IN
# ohe = FILL_THIS_IN

# COMMAND ----------

# ANSWER
ss = StandardScaler()
ohe = OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore')

# COMMAND ----------

# TODO
# # Create Spark Reference to Tables
#
# # Reference these tables:
# # - user_profile_data
# # - health_profile_data
#
# user_profile_df = spark.read FILL_THIS_IN
# health_profile_df = spark.read FILL_THIS_IN

# COMMAND ----------

# ANSWER
# Create Spark Reference to Tables
user_profile_df = spark.read.table("user_profile_data")
health_profile_df = spark.read.table("health_profile_data")

# COMMAND ----------

# TODO
# # Sample Users and Join to Health Profile
#
# # Your sample should be a 10% sample
# #
# # Join the Sample DataFrame to the health_profile_df on "_id"
#
# user_profile_sample_df = user_profile_df.sample FILL_THIS_IN
#
# health_profile_sample_df = (
#   user_profile_sample_df
#   .join FILL_THIS_IN
# )

# COMMAND ----------

# ANSWER
# Sample Users and Join to Health Profile
user_profile_sample_df = user_profile_df.sample(0.1)
health_profile_sample_df = (
  user_profile_sample_df
  .join(health_profile_df, "_id")
)

# COMMAND ----------

assert 365*user_profile_sample_df.count() == health_profile_sample_df.count()

# COMMAND ----------

# TODO
# # Aggregate Over Daily Profile Data
# # You should perform the following aggregations:
# # - mean `BMI` aliased to `mean_BMI`
# # - mean `active_heartrate` aliased to `mean_active_heartrate`
# # - mean `resting_heartrate` aliased to `mean_resting_heartrate`
# # - mean `VO2_max` aliased to `mean_VO2_max`
# # - mean `workout_minutes` aliased to `mean_workout_minutes`
#
# from pyspark.sql.functions import mean, col
#
# health_tracker_sample_agg_df = (
#     health_profile_sample_df.groupBy("_id")
#     .agg(
#        FILL_THIS_IN
#     )
# )

# COMMAND ----------

# ANSWER
# Aggregate Over Daily Profile Data
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

# TODO
# # Join the Aggregate Data to User Data to Augment with Categorical Features
# # 1. Join the two dataframes, `health_tracker_sample_agg_df` and `user_profile_df`
# # 2. Perform the join using the `_id` column.
# health_tracker_augmented_df = FILL_THIS_IN

# COMMAND ----------

# ANSWER
health_tracker_augmented_df = (
  health_tracker_sample_agg_df
  .join(user_profile_df, "_id")
)

# COMMAND ----------

# TODO
# # Select Salient Columns
#
# # Select only the following features, in this order:
# # - `mean_BMI`
# # - `mean_active_heartrate`
# # - `mean_resting_heartrate`
# # - `mean_VO2_max`
# # - `mean_workout_minutes`
# # - `female`
# # - `country`
# # - `occupation`
# # - `lifestyle`
#
# health_tracker_augmented_df = (
#   health_tracker_augmented_df
#     FILL_THIS_IN
# )

# COMMAND ----------

# ANSWER
# Select Salient Columns
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

# TODO
# # Convert the Augmented Spark DataFrame to a Pandas DataFrame
# health_tracker_augmented_pandas_df = health_tracker_augmented_df FILL_THIS_IN

# COMMAND ----------

# ANSWER
# Convert the Augmented Spark DataFrame to a Pandas DataFrame
health_tracker_augmented_pandas_df = health_tracker_augmented_df.toPandas()

# COMMAND ----------

# TODO
# # Prepare Feature and Target
#
# # 1. Create the features dataframe by dropping the "lifestyle" column along axis 1
# # 2. Create the target Vector from the "lifestyle" column
# # 3. Label encode the target vector
#
# from sklearn.preprocessing import LabelEncoder
#
# features = health_tracker_augmented_pandas_df.drop(FILL_THIS_IN)
# target = health_tracker_augmented_pandas_df["lifestyle"]
# le = FILL_THIS_IN
# target = le.FILL_THIS_IN(target)

# COMMAND ----------

# ANSWER
# Prepare Feature and Target
from sklearn.preprocessing import LabelEncoder

features = health_tracker_augmented_pandas_df.drop("lifestyle", axis=1)
target = health_tracker_augmented_pandas_df["lifestyle"]
le = LabelEncoder()
target = le.fit_transform(target)

# COMMAND ----------

# Perform Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, target)

# COMMAND ----------

# TODO
# # Split Data into Numerical and Categorical Sets
#
# # 1. Prepare the numerical sets by excluding the "object" type
# # 2. Prepare the categorical sets by including the "object" type
#
# X_train_numerical = X_train.select_dtypes(exclude=[FILL_THIS_IN])
# X_test_numerical = X_test.select_dtypes(exclude=[FILL_THIS_IN])
# X_train_categorical = X_train.select_dtypes(include=[FILL_THIS_IN])
# X_test_categorical = X_test.select_dtypes(include=[FILL_THIS_IN])

# COMMAND ----------

# ANSWER
# Split Data into Numerical and Categorical Sets
X_train_numerical = X_train.select_dtypes(exclude=["object"])
X_test_numerical = X_test.select_dtypes(exclude=["object"])
X_train_categorical = X_train.select_dtypes(include=["object"])
X_test_categorical = X_test.select_dtypes(include=["object"])

# COMMAND ----------

# TODO
# # Create One-Hot Encoded Categorical DataFrames
#
# # 1. use the ohe to convert the train and test categorical features
#
# X_train_ohe = pd.DataFrame(
#   ohe.fit_transform(FILL_THIS_IN),
#   columns=ohe.get_feature_names(),
#   index=X_train_numerical.index
# )
# X_test_ohe = pd.DataFrame(
#   ohe.transform(FILL_THIS_IN),
#   columns=ohe.get_feature_names(),
#   index=X_test_numerical.index
# )

# COMMAND ----------

# ANSWER
# Create One-Hot Encoded Categorical DataFrames
X_train_ohe = pd.DataFrame(
  ohe.fit_transform(X_train_categorical),
  columns=ohe.get_feature_names(),
  index=X_train_numerical.index
)
X_test_ohe = pd.DataFrame(
  ohe.transform(X_test_categorical),
  columns=ohe.get_feature_names(),
  index=X_test_numerical.index
)

# COMMAND ----------

# TODO
# # Merge Numerical and One-Hot Encoded Categorical
#
# # 1. Merge the training numerical data to the training ohe data
# # 2. Merge the testing numerical data to the testing ohe data
#
# X_train = FILL_THIS_IN.merge(FILL_THIS_IN, left_index=True, right_index=True)
# X_test = FILL_THIS_IN.merge(FILL_THIS_IN, left_index=True, right_index=True)

# COMMAND ----------

# ANSWER
# Merge Numerical and One-Hot Encoded Categorical
X_train = X_train_numerical.merge(X_train_ohe, left_index=True, right_index=True)
X_test = X_test_numerical.merge(X_test_ohe, left_index=True, right_index=True)

# COMMAND ----------

# Standardize Data
X_train = pd.DataFrame(
  ss.fit_transform(X_train),
  index=X_train_ohe.index,
  columns=X_train.columns)
X_test = pd.DataFrame(
  ss.transform(X_test),
  index=X_test_ohe.index,
  columns=X_train.columns)
