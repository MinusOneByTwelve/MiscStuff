# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Exploring the Augmented Sample Data

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------
# MAGIC %run ./includes/configuration

# COMMAND ----------

# MAGIC %md ## Load the Sample Data as a Pandas DataFrame
# MAGIC
# MAGIC Recall that we wrote the sample data as a Delta table to
# MAGIC the path, `goldPath + "health_tracker_augmented"`.
# MAGIC
# MAGIC 1. Use `spark.read` to read the Delta table as a Spark DataFrame.
# MAGIC 2. Use the `.toPandas()` DataFrame method to load the Spark
# MAGIC    DataFrame as a Pandas DataFrame.

# COMMAND ----------

# TODO
# health_tracker_augmented_df = (
#   spark.read
#   .format("delta")
#   FILL_THIS_IN
# )

# COMMAND ----------

# ANSWER
health_tracker_augmented_df = (
  spark.read
  .format("delta")
  .load(goldPath + "health_tracker_augmented")
  .toPandas()
)

# COMMAND ----------

# MAGIC %md ### Load Scipy Libraries

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# COMMAND ----------

# MAGIC %md ### Display the Unique Lifestyles
# MAGIC
# MAGIC 😺 Remember, Pandas DataFrames use the `.unique()` method to do this.
# MAGIC Spark DataFrames use the `.distinct()` method.
# MAGIC
# MAGIC Make sure to specify the correct column, `lifestyle`.

# COMMAND ----------

# TODO
# lifestyles = health_tracker_augmented_df FILL_THIS_IN
# lifestyles

# COMMAND ----------

# ANSWER
lifestyles = health_tracker_augmented_df.lifestyle.unique()
lifestyles

# COMMAND ----------

# MAGIC %md ### Create Feature and Target Objects

# COMMAND ----------

features = health_tracker_augmented_df.drop("lifestyle", axis=1)
target = health_tracker_augmented_df[["lifestyle"]].copy()

# COMMAND ----------

# MAGIC %md ### Display a `.sample()` of the Features DataFrame

# COMMAND ----------

features.sample(10)

# COMMAND ----------

# MAGIC %md ### Display the `.dtypes` of the Features DataFrame

# COMMAND ----------

features.dtypes

# COMMAND ----------

# TODO
# features_numerical = features.select_dtypes(include=[FILL_THIS_IN])
# features_categorical = features.select_dtypes(exclude=[FILL_THIS_IN])

# COMMAND ----------

# ANSWER
features_numerical = features.select_dtypes(include=[float])
features_categorical = features.select_dtypes(exclude=[float])

# COMMAND ----------


# MAGIC %md ### Use `seaborn` to Display a Distribution Plot for Each Feature On the Same Scale
# MAGIC
# MAGIC 1. Generate a `distplot` for each feature.
# MAGIC 1. Set the `xlim` for each axis of the subplot to `0,250`

# COMMAND ----------

# TODO
# fig, ax = plt.subplots(1,5, figsize=(25,5))
#
# for i, feature in enumerate(features_numerical):
#   sns.FILL_THIS_IN(features[feature], ax=ax[i])
#   ax[i].set_xlim(FILL_THIS_IN)

# COMMAND ----------

# ANSWER
fig, ax = plt.subplots(1,5, figsize=(25,5))

for i, feature in enumerate(features_numerical):
  sns.distplot(features[feature], ax=ax[i])
  ax[i].set_xlim(0,250)

# COMMAND ----------

# MAGIC %md ### Use `seaborn` to Display a Scaled Distribution Plot for Each Feature
# MAGIC
# MAGIC 1. Create the scaled series by subtracting the mean and dividing by the standard deviation
# MAGIC 2. Generate a `distplot` for each feature.
# MAGIC 3. Set the `xlim` for each axis of the subplot to `-5, 5`
# MAGIC
# MAGIC 🧑🏼‍🎤 This can also be done using `sklearn.preprocessing.StandardScaler`
# MAGIC
# MAGIC e.g. `ss = StandardScaler()`
# MAGIC      `feature_scaled = ss.fit_transform(feature_series)`

# COMMAND ----------

# TODO
# fig, ax = plt.subplots(1,5, figsize=(25,5))
#
# for i, feature in enumerate(features_numerical):
#   feature = features[feature]
#   feature_scaled = (feature - FILL_THIS_IN)/FILL_THIS_IN
#   sns.FILL_THIS_IN(feature_scaled, ax=ax[i])
#   ax[i].set_xlim(FILL_THIS_IN)

# COMMAND ----------

# ANSWER
fig, ax = plt.subplots(1,5, figsize=(25,5))

for i, feature in enumerate(features_numerical):
  feature_series = features[feature]
  feature_scaled = (feature_series - feature_series.mean())/feature_series.std()
  sns.distplot(feature_scaled, ax=ax[i])
  ax[i].set_xlim(-5, 5)

# COMMAND ----------

# MAGIC %md ### Use `seaborn` to Display a Distribution Plot for Each Feature, Colored by Lifestyle
# MAGIC
# MAGIC 1. Filter on the `lifestyle` column from the `target` DataFrame.
# MAGIC 1. Set this column equal to the `lifestyle` variable in the for-loop
# MAGIC 1. Generate a `distplot` for each feature.

# COMMAND ----------

# TODO
# fig, ax = plt.subplots(1,5, figsize=(25,5))
#
# for i, feature in enumerate(features_numerical):
#   for lifestyle in lifestyles:
#     subset = features[target[FILL_THIS_IN] == FILL_THIS_IN]
#     sns.FILL_THIS_IN(subset[feature], ax=ax[i], label=lifestyle)
#   ax[i].legend()

# COMMAND ----------

# ANSWER
fig, ax = plt.subplots(1,5, figsize=(25,5))

for i, feature in enumerate(features_numerical):
  for lifestyle in lifestyles:
    subset = features[target["lifestyle"] == lifestyle]
    sns.distplot(subset[feature], ax=ax[i], label=lifestyle)
  ax[i].legend()

# COMMAND ----------

# MAGIC %md ### Use `seaborn` to Display a Distribution Plot for Resting Heart Rate, Colored by Categorical Feature
# MAGIC
# MAGIC 1. exclude the numerical columns when selecting `dtypes`
# MAGIC 2. use the seaborn `distplot`

# COMMAND ----------

# TODO
# fig, ax = plt.subplots(1,3, figsize=(27,5))
#
# for i, feature in enumerate(features_categorical):
#   for value in features[feature].unique():
#     subset = features[features[feature] == value]
#     sns.FILL_THIS_IN(subset["mean_resting_heartrate"], ax=ax[i], label=value)
#   ax[i].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

# COMMAND ----------

# ANSWER
fig, ax = plt.subplots(1,3, figsize=(27,5))

for i, feature in enumerate(features_categorical):
  for value in features[feature].unique():
    subset = features[features[feature] == value]
    sns.distplot(subset["mean_resting_heartrate"], ax=ax[i], label=value)
  ax[i].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

# COMMAND ----------

# MAGIC %md ### Use Pandas Plotting to Display a Bar Plot for Each Categorical Feature by Lifestyle
# MAGIC
# MAGIC 1. Use 1 row of 3 subplots
# MAGIC 1. Group the the three subplots by the columns `female`, `country`, and `occupation`.
# MAGIC 1. Use plot kind, `bar` and each subplot to axes, `0`, `1`, and `2`

# COMMAND ----------

# TODO
# fig, ax = plt.subplots(FILL_THIS_IN, FILL_THIS_IN, figsize=(27,5))
# (
#   health_tracker_augmented_df
#   .groupby(FILL_THIS_IN)
#   .lifestyle.value_counts()
#   .unstack(0)
#   .plot(kind=FILL_THIS_IN, ax=ax[FILL_THIS_IN])
#   .legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
# )
# (
#   health_tracker_augmented_df
#   .groupby(FILL_THIS_IN)
#   .lifestyle.value_counts()
#   .unstack(0)
#   .plot(kind=FILL_THIS_IN, ax=ax[FILL_THIS_IN])
#   .legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
# )
# (
#   health_tracker_augmented_df
#   .groupby(FILL_THIS_IN)
#   .lifestyle.value_counts()
#   .unstack(0)
#   .plot(kind=FILL_THIS_IN, ax=ax[FILL_THIS_IN])
#   .legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
# )

# COMMAND ----------

# ANSWER
fig, ax = plt.subplots(1, 3, figsize=(27,5))
(
  health_tracker_augmented_df
  .groupby("female")
  .lifestyle.value_counts()
  .unstack(0)
  .plot(kind="bar", ax=ax[0]).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
)
(
  health_tracker_augmented_df
  .groupby("country")
  .lifestyle.value_counts()
  .unstack(0)
  .plot(kind="bar", ax=ax[1]).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
)
(
  health_tracker_augmented_df
  .groupby("occupation")
  .lifestyle.value_counts()
  .unstack(0)
  .plot(kind="bar", ax=ax[2]).legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
)

# COMMAND ----------

# MAGIC %md ### One-Hot Encoding
# MAGIC
# MAGIC Just as we have to numerically encode our target using
# MAGIC `sklearn.preprocessing.LabelEncoder`,  in order to use our categorical
# MAGIC features — encoded as strings — we are going to need to apply preprocessing.
# MAGIC
# MAGIC Categorical features require special handling. They can not be simply
# MAGIC converted to numbers. Rather, we will need to convert them to one-hot encoded
# MAGIC columns.

# COMMAND ----------

# MAGIC %md ### Extract Numerical and Categorical Features
# MAGIC
# MAGIC 1. Prepare the numerical sets by excluding the "object" type
# MAGIC 2. Prepare the categorical sets by including the "object" type

# COMMAND ----------

# MAGIC %md ### Use `pandas.get_dummies`
# MAGIC
# MAGIC We use this here for visualization. When we go to our experiment, we will
# MAGIC need finer control and will use `sklearn.preprocessing.OneHotEncoder`.

# COMMAND ----------

pd.get_dummies(features_categorical)

# COMMAND ----------

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False, drop=None, handle_unknown='ignore')
ohe.fit_transform(features_categorical)
