# Databricks notebook source
import my_package.math

# COMMAND ----------

from my_package.math import squared
print(squared(4))

# COMMAND ----------

# MAGIC %run ./my_package/math

# COMMAND ----------

print(squared(4))

# COMMAND ----------

from my_package import strings

# COMMAND ----------

print(strings.concatenate("foo", "bar"))

# COMMAND ----------


