# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
np.set_printoptions(precision=2)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)
DA.reset_lesson()
DA.init()

DA.init_mlflow_as_job()

DA.paths.working_path = DA.paths.to_vm_path(DA.paths.working_dir)
DA.paths.datasets_path = DA.paths.to_vm_path(DA.paths.datasets)

DA.conclude_setup()

