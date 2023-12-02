# Databricks notebook source
# !pip install faker
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from faker import Faker
from uuid import uuid1
from datetime import datetime, timedelta
import time

# COMMAND ----------

# inactivity level source: https://www.who.int/gho/ncd/risk_factors/physical_activity/en/
def gen_country_sample(active):
  COUNTRIES = pd.DataFrame([
      ("BRA", 0.4, 212),
      ("GHA", 0.19, 31),
      ("ITA", 0.36, 60),
      ("JPN", 0.33, 126),
      ("KEN", 0.14, 54),
      ("MYS", 0.34, 32),
      ("NZL", 0.39, 5),
      ("PRT", 0.37, 10),
      ("PRY", 0.38, 7),
      ("USA", 0.31, 331)
  ], columns=["country_code_alpha_3", "inactivity_level", "population"])
  COUNTRIES["activity_level"] = 1 - COUNTRIES.inactivity_level
  COUNTRIES["inactive_probs"] = COUNTRIES["inactivity_level"]*COUNTRIES["population"]
  COUNTRIES["inactive_probs"] = COUNTRIES["inactive_probs"]/np.sum(COUNTRIES["inactive_probs"])
  COUNTRIES["active_probs"] = COUNTRIES["activity_level"]*COUNTRIES["population"]
  COUNTRIES["active_probs"] = COUNTRIES["active_probs"]/np.sum(COUNTRIES["active_probs"])
  if active:
      return COUNTRIES.sample(weights=COUNTRIES["active_probs"]).country_code_alpha_3.values[0]
  return COUNTRIES.sample(weights=COUNTRIES["inactive_probs"]).country_code_alpha_3.values[0]

# COMMAND ----------

def generate_occupation():
  return np.random.choice(
    [
      "Tech-support",
      "Craft-repair",
      "Other-service",
      "Sales",
      "Exec-managerial",
      "Prof-specialty",
      "Handlers-cleaners",
      "Machine-op-inspct",
      "Adm-clerical",
      "Farming-fishing",
      "Transport-moving",
      "Priv-house-serv",
      "Protective-serv",
      "Armed-Forces"
    ]
  )

# COMMAND ----------

def gen_user_profile(ranges, cors, lifestyle):
  """Generates a single user of the platform."""
  active = True
  if lifestyle == "sedentary":
    active = False

  female = np.random.binomial(1, 0.5) == 1
  fake = Faker()
  means = np.array([(mx + mn)/2 for mx, mn in ranges])
  stds =  np.array([abs(mx - mn)/6 for mx, mn in ranges])
  covs = np.eye(4)
  covs[0,0] = stds[0]*stds[0]
  covs[1,1] = stds[1]*stds[1]
  covs[2,2] = stds[2]*stds[2]
  covs[3,3] = stds[3]*stds[3]
  covs[0,1] = cors[0]*covs[0,0]*covs[1,1]
  covs[1,0] = cors[0]*covs[0,0]*covs[1,1]
  covs[0,2] = cors[1]*covs[0,0]*covs[2,2]
  covs[2,0] = cors[1]*covs[0,0]*covs[2,2]
  covs[0,3] = cors[2]*covs[0,0]*covs[3,3]
  covs[3,0] = cors[2]*covs[0,0]*covs[3,3]
  covs[1,2] = cors[3]*covs[1,1]*covs[2,2]
  covs[2,1] = cors[3]*covs[1,1]*covs[2,2]
  covs[1,3] = cors[4]*covs[1,1]*covs[3,3]
  covs[3,1] = cors[4]*covs[1,1]*covs[3,3]
  covs[2,3] = cors[5]*covs[2,2]*covs[3,3]
  covs[3,2] = cors[5]*covs[2,2]*covs[3,3]
  data = list(np.random.multivariate_normal(means, covs))
  if female:
    data.append(fake.first_name_female())
  else:
    data.append(fake.first_name_male())
  data.append(fake.last_name())
  data.append(lifestyle)
  data.append(str(uuid1()))
  data.append(female)
  data.append(gen_country_sample(active))
  data.append(generate_occupation())
  return data

def gen_data(ranges, cors, lifestyle, size):
  data = pd.DataFrame([
    gen_user_profile(ranges, cors, lifestyle)
    for _ in range(size)
  ])
  data.columns = [
    "resting_heartrate",
    "active_heartrate",
    "BMI",
    "VO2_max",
    "first_name",
    "last_name",
    "lifestyle",
    "_id",
    "female",
    "country",
    "occupation"
  ]
  return data

def gen_month_year(year, month, data_df):
  start = datetime(year, month, 1)
  if month == 12:
    end = datetime(year+1, 1, 1)
  else:
    end = datetime(year, month+1, 1)
  n_days = (end-start).days
  data = None
  ["resting heartrate", "active heartrate", "BMI", "VO2_max"]

  for (
    rst_hr_mean,
    act_hr_mean,
    BMI_mean,
    VO2_max_mean,
    first_name,
    last_name,
    lifestyle,
    _id
  ) in data_df.to_records(index=False):
      dates = pd.Series([start + timedelta(days=i) for i in range(n_days)])
      tmp_data = pd.DataFrame({"dte" : dates})
      tmp_data["_id"] = _id
      tmp_data["first_name"] = first_name
      tmp_data["last_name"] = last_name
      tmp_data["resting_heartrate"] = np.random.normal(rst_hr_mean, 7, size=n_days)
      tmp_data["active_heartrate"] = np.random.normal(act_hr_mean, 7, size=n_days)
      tmp_data["BMI"] = np.random.normal(BMI_mean, 0.04*BMI_mean, size=n_days)
      tmp_data["VO2_max"] = np.random.normal(VO2_max_mean, 0.01*VO2_max_sd, size=n_days)
      tmp_data["lifestyle"] = lifestyle
      if data is not None:
          data = pd.concat([data, tmp_data])
      else:
          data = tmp_data
  return data
