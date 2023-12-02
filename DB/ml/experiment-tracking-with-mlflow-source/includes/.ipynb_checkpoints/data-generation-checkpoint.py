# Databricks notebook source
%run ./data-generation-utilities

# COMMAND ----------

# ranges for each attribute, correlations between attributes
lft, lft_cors = (
  ((56,61), (98,138), (21,28), (31,37)),
  (0.8, -0.4, 0.3, 0.4, -0.5, -0.2)
)
rce, rce_cors = (
  ((49,53), (87,120), (16,23), (35,42)),
  (0.8, -0.6, -0.5, -0.5, -0.6, -0.5)
)
sed, sed_cors = (
  ((73,82), (115,162), (21,28), (21,28)),
  (0.8, -0.8, -0.7, -0.6, -0.6, -0.4)
)

# COMMAND ----------

n = 1000
X_lft = gen_data(lft, lft_cors, "weight trainer", n)
X_rce = gen_data(rce, rce_cors, "cardio trainer", n)
X_sed = gen_data(sed, sed_cors, "sedentary", n)
full_data_df = pd.concat([
    X_lft,
    X_rce,
    X_sed
])
full_data_df = full_data_df.reset_index(drop=True)

# COMMAND ----------

data_2019_01 = gen_month_year(2019,  1, full_data_df)
data_2019_02 = gen_month_year(2019,  2, full_data_df)
data_2019_03 = gen_month_year(2019,  3, full_data_df)
data_2019_04 = gen_month_year(2019,  4, full_data_df)
data_2019_05 = gen_month_year(2019,  5, full_data_df)
data_2019_06 = gen_month_year(2019,  6, full_data_df)
data_2019_07 = gen_month_year(2019,  7, full_data_df)
data_2019_08 = gen_month_year(2019,  8, full_data_df)
data_2019_09 = gen_month_year(2019,  9, full_data_df)
data_2019_10 = gen_month_year(2019, 10, full_data_df)
data_2019_11 = gen_month_year(2019, 11, full_data_df)
data_2019_12 = gen_month_year(2019, 12, full_data_df)

user_data = data_2019_09[["_id", "first_name", "last_name", "lifestyle"]].drop_duplicates()
