# Databricks notebook source
course_name = "smlp"

username = spark.sql("SELECT current_user()").first()[0].lower()

base_read_path = "wasbs://courseware@dbacademy.blob.core.windows.net/machine-learning-engineering-with-databricks/scaling-machine-learning-pipelines/v01"

london_read_path = f"{base_read_path}/london"
tokyo_read_path =  f"{base_read_path}/tokyo"

base_write_path = f"dbfs:/user/{username}/dbacademy/{course_name}"
dbutils.fs.rm(base_write_path, True)
dbutils.fs.mkdirs(base_write_path)

# Lesson 2 paths
lesson_2_path =                      f"{london_read_path}/london-listings-2021-01-31-lesson-2"
lesson_2_train_path =                f"{base_write_path}/lesson_2_train_df"
lesson_2_test_path =                 f"{base_write_path}/lesson_2_test_df"
lesson_2_train_prepared_path =       f"{base_write_path}/lesson_2_train_prepared_df"
lesson_2_train_feature_vector_path = f"{base_write_path}/lesson_2_train_feature_vector_df"

lab_2_path =                      f"{tokyo_read_path}/tokyo-listings-2021-01-31-lab-2"
lab_2_train_path =                f"{base_write_path}/lab_2_train_df"
lab_2_test_path =                 f"{base_write_path}/lab_2_test_df"
lab_2_train_prepared_path =       f"{base_write_path}/lab_2_train_prepared_df"
lab_2_train_feature_vector_path = f"{base_write_path}/lab_2_train_feature_vector_df"

# Lesson 3 paths
lesson_3_train_feature_vector_path = f"{london_read_path}/london-listings-2021-01-31-lesson-3-train-feature-vector"
lesson_3_train_path =                f"{london_read_path}/london-listings-2021-01-31-lesson-3-train"
lesson_3_test_path =                 f"{london_read_path}/london-listings-2021-01-31-lesson-3-test"
lesson_3_train_tree_path =           f"{london_read_path}/london-listings-2021-01-31-lesson-3-tree-train"
lesson_3_train_tree_cat_path =       f"{london_read_path}/london-listings-2021-01-31-lesson-3-train-tree-train-cat"
lesson_3_train_path_imp =            f"{london_read_path}/london-listings-2021-01-31-lesson-3-train-imputed"
lesson_3_test_path_imp =             f"{london_read_path}/london-listings-2021-01-31-lesson-3-test-imputed"

lab_3_train_path = f"{tokyo_read_path}/tokyo-listings-2021-01-31-lab-3-train"
lab_3_test_path =  f"{tokyo_read_path}/tokyo-listings-2021-01-31-lab-3-test"

# Lesson 4 paths
lesson_4_train_path = f"{london_read_path}/london-listings-2021-01-31-lesson-4-train"
lesson_4_test_path =  f"{london_read_path}/london-listings-2021-01-31-lesson-4-test"

lab_4_train_path = f"{tokyo_read_path}/tokyo-listings-2021-01-31-lab-4-train"
lab_4_test_path =  f"{tokyo_read_path}/tokyo-listings-2021-01-31-lab-4-test"

# Lesson 5 paths
lesson_5_model_path =     f"{london_read_path}/london-listings-2021-01-31-lesson-5-model"
lesson_5_inference_path = f"{london_read_path}/london-listings-2021-01-31-lesson-5-inference"

lab_5_model_path =     f"{tokyo_read_path}/tokyo-listings-2021-01-31-lab-5-model"
lab_5_inference_path = f"{tokyo_read_path}/tokyo-listings-2021-01-31-lab-5-inference"

displayHTML("Initialized classroom variables & functions...")

