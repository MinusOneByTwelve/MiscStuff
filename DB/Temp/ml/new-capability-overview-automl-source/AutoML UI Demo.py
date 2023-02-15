# Databricks notebook source
# MAGIC %md
# MAGIC # Creating an AutoML Experiment - UI
# MAGIC AutoML can be used both via the user interface and via a Python-based API.
# MAGIC 
# MAGIC In this demonstration, we're going to develop a baseline model using the user interface.
# MAGIC 
# MAGIC ##### Objectives
# MAGIC 1. Navigate to AutoML UI
# MAGIC 1. Set up and run an experiment
# MAGIC 1. Evaluate the results
# MAGIC 1. View the best model

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC 
# MAGIC First, we'll run the `Classroom-Setup` notebook to set up our environment.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup-UI" $mode=reset

# COMMAND ----------

# MAGIC %md
# MAGIC ### Navigate to AutoML UI
# MAGIC 
# MAGIC To do this, start by clicking on the **Experiments** tab in the left sidebar of the Databricks Machine Learning platform.
# MAGIC 
# MAGIC Next, click on the **Start AutoML Experiment** button to begin a new AutoML Experiment.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set up an experiment
# MAGIC 
# MAGIC When prompted, enter the details below – be sure to look for the table in your **own** database!
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/idbml/automl-1.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Run the experiment
# MAGIC 
# MAGIC When you're ready to run the experiment, click **Start AutoML**. At this point, AutoML will start automatically generating models to predict price based on the features in the feature table.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/idbml/automl-2.png">
# MAGIC 
# MAGIC As the experiment runs, you will start to see MLflow runs appear in the experiment page indicating the models have been completed.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/idbml/automl-3.png">
# MAGIC 
# MAGIC You will also see the **Stop Experiment** button. You can stop the AutoML process by clicking this button at any time.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/idbml/automl-4.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluate the Results
# MAGIC 
# MAGIC AutoML will automatically evaluate each of your models. You can view each of these model's results in the experiment page. AutoML will also provide an easy link for you to view the notebook that generated the best model.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/idbml/automl-5.png">

# COMMAND ----------

# MAGIC %md
# MAGIC ### View the best model
# MAGIC 
# MAGIC When you view the notebook for the best model, you're able to copy code, edit code, and even clone the exact notebook into your production workflow.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/idbml/automl-6.png">
