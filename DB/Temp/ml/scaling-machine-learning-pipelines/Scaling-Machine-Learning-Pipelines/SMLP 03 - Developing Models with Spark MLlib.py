# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Developing Models with Spark MLlib
# MAGIC 
# MAGIC In this notebook, we'll demonstrate how scale machine learning model development using Spark MLlib.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup
# MAGIC 
# MAGIC Run the classroom-setup notebook to initialize all of our variables and load our course data.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> You will need to run this in every notebook of the course.

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Basic Spark MLlib Model Workflow
# MAGIC 
# MAGIC In this part of the demo, we will demonstrate a basic workflow to develop models using the Spark MLlib API.
# MAGIC 
# MAGIC #### Load prepared data
# MAGIC 
# MAGIC First, we are going to load the data we prepared in the last lab. Remember that this is one-hot encoded, imputed, and has a features column.

# COMMAND ----------

feature_vector_train_df = spark.read.format("delta").load(lesson_3_train_feature_vector_path)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Training
# MAGIC 
# MAGIC Next, we're going to train our model. Each modeling algorithm in Spark MLlib is an **estimator**, so we need to first instantiate it with the necessary arguments and then call its `fit` method on the training DataFrame.
# MAGIC 
# MAGIC In this case, we're going to use the [**`LinearRegression`** class](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.LinearRegression.html#pyspark.ml.regression.LinearRegression) to predict the `price` target variable.

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol="features", labelCol="price")
lr_model = lr.fit(feature_vector_train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Inference
# MAGIC 
# MAGIC In order to evaluate our model's quality, we need to first get the predictions for our training DataFrame.
# MAGIC 
# MAGIC We can use the [fitted **`LinearRegressionModel`**](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.LinearRegressionModel.html#pyspark.ml.regression.LinearRegressionModel) `lr_model` to complete this – it's a special kind of **transformer** called a **model**, so we can use its `transform` method on the training DataFrame.

# COMMAND ----------

predictions_train_df = lr_model.transform(feature_vector_train_df).select("price", "prediction")
display(predictions_train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Model Evaluation
# MAGIC 
# MAGIC Now that we have our predictions for our training DataFrame, we need to evaluate the quality of our model.
# MAGIC 
# MAGIC We can do this using an **evaluator** &mdash; another type of Spark MLlib.
# MAGIC 
# MAGIC There are a number of [types of evaluators in Spark MLlib](https://spark.apache.org/docs/latest/api/python/reference/pyspark.ml.html#evaluation), but three that are commonly used are:
# MAGIC 
# MAGIC 1. [The `RegressionEvaluator`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.RegressionEvaluator.html#pyspark.ml.evaluation.RegressionEvaluator)
# MAGIC 2. [The `BinaryClassificationEvaluator`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.RegressionEvaluator.html#pyspark.ml.evaluation.RegressionEvaluator)
# MAGIC 3. [The `MulticlassClassificationEvaluator`](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.evaluation.MulticlassClassificationEvaluator.html#pyspark.ml.evaluation.MulticlassClassificationEvaluator)
# MAGIC 
# MAGIC Since we are working on a regression problem here, we'll use the `RegressionEvaluator` class. We can set a specific evaluation metric and use the `evaluate` method compute that evaluation metric.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price", metricName="rmse")
rmse = evaluator.evaluate(predictions_train_df)
r2 = evaluator.setMetricName("r2").evaluate(predictions_train_df)
print(f"The training RMSE = {rmse}")
print(f"The training R^2 = {r2}")

# COMMAND ----------

# MAGIC %md
# MAGIC You might be wondering why we aren't working with the test DataFrame at this point &mdash; we'll cover it now with the `Pipeline` class.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Spark MLlib Pipeline API
# MAGIC 
# MAGIC In this part of the demo, we'll demonstrate how to use the [Spark MLlib Pipeline API](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Pipeline.html#pyspark.ml.Pipeline). We can use Pipelines for data preparation stages only, or we can create full-on modeling pipelines.
# MAGIC 
# MAGIC To utilize the Pipeline API, we must first instantiate the necessary estimator/transformer objects. 
# MAGIC 
# MAGIC We have three key classes we're going to use:
# MAGIC 
# MAGIC 1. `Imputer` &mdash; Imputing missing values
# MAGIC 2. `VectorAssembler` &mdash; Assembling a feature vector
# MAGIC 3. `LinearRegression` &mdash; Training a linear regression model
# MAGIC 
# MAGIC First, we'll load in our unprepared training and test sets &mdash; note that these versions only contain numeric features.

# COMMAND ----------

train_df = spark.read.format("delta").load(lesson_3_train_path)
test_df = spark.read.format("delta").load(lesson_3_test_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we'll create the imputer. **Notice that we aren't fitting the imputer here.**
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> We still need to programmatically determine which feature columns we should use!

# COMMAND ----------

from pyspark.sql.functions import col, count, when
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import Imputer

# Identify columns with missing values
double_cols = [column.name for column in train_df.schema.fields if column.dataType == DoubleType() and column.name != "price"]
missing_values_logic = [count(when(col(column).isNull(), column)).alias(column) for column in double_cols]
row_dict = train_df.select(missing_values_logic).first().asDict()
missing_cols = [column for column in row_dict if row_dict[column] > 0]

# Create the imputer
imputer = Imputer(strategy="median", inputCols=missing_cols, outputCols=missing_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we'll create the vector assembler. Again, notice that we are not transforming the data here.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Create the vector assembler
feature_cols = [column.name for column in train_df.schema.fields if column.dataType == DoubleType() and column.name != "price"]
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# COMMAND ----------

# MAGIC %md
# MAGIC And finally, we create the untrained linear regression estimator.

# COMMAND ----------

# Create linear regression
lr = LinearRegression(featuresCol="features", labelCol="price")

# COMMAND ----------

# MAGIC %md
# MAGIC Once we have each of our untrained estimators and transformers created, we can put them into stages as a Spark MLlib Pipeline.
# MAGIC 
# MAGIC We can then call the [Pipeline's **`fit`** method](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.Pipeline.html#pyspark.ml.Pipeline.fit) on the training data to fit each of the estimators.

# COMMAND ----------

from pyspark.ml import Pipeline

pipeline = Pipeline(stages=[imputer, vector_assembler, lr])
pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC And then we can use the fit [PipelineModel](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.PipelineModel.html#pyspark.ml.PipelineModel) to transform our initial training DataFrame. 
# MAGIC 
# MAGIC This will pass our data through each of the fitted transformers, including the linear regression model.

# COMMAND ----------

train_preds_df = pipeline_model.transform(train_df).select("price", "prediction")
display(train_preds_df)

# COMMAND ----------

# MAGIC %md
# MAGIC We can also use the fit PipelineModel to transform the test data!

# COMMAND ----------

test_preds_df = pipeline_model.transform(test_df).select("price", "prediction")
display(test_preds_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distributed Decision Trees
# MAGIC 
# MAGIC Next, we'll investigate the implications of distributed algorithms by creating a decision tree using Spark MLlib.
# MAGIC 
# MAGIC Based on the prerequisites for this course, we're assuming that you are familiar with decision trees. If you need a refreshed, we recommend checking out the [R2D3's explanation](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/).
# MAGIC 
# MAGIC #### Load prepared data
# MAGIC 
# MAGIC We are going to load a version of our London-based listing data that's been prepared for tree-based models.
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Remember that tree-based models don't perform well with one-hot encoded features!

# COMMAND ----------

# Load data
tree_train_df = spark.read.format("delta").load(lesson_3_train_tree_path)

# Prepare Spark data
feature_cols = [column for column in tree_train_df.columns if column != "price"]
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
spark_tree_train_df = vector_assembler.transform(tree_train_df)

# Prepare Sklearn data
pandas_tree_train_df = tree_train_df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Spark MLlib vs. Scikit-learn
# MAGIC 
# MAGIC As we mentioned, sometimes learning algorithms differ when they're distributed vs. single-node. In order to demonstrate this, we're going to train a decision tree in Spark MLlib and Scikit-learn and compare their predictions.
# MAGIC 
# MAGIC We'll start with [Spark MLlib's **`DecisionTreeRegressor`**](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.DecisionTreeRegressor.html#pyspark.ml.regression.DecisionTreeRegressor).
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> Recognize that Spark MLlib's API workflow is similar across model types.

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

dtr = DecisionTreeRegressor(featuresCol="features", labelCol="price", maxDepth=10)
dtr_model = dtr.fit(spark_tree_train_df)

train_preds_sparkmllib = dtr_model.transform(spark_tree_train_df).select("price", "prediction")

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we'll build a single-node decision tree with [Scikit-learn's **`DecisionTreeRegressor`**](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html).
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/icon_note_24.png"/> For comparison, it's important to note that these runs have the same tree depth.

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

X = pandas_tree_train_df.drop("price", axis=1)
y = pandas_tree_train_df["price"]

dtr = DecisionTreeRegressor(max_depth=10)
dtr_model = dtr.fit(X, y)

train_preds_sklearn = dtr_model.predict(X)

# COMMAND ----------

# MAGIC %md
# MAGIC And once we have both models, we can compute the correlation between the predictions for each model.
# MAGIC 
# MAGIC **Question:** What do you think the correlation will be?

# COMMAND ----------

import pandas as pd

full_preds_df = train_preds_sparkmllib \
    .select(col("price").alias("Actual"), col("prediction").alias("Spark MLlib")) \
    .toPandas()

full_preds_df["Scikit-learn"] = train_preds_sklearn

display(full_preds_df.corr())

# COMMAND ----------

# MAGIC %md
# MAGIC There's a strong correlation &mdash; but they aren't identical. 
# MAGIC 
# MAGIC There could be a few reasons for this, but a big one is that **tweaks have been made to the distributed algorithm**.
# MAGIC 
# MAGIC #### Discretizing Continuous Features
# MAGIC 
# MAGIC A typical decision tree algorithm will consider and assess **all unique values in a continuous feature** for a split. This is what Scikit-learn is doing.
# MAGIC 
# MAGIC ![Single-node tree splits](https://s3-us-west-2.amazonaws.com/files.training.databricks.com/images/mlewd/Scaling-Machine-Learning-Pipelines/single-node-tree-splits.png)
# MAGIC 
# MAGIC But when your data is large and is distributed across multiple worker nodes in a cluster, assessing each unique value for each split can be **prohibitively computationally expensive**. 
# MAGIC 
# MAGIC As a result, Spark MLlib [shortcuts this process](https://spark.apache.org/docs/latest/mllib-decision-tree.html#split-candidates) by **discretizing continuous features** into a specified number of bins, denoted by [the **`maxBins`** parameter](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.DecisionTreeRegressor.html#pyspark.ml.regression.DecisionTreeRegressor.maxBins). This reduces the number of split candidates considered which makes the training of the decision tree on large, distributed data more efficient.
# MAGIC 
# MAGIC ![Distributed tree splits](https://s3-us-west-2.amazonaws.com/files.training.databricks.com/images/mlewd/Scaling-Machine-Learning-Pipelines/distributed-tree-splits.png)
# MAGIC 
# MAGIC Now, `maxBins` is a tunable parameter just like other hyperparameters:
# MAGIC 
# MAGIC * As `maxBins` increases, the algorithm will consider a larger number of split candidates. It also increases the need for communication among workers and reduces the speed at which the algorithm runs.
# MAGIC * As `maxBins` decreases, the algorithm with consider fewer split candidates likely reducing the predictive power of the model. However, the algorithm will run more efficiently.
# MAGIC 
# MAGIC The default value of `maxBins` is 32, but the optimal value depends on your use case and relative need for speed vs. predictive performance.
# MAGIC 
# MAGIC #### Binning Categorical Features
# MAGIC 
# MAGIC The training data we've been using for decision trees so far has only been made up of numeric features, but categorical features can be really helpful!
# MAGIC 
# MAGIC Spark MLlib's decision tree implementation handles categorical features really well &mdash; the recommended best practice is to simply apply the `StringIndexer` to categorical features and the algorithm will recognize that they are categorical rather than ordinally discrete.
# MAGIC 
# MAGIC In the below cell, we load in data that's been prepared using the `StringIndexer` class on categorical features:

# COMMAND ----------

tree_cat_train_df = spark.read.format("delta").load(lesson_3_train_tree_cat_path)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, let's train a Spark MLlib decision tree using that data:

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

dtr = DecisionTreeRegressor(featuresCol="features", labelCol="price")

# Uncomment below line to fit the tree
# dtr_model = dtr.fit(tree_cat_train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Oops! We see the below error here: 
# MAGIC 
# MAGIC ```
# MAGIC DecisionTree requires maxBins (= 32) to be at least as large as the number of 
# MAGIC values in each categorical feature, but categorical feature 49 has 515 values. 
# MAGIC Consider removing this and other categorical features with a large number of 
# MAGIC values, or add more training examples.
# MAGIC ```
# MAGIC 
# MAGIC We received this error because Spark MLlib [uses bins for the indexed categorical features](https://spark.apache.org/docs/latest/mllib-decision-tree.html#split-candidates), as well &mdash; and it doesn't know what to do if there are more categories than available bins for a given feature.
# MAGIC 
# MAGIC ![Categorical binning](https://s3-us-west-2.amazonaws.com/files.training.databricks.com/images/mlewd/Scaling-Machine-Learning-Pipelines/distributed-categorical-tree-bins.png)
# MAGIC 
# MAGIC As a result, we need to make sure that **`maxBins` is at least as large as the greatest cardinality of all of our categorical features**:

# COMMAND ----------

dtr = DecisionTreeRegressor(featuresCol="features", labelCol="price", maxBins=515)
dtr_model = dtr.fit(feature_vector_train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC Great! That fixed our problem.
# MAGIC 
# MAGIC Issues like this are why it's important to understand that distributed algorithms can differ from their single-node counterparts.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scaling Hyperparameter Tuning with Spark MLlib
# MAGIC 
# MAGIC In this final part of this lesson's demo, we're going to show how to speed up hyperparameter tuning for Spark MLlib models.
# MAGIC 
# MAGIC #### Decision Tree Pipeline
# MAGIC 
# MAGIC For this demo, we'll be using Spark MLlib's decision tree algorithm.
# MAGIC 
# MAGIC This is our first step in scaling hyperparameter tuning of Spark MLlib models.
# MAGIC 
# MAGIC :NOTE: Because this is a tree-based model, we'll be creating a tree-based data preparation pipeline with a decision tree as its last stage. 

# COMMAND ----------

from pyspark.sql.types import StringType
from pyspark.ml.feature import StringIndexer

# Load imputed data
train_df = spark.read.format("delta").load(lesson_3_train_path_imp)
test_df = spark.read.format("delta").load(lesson_3_test_path_imp)

# StringIndexer
categorical_cols = [column.name for column in train_df.schema.fields if column.dataType == StringType()]
index_cols = [column + "_index" for column in categorical_cols]
string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_cols, handleInvalid="skip")

# VectorAssembler
numeric_cols = [column.name for column in train_df.schema.fields if column.dataType == DoubleType() and column.name != "price"]
feature_cols = numeric_cols + index_cols
vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# DecisionTreeRegressor
dtr = DecisionTreeRegressor(featuresCol="features", labelCol="price", maxBins=600)

# Pipeline
pipeline = Pipeline(stages=[string_indexer, vector_assembler, dtr])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Parameter Grid
# MAGIC 
# MAGIC Spark MLlib supports **grid search for hyperparameter tuning**.
# MAGIC 
# MAGIC ![grid-search](https://s3-us-west-2.amazonaws.com/files.training.databricks.com/images/mlewd/Scaling-Machine-Learning-Pipelines/grid-search.png)
# MAGIC 
# MAGIC So the first thing we need to do is build a hyperparameter value grid with [Spark MLlib's **`ParamGridBuilder`**](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.ParamGridBuilder.html#pyspark.ml.tuning.ParamGridBuilder). This is where we specify the hyperparameter values we'd like to assess.
# MAGIC 
# MAGIC Because we are using [the **`DecisionTreeRegressor`** class](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.regression.DecisionTreeRegressor.html#pyspark.ml.regression.DecisionTreeRegressor), we'll test the following values:
# MAGIC 
# MAGIC * `maxDepth` (max depth of each decision tree): 2, 5
# MAGIC * `minInfoGain` (minimum information gain for a split to be considered at a tree node): 0, 0.1
# MAGIC 
# MAGIC The `addGrid` method accepts the name of the parameter (e.g. `dtr.maxDepth`), and a list of the possible values (e.g. `[2, 5]`).

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder

param_grid = (ParamGridBuilder()
    .addGrid(dtr.maxDepth, [5, 8])
    .addGrid(dtr.minInfoGain, [0, 0.1])
    .build())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Cross-validation
# MAGIC 
# MAGIC Since we are tuning hyperparameters, **we need a validation set** to assess model performance within our optimization process &mdash; this keeps us from optimizing on our test or holdout set!
# MAGIC 
# MAGIC To do this, we'll use 3-fold cross-validation with Spark MLlib's `CrossValidator`.
# MAGIC 
# MAGIC ![3-fold-CV](https://s3-us-west-2.amazonaws.com/files.training.databricks.com/images/mlewd/Scaling-Machine-Learning-Pipelines/3-fold-cv.png)
# MAGIC 
# MAGIC With 3-fold cross-validation, we train on 2/3 of the data (the training set), and evaluate with the remaining (validation) 1/3 (the validation set). **We repeat this process 3 times**, so each fold gets the chance to act as the validation set. We then **average the results of the three rounds** to identify which hyperparameter perform the best, on average.
# MAGIC 
# MAGIC When putting together a [Spark MLlib **`CrossValidator`**](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html#pyspark.ml.tuning.CrossValidator), we pass in the `estimator` (pipeline), `evaluator`, and `estimatorParamMaps` (parameter grid) to `CrossValidator` so that it knows:
# MAGIC * Which model to use
# MAGIC * How to evaluate the model
# MAGIC * What hyperparameter values to assess
# MAGIC 
# MAGIC We can also set the number of folds we want to split our data into (3), as well as setting a seed so we all have the same split in the data.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator

evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")

cv = CrossValidator(
    estimator=pipeline, 
    evaluator=evaluator, 
    estimatorParamMaps=param_grid,              
    numFolds=3,
    parallelism=2,
    seed=42
)

# COMMAND ----------

# MAGIC %md
# MAGIC The [**`parallelism`** parameter](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidator.html#pyspark.ml.tuning.CrossValidator.parallelism) here is defined in the Spark documentation as: 
# MAGIC 
# MAGIC > the number of threads to use when running parallel algorithms (>= 1).
# MAGIC 
# MAGIC **Question**: Will increasing the `parallelism` parameter always help speed up the tuning process?
# MAGIC 
# MAGIC It depends, increasing the parallelism will:
# MAGIC 
# MAGIC 1. Increase the number of models that can run concurrently
# MAGIC 2. Decrease the amount of resources available to each model
# MAGIC 
# MAGIC The optimal value for `parallelism` depends on things like the size of your data, your cluster configuration, and the type of modeling algorithm you're using. It's recommended that you **tune this `parallelism` parameter to optimize on speed for your given pipeline**.
# MAGIC 
# MAGIC Next, we can call the `CrossValidator` object's `fit` method to fit the entire pipeline for each unique combination of hyperparameter values.

# COMMAND ----------

cv_model = cv.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC By default, [our **`CrossValidatorModel`**](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.tuning.CrossValidatorModel.html#pyspark.ml.tuning.CrossValidatorModel) will retrain on the entire `train_df` with the optimal hyperparameters it identified during the cross-validation and grid-search processes.
# MAGIC 
# MAGIC We can access this retrained model for inference using the `transform` method.

# COMMAND ----------

train_preds_df = cv_model.transform(train_df)
display(train_preds_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pipeline and CrossValidator Placement
# MAGIC 
# MAGIC In the above demonstration, we placed our `Pipeline` object inside of our `CrossValidator` object as the estimator.
# MAGIC 
# MAGIC There's a major scale-related consequence of this: **the entire data preparation pipeline will be fit and run with each model during the hyperparameter tuning and cross-validation process!** 
# MAGIC 
# MAGIC ![pipeline-in-crossvalidator](https://s3-us-west-2.amazonaws.com/files.training.databricks.com/images/mlewd/Scaling-Machine-Learning-Pipelines/pipeline-in-crossvalidator.png)
# MAGIC 
# MAGIC If the data preparation stages of the pipeline are time-consuming, they will dramatically slow down the cross-validation process.
# MAGIC 
# MAGIC An alternative approach is to place the `CrossValidator` object inside of the `Pipeline` object. This way, **we only need to prepare the data preparation pipeline one time**.
# MAGIC 
# MAGIC ![crossvalidator-in-pipeline](https://s3-us-west-2.amazonaws.com/files.training.databricks.com/images/mlewd/Scaling-Machine-Learning-Pipelines/crossvalidator-in-pipeline.png)
# MAGIC 
# MAGIC Here's a demonstration of this process:

# COMMAND ----------

# Put the model estimator in the cross-validator 
cv = CrossValidator(
    estimator=dtr, 
    evaluator=evaluator, 
    estimatorParamMaps=param_grid,              
    numFolds=3,
    parallelism=2,
    seed=42
)

# Put the cross validator in the pipeline
pipeline = Pipeline(stages=[string_indexer, vector_assembler, cv])
pipeline_model = pipeline.fit(train_df)

train_preds_df = pipeline_model.transform(train_df).select("features", "price", "prediction")
display(train_preds_df)

# COMMAND ----------

# MAGIC %md
# MAGIC **Question:** So should we always put the cross-validator within the pipeline?
# MAGIC 
# MAGIC Not necessarily. When we put the cross-validator in the pipeline, we are opening ourselves up to **feature data leakage** in the data preparation stages of our pipeline. With the cross-validator at the end of the pipeline, our early stages will fit on the entirety of the cross-validation data &mdash; this will leak feature information from the validation set to the training set for each model.
# MAGIC 
# MAGIC Whether you choose to put the pipeline in the cross-validator or the cross-validator in the pipeline **depends on your relative needs for scale and avoiding leakage of feature information**.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="http://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="http://help.databricks.com/">Support</a>
