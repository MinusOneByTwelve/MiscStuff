# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Document Classification
# MAGIC 
# MAGIC In this lab, you will use transfer learning to classify <a href="https://newscatcherapi.com/" target="_blank">news topics collected by the NewsCatcher team</a>, who collect and index news articles and release them to the open-source community. The dataset can be downloaded from <a href="https://www.kaggle.com/kotartemiy/topic-labeled-news-dataset" target="_blank">Kaggle</a>.
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lab you will:<br>
# MAGIC - Fine-tune a pretrained model to classify the topics of the news

# COMMAND ----------

# MAGIC %pip install spark-nlp==4.2.5

# COMMAND ----------

# MAGIC %scala
# MAGIC // Verify that com.johnsnowlabs.nlp:spark-nlp 
# MAGIC // is properly installed by importing it.
# MAGIC import com.johnsnowlabs.nlp.SparkNLP

# COMMAND ----------

# MAGIC %run "../Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Read in training data
# MAGIC The dataset contains 6 columns:
# MAGIC 
# MAGIC - topic: topic of the news article
# MAGIC - link: link of the news article
# MAGIC - domain: domain of the websites on which the news articles are published
# MAGIC - published_date: the date when the articles are published 
# MAGIC - title: title of the news article
# MAGIC - lang: language of the news article
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC For the purpose of this lab, we will predict the **`topic`** of the news.
# MAGIC 
# MAGIC The possible topics are: 
# MAGIC 
# MAGIC 1. Business
# MAGIC 2. Entertainment
# MAGIC 3. Health
# MAGIC 4. Nation
# MAGIC 5. Science
# MAGIC 6. Sports
# MAGIC 7. Technology
# MAGIC 8. World

# COMMAND ----------

df = (spark.read
      .option("header", True)
      .option("sep", ";")
      .format("csv")
      .load(f"{DA.paths.datasets}/news/labelled_newscatcher_dataset.csv"))

train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
display(train_df)

# COMMAND ----------

train_df.count()

# COMMAND ----------

display(train_df.groupby("topic").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Lab: Build the text classifier pipeline
# MAGIC 
# MAGIC Instructions: 
# MAGIC 1. Use **`DocumentAssembler`**
# MAGIC 2. You will use **`UniversalSentenceEncoder`** for the first time. Refer to the <a href="https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.UniversalSentenceEncoder.html" target="_blank">documentation</a> on Universal Sentence Encoder to see an example of how you can build such an encoder.  This is a <a href="https://arxiv.org/pdf/1803.11175.pdf" target="_blank">pretrained model published by Google on TFhub</a> and SparkNLP provides a wrapper to use this model efficiently with Spark.
# MAGIC 3. You will need to build a <a href="https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.annotator.ClassifierDLApproach.html" target="_blank">text classifier</a>. 
# MAGIC This model uses embeddings from the **`UniversalSentenceEncoder`** and was originally trained on the TREC6 dataset, which is another common benchmarking dataset that classifies fact-based questions into different semantic categories. TREC stands for Text REtrieval Conference. More about the TREC dataset <a href="https://trec.nist.gov/data/qa.html" target="_blank">here</a>.
# MAGIC   - Set the **`maxEpochs`** to be **`3`**. This number determines how many epochs you want to fine-tune the pretrained model.
# MAGIC 4. Compile the stages above into a pipeline.

# COMMAND ----------

# ANSWER
from sparknlp.base import DocumentAssembler
from sparknlp.annotator import UniversalSentenceEncoder, ClassifierDLApproach

from pyspark.ml import Pipeline

document = (DocumentAssembler()
           .setInputCol("title")
           .setOutputCol("document")
           )

encoder = (UniversalSentenceEncoder.pretrained()
          .setInputCols(["document"])
          .setOutputCol("sentence_embeddings")
          )

text_classifier = (ClassifierDLApproach()
                   .setInputCols(["sentence_embeddings"])
                   .setOutputCol("class")
                   .setLabelColumn("topic")
                   .setMaxEpochs(3)
                  )

classifier_pipeline = (Pipeline(stages=[document, encoder, text_classifier]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train the model
# MAGIC 
# MAGIC Use **`classifier_pipeline`** to train the model and name it as **`pipeline_model`**.

# COMMAND ----------

# ANSWER
pipeline_model = classifier_pipeline.fit(train_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model inference

# COMMAND ----------

display(test_df)

# COMMAND ----------

test_df.count()

# COMMAND ----------

import pyspark.sql.functions as F

prediction_df = pipeline_model.transform(test_df)
display(prediction_df.select("title", "topic", "class.result", "class.metadata"))

# COMMAND ----------

# MAGIC %md
# MAGIC Notice that the **`result`** column is an array because SparkNLP allows you to have multiple sentences in each row. Let's extract the prediction from the array below.

# COMMAND ----------

prediction_df = (prediction_df
                 .select("title", "topic", "class.result")
                 .withColumn("result", F.col("result")[0])
                )   
display(prediction_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluation
# MAGIC 
# MAGIC Complete the cell below to use <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html" target="_blank">sklearn.metrics.classification_report</a> to generate the evaluation metrics on the prediction dataframe.

# COMMAND ----------

# ANSWER
from sklearn.metrics import classification_report

prediction_pdf = prediction_df.toPandas()
print(classification_report(prediction_pdf["result"], prediction_pdf["topic"]))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
