# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning" style="width: 600px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # Named Entity Recognition (NER)
# MAGIC 
# MAGIC ## ![Spark Logo Tiny](https://files.training.databricks.com/images/105/logo_spark_tiny.png) In this lesson you will:<br>
# MAGIC - Extract facts and relevant information from an otherwise unstructured data (raw, unprocessed text)
# MAGIC - Explore example applications: 
# MAGIC   - information management
# MAGIC   - question answering 
# MAGIC - Use groups of words as a single entity rather than single words
# MAGIC - Detect named entities and their types 
# MAGIC   - people, locations, organizations 
# MAGIC - Use SparkNLP - an open source library created by <a href="https://nlp.johnsnowlabs.com/api/python/reference/index.html" target="_blank">John Snow Labs</a>
# MAGIC 
# MAGIC 
# MAGIC <img src="https://tse4.mm.bing.net/th?id=OIP.VIsqw-pcH4m6rMKvFteEaQAAAA&pid=Api" width="250px" height="50px">
# MAGIC 
# MAGIC Library Prerequisites: 
# MAGIC - **`spark-nlp-display==4.1`** 
# MAGIC - **`spark-nlp==4.2.1`**
# MAGIC 
# MAGIC **You need to additionally install a <a href="https://docs.databricks.com/libraries/cluster-libraries.html" target="_blank">Maven package on the cluster</a> as well.**
# MAGIC 
# MAGIC When on Apache Spark versions 3.x: 
# MAGIC - **`com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.5`**
# MAGIC 
# MAGIC Note: John Snow Labs updates the SparkNLP package periodically. When Apache Spark/Databricks runtime versions are updated, please refer to JSL's [package cheatsheet](https://github.com/JohnSnowLabs/spark-nlp#packages-cheatsheet) to get the latest Maven version to install. 

# COMMAND ----------

# MAGIC %pip install spark-nlp==4.2.1
# MAGIC %pip install spark-nlp-display==4.1

# COMMAND ----------

# MAGIC %scala
# MAGIC // Verify that com.johnsnowlabs.nlp:spark-nlp 
# MAGIC // is properly installed by importing it.
# MAGIC import com.johnsnowlabs.nlp.SparkNLP

# COMMAND ----------

# MAGIC %md
# MAGIC ## Using pre-trained model without fine-tuning
# MAGIC 
# MAGIC What is the data? A set of financial documents. 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/nlp_sec_fillings.png" height="5px" width="500px">
# MAGIC <br>
# MAGIC <br>
# MAGIC - Data source: publicly disclosed SEC fillings
# MAGIC - Goal:
# MAGIC   - assess loan agreement, loan amount, value of collateral
# MAGIC   - perform risk assessment
# MAGIC - Research paper published <a href="https://aclanthology.org/U15-1010.pdf" target="_blank">here</a> by a group of researchers at the University of Melbourne, Australia. 
# MAGIC - Data can be downloaded <a href="https://people.eng.unimelb.edu.au/tbaldwin/#resources" target="_blank">here</a>

# COMMAND ----------

# MAGIC %run "./Includes/Classroom-Setup"

# COMMAND ----------

# MAGIC %md
# MAGIC CoNLL is the conventional name for TSV (tab-separated values) data in NLP. It originally refers to the shared NLP tasks organized by the Conferences of Natural Language Learning (CoNLL). 
# MAGIC You can <a href="https://raw.githubusercontent.com/patverga/torch-ner-nlp-from-scratch/master/data/conll2003/eng.train" target="_blank">go to this page</a> to look at how the data looks like: this particular dataset - that contains Reuter stories - is one of the most widely used NER datasets in the CoNLL format. But we will be using financial data for this notebook! 
# MAGIC 
# MAGIC <img src="https://files.training.databricks.com/images/conll_sample.png" height="1px" width="300px">
# MAGIC <img src="https://www.researchgate.net/profile/Mitchell-Marcus-2/publication/2873803/figure/tbl1/AS:669991049392137@1536749722377/1-The-Penn-Treebank-POS-tagset.png" height="5px" width="500px">
# MAGIC 
# MAGIC <a href="https://www.researchgate.net/figure/1-The-Penn-Treebank-POS-tagset_tbl1_2873803" target="_blank">Here is the source for the image on the right.</a>

# COMMAND ----------

from sparknlp.training import CoNLL

training_data = CoNLL().readDataset(spark, f"{DA.paths.datasets}/sec-fillings/train-fin5.txt", read_as="text")
display(training_data)

# COMMAND ----------

test_data = CoNLL().readDataset(spark, f"{DA.paths.datasets}/sec-fillings/test-fin3.txt", read_as="text")

# COMMAND ----------

# MAGIC %md
# MAGIC Here are all the available pretrained <a href="https://nlp.johnsnowlabs.com/docs/en/pipelines" target="_blank">pipelines</a> in SparkNLP. This **`recognize_entities_dl`** pipeline was trained on <a href="https://nlp.johnsnowlabs.com/2020/01/22/glove_100d.html" target="_blank">Glove</a> embeddings and this <a href="https://nlp.johnsnowlabs.com/2020/03/19/ner_dl_en.html" target="_blank">NER</a> model that was trained on the CoNLL 2003 Reuters text corpus.

# COMMAND ----------

from sparknlp.pretrained import PretrainedPipeline

rec_entities_pipeline = PretrainedPipeline("recognize_entities_dl")
sample_text = training_data.first()[0]
single_example = rec_entities_pipeline.fullAnnotate(sample_text)[0]
single_example.keys()

# COMMAND ----------

rec_entities_pipeline.model.stages

# COMMAND ----------

# MAGIC %md
# MAGIC Visualizations are possible with the use of the library **`spark-nlp-display`**. Documentation is <a href="https://github.com/JohnSnowLabs/spark-nlp-display/blob/main/tutorials/Spark_NLP_Display.ipynb" target="_blank">here</a>.

# COMMAND ----------

from sparknlp_display import NerVisualizer

visualizer = NerVisualizer()
ner_vis = visualizer.display(single_example, label_col="entities", document_col="document", return_html=True)

displayHTML(ner_vis)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Transfer Learning
# MAGIC 
# MAGIC What if we'd like to pick a set of embeddings which we fine tune later on?

# COMMAND ----------

from sparknlp.annotator import BertEmbeddings, NerDLApproach
from pyspark.ml import Pipeline

# Define the pretrained BERT model 
bert = (BertEmbeddings.pretrained("small_bert_L2_128", "en")
        .setInputCols("document", "token")
        .setOutputCol("bert"))

# Define the NER approach
ner_dl = (NerDLApproach()
          .setInputCols(["document", "token", "bert"])
          .setLabelColumn("label")
          .setOutputCol("ner")
          .setMaxEpochs(10)
          .setLr(0.001)
          .setValidationSplit(0.2)
          .setBatchSize(16)
          .setRandomSeed(0)
         )

# put everything into the pipeline
ner_pipeline = (Pipeline(stages=[bert, ner_dl]))

# COMMAND ----------

ner_model = ner_pipeline.fit(training_data.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Inference using the fine-tuned model

# COMMAND ----------

display(ner_model.transform(test_data.limit(1)).select("text", "ner"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Use MLflow to log hyperparameters and the model object

# COMMAND ----------

key_list = []
value_list = []

for key, value in ner_dl.extractParamMap().items():
    key_list.append(str(key).partition("__")[-1])
    value_list.append(value)

hyperparam_dict = dict(zip(key_list, value_list))
hyperparam_dict

# COMMAND ----------

import mlflow

with mlflow.start_run() as run:
    mlflow.log_params(hyperparam_dict)
    mlflow.spark.log_model(ner_model, "ner_model")

# COMMAND ----------

# MAGIC %md
# MAGIC Load the model using **`mlflow.spark.load_model`**

# COMMAND ----------

loaded_model = mlflow.spark.load_model(f"runs:/{run.info.run_id}/ner_model")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Inference
# MAGIC 
# MAGIC Note that computing evaluation metrics is not available on the open-sourced version of SparkNLP. It's only available for the enterprise edition of SparkNLP. Refer to this <a href="https://nlp.johnsnowlabs.com/docs/en/evaluation#evaluating-ner-dl" target="_blank">documentation</a>.

# COMMAND ----------

display(loaded_model.transform(test_data.limit(1)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### What if we would like to view the visualizations as well? 

# COMMAND ----------

from sparknlp.base import LightPipeline
light_model = LightPipeline(ner_model)
ann_text = light_model.fullAnnotate(sample_text)[0]
ann_text.keys()

# COMMAND ----------

### Uncomment to run this cell
### This cell is expected to error out.  
# displayHTML(visualizer.display(ann_text, label_col="entities", document_col="document", return_html=True))

# COMMAND ----------

# MAGIC %md
# MAGIC The error occurred because the visualization is expecting the column **`document`** in the data. However, the column is not present because our custom **`ner_pipeline`** only produced **`bert`** and **`ner`** as our output columns! To allow visualizations to work, we will need to construct the complete NLP pipeline, starting from assembling the text data, to tokenizing data, and then finally continue with the rest of the NER pipeline we have made above! 

# COMMAND ----------

from sparknlp.base import DocumentAssembler
from sparknlp.annotator import SentenceDetector, Tokenizer, NerConverter

## Step 1: DocumentAssembler
## Prepares data into a format that is processable by Spark NLP. This is the entry point for every Spark NLP pipeline.
step1_document = (DocumentAssembler()
                  .setInputCol("text")
                  .setOutputCol("document")
                 )

### Step 2: SentenceDetector
### Annotator that detects sentence boundaries using any provided approach. 
step2_sentence = (SentenceDetector()
                  .setInputCols("document")
                  .setOutputCol("sentence")
                 )

# ### Step 3: Tokenizer
step3_tokenizer = (Tokenizer()
                   .setInputCols("sentence")
                   .setOutputCol("token")
                  )

# ### Step 4: Apply embeddings
step4_embeddings = (BertEmbeddings.pretrained('small_bert_L2_128', 'en')
                    .setInputCols(["sentence", "token"])
                    .setOutputCol("bert")
                   )

### Step 5: NER 
step5_ner_dl = loaded_model.stages[-1]

### NER Converter
### This converts the NER column to the right structure for visualizations.
step6_converter = (NerConverter()
                   .setInputCols(["document", "token", "ner"])
                   .setOutputCol("ner_span")
                  )

full_ner_pipeline = Pipeline(stages=[step1_document, step2_sentence, 
                                     step3_tokenizer, step4_embeddings, 
                                     step5_ner_dl, step6_converter
                                    ])

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC According to the documentation, **`LightPipeline`** is equivalent to SparkML Pipeline but much faster with small amounts of data. 
# MAGIC 
# MAGIC Refer to <a href="https://nlp.johnsnowlabs.com/api/python/reference/autosummary/sparknlp.base.LightPipeline.html" target="_blank">documentation here</a>.

# COMMAND ----------

prediction_model = full_ner_pipeline.fit(training_data.limit(100))
light_model = LightPipeline(prediction_model)
ann_text = light_model.fullAnnotate(sample_text)[0]
ann_text.keys()

# COMMAND ----------

ner_vis_complete = visualizer.display(ann_text, label_col="ner_span", document_col="document", return_html=True)
displayHTML(ner_vis_complete)

# COMMAND ----------

# MAGIC %md
# MAGIC Tada! What differences did you notice between using the out-of-the-box pretrained model vs. fine-tuning BERT embeddings for an additional 10 epochs in the results of NER? 
# MAGIC 
# MAGIC How do the results differ from the ground truth labels? 

# COMMAND ----------

import pyspark.sql.functions as F

ground_truth = (training_data
                .limit(1)
                .select("label")
                .select(F.explode("label"))
                .select(F.expr("col.metadata['word']").alias("word"), F.expr("col.result"))
                .filter(F.col("result") != "O")
               )
display(ground_truth)

# COMMAND ----------

# MAGIC %md
# MAGIC [HuggingFace](https://huggingface.co/) is another popular NLP library. However, Hugging Face does not integrate with Spark. If your data is small enough to fit on an arbitrarily large node, you might consider using Hugging Face. Also note that, MLflow does not have built-in integration with Hugging Face. If you are curious how to use Hugging Face on Databricks with MLflow, you can refer to the following resources developed by Databricks employees. Notice that you will need to create custom Transformer model using `mlflow.pyfunc.PythonModel`:
# MAGIC 
# MAGIC * [Rapid NLP Development with Databricks, Delta, and Transformers](https://www.databricks.com/blog/2022/09/09/rapid-nlp-development-databricks-delta-and-transformers.html)
# MAGIC * [Making Transformer-Based Models First-Class Citizens in the Lakehouse](https://medium.com/@tim.lortz/making-transformer-based-models-first-class-citizens-in-the-lakehouse-cf900cf604d4)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## Classroom Cleanup
# MAGIC 
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson:

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/>
# MAGIC <a href="https://databricks.com/privacy-policy">Privacy Policy</a> | <a href="https://databricks.com/terms-of-use">Terms of Use</a> | <a href="https://help.databricks.com/">Support</a>
