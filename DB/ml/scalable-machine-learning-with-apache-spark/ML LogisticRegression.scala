// Databricks notebook source
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.mlflow.tracking.ActiveRun
import org.mlflow.tracking.MlflowContext

// COMMAND ----------

val df = spark.read.option("header", "true").option("inferSchema", true).csv("dbfs:/FileStore/tables/ML/Data/adcampaign.csv")
df.printSchema()
df.show()

// COMMAND ----------

val splits = df.randomSplit(Array(0.8, 0.2), seed = 1234L)
val train = splits(0)
val test = splits(1)

// COMMAND ----------

val genderIndexer = new StringIndexer().setInputCol("Gender").setOutputCol("GenderIndex")
val genderOneHotEncoder = new OneHotEncoder().setInputCol("GenderIndex").setOutputCol("GenderOHE")

// COMMAND ----------

val features = Array("GenderOHE", "Age", "EstimatedSalary")
val dependetVariable = "Purchased"

// COMMAND ----------

val vectorAssembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")

// COMMAND ----------

val logisticRegression = new LogisticRegression()
.setFeaturesCol("scaledFeatures")
.setLabelCol(dependetVariable)

// COMMAND ----------

val stages = Array(genderIndexer, genderOneHotEncoder, vectorAssembler, scaler, logisticRegression)
val pipeline = new Pipeline().setStages(stages)

// COMMAND ----------

val mlflowContext = new MlflowContext()
val experimentName = "/Users/minus1by12@mailbox.org/Test1"
val client = mlflowContext.getClient()
val experimentOpt = client.getExperimentByName(experimentName);
if (!experimentOpt.isPresent()) {
  client.createExperiment(experimentName)
}
mlflowContext.setExperimentName(experimentName)

// COMMAND ----------

val run = mlflowContext.startRun("Iteration1")

// COMMAND ----------

run.logParam("param1", "5")
 
run.logMetric("foo1", 2.0, 1)
run.logMetric("foo2", 4.0, 2)
run.logMetric("foo3", 6.0, 3)

import java.io.{File,PrintWriter}
import java.nio.file.Paths
new PrintWriter("/tmp/output.txt") { write("Hello, world!") ; close }
run.logArtifact(Paths.get("/tmp/output.txt"))

// COMMAND ----------

val model = pipeline.fit(train)
val results = model.transform(test)
results.show()

// COMMAND ----------

val evaluator = new BinaryClassificationEvaluator()
.setLabelCol(dependetVariable)
.setRawPredictionCol("rawPrediction")
.setMetricName("areaUnderROC")

val accuracy = evaluator.evaluate(results)
println(s"Accuracy of Model : ${accuracy}")

// COMMAND ----------

run.endRun()
