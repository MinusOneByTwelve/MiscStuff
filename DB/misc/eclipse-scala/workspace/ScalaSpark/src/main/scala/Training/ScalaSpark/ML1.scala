package Training.ScalaSpark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.sql.SparkSession

/*
spark-submit --class "Training.ScalaSpark.ML1" --master local[*] C:\\Users\\Windows10\\Downloads\\ML1.jar D:\\Work\\BigDataNew\\DataSet\\adcampaign.csv 
*/

object ML1 {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder.appName("ML1").getOrCreate()
    Logger.getLogger("org").setLevel(Level.ERROR)
    spark.sparkContext.setLogLevel("ERROR")

    val df = spark.read.option("header", "true").option("inferSchema", true).csv(args(0))
    df.printSchema()

    val splits = df.randomSplit(Array(0.8, 0.2), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    val genderIndexer = new StringIndexer().setInputCol("Gender").setOutputCol("GenderIndex")
    val genderOneHotEncoder = new OneHotEncoder().setInputCol("GenderIndex").setOutputCol("GenderOHE")

    val features = Array("GenderOHE", "Age", "EstimatedSalary")
    val dependetVariable = "Purchased"

    val vectorAssembler = new VectorAssembler().setInputCols(features).setOutputCol("features")
    val scaler = new StandardScaler().setInputCol("features").setOutputCol("scaledFeatures")

    val logisticRegression = new LogisticRegression()
      .setFeaturesCol("scaledFeatures")
      .setLabelCol(dependetVariable)

    val stages = Array(genderIndexer, genderOneHotEncoder, vectorAssembler, scaler, logisticRegression)
    val pipeline = new Pipeline().setStages(stages)

    val model = pipeline.fit(train)

    val results = model.transform(test)
    results.show()
    
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol(dependetVariable)
      .setRawPredictionCol("rawPrediction")
      .setMetricName("areaUnderROC")

    val accuracy = evaluator.evaluate(results)

    println(s"Accuracy of Model : ${accuracy}")
  }
}