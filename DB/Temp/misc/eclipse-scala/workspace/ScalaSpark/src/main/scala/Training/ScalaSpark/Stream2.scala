package cts.analytics.Learn

import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.log4j._

//spark-submit --class "baci.sprk.Stream2" --master local str2.jar

object Stream2 {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val sparkConf = new SparkConf().setAppName("Streaming Word Count From HDFS")
    val ssc = new StreamingContext(sparkConf, Seconds(15))

    val lines = ssc.textFileStream("hdfs://10.142.1.1:8020/user/contactrkk4095/strmread")
    val words = lines.flatMap(_.split(" "))
    val wordCounts = words.map(x => (x, 1)).reduceByKey(_ + _)
    wordCounts.print()
    ssc.start()
    ssc.awaitTermination()
  }
}
