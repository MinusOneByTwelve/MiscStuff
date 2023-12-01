package cts.analytics.Learn

import org.apache.spark.SparkConf
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.apache.log4j._

//while read line; do echo -e "$line\n"; sleep .01; done < shakespeare.txt | nc -lk 9998
//spark-submit --class "cts.analytics.Learn.Stream1" --master local Stream1.jar localhost 9998 5

object WordCount {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val sparkConf = new SparkConf().setAppName("cts.analytics.WordCount")
    val ssc = new StreamingContext(sparkConf, Seconds(args(2).toInt))

    val lines = ssc.socketTextStream(args(0), args(1).toInt, StorageLevel.MEMORY_AND_DISK_SER)
    val words = lines.flatMap(_.split(" "))
    val wordCounts = words.map(x => (x, 1)).reduceByKey(_ + _)    
    wordCounts.print()
    
    ssc.start()
    ssc.awaitTermination()
  }
}