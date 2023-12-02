package Training.ScalaSpark

import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.apache.spark.streaming.{Seconds, StreamingContext, Time}
import org.apache.log4j._

//while read line; do echo -e "$line\n"; sleep .01; done < shakespeare.txt | nc -lk 9998
//spark-submit --class "Training.ScalaSpark.Stream3" --master local Stream3.jar localhost 9998 5

object Stream3 {
  case class Record(word: String)
  
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val sparkConf = new SparkConf().setAppName("Streaming Word Count From SQL")
    val ssc = new StreamingContext(sparkConf, Seconds(5))

    val lines = ssc.socketTextStream(args(0), args(1).toInt, StorageLevel.MEMORY_AND_DISK_SER)
    val words = lines.flatMap(_.split(" "))

    words.foreachRDD 
    { 
      (rdd: RDD[String], time: Time) =>
      val spark = SparkSessionSingleton2.getInstance(rdd.sparkContext.getConf)
      import spark.implicits._

      val wordsDataFrame = rdd.map(w => Record(w)).toDF()

      wordsDataFrame.createOrReplaceTempView("words")

      val wordCountsDataFrame =
        spark.sql("select word, count(*) as total from words group by word order by total desc limit 5")
        
      println(s"========= $time =========")
      
      wordCountsDataFrame.show()
    }
    
    ssc.start()
    ssc.awaitTermination()
  }
}
object SparkSessionSingleton2 {

  @transient  private var instance: SparkSession = _

  def getInstance(sparkConf: SparkConf): SparkSession = {
    if (instance == null) {
      instance = SparkSession
        .builder
        .config(sparkConf)
        .getOrCreate()
    }
    instance
  }
}