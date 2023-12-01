package Training.ScalaSpark

import org.apache.spark._

object WordCnt {
  
  def main(args: Array[String]) {
      val conf = new SparkConf().setAppName("sprint56-story59")
      val sc = new SparkContext(conf)
      val input = sc.textFile("hdfs://"+args(0)+":8020/user"+args(1))
      val words = input.flatMap(line => line.split(' '))
      val lowerCaseWords = words.map(word => word.toLowerCase())
      val counts2 = lowerCaseWords.map(word => (word, 1))
      val counts3 = counts2.reduceByKey(_ + _)
      counts3.saveAsTextFile("hdfs://"+args(0)+":8020/user"+args(2))      
      sc.stop()
    }  
}