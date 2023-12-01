package Training.ScalaSpark

import org.apache.spark._

object FilterEmps {
  
  def main(args: Array[String]) {
      val conf = new SparkConf().setAppName("sprint56-story59")
      val sc = new SparkContext(conf)
      val rdd1 = sc.textFile("file:///opt/DataSet/EmployeesAll.csv",4)
      val rdd2 = rdd1.filter(!_.startsWith("emp_no"))
      val rdd3 = rdd2.map(X => (X.split(',')(2),X.split(',')(3),X.split(',')(4)))
      val rdd4 = rdd3.filter(x => (x._1.startsWith("Ara") && x._2.startsWith("Ba") && x._3.equals("M")))
      val rdd5 = rdd4.sortBy(_._2, true)
      val rdd6 = rdd5.map(tuple => "%s,%s,%s".format(tuple._1, tuple._2, tuple._3))
      rdd6.collect()
      rdd6.saveAsTextFile("file:///opt/DataSet/output4") 
      sc.stop()
    }  
}