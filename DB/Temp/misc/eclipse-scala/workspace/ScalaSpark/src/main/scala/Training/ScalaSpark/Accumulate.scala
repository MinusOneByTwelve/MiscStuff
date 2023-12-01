package Training.ScalaSpark

import org.apache.spark._
import org.apache.log4j._

object Accumulate {     
  case class Rating(UserID:Int, MovieID:Int, Rating:Int)
   
  def main(args: Array[String]) {
      Logger.getLogger("org").setLevel(Level.ERROR)
      
      val conf = new SparkConf().setAppName("Accumulate")
      val sc = new SparkContext(conf)
      var count5s: Int = 0 
      /*The problem with the above code is that when the driver prints the 
       * variable count5s its value will be zero. 
       * This is because when Spark ships this code to every executor the 
       * 
       * variables become local to that executor and its updated value is
       *  not relayed back to the driver.
       *  
       *  Computations inside transformations are evaluated lazily, so unless an action happens on an RDD the transformationsare not executed. 
       *  As a result of this, accumulators used inside functions like map() or filter() wont get executed unless some action happen on the RDD.
Spark guarantees to update accumulators inside actionsonly once. So even if a task is restarted and the lineage is recomputed, 
the accumulators will be updated only once.
Spark does not guarantee this for transformations. So if a task is restarted and the lineage is recomputed, 
there are chances of undesirable side effects when the accumulators will be updated more than once.
To be on the safe side, always use accumulators inside actions ONLY.
      */
      val _5Counts = sc.longAccumulator("All 5 Counts")
      
      //val ratinglines = sc.textFile("hdfs://quickstart.cloudera:8020/user/cloudera/ratings.dat")
      sc.textFile("hdfs://ifmr-bigdata-164-52-215-55-e2e7-73671-ncr.cluster:8020/training/ratings.dat", 4)
			.foreach { line =>
			 val fields = line.split("::");
			 val rating:Rating = Rating(Integer.parseInt(fields(0)),Integer.parseInt(fields(1)), 
			     Integer.parseInt(fields(2)));
			 var x = Integer.parseInt(fields(2));
			 if(x == 5){
			   count5s = count5s+1
			   _5Counts.add(1)
			   }			 
				}
 
      println(s"\tTotal 5 Ratings Accumulator=${_5Counts.value}")
      println(s"\tTotal 5 Ratings Variable=${count5s}")
            
      sc.stop()
    }  
}