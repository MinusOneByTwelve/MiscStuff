package Training.ScalaSpark

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import scala.io.Source
import java.nio.charset.CodingErrorAction
import scala.io.Codec
import org.apache.spark.storage.StorageLevel._

object Broadcast {
  var movieNames:Map[Int, String] = Map()
   
  case class Rating(UserID:Int, MovieID:Int, Rating:Int)
  
    def MapRating(line:String): Rating = {
    val fields = line.split("::")    
    val rating:Rating = Rating(Integer.parseInt(fields(0)), 
        Integer.parseInt(fields(1)), Integer.parseInt(fields(2)))
    
    return rating
  } 
  
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("Broadcast")
    val sc = new SparkContext(conf) 
       
    val ratinglines = sc.textFile("hdfs://ifmr-bigdata-164-52-215-55-e2e7-73671-ncr.cluster:8020/training/ratings.dat")
    val movielines = sc.textFile("hdfs://ifmr-bigdata-164-52-215-55-e2e7-73671-ncr.cluster:8020/training/movies.dat")
			.foreach { line =>
			 val fields = line.split(":");
			 movieNames += (fields(0).toInt -> fields(1))			 			 
				}    
    var nameDict = sc.broadcast(movieNames)
    
    val rating = ratinglines.map(MapRating).
    map(movie => (movie.MovieID, 1)).reduceByKey(_ + _).
    filter(record => record._2 >= 40).sortBy(_._2,false)
    //https://spark.apache.org/docs/2.4.0/rdd-programming-guide.html#rdd-persistence
    rating.persist(MEMORY_AND_DISK_SER)
    //rating.unpersist()
    
    val ratingWithNames = rating.map( x  => (nameDict.value(x._1), x._2) )
    val results = ratingWithNames.collect().take(20)
    results.foreach(println)
  }  
}

