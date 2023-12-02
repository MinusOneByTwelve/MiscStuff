package FragmaData.MovieLens

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.log4j._
import org.apache.spark.sql._

object Spark
{
  case class Rating(UserID:Int, MovieID:Int, Rating:Int)
  case class User(UserID:Int,Age:String,Occupation:String,CustomAgeGroup:String)
  case class Movie(MovieID:Int, Title:String, Genres:List[String])
  
  def MapRating(line:String): Rating = {
    val fields = line.split("::")    
    val rating:Rating = Rating(Integer.parseInt(fields(0)), Integer.parseInt(fields(1)), Integer.parseInt(fields(2)))
    
    return rating
  }  

  def MapMovie(line:String): Movie = {
    val fields = line.split("::")    
    val movie:Movie = Movie(Integer.parseInt(fields(0)), fields(1), fields(2).split('|').toList)
    
    return movie
  } 
  
  def MapUser(line:String): User = {
    val fields = line.split("::")  
    
    val Age = Integer.parseInt(fields(2)) match {
      case 1   => "Under 18"
      case 18  => "18-24"
      case 25  => "25-34"
      case 35  => "35-44"
      case 45  => "45-49"
      case 50  => "50-55"
      case 56  => "56+"      
      case _   => "-"
    } 
    
    val CustomAgeGroup = Integer.parseInt(fields(2)) match {
      case 18  => "18-35"
      case 25  => "18-35"
      case 35  => "36-50"
      case 45  => "36-50"
      case 50  => "50+"
      case 56  => "50+"      
      case _   => "-"
    }
    
    val Occupation = Integer.parseInt(fields(3)) match {
      case  0 => "\"other\" or not specified"
      case  1 => "academic/educator"
      case  2 => "artist"
      case  3 => "clerical/admin"
      case  4 => "college/grad student"
      case  5 => "customer service"
      case  6 => "doctor/health care"
      case  7 => "executive/managerial"
      case  8 => "farmer"
      case  9 => "homemaker"
      case 10 => "K-12 student"
      case 11 => "lawyer"
      case 12 => "programmer"
      case 13 => "retired"
      case 14 => "sales/marketing"
      case 15 => "scientist"
      case 16 => "self-employed"
      case 17 => "technician/engineer"
      case 18 => "tradesman/craftsman"
      case 19 => "unemployed"
      case 20 => "writer"     
      case _   => "-"
    }
    
    val user:User = User(Integer.parseInt(fields(0)), Age, Occupation, CustomAgeGroup)
    
    return user
  }  
    
  def main(args: Array[String]) 
  {
    Logger.getLogger("org").setLevel(Level.ERROR)
        
    val SparkConf = new SparkConf().setAppName("FragmaData.MovieLens.Spark")
    val sc = new SparkContext(SparkConf)
   
    val ratinglines = sc.textFile("hdfs://" + args(0) + ":8020/user" + args(1))
    val movielines = sc.textFile("hdfs://" + args(0) + ":8020/user" + args(2))   
    val userlines = sc.textFile("hdfs://" + args(0) + ":8020/user" + args(3))
    val rating = ratinglines.map(MapRating)
    val movie = movielines.map(MapMovie)
    val people = userlines.map(MapUser) 

    println("-----------")    
    println("Question 1")
    println("-----------")    
    
    rating.map(movie => (movie.MovieID, 1)).reduceByKey(_ + _).keyBy(_._1).
    join(movie.map(m => (m.MovieID,m.Title)).keyBy(_._1)).distinct().map(x => (x._2._2._2,x._2._1._2)).sortBy(_._2,false).take(10).foreach(println)
 
    println("-----------")    
    println("Question 2")
    println("-----------")  
    
    ((rating.map(movie => (movie.MovieID, 1)).reduceByKey(_ + _).filter(record => record._2 >= 40)).keyBy(_._1).join(
    (rating.map(movie => (movie.MovieID, movie.Rating)).aggregateByKey((0, 0))(
        (id, rating) => (id._1 + rating, id._2 + 1),
        (x, y) => (x._1 + y._1, x._2 + y._2))
    .mapValues(ratingavg => 1.0 * ratingavg._1 / ratingavg._2).keyBy(_._1).
    join(movie.map(m => (m.MovieID,m.Title)).keyBy(_._1)).distinct()).keyBy(_._1))).
    map(x => (x._2._2._2._2._2,BigDecimal(x._2._2._2._1._2).setScale(2, BigDecimal.RoundingMode.UP))).sortBy(_._2,false).take(20).foreach(println)
    
    println("-----------")    
    println("Question 3")
    println("-----------") 
    
    val RDD = (((people.filter(record => (record.CustomAgeGroup == "18-35" || record.CustomAgeGroup == "36-50" || record.CustomAgeGroup == "50+")))
    .map(record => (record.UserID,record.CustomAgeGroup,record.Occupation)))
    .keyBy(_._1).join(rating.keyBy(_.UserID)).map(record => (record._2._1._3,record._2._1._2,record._2._2.MovieID,record._2._2.Rating,record._2._1._1))).
    keyBy(_._3).join(movie.keyBy(_.MovieID)).map(record => ((record._2._1._1,record._2._1._2,record._2._1._3,record._2._1._4),record._2._2.Genres)).
    flatMap {case (key, value) => value.map(v => (key, v))}.
    map(record => ((record._1._1,record._1._2,record._2),record._1._4)).
    aggregateByKey((0, 0))(
        (identity, rating) => (identity._1 + rating, identity._2 + 1),
        (x, y) => (x._1 + y._1, x._2 + y._2))
    .mapValues(ratingavg => BigDecimal(1.0 * ratingavg._1 / ratingavg._2).setScale(2, BigDecimal.RoundingMode.UP).toDouble).
    map(record => (record._1._1,record._1._2,record._1._3,record._2))
    
    val sqlContext = new SQLContext(sc) 
    import sqlContext.implicits._    
    import org.apache.spark.sql.types._
    import org.apache.spark.sql.expressions.Window
    import org.apache.spark.sql.functions.row_number      
    
    val DF = RDD.toDF("Occupation","AgeGroup","Genre","AvgRating")
    
    val DF_Final = DF.withColumn("Rank", row_number().over(Window.partitionBy($"Occupation",$"AgeGroup").orderBy($"AvgRating".desc))).toDF()
    
    DF_Final.filter(DF_Final("Rank") < 6).sort(DF_Final("Occupation"),DF_Final("AgeGroup"),DF_Final("Rank")).limit(15).show()
    DF_Final.filter(DF_Final("Rank") < 6).sort(DF_Final("Occupation"),DF_Final("AgeGroup"),DF_Final("Rank")).
    coalesce(1).write.csv("hdfs://" + args(0) + ":8020/user" + args(4) + "V")

    val DF_Pivot = DF_Final.filter(DF_Final("Rank") < 6).groupBy("Occupation","AgeGroup")
    .pivot("Rank").agg(org.apache.spark.sql.functions.first("Genre")).toDF()
    
    DF_Pivot.sort(DF_Pivot("Occupation"),DF_Pivot("AgeGroup")).limit(18).show()    
    DF_Pivot.sort(DF_Pivot("Occupation"),DF_Pivot("AgeGroup")).coalesce(1).write.csv("hdfs://" + args(0) + ":8020/user" + args(4))
    
    sc.stop()
  }
}