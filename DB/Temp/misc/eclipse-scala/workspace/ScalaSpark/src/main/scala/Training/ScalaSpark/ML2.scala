package Training.ScalaSpark

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.SparkSession

/*
spark-submit --class "Training.ScalaSpark.ML2" --master local[*] C:\\Users\\Windows10\\Downloads\\ML2.jar D:\\Work\\BigDataNew\\DataSet\\bookrating\\Books.csv D:\\Work\\BigDataNew\\DataSet\\bookrating\\Ratings.csv 276747 5 
*/

/*To provide recommendations based on the ratings given by users, 
 * we can use a technique called Collaborative Filtering. 
 * This is based on the concept that if person A and B have given similar ratings to the same objects, 
 * then they must have similar taste. Therefore, there is a higher probability that person A will like an object they haven’t 
 * come across but is rated highly by B.

To perform collaborative filtering, we will use an algorithm called ALS (Alternating Least Squares), 
which will make predictions about how much each user would rate each book and ultimately provide recommendations
 for every user listed in the dataset.*/

object ML2 {
    def main(args: Array[String]): Unit = {
      val spark = SparkSession.builder.appName("ML2").getOrCreate()
      Logger.getLogger("org").setLevel(Level.ERROR)  
      spark.sparkContext.setLogLevel("ERROR")
      
      val books_df = spark.read.option("delimiter", ";").
      option("header", "true").csv(args(0))
      books_df.createOrReplaceTempView("books")
      books_df.show()  
      
      val user_ratings_df = spark.read.option("delimiter", ";").
      option("header", "true").csv(args(1))
      user_ratings_df.createOrReplaceTempView("userrat")      
      val ratings_df = spark.sql("select * from (select cast(`User-ID` as int) as UserId,cast(ISBN as int) as ISBN,cast(Rating as int) as Rating from userrat)obj where ISBN is not null")
      ratings_df.createOrReplaceTempView("userrata")
      ratings_df.show()
      
      spark.sql("select * from userrata a join books b on a.ISBN = b.ISBN and a.UserId ="+args(2)).show()
      
      val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("UserId").setItemCol("ISBN").setRatingCol("Rating")
      val dataframemodel = als.fit(ratings_df) 
      
      import spark.implicits._
      val dfWithSchema = spark.sparkContext.parallelize(List(args(2).toInt)).toDF("UserId")
      
      val recommendations = dataframemodel.recommendForUserSubset(dfWithSchema , args(3).toInt)
      recommendations.show(false)
    }
}