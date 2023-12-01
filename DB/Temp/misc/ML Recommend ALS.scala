// Databricks notebook source
import org.apache.spark.ml.recommendation.ALS

// COMMAND ----------

val books_df = spark.read.option("delimiter", ";").
option("header", "true").csv("dbfs:/FileStore/tables/ML/Data/Books.csv")
books_df.createOrReplaceTempView("books")
books_df.show() 

// COMMAND ----------

val user_ratings_df = spark.read.option("delimiter", ";").
option("header", "true").csv("dbfs:/FileStore/tables/ML/Data/Ratings.csv")
user_ratings_df.createOrReplaceTempView("userrat")      
val ratings_df = spark.sql("select * from (select cast(`User-ID` as int) as UserId,cast(ISBN as int) as ISBN,cast(Rating as int) as Rating from userrat)obj where ISBN is not null")
ratings_df.createOrReplaceTempView("userrata")
ratings_df.show()

// COMMAND ----------

spark.sql("select * from userrata a join books b on a.ISBN = b.ISBN and a.UserId =276747").show()

// COMMAND ----------

/*To provide recommendations based on the ratings given by users, 
* we can use a technique called Collaborative Filtering. 
* This is based on the concept that if person A and B have given similar ratings to the same objects, 
* then they must have similar taste. Therefore, there is a higher probability that person A will like an object they haven’t 
* come across but is rated highly by B.

To perform collaborative filtering, we will use an algorithm called ALS (Alternating Least Squares), 
which will make predictions about how much each user would rate each book and ultimately provide recommendations
for every user listed in the dataset.*/

val als = new ALS().setMaxIter(5).setRegParam(0.01).setUserCol("UserId").setItemCol("ISBN").setRatingCol("Rating")
val dataframemodel = als.fit(ratings_df) 

// COMMAND ----------

import spark.implicits._
val dfWithSchema = spark.sparkContext.parallelize(List(276747)).toDF("UserId")

// COMMAND ----------

val recommendations = dataframemodel.recommendForUserSubset(dfWithSchema , 5)
recommendations.show(false)

// COMMAND ----------


