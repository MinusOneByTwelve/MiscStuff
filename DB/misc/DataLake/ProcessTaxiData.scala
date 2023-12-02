// Databricks notebook source
// MAGIC %md ###Section 1: Exploration Operations

// COMMAND ----------

display(
  dbutils.fs.ls("mnt/DatalakeGen2/")
)

// COMMAND ----------

 dbutils.fs.head("mnt/DatalakeGen2/YellowTaxiTripData.csv")

// COMMAND ----------

var yellowTaxiTripDataDF = spark
    .read
  //  .option("header", "true")    
    .csv("/mnt/DatalakeGen2/YellowTaxiTripData.csv")

// COMMAND ----------

display(yellowTaxiTripDataDF)

// COMMAND ----------

// MAGIC %md ###Section 2: Analyse Data

// COMMAND ----------

display(
    yellowTaxiTripDataDF.describe(
                                     "passenger_count",                                     
                                     "trip_distance"                                     
                                 )
)

// COMMAND ----------

// MAGIC %md ###Section 3: Clean Data

// COMMAND ----------

// Display the count before filtering
println("Before filter = " + yellowTaxiTripDataDF.count())

// Filter inaccurate data
yellowTaxiTripDataDF = yellowTaxiTripDataDF
                          .where("passenger_count > 0")
                          .filter($"trip_distance" > 0.0)

// Display the count after filtering
println("After filter = " + yellowTaxiTripDataDF.count())

// COMMAND ----------



// COMMAND ----------

// Display the count before filtering
println("Before filter = " + yellowTaxiTripDataDF.count())

// Drop rows with nulls in PULocationID or DOLocationID
yellowTaxiTripDataDF = yellowTaxiTripDataDF
                          .na.drop(
                                    Seq("PULocationID", "DOLocationID")
                                  )

// Display the count after filtering
println("After filter = " + yellowTaxiTripDataDF.count())

// COMMAND ----------

// MAGIC %md ###Section 4: Transformation

// COMMAND ----------

// Rename the columns
yellowTaxiTripDataDF = yellowTaxiTripDataDF                                                
                        .withColumnRenamed("PUlocationID", "PickupLocationId")
                        .withColumnRenamed("DOlocationID", "DropLocationId")                        
yellowTaxiTripDataDF.printSchema

// COMMAND ----------

// MAGIC %md ###Section 5: Loading Data

// COMMAND ----------

// Load the dataframe as CSV to data lake
yellowTaxiTripDataDF  
    .write
    .option("header", "true")
    .option("dateFormat", "yyyy-MM-dd HH:mm:ss.S")
    .mode(SaveMode.Overwrite)
    .csv("/mnt/DatalakeGen2/ProcessedTaxiData/YellowTaxiData.csv")

// COMMAND ----------

// Load the dataframe as CSV to data lake
yellowTaxiTripDataDF  
    .write
    .option("header", "true")
    .option("dateFormat", "yyyy-MM-dd HH:mm:ss.S")
    .mode(SaveMode.Overwrite)
    .parquet("/mnt/DatalakeGen2/ProcessedTaxiDataParquet/YellowTaxiData.parquet")


// COMMAND ----------


