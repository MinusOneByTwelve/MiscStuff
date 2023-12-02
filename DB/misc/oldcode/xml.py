#spark-submit --jars spark-csv_2.11-1.5.0.jar,commons-csv-1.8.jar,spark-xml_2.11-0.9.0.jar --master local[*] xml.py
#export PYTHONIOENCODING=utf8

import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('PySparkXML').getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

DF_XML = spark.read.format("com.databricks.spark.xml").option("rowTag", "person").load("hdfs://randstad-niit-bigdataspark-164-52-214-120-e2e7-70155-ncr.cluster:8020/user/bigdata/person.xml");
DF_XML.show(truncate=False)
DF_XML.printSchema()

DF_XML2 = DF_XML.withColumn("Amount",DF_XML["salary._VALUE"]).withColumn("Currency",DF_XML["salary._currency"]).withColumn("AddressList",DF_XML["addresses.address"])
DF_XML2.show(truncate=False)
DF_XML2.printSchema()

from pyspark.sql.functions import explode
DF_XML_Expand = DF_XML2.select(DF_XML2["_id"].alias("Id"),explode(DF_XML2["AddressList"]).alias("Addresses"))
#DF_XML_Expand = DF_XML_Expand.select(DF_XML_Expand.col("Id"),
#DF_XML_Expand.col("Addresses").getField("street").as("Street"), 
#DF_XML_Expand.col("Addresses").getField("city").as("City"));
DF_XML_Expand.show(truncate=False)
DF_XML_Expand.printSchema()

DF_XML2 = spark.read.format("com.databricks.spark.xml").option("rootTag", "users").option("rowTag", "user").load("hdfs://randstad-niit-bigdataspark-164-52-214-120-e2e7-70155-ncr.cluster:8020/user/bigdata/person2.xml");
DF_XML2.show(truncate=False)
DF_XML2.printSchema()
