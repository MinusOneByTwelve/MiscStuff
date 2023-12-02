#spark-submit --jars spark-csv_2.11-1.5.0.jar,commons-csv-1.8.jar,spark-xml_2.11-0.9.0.jar --master local[*] xml.py
#export PYTHONIOENCODING=utf8

import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('PySparkXML').getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

DF_XML = spark.read.format("com.databricks.spark.xml").option("rowTag", "Employee").load("dbfs:/FileStore/tables/employees.xml");
DF_XML.show(truncate=False)
DF_XML.printSchema()
