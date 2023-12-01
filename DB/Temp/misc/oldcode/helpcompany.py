#spark-submit --jars spark-csv_2.11-1.5.0.jar,commons-csv-1.8.jar,spark-xml_2.11-0.9.0.jar --master local[*] helpcompany.py
#export PYTHONIOENCODING=utf8

import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('HelpCompany').getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

DF_XML = spark.read.format("com.databricks.spark.xml").option("rootTag", "Employees").option("rowTag", "Employee").load("hdfs://randstad-niit-bigdataspark-164-52-214-120-e2e7-70155-ncr.cluster:8020/user/bigdata/employees.xml");
DF_XML.show(truncate=False)
DF_XML.printSchema()

DF_CSV = spark.read.options(header='True', delimiter=',').csv("hdfs://randstad-niit-bigdataspark-164-52-214-120-e2e7-70155-ncr.cluster:8020/user/bigdata/employees.csv")
DF_CSV.show(truncate=False)
DF_CSV.printSchema()

DF_JSON = spark.read.option("multiline","true").json("hdfs://randstad-niit-bigdataspark-164-52-214-120-e2e7-70155-ncr.cluster:8020/user/bigdata/employees.json")
DF_JSON.show(truncate=False)
DF_JSON.printSchema()

from pyspark.sql.functions import explode
DF_JSON_Expand = DF_JSON.select(explode(DF_JSON["Employees"]).alias("Employees"))
#DF_XML_Expand = DF_XML_Expand.select(DF_XML_Expand.col("Id"),
#DF_XML_Expand.col("Addresses").getField("street").as("Street"), 
#DF_XML_Expand.col("Addresses").getField("city").as("City"));
DF_JSON_Expand.show(truncate=False)
DF_JSON_Expand.printSchema()

DF_JSON_Final = DF_JSON_Expand.withColumn("emailAddress",DF_JSON_Expand["Employees.emailAddress"]).withColumn("empId",DF_JSON_Expand["Employees.empId"]).withColumn("phoneNumber",DF_JSON_Expand["Employees.phoneNumber"])
DF_JSON_Final.show(truncate=False)
DF_JSON_Final.printSchema()

DF_XML.createOrReplaceTempView("dataxml")
DF_CSV.createOrReplaceTempView("datacsv")
DF_JSON_Final.createOrReplaceTempView("datajson")

#DF_RESULT = spark.sql("")
#DF_RESULT.coalesce(1).write.mode('Overwrite').csv("/user/bigdata/companyresult") 