#spark-submit --jars mysql-connector-java-5.1.49.jar --master local[*] mysqldf.py
#export PYTHONIOENCODING=utf8

import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('DFMySQL').getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

DF_MySql=spark.read.format("jdbc").option("url", "jdbc:mysql://ifmr-bigdata-164-52-215-39-e2e7-73669-ncr.cluster/retail_db").option("driver", "com.mysql.jdbc.Driver").option("dbtable", "customers").option("user", "mysqluser").option("password", "mysqluser123").load()
DF_MySql.registerTempTable("Customers")
DF_MySql.printSchema()
DF_MySql.persist( pyspark.StorageLevel.MEMORY_AND_DISK )
#DF_MySql.getStorageLevel()  
#print(DF_MySql.getStorageLevel())
DF_MySql = spark.sql("Select customer_fname as FirstName,customer_lname as LastName,customer_street Location from Customers where customer_city like '%New York%' order by LastName desc,FirstName").show(truncate=False)  