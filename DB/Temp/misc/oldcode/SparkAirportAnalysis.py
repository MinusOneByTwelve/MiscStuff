#export PYTHONIOENCODING=utf8
#spark-submit --master local[*] SparkAirportAnalysis.py
#spark-submit --master yarn --deploy-mode cluster --num-executors 1 --driver-memory 512m --executor-memory 512m --executor-cores 1 SparkAirportAnalysis.py
from pyspark.sql.types import *
from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("Sprint79-Story56")
sc = SparkContext(conf=conf) 
sc.setLogLevel("ERROR")
from pyspark.sql import SQLContext
sqlContext = SQLContext(sc)

A1__ = sc.textFile("hdfs://bigdata-216-48-184-61-e2e7-100461-ncr.cluster:8020/user/bigdata/Final_airlines")
A1_ = A1__.map(lambda l: l.encode('utf8').split(","))
A1 = A1_.map(lambda p: (p[0], p[1]))
A1Schema_ = "Airline_ID Name"
A1SchemaFields = [StructField(field_name, StringType(), True) for field_name in A1Schema_.split()]
A1Schema = StructType(A1SchemaFields)
A1_DF = sqlContext.createDataFrame(A1, A1Schema)
A1_DF.createOrReplaceTempView("Airlines")

A2__ = sc.textFile("hdfs://bigdata-216-48-184-61-e2e7-100461-ncr.cluster:8020/user/bigdata/routes.dat")
A2_ = A2__.map(lambda l: l.encode('utf8').split(","))
A2 = A2_.map(lambda p: (p[1], p[7]))
A2Schema_ = "Airline_ID Stops"
A2SchemaFields = [StructField(field_name, StringType(), True) for field_name in A2Schema_.split()]
A2Schema = StructType(A2SchemaFields)
A2_DF = sqlContext.createDataFrame(A2, A2Schema)
A2_DF.createOrReplaceTempView("Routes")

A3__ = sc.textFile("hdfs://bigdata-216-48-184-61-e2e7-100461-ncr.cluster:8020/user/bigdata/airports_mod.dat")
A3_ = A3__.map(lambda l: l.encode('utf8').split("\t"))
A3 = A3_.map(lambda p: (p[0], p[1], p[2], p[3]))
A3Schema_ = "Airline_ID Name City Country"
A3SchemaFields = [StructField(field_name, StringType(), True) for field_name in A3Schema_.split()]
A3Schema = StructType(A3SchemaFields)
A3_DF = sqlContext.createDataFrame(A3, A3Schema)
A3_DF.createOrReplaceTempView("Airports")

sqlContext.sql("select distinct(a.Name)as Airlines from Airlines a join Routes b on a.Airline_ID=b.Airline_ID where b.Stops=0 order by Airlines").show()
sqlContext.sql("select a.Name,Stops from Airlines a join Routes r on a.Airline_ID=r.Airline_ID where Stops=0 group by a.Name,Stops order by a.Name").show()
