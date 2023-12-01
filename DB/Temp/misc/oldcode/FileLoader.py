#spark-submit --master local[*] FileLoader.py
#spark-submit --jars spark-csv_2.11-1.5.0.jar,commons-csv-1.8.jar,spark-xml_2.11-0.9.0.jar --master local[*] FileLoader.py
#export PYTHONIOENCODING=utf8

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, ArrayType, DoubleType, BooleanType

spark = SparkSession.builder.appName('FileLoader').getOrCreate()
#spark  = SparkSession.builder.appName('FileLoader').master("local").enableHiveSupport().getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

filename="hdfs://randstad-niit-bigdataspark-164-52-214-120-e2e7-70155-ncr.cluster:8020/user/bigdata/EmployeesAll.csv"

df = spark.read.csv(filename)
df.printSchema()

df2 = spark.read.option("header",True).csv(filename)
df2.printSchema()

df3 = spark.read.options(header='True', delimiter=',').csv(filename)
df3.printSchema()

schema = StructType().add("EmployeeId",IntegerType(),True).add("BirthDate",StringType(),True).add("FirstName",StringType(),True).add("LastName",StringType(),True).add("Gender",StringType(),True).add("JoiningDate",StringType(),True)
df4 = spark.read.format("csv").option("header", True).schema(schema).load(filename)
df4.printSchema()
df4.coalesce(1).write.mode('Overwrite').option("header",True).csv("/user/bigdata/EmpNice")

df2.write.csv("/user/bigdata/res1")
df2.write.mode('Overwrite').csv("/user/bigdata/res1")
df2.write.mode('Overwrite').option("header",True).csv("/user/bigdata/res1")
df2.write.mode('Overwrite').json("/user/bigdata/res2")
df2.coalesce(1).write.mode('Overwrite').parquet("/user/bigdata/res3") 
df2.coalesce(1).write.mode('Overwrite').orc("/user/bigdata/res4")
df2.coalesce(1).write.mode('Overwrite').format("com.databricks.spark.xml").option("rootTag", "Employees").option("rowTag", "Employee").save("/user/bigdata/res5")

multiline_df = spark.read.option("multiline","true").json("hdfs://randstad-niit-bigdataspark-164-52-214-120-e2e7-70155-ncr.cluster:8020/user/bigdata/sample.json")
multiline_df.show()

rdf1=spark.read.parquet("/user/bigdata/res3/*")
rdf1.createOrReplaceTempView("emps")
spark.sql("select * from emps where gender = 'M' and first_name like '%Ara%' and last_name like '%Ba%'").show(truncate=False)

rdf2=spark.read.orc("/user/bigdata/res4/*")
rdf2.createOrReplaceTempView("emps2")
spark.sql("select * from emps2 where gender = 'M' and first_name like '%Ara%' and last_name like '%Ba%'").show(truncate=False)

rdf3=spark.read.json("hdfs://randstad-niit-bigdataspark-164-52-214-120-e2e7-70155-ncr.cluster:8020/user/bigdata/olympic.json")
rdf3.createOrReplaceTempView("olympic")
spark.sql("select * from olympic limit 10").show(truncate=False)
rdf3.coalesce(1).write.mode('Overwrite').format("com.databricks.spark.avro").save("/user/bigdata/res6")

rdf4 = spark.read.format("com.databricks.spark.avro").load("/user/bigdata/res6/*.avro")
rdf4.createOrReplaceTempView("olympic2")
rdf4.printSchema()
spark.sql("select * from olympic2 limit 5").show(truncate=False)
