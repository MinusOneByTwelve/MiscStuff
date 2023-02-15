#spark-submit --jars spark-csv_2.11-1.5.0.jar,commons-csv-1.8.jar,spark-xml_2.11-0.9.0.jar --master local[*] xml.py
#export PYTHONIOENCODING=utf8

import pyspark
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("assignmentxml").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

Df_xml=spark.read.format("com.databricks.spark.xml").option("rootTag","Employees").option("rowTag",'Employees').load("hdfs://randstad-niit-bigdataspark-164-52-214-120-e2e7-70155-ncr.cluster:8020/training/deeparani/employees.xml")
Df_xml.show(truncate=False)
Df_xml.createOrReplaceTempView("employeexml")
Df_xml.printSchema()


Df_json=spark.read.option("multilines","true").json("hdfs://randstad-niit-bigdataspark-164-52-214-120-e2e7-70155-ncr.cluster:8020/training/deeparani/employees.json")
Df_json.show(truncate=False)
Df_json_Final.createOrReplaceTempView("employeejson")
Df_json.printSchema()

Df_csv=spark.read.options(header='True',delimeter=',').csv("hdfs://randstad-niit-bigdataspark-164-52-214-120-e2e7-70155-ncr.cluster:8020/training/deeparani/employees.csv")
Df_csv.show(truncate=False)
Df_csv.createOrReplaceTempView("employeecsv")
Df_csv.printSchema()


from pyspark.sql.functions import explode
Df_json_json_Expand = Df_json.select(explode(Df_json["Employees"]).alias("Employees"))
#DF_XML_Expand = DF_XML_Expand.select(DF_XML_Expand.col("Id"),
#DF_XML_Expand.col("Addresses").getField("street").as("Street"), 
#DF_XML_Expand.col("Addresses").getField("city").as("City"));
Df_json_Expand.show(truncate=False)
Df_json_Expand.printSchema()

Df_json_Final = Df_json_Expand.withColumn("emailAddress",Df_json_Expand["Employees.emailAddress"]).withColumn("empId",Df_json_Expand["Employees.empId"]).withColumn("phoneNumber",Df_json_Expand["Employees.phoneNumber"])
Df_json_Final.show(truncate=False)
Df_json_Final.printSchema()


df=spark.sql("select employeeid,employeefirtsname,employeelastname,gender,salary from employeesxml inner join employeejson on employeesxml.id=employeesjson.id inner join employeescsv on employeescsv.id=employeesjson.id")
df.coalesce(1).write.mode('Overwrite').option("header",True).csv("/training/deeparani/assignmentxml")

