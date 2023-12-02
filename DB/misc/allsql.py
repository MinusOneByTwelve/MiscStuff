#spark-submit --jars spark-csv_2.11-1.5.0.jar,commons-csv-1.8.jar,spark-xml_2.11-0.9.0.jar,mysql-connector-java-5.1.49.jar,Tax.jar --master local[*] allsql.py
#export PYTHONIOENCODING=utf8

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, ArrayType, DoubleType, BooleanType, FloatType
from pyspark.sql.functions import udf
from pyspark.sql import Row

spark = SparkSession.builder.appName('SparkAllTypeSqlStuff').getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

PerfBonusPyFunc = udf(lambda x: PerfBonusPy(x), FloatType())

def PerfBonusPy(s):
    return (s*5.5)/100

spark.udf.register("GetPerfBonus", PerfBonusPyFunc)

DF_XML = spark.read.format("com.databricks.spark.xml").option("rootTag", "Employees").option("rowTag", "Employee").load("file:///opt/DataSet/employees.xml");
DF_XML.show(truncate=False)
DF_XML.printSchema()
DF_XML.coalesce(1).write.mode("Overwrite").option("header", "true").format("com.databricks.spark.avro").save("file:///opt/DataSet/res1")

DF_CSV = spark.read.format("com.databricks.spark.csv").option("delimiter", ",").option("header", "True").option("inferSchema", "True").load("file:///opt/DataSet/employees.csv");
DF_CSV.show(truncate=False)
DF_CSV.printSchema()
DF_CSV.coalesce(1).write.mode("Overwrite").option("header", "true").parquet("file:///opt/DataSet/res2") 

DF_JSON = spark.read.json("file:///opt/DataSet/emp.json");
DF_JSON.show(truncate=False)
DF_JSON.printSchema() 
DF_JSON.coalesce(1).write.mode("Overwrite").option("header", "true").parquet("file:///opt/DataSet/res3") 

DF_MYSQL=spark.read.format("jdbc").option("url", "jdbc:mysql://91.203.133.229/retail_db").option("driver", "com.mysql.jdbc.Driver").option("dbtable", "employeesql").option("user", "mysqluser").option("password", "mysqluser123").load()
DF_MYSQL.show(truncate=False)
DF_MYSQL.printSchema()
DF_MYSQL.persist( pyspark.StorageLevel.MEMORY_AND_DISK )
DF_MYSQL.coalesce(1).write.mode("Overwrite").option("header", "true").orc("file:///opt/DataSet/res4")

DF_XML_1 = spark.read.format("com.databricks.spark.avro").load("file:///opt/DataSet/res1/*");
DF_CSV_1 = spark.read.parquet("file:///opt/DataSet/res2/*");
DF_JSON_1 = spark.read.parquet("file:///opt/DataSet/res3/*");
DF_MYSQL_1 = spark.read.orc("file:///opt/DataSet/res4/*"); 

DF_XML_1.createOrReplaceTempView("employeexml")    
DF_JSON_1.createOrReplaceTempView("employeejson") 
DF_CSV_1.createOrReplaceTempView("employeecsv") 
DF_MYSQL_1.createOrReplaceTempView("employeesql")

spark.udf.registerJavaFunction("Tax", "Training.CustomFunc.Tax", StringType())

DF_Result = spark.sql("select /*+ BROADCAST(d) */ a.empId as EmployeeId,c.first_name as FirstName,upper(a.lastName) as LastName,a.gender as Gender,b.emailAddress as Email,b.phoneNumber as Contact,d.dept as Department,a.salary as Salary,GetPerfBonus(a.salary*12) as AnnualBonus,Tax(Salary+GetPerfBonus(a.salary*12)) as Tax,e.birth_date as BirthDate,e.hire_date as HireDate from employeexml a inner join employeejson b on a.empId=b.empId inner join employeecsv c on c.emp_no=b.empId inner join employeesql d on d.emp_no=c.emp_no inner join default.emps2 e on e.emp_no=d.emp_no") 
DF_Result.show(truncate=False)
DF_Result.printSchema() 

