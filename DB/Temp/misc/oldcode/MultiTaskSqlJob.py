# -*- coding: utf-8 -*-

"""
dbfs:/FileStore/tables/MultiTaskData/employees.csv
dbfs:/FileStore/tables/MultiTaskData/employees.json
dbfs:/FileStore/tables/MultiTaskData/employees.xml
dbfs:/FileStore/tables/MultiTaskFiles/MultiTaskSqlJob.py
/FileStore/MultiTaskJob
/FileStore/MultiTaskJob/XML
/FileStore/MultiTaskJob/JSON
/FileStore/MultiTaskJob/CSV
/FileStore/MultiTaskJob/MYSQL

["MultiTaskDBJob","A","/FileStore/MultiTaskJob"]
["MultiTaskDBJob","B1","/FileStore/tables/MultiTaskData/employees.xml","Employee","/FileStore/MultiTaskJob/XML"]
["MultiTaskDBJob","B2","/FileStore/tables/MultiTaskData/employees.csv","/FileStore/MultiTaskJob/CSV"]
["MultiTaskDBJob","B3","/FileStore/tables/MultiTaskData/employees.json","/FileStore/MultiTaskJob/JSON"]
["MultiTaskDBJob","B4","/FileStore/MultiTaskJob/MYSQL","216.48.181.38","retail_db","employeesql","mysqluser","mysqluser123"]
["MultiTaskDBJob","C","/FileStore/MultiTaskJob","/FileStore/MultiTaskJob/XML","/FileStore/MultiTaskJob/JSON","/FileStore/MultiTaskJob/CSV","/FileStore/MultiTaskJob/MYSQL"]
["MultiTaskDBJob","D","/FileStore/MultiTaskJob/FinalResult","/FileStore/MultiTaskJob/XML","/FileStore/MultiTaskJob/JSON","/FileStore/MultiTaskJob/CSV","/FileStore/MultiTaskJob/MYSQL"]
"""

import os
import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
from pyspark.sql import DataFrame
  
print("Start Of Stage -: "+sys.argv[2])
print("")

def main():
  print("Main Start")
  print("") 
    
  SparkSQLContext = SparkSession.builder.appName(sys.argv[1]).getOrCreate() 
  
  if(sys.argv[2] == "A"): 
    cleanup(SparkSQLContext)
  if(sys.argv[2] == "C"): 
    checkfiles(SparkSQLContext)    
  if(sys.argv[2] == "D"): 
    finalwork(SparkSQLContext)
  if(sys.argv[2] == "B1"): 
    xmlwork(SparkSQLContext)
  if(sys.argv[2] == "B2"): 
    csvwork(SparkSQLContext)
  if(sys.argv[2] == "B3"): 
    jsonwork(SparkSQLContext)
  if(sys.argv[2] == "B4"): 
    mysqlwork(SparkSQLContext)
                      
  print("Main End") 
  print("")   

def finalwork(SQLContext):
  print("FinalWork Start")
  
  DF_XML = SQLContext.read.format("com.databricks.spark.avro").load("dbfs:"+sys.argv[4]+"/*");
  DF_CSV = SQLContext.read.parquet("dbfs:"+sys.argv[5]+"/*");
  DF_JSON = SQLContext.read.parquet("dbfs:"+sys.argv[6]+"/*");
  DF_MYSQL = SQLContext.read.orc("dbfs:"+sys.argv[7]+"/*"); 
  SQLContext.udf.registerJavaFunction("PerfBonus", "Training.CustomFunc.PerfBonus", IntegerType())
  
  DF_XML.createOrReplaceTempView("employeexml")    
  DF_CSV.createOrReplaceTempView("employeejson") 
  DF_JSON.createOrReplaceTempView("employeecsv") 
  DF_MYSQL.createOrReplaceTempView("employeesql")
  
  DF_Result = SQLContext.sql("select a.empId as EmployeeId,c.first_name as FirstName,a.lastName as LastName,a.gender as Gender,b.emailAddress as Email,b.phoneNumber as Contact,d.dept as Department,a.salary as Salary,PerfBonus(c.first_name) as Bonus from employeexml a inner join employeejson b on a.empId=b.empId inner join employeecsv c on c.emp_no=b.empId inner join employeesql d on d.emp_no=c.emp_no") 
  DF_Result.show(truncate=False)
  DF_Result.printSchema() 
  DF_Result.coalesce(1).write.mode("Overwrite").option("header", "true").parquet("dbfs:"+sys.argv[3])   
             
  print("FinalWork End") 
  print("")
  
def xmlwork(SQLContext):
  print("XML Start")
  
  DF_XML = SQLContext.read.format("com.databricks.spark.xml").option("rowTag", sys.argv[4]).load("dbfs:"+sys.argv[3]);
  DF_XML.show(truncate=False)
  DF_XML.printSchema()    
  DF_XML.coalesce(1).write.mode("Overwrite").option("header", "true").format("com.databricks.spark.avro").save("dbfs:"+sys.argv[5])  
  
  print("XML End") 
  print("") 

def csvwork(SQLContext):
  print("CSV Start")
  
  DF_CSV = SQLContext.read.format("com.databricks.spark.csv").option("header", "True").load("dbfs:"+sys.argv[3]);
  DF_CSV.show(truncate=False)
  DF_CSV.printSchema() 
  DF_CSV.coalesce(1).write.mode("Overwrite").option("header", "true").parquet("dbfs:"+sys.argv[4])   
  
  print("CSV End") 
  print("")
  
def jsonwork(SQLContext):
  print("JSON Start")
  
  DF_JSON = SQLContext.read.json("dbfs:"+sys.argv[3]);
  DF_JSON.show(truncate=False)
  DF_JSON.printSchema() 
  DF_JSON.coalesce(1).write.mode("Overwrite").option("header", "true").parquet("dbfs:"+sys.argv[4])   
  
  print("JSON End") 
  print("")  

def mysqlwork(SQLContext):
  print("MYSQL Start")

  DF_MYSQL=SQLContext.read.format("jdbc").option("url", "jdbc:mysql://"+sys.argv[4]+"/"+sys.argv[5]).option("driver", "com.mysql.jdbc.Driver").option("dbtable", sys.argv[6]).option("user", sys.argv[7]).option("password", sys.argv[8]).load()
  DF_MYSQL.show(truncate=False)
  DF_MYSQL.printSchema()
  DF_MYSQL.persist( pyspark.StorageLevel.MEMORY_AND_DISK )
  DF_MYSQL.coalesce(1).write.mode("Overwrite").option("header", "true").orc("dbfs:"+sys.argv[3])   
 
  print("MYSQL End") 
  print("") 
      
def cleanup(spark):
  print("CleanUp Start")
  
  dbutils1 = get_db_utils(spark)
  dbutils1.fs.rm(sys.argv[3], True)
  dbutils.fs.mkdirs(sys.argv[3])  
  
  print("CleanUp End") 
  print("") 
    
def checkfiles(SparkSQLContext):
  print("CheckFiles Start")
  
  Sum = 0
  Done1 = 0
  Done2 = 0
  Done3 = 0
  Done4 = 0  
  dbutils2 = get_db_utils(SparkSQLContext)
  
  while True:
    if(file_exists(sys.argv[4],SparkSQLContext)): 
      if(Done1 == 0):
        print(sys.argv[4]+" Ready");
        dbutils2.fs.ls(sys.argv[4])
        Sum += 1
        Done1 = 1

    if(file_exists(sys.argv[5],SparkSQLContext)): 
      if(Done2 == 0):
        print(sys.argv[5]+" Ready");
        dbutils2.fs.ls(sys.argv[5])
        Sum += 1
        Done2 = 1
      
    if(file_exists(sys.argv[6],SparkSQLContext)): 
      if(Done3 == 0):
        print(sys.argv[6]+" Ready");
        dbutils2.fs.ls(sys.argv[6])
        Sum += 1
        Done3 = 1
      
    if(file_exists(sys.argv[7],SparkSQLContext)): 
      if(Done4 == 0):
        print(sys.argv[7]+" Ready");
        dbutils2.fs.ls(sys.argv[7])
        Sum += 1
        Done4 = 1             
       
    if Sum == 4:
        break  
        
  print("CheckFiles End") 
  print("")        

def get_db_utils(spark):
  dbutils = None
  if spark.conf.get("spark.databricks.service.client.enabled") == "true":
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)
  else:
    import IPython
    dbutils = IPython.get_ipython().user_ns["dbutils"]
  return dbutils

def file_exists(path,spark):
  if path[:5] == "/dbfs":
    import os
    return os.path.exists(path)
  else:
    try:
      dbutils = get_db_utils(spark)
      dbutils.fs.ls(path)
      return True
    except Exception as e:
      if 'java.io.FileNotFoundException' in str(e):
        return False
      else:
        return False

def ComplexBusinessLogic(input_df: DataFrame) -> DataFrame:
    inter_df = input_df.where(input_df['Category'] == \
                              F.lit('Expenses')).groupBy('Department').agg(F.sum('Percentage').alias('FinalPerc'))
    output_df = inter_df.select('Department', 'FinalPerc', \
                                F.when(F.col('FinalPerc') > 10, 'yes').otherwise('no').alias('Indicator')).where(
                F.col('Indicator') == F.lit('yes'))
    return output_df
    
main()

print("End Of Stage -: "+sys.argv[2])
print("")
