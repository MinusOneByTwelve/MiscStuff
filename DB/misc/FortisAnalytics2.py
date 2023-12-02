#while read line; do echo -e "$line\n"; sleep .1; done < SampleData2.csv | nc -lk 9996
#spark-submit --jars mysql-connector-java-5.1.49.jar --master local[*] FortisAnalytics2.py localhost 9996 5 /tmp/FortisAnalytics2 15 5
    
import sys
import pyspark
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.sql import Row, SparkSession

def getSparkSessionInstance(sparkConf):
    if ('sparkSessionSingletonInstance' not in globals()):
        globals()['sparkSessionSingletonInstance'] = SparkSession\
            .builder\
            .config(conf=sparkConf)\
            .getOrCreate()
    return globals()['sparkSessionSingletonInstance']

def CheckAndTake(line):
    columns = line.split(',')
    colslen = len(columns)
    
    if colslen == 7:
        if line.startswith("PATIENT"):
            return ("NA", "NA", "NA", 0, 0, 0, 0)
        else:
            return (columns[0],columns[1],columns[2],int(columns[3]),int(columns[4]),int(columns[5]),int(columns[6]))    
    else:
        return ("NA", "NA", "NA", 0, 0, 0, 0)
    return ("NA", "NA", "NA", 0, 0, 0, 0)

def PrintRequiredDStream(time, rdd):
    try:
        if rdd.isEmpty():
            print ""
        else:
            spark = getSparkSessionInstance(rdd.context.getConf())
            
            DF_Patients = getPatientMaster(spark)
            DF_Patients.createOrReplaceTempView("Patients")
                    
            DF = spark.createDataFrame(rdd)
            DF.createOrReplaceTempView("SensorInference") 
            SensorInferenceAvg = spark.sql("select CONCAT(b.first_name, ' ',b.last_name) as PatientName,a.Alert as Condition from SensorInference a inner join Patients b on a.Patient = b.id order by PatientName") 
            print "=== Sensor Alert === [ " + str(time) + " ] === [ Batch Size : " + sys.argv[3] + " Seconds ] ==="
            SensorInferenceAvg.show(truncate=False)            
    except Exception as e: print(e) 

def getPatientMaster(spark):
    if ('PatientMaster' not in globals()):        
        #DF_MySql=spark.read.format("jdbc").option("url", "jdbc:mysql://Randstad-NIIT-Repo/retail_db").option("driver", "com.mysql.jdbc.Driver").option("dbtable", "PatientMaster").option("user", "mysqluser").option("password", "mysqluser123").load()
        DF_MySql=spark.read.format("jdbc").option("url", "jdbc:mysql://216.48.181.38/retail_db").option("driver", "com.mysql.jdbc.Driver").option("dbtable", "Drivers").option("user", "mysqluser").option("password", "mysqluser123").load()
        DF_MySql.persist( pyspark.StorageLevel.MEMORY_AND_DISK )
        globals()['PatientMaster'] = DF_MySql
    return globals()['PatientMaster']

def PatientCondition(Temp,HeartRate,SYS,DIA):
    if Temp >=35 and Temp <=97 and HeartRate >=60 and HeartRate <=100 and SYS >=90 and SYS <=120 and DIA >=60 and DIA <=80:
        return "Normal"
    elif Temp >=35 and Temp <=97 and HeartRate > 100 and SYS >=90 and SYS <=120 and DIA >=60 and DIA <=80:
        return "Breathing Difficulty"
    elif Temp >=35 and Temp <=97 and HeartRate >=60 and HeartRate <=100 and SYS < 90 and DIA < 60:
        return "Low BP"        
    elif Temp >=35 and Temp <=97 and HeartRate >=60 and HeartRate <=100 and SYS > 130 and DIA >= 80:
        return "High BP"        
    elif (Temp < 35 or Temp > 97) and HeartRate >=60 and HeartRate <=100 and SYS >=90 and SYS <=120 and DIA >=60 and DIA <=80:
        return "Temperature"
    else:
        return "No Clue..."
    
def TransformData(rdd):
    if rdd.isEmpty():
        return rdd
    else:        
        spark = getSparkSessionInstance(rdd.context.getConf())
        rowRdd = rdd.map(lambda x: Row(Patient=x[0],Date=x[1],Time=x[2],Temp=x[3],HeartRate=x[4],SYS=x[5],DIA=x[6]))
        newformatdataDF = spark.createDataFrame(rowRdd)
        newformatdataDF.createOrReplaceTempView("SensorInference")       
        SensorInferenceAvg = spark.sql("select * from(select Patient,avg(Temp) as Temp,avg(HeartRate) as HeartRate, avg(SYS) as SYS,avg(DIA) as DIA from SensorInference group by Patient)obj")
        SensorInferenceAvgRDD = SensorInferenceAvg.rdd.map(lambda x: Row(Patient=x['Patient'],Temp=x['Temp'],HeartRate=x['HeartRate'],SYS=x['SYS'],DIA=x['DIA'],Alert=PatientCondition(x['Temp'],x['HeartRate'],x['SYS'],x['DIA'])))       
        return SensorInferenceAvgRDD
    
def CreateSSC(timeinterval,cp):
    print "Creating SSC..."
    sparkConf = SparkConf().setAppName("FortisAnalytics2")
    sc = SparkContext.getOrCreate(sparkConf)
    sc.setLogLevel("ERROR")
        
    ssc = StreamingContext(sc, timeinterval)
    
    datastream = ssc.socketTextStream(sys.argv[1], int(sys.argv[2]), pyspark.StorageLevel.MEMORY_AND_DISK)
    properdata = datastream.map(CheckAndTake).filter(lambda x: ("NA" not in x[0]))
        
    RequiredDStream = properdata.transform(lambda rdd: TransformData(rdd))
    #RequiredDStream.pprint()
    RequiredDStream.foreachRDD(PrintRequiredDStream)
                          
    return ssc
    
if __name__ == "__main__":    
    ssc = StreamingContext.getOrCreate(sys.argv[4],lambda: CreateSSC(int(sys.argv[3]), sys.argv[4]))
    scnow = ssc.sparkContext
    scnow.setLogLevel("ERROR")
      
    ssc.start()
    ssc.awaitTermination()
