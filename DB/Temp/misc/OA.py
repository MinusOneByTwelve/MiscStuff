#while read line; do echo -e "$line\n"; sleep .1; done < olarides.csv | nc -lk 9998
#spark-submit --jars mysql-connector-java-5.1.49.jar --master local[*] OlaAnalytics.py localhost 9998 5 /tmp/OlaAnalytics 15 5
    
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
    
    if colslen == 5:
        if line.startswith("from"):
            return ("NA", "NA", 0, "NA", 0)
        else:
            return (columns[0],columns[1],int(columns[2]),columns[3],int(columns[4]))    
    else:
        return ("NA", "NA", 0, "NA", 0)
    return ("NA", "NA", 0, "NA", 0)

def PrintPopularRoutesWindow(time, rdd):
    try:
        if rdd.isEmpty():
            print ""
        else:
            spark = getSparkSessionInstance(rdd.context.getConf())
        
            rowRdd = rdd.map(lambda x: Row(Route=x[0], Revenue=x[1]))
        
            RouteDF = spark.createDataFrame(rowRdd)
            RouteDF.createOrReplaceTempView("RouteAnalysis")
        
            TopRoutes = spark.sql("select Route,sum(Revenue)as RevenueTotal from RouteAnalysis group by Route order by RevenueTotal desc limit 5")
        
            print "=== MOST POPULAR ROUTES === [ " + str(time) + " ] === [ Window Length : " + sys.argv[5] + " Seconds / Slide Interval : " + sys.argv[6] + " Seconds ] ==="
            TopRoutes.select("Route", "RevenueTotal").show(truncate=False)        
    except Exception as e: print(e)    

def getDrivers(spark):
    if ('DriversMaster' not in globals()):        
        DF_MySql=spark.read.format("jdbc").option("url", "jdbc:mysql://216.48.190.239/retail_db").option("driver", "com.mysql.jdbc.Driver").option("dbtable", "Drivers").option("user", "mysqluser").option("password", "mysqluser123").load()
        DF_MySql.persist( pyspark.StorageLevel.MEMORY_AND_DISK )
        globals()['DriversMaster'] = DF_MySql
    return globals()['DriversMaster']
    
def PrintDriverFinalDStream(time, rdd):
    try:
        if rdd.isEmpty():
            print ""
        else:
            spark = getSparkSessionInstance(rdd.context.getConf())
            
            DF_Drivers = getDrivers(spark)
            DF_Drivers.createOrReplaceTempView("Drivers")
            
            rowRdd = rdd.map(lambda x: Row(Driver=x[0], Revenue=x[1]))
        
            DriverFinalDF = spark.createDataFrame(rowRdd)
            DriverFinalDF.createOrReplaceTempView("DriverRevenue")
                           
            DriverFinal = spark.sql("select * from DriverRevenue a inner join Drivers b on a.Driver = b.id order by a.Revenue desc limit 5")
        
            print "=== Top Drivers === [ " + str(time) + " ] ==="
            DriverFinal.select("Driver", "first_name", "last_name", "email", "Revenue").show(truncate=False)
            #DriverFinal.coalesce(1).write.mode('append').csv("/user/bigdata/OlaAnalytics")        
    except Exception as e: print(e)
    
def PrintCabTypeDStream(time, rdd):
    try:
        if rdd.isEmpty():
            print ""
        else:
            spark = getSparkSessionInstance(rdd.context.getConf())
        
            rowRdd = rdd.map(lambda x: Row(CabType=x[0], Booked=x[1]))
        
            CabTypeDF = spark.createDataFrame(rowRdd)
            CabTypeDF.createOrReplaceTempView("Cabs")
        
            CabTypeFinal = spark.sql("select * from Cabs order by Booked desc limit 10")
        
            print "=== Booked Cabs === [ " + str(time) + " ] === [ Batch Size : " + sys.argv[3] + " Seconds ] ==="
            CabTypeFinal.select("CabType", "Booked").show(truncate=False)        
    except Exception as e: print(e)    
        
def updateFunc(new_values, last_sum):
    return sum(new_values) + (last_sum or 0)
    
def CreateSSC(timeinterval,cp):
    print "Creating SSC..."
    sparkConf = SparkConf().setAppName("OlaAnalytics")
    sc = SparkContext.getOrCreate(sparkConf)
    sc.setLogLevel("ERROR")
        
    ssc = StreamingContext(sc, timeinterval)
    ssc.checkpoint(cp)
    
    datastream = ssc.socketTextStream(sys.argv[1], int(sys.argv[2]), pyspark.StorageLevel.MEMORY_AND_DISK)
    properdata = datastream.map(CheckAndTake).filter(lambda x: ("NA" not in x[0]))
    
    RouteDStream = properdata.map(lambda x : (x[0]+"-"+x[1],x[4]))
    DriverDStream = properdata.map(lambda x : (x[2],x[4]))
    
    CabTypeDStream = properdata.map(lambda x : (x[3])).map(lambda x: (x, 1)).reduceByKey(lambda a, b: a+b)
    CabTypeDStream.foreachRDD(PrintCabTypeDStream)
    
    PopularRoutesWindow = RouteDStream.reduceByKeyAndWindow(lambda a, b: a + b, lambda a, b: a - b, int(sys.argv[5]), int(sys.argv[6]))
    PopularRoutesWindow.foreachRDD(PrintPopularRoutesWindow)
    
    DriverFinalDStream = DriverDStream.updateStateByKey(updateFunc)
    DriverFinalDStream.foreachRDD(PrintDriverFinalDStream)
              
    return ssc
    
if __name__ == "__main__":    
    ssc = StreamingContext.getOrCreate(sys.argv[4],lambda: CreateSSC(int(sys.argv[3]), sys.argv[4]))
    scnow = ssc.sparkContext
    scnow.setLogLevel("ERROR")
      
    ssc.start()
    ssc.awaitTermination()
