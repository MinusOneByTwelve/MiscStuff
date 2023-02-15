#while read line; do echo -e "$line\n"; sleep .1; done < SampleData1.csv | nc -lk 9998
#spark-submit --master local[*] FortisAnalytics1.py localhost 9998 5 /tmp/FortisAnalytics1 15 5
    
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
    
    if colslen == 2:
        if line.startswith("PATIENT"):
            return ("NA", "NA")
        else:
            return (columns[0],columns[1])    
    else:
        return ("NA", "NA")
    return ("NA", "NA")

def PrintDiseaseWindow(time, rdd):
    try:
        if rdd.isEmpty():
            print ""
        else:
            spark = getSparkSessionInstance(rdd.context.getConf())
        
            rowRdd = rdd.map(lambda x: Row(PrincipalDiagnosis=x[0], Admissions=x[1]))
        
            EmergencyDF = spark.createDataFrame(rowRdd)
            EmergencyDF.createOrReplaceTempView("Emergency")
        
            EmergencyCases = spark.sql("select * from Emergency where PrincipalDiagnosis in('Covid','Low BP','Food Poisoning') order by Admissions desc")
        
            print "=== Emergency Cases === [ " + str(time) + " ] === [ Window Length : " + sys.argv[5] + " Seconds / Slide Interval : " + sys.argv[6] + " Seconds ] ==="
            EmergencyCases.select("PrincipalDiagnosis", "Admissions").show(truncate=False)       
    except Exception as e: print(e)    
   
def PrintDiseaseTDStream(time, rdd):
    try:
        if rdd.isEmpty():
            print ""
        else:
            spark = getSparkSessionInstance(rdd.context.getConf())
                                  
            rowRdd = rdd.map(lambda x: Row(PrincipalDiagnosis=x[0], Admissions=x[1]))
        
            AllTypesDF = spark.createDataFrame(rowRdd)
            AllTypesDF.createOrReplaceTempView("AllTypes")
                           
            AllTypesCases = spark.sql("select * from AllTypes order by Admissions desc limit 5")
        
            print "=== Total Cases === [ " + str(time) + " ] ==="
            AllTypesCases.select("PrincipalDiagnosis", "Admissions").show(truncate=False)        
    except Exception as e: print(e)
         
def updateFunc(new_values, last_sum):
    return sum(new_values) + (last_sum or 0)

def DiseaseType(Symptom):
    if Symptom == "YYYYYNNNNNNYNNN":
        return "Covid"
    elif Symptom == "NNNNNNNNYNNYNYY":
        return "Diabetes"
    elif Symptom == "NNNNNNNNYNNNYNN":
        return "Low BP"        
    elif Symptom == "NYNYNNNNNYNYNNN":
        return "Cold"        
    elif Symptom == "NNYNNNNNNYNYNNN":
        return "Flu"        
    elif Symptom == "NNNNYNNYNNNNYNN":
        return "Anaemia"        
    elif Symptom == "NYYNNNNNNNNNNNN":
        return "Malaria"        
    elif Symptom == "NNNNNYNYNNNNYNN":
        return "Food Poisoning"
    elif Symptom == "NNYYNNNYNNNYYNN":
        return "Typhoid"        
    elif Symptom == "NNNYNNNNNNNYNNN":
        return "General"
    else:
        return "NA"
    
def CreateSSC(timeinterval,cp):
    print "Creating SSC..."
    sparkConf = SparkConf().setAppName("FortisAnalytics1")
    sc = SparkContext.getOrCreate(sparkConf)
    sc.setLogLevel("ERROR")
        
    ssc = StreamingContext(sc, timeinterval)
    
    datastream = ssc.socketTextStream(sys.argv[1], int(sys.argv[2]), pyspark.StorageLevel.MEMORY_AND_DISK)
    properdata = datastream.map(CheckAndTake).filter(lambda x: ("NA" not in x[0]))
    
    WindowDStream = properdata.map(lambda x : (DiseaseType(x[1]),1))
    TotalDStream = properdata.map(lambda x : (DiseaseType(x[1]),1))
        
    DiseaseWindow = WindowDStream.reduceByKeyAndWindow(lambda a, b: a + b, lambda a, b: a - b, int(sys.argv[5]), int(sys.argv[6]))
    DiseaseWindow.foreachRDD(PrintDiseaseWindow)
    
    DiseaseTDStream = TotalDStream.updateStateByKey(updateFunc)
    DiseaseTDStream.foreachRDD(PrintDiseaseTDStream)
              
    return ssc
    
if __name__ == "__main__":    
    ssc = StreamingContext.getOrCreate(sys.argv[4],lambda: CreateSSC(int(sys.argv[3]), sys.argv[4]))
    scnow = ssc.sparkContext
    scnow.setLogLevel("ERROR")
      
    ssc.start()
    ssc.awaitTermination()
