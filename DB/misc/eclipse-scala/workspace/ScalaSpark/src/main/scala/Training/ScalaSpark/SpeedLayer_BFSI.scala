package Training.ScalaSpark

import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.storage.StorageLevel
import org.apache.log4j._
import org.apache.spark.sql.functions._
import org.apache.spark.streaming._
/*import org.apache.kafka.clients.consumer.ConsumerRecord
import org.apache.kafka.common.serialization.StringDeserializer
import org.apache.spark.streaming.kafka010._
import org.apache.spark.streaming.kafka010.LocationStrategies.PreferConsistent
import org.apache.spark.streaming.kafka010.ConsumerStrategies.Subscribe*/

object SpeedLayer_BFSI {

/*
while read line; do echo -e "$line\n"; sleep .1; done < atmfeed.csv | nc -lk 9998

spark-submit --class "Training.ScalaSpark.SpeedLayer_BFSI" --master local[*] SpeedLayer_BFSI.jar localhost 9998 10 30 10 1 ATM_Transactions /tmp/BFSI e2e-72-55.e2enetworks.net.in:9092 1
*/

  case class Trans(CardNo:String, Amount:Int, ATM_Code:String, ATM_Type:String, 
      ATM_Lat:Double, ATM_Long:Double, TimeStamp:java.sql.Timestamp)

  def model_mapper(T:(String,(Any,String,String,Any,Any,String))): Trans = {
    val format = new java.text.SimpleDateFormat("yyyy-MM-dd HH:mm:ss")
    
    val trans:Trans = Trans(T._1,T._2._1.toString().toInt,T._2._2,T._2._3,
        T._2._4.toString().toDouble,T._2._5.toString().toDouble,
        new java.sql.Timestamp(format.parse(T._2._6).getTime()))
        
    return trans
  }
  
  def generic_mapper(line:String) = {
    val fields = line.split(',')
    if(fields.length == 8){(fields(1),(fields(0),fields(3),fields(4),fields(5),fields(6),fields(7)))}
    else{("NA",(0,"NA","NA",0.0,0.0,"NA"))}
  }

  def DistanceInKM = udf((Lat1:Double,Long1:Double,Lat2:Double,Long2:Double) => {
    val AVERAGE_RADIUS_OF_EARTH_KM = 6371
    val latDistance = Math.toRadians(Lat1 - Lat2)
    val lngDistance = Math.toRadians(Long1 - Long2)
    val sinLat = Math.sin(latDistance / 2)
    val sinLng = Math.sin(lngDistance / 2)
    val a = sinLat * sinLat +
      (Math.cos(Math.toRadians(Lat1))
        * Math.cos(Math.toRadians(Lat2))
        * sinLng * sinLng)
    val c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
    (AVERAGE_RADIUS_OF_EARTH_KM * c).toInt    
  })

  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.ERROR)
  
    val sparkConf = new SparkConf().setAppName("Training.ScalaSpark.BFSI.SpeedLayer")
    val sc = new StreamingContext(sparkConf, Seconds(args(2).toInt))
    sc.checkpoint(args(7));
    sc.sparkContext.setLogLevel("ERROR");
        
    /* FOR KAFKA */
   /* val kafkaParams = Map[String, Object](
      "bootstrap.servers" -> args(8),
      "key.deserializer" -> classOf[StringDeserializer],
      "value.deserializer" -> classOf[StringDeserializer],
      "group.id" -> "TrainingScalaSpark_BFSI_SpeedLayer",
      "auto.offset.reset" -> "earliest",
      //"fetch.min.bytes" -> "1",
      //"max.poll.records" -> "10",
      //"max.partition.fetch.bytes" -> "1",
      //"fetch.max.bytes" -> "50",      
      "enable.auto.commit" -> (false: java.lang.Boolean)
    )    
    val topics = Array(args(6))
    val datastream_ = KafkaUtils.createDirectStream[String, String](
      sc,
      PreferConsistent,
      Subscribe[String, String](topics, kafkaParams)
    ) 
    var datastream = datastream_.map(record => record.value)*/
    /* FOR KAFKA */
    
    //if (args(9).toInt > 0) {
    /* FROM TEXTFILE */
    //val datastream = sc.socketTextStream(args(0), args(1).toInt, StorageLevel.MEMORY_AND_DISK_SER)   
    /* FROM TEXTFILE */
    //}
    
    val datastream = sc.socketTextStream(args(0), args(1).toInt, StorageLevel.MEMORY_AND_DISK_SER)
    
    val mapdatatopair =  datastream.map(generic_mapper)   
    val atmusagecount =  mapdatatopair.filter(t => (t._1 != "NA")).map(t => (t._1,1)).reduceByKey(_+_).filter(t => (t._2 > 1))           
    val takerelevantdata = atmusagecount.join(mapdatatopair).map(t => (t._1,t._2._2))  
       
    val RequiredDStream = takerelevantdata.transform { rdd =>
      {
        val spark = SparkSessionSingleton.getInstance(rdd.sparkContext.getConf)
        import spark.implicits._ 

        val PotentialFraudulentDF = rdd.map(model_mapper).toDF()
        PotentialFraudulentDF.createOrReplaceTempView("FraudSuspect1")
  
        val PotentialFraudDetection_1 =
          spark.sql("""
            select * from
            (
              SELECT CardNo,Amount,ATM_Code,ATM_Type,TimeStamp,
              ATM_Lat,LAG(ATM_Lat, 1) OVER (PARTITION BY CardNo ORDER BY TimeStamp) Last_ATM_Lat,
              ATM_Long,LAG(ATM_Long, 1) OVER (PARTITION BY CardNo ORDER BY TimeStamp) Last_ATM_Long,
              LAG(ATM_Code, 1) OVER (PARTITION BY CardNo ORDER BY TimeStamp) Last_ATM,
              LAG(ATM_Type, 1) OVER (PARTITION BY CardNo ORDER BY TimeStamp) Last_ATM_Type 
              FROM FraudSuspect1
            )Obj 
            where (Last_ATM_Lat is not null and Last_ATM_Long is not null)
            and (Last_ATM_Lat <> ATM_Lat and Last_ATM_Long <> ATM_Long)    
            """)
                
        val PotentialFraudDetection_2 = PotentialFraudDetection_1.withColumn("DistanceInKM",
            DistanceInKM($"Last_ATM_Lat",$"Last_ATM_Long",$"ATM_Lat",$"ATM_Long"))
        PotentialFraudDetection_2.createOrReplaceTempView("FraudSuspect2")
        
        val PotentialFraudDetection_3 =
          spark.sql("select CONCAT(Last_ATM, \"_\", Last_ATM_Type, \"=>\", ATM_Code, \"_\", ATM_Type) AS Origin_Fraud,CardNo from FraudSuspect2 where DistanceInKM > "+args(5))      
        
        PotentialFraudDetection_3.rdd
      }
    } 
    
    RequiredDStream.foreachRDD 
    { 
      (rdd, time) =>
        if(!rdd.isEmpty()){
          val spark = SparkSessionSingleton.getInstance(rdd.sparkContext.getConf)
          import spark.implicits._
    
          val FraudulentCardsDF = rdd.map(R => (R.get(0).toString(),R.get(1).toString())).toDF("Origin_Fraud","CardNo")
          
          println(s"=== Compromised Cards === [ $time ] === [ Batch Size : "+args(2)+" Seconds ] ===")
          
          FraudulentCardsDF.select(FraudulentCardsDF("CardNo")).show(false)
      }
    }
    
    val SuspectATMDStream = RequiredDStream.map(T => (T.getAs("Origin_Fraud").toString().split('_')(0),1))
    val SuspectATMWindow =
        		SuspectATMDStream.reduceByKeyAndWindow((a:Int,b:Int) => (a + b),(a:Int,b:Int) => (a - b),
        		    Durations.seconds(args(3).toInt), Durations.seconds(args(4).toInt))
    SuspectATMWindow.foreachRDD 
    { 
      (rdd, time) =>
        if(!rdd.isEmpty()){        
          val spark = SparkSessionSingleton.getInstance(rdd.sparkContext.getConf)
          import spark.implicits._
    
          val ATMAffectedDF = rdd.toDF("ATM","Incidents")
          ATMAffectedDF.createOrReplaceTempView("ATMAffected")
    
          val ATMAffected =
            spark.sql("select * from ATMAffected order by Incidents desc")
              
          println(s"=== Compromised ATM === [ $time ] === [ Window Length : "+args(3)+" Seconds / Slide Interval : "+args(4)+" Seconds ] ===")
          
          ATMAffected.select(ATMAffected("ATM"),ATMAffected("Incidents")).show(false)   
      }
    }
    
    val BankAffectedDStream = RequiredDStream.map(T => 
      ({
        val t0 = T.getAs("Origin_Fraud").toString();
        val t1 = t0.split('_')(0);
        val t2 = t1.takeRight(3);
        t1.replace(t2, "")+"_"+t0.split("=>")(0).split('_')(1)
        },1)
      )
    def BankAffected(newData: Seq[Int], state: Option[Int]) = {
      val newState = state.getOrElse(0) + newData.sum
      Some(newState)
    }
    val BankAffectedFinalDStream = BankAffectedDStream.updateStateByKey(BankAffected)
    BankAffectedFinalDStream.foreachRDD 
    { 
      (rdd, time) =>
        if(!rdd.isEmpty()){        
          val spark = SparkSessionSingleton.getInstance(rdd.sparkContext.getConf)
          import spark.implicits._
    
          val BankAffectedDF = rdd.toDF("Bank_TypeOfATM","Incidents")
          BankAffectedDF.createOrReplaceTempView("BankAffected")
    
          val BankAffected =
            spark.sql("select * from BankAffected order by Incidents desc limit 5")
              
          println(s"=== Most Vulnerable OverAll === [ $time ] === [ Top 5 ] ===")
          
          BankAffected.select(BankAffected("Bank_TypeOfATM"),BankAffected("Incidents")).show(false)   
      }
    }
    
    sc.start()
    sc.awaitTermination()      
  }
}