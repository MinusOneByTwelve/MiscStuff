package Training.CustomFunc
//spark.udf.registerJavaFunction("Tax", "Training.CustomFunc.Tax", StringType())
import scala.util.Try
import org.apache.spark.sql.api.java.UDF1

class Tax extends UDF1[Float, String] {

  override def call(Income: Float): String = {
    //val r = new scala.util.Random
    //val r1 = 5 + r.nextInt(( 30 - 5) + 1)

    val tax = Income*(9.7/100)
    
    ((tax * 100).round / 100.toDouble).toString()
  }
}