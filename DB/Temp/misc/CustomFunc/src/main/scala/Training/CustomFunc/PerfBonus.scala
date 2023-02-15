package Training.CustomFunc
//SQLContext.udf.registerJavaFunction("PerfBonus", "Training.CustomFunc.PerfBonus", IntegerType())
import scala.util.Try
import org.apache.spark.sql.api.java.UDF1

class PerfBonus extends UDF1[String, Int] {

  override def call(Dept: String): Int = {
    val rand = new scala.util.Random
    rand.nextInt(50000)
  }
}