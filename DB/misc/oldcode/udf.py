#spark-submit --master local[*] udf.py
#export PYTHONIOENCODING=utf8

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, ArrayType, DoubleType, BooleanType, FloatType
from pyspark.sql.functions import udf
from pyspark.sql import Row

spark = SparkSession.builder.appName('PySparkUDF').getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

schema = StructType([
    StructField("amount", FloatType(),True),    
    StructField("customer", StringType(),True),
    StructField("orderid", IntegerType(),True)
])

data = [[ 3456.56, "Rakesh",123]]

df = spark.createDataFrame(data,schema=schema)

newcolNameCaps = udf(lambda x: NameCaps(x), StringType())
newcolDiscount = udf(lambda x: Discount(x), FloatType())
spark.udf.register("NameInCapital", newcolNameCaps)
spark.udf.register("GetDiscount", newcolDiscount)

def NameCaps(s):
    return s.upper()

def Discount(s):
    return (s*5.6)/100

df2 = df.withColumn( 'CapitalName',newcolNameCaps('customer')).withColumn( 'TotalDiscount',newcolDiscount('amount'))
df2.printSchema()
df2.show(truncate=False)