from pyspark.sql.types import (
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

raw = StructType([StructField("value", StringType(), False)])

bronze = StructType(
    [
        StructField("datasource", StringType(), False),
        StructField("ingesttime", TimestampType(), False),
        StructField("value", StringType(), True),
        StructField("p_ingestdate", DateType(), False),
    ]
)

silver = StructType(
    [
        StructField("device_id", IntegerType(), True),
        StructField("device_type", StringType(), True),
        StructField("heartrate", DoubleType(), True),
        StructField("eventtime", TimestampType(), True),
        StructField("name", StringType(), True),
        StructField("p_eventdate", DateType(), True),
    ]
)
