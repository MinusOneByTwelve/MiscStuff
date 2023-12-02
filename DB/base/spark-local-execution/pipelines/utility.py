import os
import time
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.types import StructType


def generate_spark_session() -> SparkSession:
    pyspark_submit_args = '--packages "io.delta:delta-core_2.12:0.7.0" '
    pyspark_submit_args += "pyspark-shell"
    os.environ["PYSPARK_SUBMIT_ARGS"] = pyspark_submit_args
    return SparkSession.builder.master("local[8]").getOrCreate()


def initialize_delta_table(
    spark: SparkSession, path: str, schema: StructType, partitionBy: str = None
):
    df = spark.createDataFrame([], schema)
    writer = df.write.format("delta").option("path", path)
    if partitionBy is not None:
        writer = writer.partitionBy(partitionBy)
    writer.save()


def load_dataframe(
    spark: SparkSession,
    format: str,
    path: str,
    alias: str = None,
    schema: StructType = None,
    streaming: bool = False,
) -> DataFrame:
    if streaming:
        spark_reader = spark.readStream
    else:
        spark_reader = spark.read

    df = spark_reader.format(format).option("path", path)
    if schema is not None:
        df = df.schema(schema)
    if alias is not None:
        df = df.alias(alias)
    return df.load()


def until_stream_is_ready(
    spark: SparkSession, named_stream: str, progressions: int = 3
) -> bool:
    queries = [stream for stream in spark.streams.active if stream.name == named_stream]
    while len(queries) == 0 or len(queries[0].recentProgress) < progressions:
        time.sleep(5)
        queries = [
            stream for stream in spark.streams.active if stream.name == named_stream
        ]
    return True
