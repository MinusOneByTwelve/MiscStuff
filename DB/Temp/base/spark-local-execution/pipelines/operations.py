from typing import List
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col,
    current_timestamp,
    from_json,
    from_unixtime,
    lag,
    lead,
    lit,
    when,
)
from pyspark.sql.session import SparkSession
from pyspark.sql.streaming import DataStreamWriter
from pyspark.sql.window import Window

from pipelines.utility import load_dataframe


def create_batch_writer(
    dataframe: DataFrame,
    path: str,
    partition_column: str,
    exclude_columns: List = [],
    mode: str = "append",
    format: str = "delta",
) -> DataFrame:
    return (
        dataframe.drop(*exclude_columns)
        .write.format(format)
        .mode(mode)
        .option("path", path)
        .partitionBy(partition_column)
    )


def create_stream_writer(
    dataframe: DataFrame,
    path: str,
    checkpoint: str,
    name: str,
    partition_column: str,
    mode: str = "append",
    format: str = "delta",
    mergeSchema: bool = False,
) -> DataStreamWriter:

    stream_writer = (
        dataframe.writeStream.format(format)
        .outputMode(mode)
        .option("path", path)
        .option("checkpointLocation", checkpoint)
        .partitionBy(partition_column)
        .queryName(name)
    )

    if mergeSchema:
        stream_writer = stream_writer.option("mergeSchema", True)
    if partition_column is not None:
        stream_writer = stream_writer.partitionBy(partition_column)
    return stream_writer


def prepare_interpolated_updates_dataframe(
    spark: SparkSession, silver_df: DataFrame
) -> DataFrame:
    dateWindow = Window.partitionBy("device_id").orderBy("p_eventdate")

    lag_lead_silver_df = silver_df.select(
        "*",
        lag(col("heartrate")).over(dateWindow).alias("prev_amt"),
        lead(col("heartrate")).over(dateWindow).alias("next_amt"),
    )
    updates = lag_lead_silver_df.where(col("heartrate") < 0)
    updates = updates.withColumn(
        "heartrate",
        when(col("prev_amt").isNull(), col("next_amt")).otherwise(
            when(col("next_amt").isNull(), col("prev_amt")).otherwise(
                (col("prev_amt") + col("next_amt")) / 2
            )
        ),
    )
    return updates.select(
        "device_id",
        "heartrate",
        "eventtime",
        "name",
        "p_eventdate",
    )


def transform_bronze(spark: SparkSession, bronze: DataFrame) -> DataFrame:

    json_schema = """
        device_id INTEGER,
        heartrate DOUBLE,
        device_type STRING,
        name STRING,
        time FLOAT
    """

    return (
        bronze.select(from_json(col("value"), json_schema).alias("nested_json"))
        .select("nested_json.*")
        .select(
            "device_id",
            "device_type",
            "heartrate",
            from_unixtime("time").cast("timestamp").alias("eventtime"),
            "name",
            from_unixtime("time").cast("date").alias("p_eventdate"),
        )
    )


def transform_raw(spark: SparkSession, raw: DataFrame) -> DataFrame:
    return raw.select(
        lit("files.training.databricks.com").alias("datasource"),
        current_timestamp().alias("ingesttime"),
        "value",
        current_timestamp().cast("date").alias("p_ingestdate"),
    )


def update_silver_table(spark: SparkSession, silverPath: str) -> bool:
    from delta.tables import DeltaTable

    silver_df = load_dataframe(spark, format="delta", path=silverPath)
    silverTable = DeltaTable.forPath(spark, silverPath)

    update_match = """
    health_tracker.eventtime = updates.eventtime
    AND
    health_tracker.device_id = updates.device_id
    """

    update = {"heartrate": "updates.heartrate"}

    updates_df = prepare_interpolated_updates_dataframe(spark, silver_df)

    (
        silverTable.alias("health_tracker")
        .merge(updates_df.alias("updates"), update_match)
        .whenMatchedUpdate(set=update)
        .execute()
    )

    return True
