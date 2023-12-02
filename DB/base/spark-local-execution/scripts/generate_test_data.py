# Databricks notebook source
from shutil import rmtree
from pipelines.config import paths, schemas
from pipelines.operations import create_batch_writer, transform_bronze, transform_raw
from pipelines.utility import generate_spark_session, load_dataframe

if __name__ == "__main__":
    spark = generate_spark_session()

    rmtree(paths.test_bronze, ignore_errors=True)
    rmtree(paths.test_silver, ignore_errors=True)

    raw_df = load_dataframe(
        spark, format="text", path=paths.test_raw, schema=schemas.raw
    )

    transformed_raw_df = transform_raw(spark, raw_df)

    raw_to_bronze_json_writer = create_batch_writer(
        dataframe=transformed_raw_df,
        path=paths.test_bronze,
        partition_column="p_ingestdate",
        format="json",
    )
    raw_to_bronze_json_writer.save()

    bronze_df = transformed_raw_df
    transformed_bronze_df = transform_bronze(spark, bronze_df)

    bronze_to_silver_json_writer = create_batch_writer(
        dataframe=transformed_bronze_df,
        path=paths.test_silver,
        partition_column="p_eventdate",
        format="json",
    )
    bronze_to_silver_json_writer.save()
