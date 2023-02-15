# -*- coding: utf-8 -*-

import os
import sys

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('SparkByExamples.com').getOrCreate()
log4jLogger = spark.sparkContext._jvm.org.apache.log4j
log = log4jLogger.LogManager.getLogger(__name__)
log.info("Hello World!")
rdd = spark.sparkContext.textFile(sys.argv[1])
rdd2 = rdd.filter(lambda x: ("emp_no" not in x))
rdd3 = rdd2.map(lambda X: (X.split(',')[2],X.split(',')[3],X.split(',')[4]))
rdd4 = rdd3.filter(lambda x: x[0].startswith("Ara") and x[1].startswith("Ba") and x[2]=="M")
rdd5 = rdd4.sortBy(lambda x: x[1], ascending=False)
rdd5.saveAsTextFile(sys.argv[2])
