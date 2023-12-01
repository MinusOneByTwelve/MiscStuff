# -*- coding: utf-8 -*-

import os
import sys
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType

SparkSQLContext = SparkSession.builder.appName("ScalaUdf").getOrCreate() 
SparkSQLContext.udf.registerJavaFunction("PerfBonus", "Training.CustomFunc.PerfBonus", IntegerType())
SparkSQLContext.sql("select PerfBonus('Rakesh')").show()
