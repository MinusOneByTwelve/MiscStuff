#spark-submit --master spark://BajajFinservIgniteDEMaster:7077 --num-executors 8 --driver-memory 512m --executor-memory 512m --executor-cores 1 --total-executor-cores 8 empfilter.py 8
#spark-submit --master spark://BajajFinservIgniteDEMaster:7077 --num-executors 4 --driver-memory 512m --executor-memory 1024m --executor-cores 2 empfilter.py 8
#spark-submit --master spark://BajajFinservIgniteDEMaster:7077 --num-executors 4 --driver-memory 512m --executor-memory 1024m --executor-cores 2 --total-executor-cores 8 empfilter.py 8

import os
import sys
from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("Spark_FilterEmps")

sc = SparkContext(conf=conf)
#sc.setLogLevel("ERROR")

rdd1 = sc.textFile("file:///opt/DataSet/EmployeesAll.csv",int(sys.argv[1]))
rdd2 = rdd1.filter(lambda x: ("emp_no" not in x))
rdd3 = rdd2.map(lambda X: (X.split(',')[2],X.split(',')[3],X.split(',')[4]))
rdd4 = rdd3.filter (lambda x: x[0].startswith("Ara") and x[1].startswith("Ba") and x[2]=="M")
rdd5 = rdd4.sortBy(lambda x: x[1], ascending=False)
rdd6 = rdd5.collect()
for row in rdd6:
  print(row[0] + "," + row[1] + "," + row[2])
