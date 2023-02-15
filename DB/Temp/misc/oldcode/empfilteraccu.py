from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("SprkAccuEmpFilter")

sc = SparkContext(conf=conf) 
sc.setLogLevel("ERROR")

male_emp = sc.accumulator(0)
female_emp = sc.accumulator(0)
me=0
fe=0

def f(x): 
   global male_emp
   global female_emp
   global me
   global fe
   
   columns = x.split(',')
   if columns[4] == 'M' and columns[2].startswith('Ara') and columns[3].startswith('Ba'):
    me+=1
    male_emp+=1
   if columns[4] == 'F' and columns[2].startswith('Ara') and columns[3].startswith('Ba'):
    fe+=1
    female_emp+=1    
   
rdd = sc.textFile("file:///opt/DataSet/EmployeesAll.csv")
rdd.foreach(f)
 
print "Male Employee Count -> %i" % (male_emp.value)
print "Female Employee Count -> %i" % (female_emp.value)
print "Male2 Employee Count -> %i" % (me)
print "Female2 Employee Count -> %i" % (fe)
