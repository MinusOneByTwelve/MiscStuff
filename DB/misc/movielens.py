#tail 10 cat /opt/DataSet/movielens/users.dat
#tail 10 cat /opt/DataSet/movielens/ratings.dat
#tail 10 cat /opt/DataSet/movielens/movies.dat
#sed -i 's/::/,/g' /opt/DataSet/movielens/ratings.dat

#spark-submit --master spark://BajajFinservIgniteDEMaster:7077 --num-executors 8 --driver-memory 512m --executor-memory 512m --executor-cores 1 --total-executor-cores 8 movielens.py 8
#spark-submit --master spark://BajajFinservIgniteDEMaster:7077 --num-executors 4 --driver-memory 512m --executor-memory 1024m --executor-cores 2 movielens.py 8
#spark-submit --master spark://BajajFinservIgniteDEMaster:7077 --num-executors 4 --driver-memory 512m --executor-memory 1024m --executor-cores 2 --total-executor-cores 8 movielens.py 8

from pyspark import SparkConf, SparkContext, StorageLevel

def loadMovieNames():
    movieNames = {}
    lines = sc.textFile("file:///opt/DataSet/movielens/movies.dat").collect()
    for line in lines:
        fields = line.split(':')
        movieNames[int(fields[0])] = fields[1]
    return movieNames

def f(x): 
    global _5Counts
    global count5s
    
    columns = x.split(',')
    rating = int(columns[2])
    
    if rating == 5:
        _5Counts+=1
        count5s+=1
        
    return (int(columns[1]), 1)

conf = SparkConf().setAppName("Spark_MovieLens_ACCU_BV")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

_5Counts = sc.accumulator(0)
count5s=0

nameDict = sc.broadcast(loadMovieNames())

lines = sc.textFile("file:///opt/DataSet/movielens/ratings.dat",int(sys.argv[1]))
movies = lines.map(f)
movieCounts = movies.reduceByKey(lambda x, y: x + y)

flipped = movieCounts.map( lambda x : (x[1], x[0]))
sortedMovies = flipped.sortByKey(False)

sortedMoviesWithNames = sortedMovies.map(lambda countMovie : (nameDict.value[countMovie[1]], countMovie[0]))

sortedMoviesWithNames.persist( StorageLevel.MEMORY_AND_DISK )

print sortedMoviesWithNames.getNumPartitions()
MWN1 = sortedMoviesWithNames.repartition(4)
print MWN1.getNumPartitions()
MWN2 = MWN1.coalesce(1)
print MWN2.getNumPartitions()

results = MWN2.take(10)

print "Total 5 Ratings Accumulator -> %i" % (_5Counts.value)
print "Total 5 Ratings Variable -> %i" % (count5s)

for result in results:
    print (result)
