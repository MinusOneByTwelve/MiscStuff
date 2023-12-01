from pyspark import SparkConf, SparkContext

def loadMovieNames():
    movieNames = {}
    lines = sc.textFile("file:///opt/DataSet/movielens/movies.dat").collect()
    for line in lines:
        fields = line.split(':')
        movieNames[int(fields[0])] = fields[1]
    return movieNames

conf = SparkConf().setAppName("SprkMovieLens")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

nameDict = sc.broadcast(loadMovieNames())

lines = sc.textFile("file:///opt/DataSet/movielens/ratings.dat")
movies = lines.map(lambda x: (int(x.split(',')[1]), 1))
movieCounts = movies.reduceByKey(lambda x, y: x + y)

flipped = movieCounts.map( lambda x : (x[1], x[0]))
sortedMovies = flipped.sortByKey(False)

sortedMoviesWithNames = sortedMovies.map(lambda countMovie : (nameDict.value[countMovie[1]], countMovie[0]))

results = sortedMoviesWithNames.take(10)

for result in results:
    print (result)
