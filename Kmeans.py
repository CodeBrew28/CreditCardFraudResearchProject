from numpy import array
from math import sqrt
import json
from pyspark.mllib.clustering import KMeans, KMeansModel
import matplotlib.pyplot as plt
import csv

def clean(x):
    if (x[29] != "Amount"):
        return x

# from pyspark import SparkContext, SparkConf
# conf = SparkConf().setAppName("KMeans")
# sc = SparkContext(conf=conf)    

# # Load and parse the data  
# rdd = sc.textFile("creditcard.csv")
# data = rdd.mapPartitions(lambda x: csv.reader(x))
# data = data.map( lambda x: clean(x) )
# data = data.filter(lambda x: x != None)
# labeled_data = data.map( lambda x: ( float(x[29]), float(x[0])))

# #train the model
# clusters = KMeans.train(labeled_data, 3, maxIterations=10, initializationMode="random")

# # compute Within Set Sum of Squared Error (WSSSE)
# # edit the number of clusters to find the elbow of tje wsse values 
# def error(point):
#     center = clusters.centers[clusters.predict(point)]
#     return sqrt(sum([x**2 for x in (point - center)]))
# wssse = [labeled_data.map(lambda point: error(point)).reduce(lambda x, y: x + y)]


# #generate the graph for the data
# data = labeled_data.map(lambda x: x[0])
# x = data.collect()
# data = labeled_data.map(lambda x: x[1])
# y = data.collect()
# data = labeled_data.map(lambda x: clusters.predict(x))
# colors = data.collect()
# plt.scatter( x, y, c=colors)
# plt.show()



rdd = sc.textFile("creditcard.csv")
data = rdd.mapPartitions(lambda x: csv.reader(x))
data = data.map( lambda x: clean(x) )
data = data.filter(lambda x: x != None)
labeled_data = data.map( lambda x: [ float(x[29]), float(x[0]) , float(x[0]) ] )

#train the model
clusters = KMeans.train(labeled_data, 2, maxIterations=10, initializationMode="random")

# compute Within Set Sum of Squared Error (WSSSE)
# edit the number of clusters to find the elbow of tje wsse values 
def error(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))
wssse = [labeled_data.map(lambda point: error(point)).reduce(lambda x, y: x + y)]


#generate the graph for the data
data = labeled_data.map(lambda x: x[0])
x = data.collect()
data = labeled_data.map(lambda x: x[1])
y = data.collect()
data = labeled_data.map(lambda x: clusters.predict(x))
colors = data.collect()
plt.scatter( x, y, c=colors)
plt.show()

