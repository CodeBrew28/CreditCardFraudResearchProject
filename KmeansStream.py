import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Vectors 
from pyspark.mllib.regression import LabeledPoint 
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD
from pyspark.mllib.clustering import StreamingKMeans

# def clean(x):
#     if (x[29] != "Amount"):
#         return x
# import csv
# rdd = sc.textFile("creditcard.csv")
# data = rdd.mapPartitions(lambda x: csv.reader(x))
# data = data.map( lambda x: clean(x) )
# data = data.filter(lambda x: x != None)
# labeled_data = data.map( lambda x: ( float(x[29]), float(x[30])))


conf = (SparkConf()
     .setMaster("local")
     .setAppName("My app")
     .set("spark.executor.memory", "1g"))
     
sc = SparkContext(conf = conf)
ssc = StreamingContext(sc, 1)

#parsing data as a stream
trainingData = sc.textFile("file:///mnt/vdatanodea/datasets/creditcards/credit/b").map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))
testinggData = sc.textFile("file:///mnt/vdatanodea/datasets/creditcards/credit/b").map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))
trainingQueue = [trainingData]
testingQueue = [testingData]
trainingStream = ssc.queueStream(trainingQueue)
testingStream = ssc.queueStream(testingQueue)

# We create a model with random clusters and specify the number of clusters to find
model = StreamingKMeans(k=2, decayFactor=1.0).setRandomCenters(3, 1.0, 0)
# Now register the streams for training and testing and start the job, printing the predicted cluster assignments on new data points as they arrive.
model.trainOn(trainingStream)
result = model.predictOnValues(testingStream.map(lambda lp: (lp.label, lp.features)))
result.pprint()

ssc.start()
ssc.stop(stopSparkContext=True, stopGraceFully=True)