from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import pandas as pd
import csv
from pyspark.ml.classification import LogisticRegression
from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel
from pyspark.mllib.regression import LabeledPoint

def clean(x):
    if (x[29] != "Amount"):
        return x

def normalize(x):
    return LabeledPoint(float(x[30]), [float(x[0]), float(x[29])/ 25691.16])

# conf = SparkConf().setAppName("LinearSVMRegression")
# sc = SparkContext(conf=conf)
# sqlContext = SQLContext(sc)

# Load and parse the data
rdd = sc.textFile("creditcard.csv")
data = rdd.mapPartitions(lambda x: csv.reader(x))
data = data.map( lambda x: clean(x) )
data = data.filter(lambda x: x != None)
normalizedData = data.map(normalize)
normalizedData = sc.parallelize(normalizedData.take(1000))

#split the training and test data
(trainingData, testData) = normalizedData.randomSplit([0.7, 0.3])
sample = sc.parallelize(trainingData.take(100000))
testsample = sc.parallelize(testData.take(78000))

# Build the model
model = SVMWithSGD.train(sample, iterations=100)
# Evaluating the model on training data
labelsAndPreds = testsample.map(lambda p: (p.label, p.features, model.predict(p.features)))

trainErr = labelsAndPreds.filter(lambda d: d[0] != d[2]).count() / float(testsample.count())
print("Training Error = " + str(trainErr))

# Save and load model
model.save(sc, "target/tmp/pythonSVMWithSGDModel")
sameModel = SVMModel.load(sc, "target/tmp/pythonSVMWithSGDModel")