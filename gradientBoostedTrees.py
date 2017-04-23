from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.util import MLUtils
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import pandas as pd
import csv
from pyspark.mllib.util import MLUtils

def clean(x):
    if (x[29] != "Amount"):
        return x

def normalize(x):
    return LabeledPoint(float(x[30]), [float(x[0]), float(x[29])/ 25691.16])

# Load and parse the data
rdd = sc.textFile("creditcard.csv")
data = rdd.mapPartitions(lambda x: csv.reader(x))
data = sc.parallelize( data.take(1000) )
data = data.map( lambda x: clean(x) )
data = data.filter(lambda x: x != None)
data = data.map(normalize)

(trainingData, testData) = data.randomSplit([0.7, 0.3])

# Train a GradientBoostedTrees model.
model = GradientBoostedTrees.trainClassifier(trainingData, categoricalFeaturesInfo={}, numIterations=3)

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testErr = labelsAndPredictions.filter(lambda d: d[0] != d[1]).count() / float(testData.count())
print("Test Error = " + str(testErr))
# Test Error = 0.002310551518601614

# Save and load model
model.save(sc, "target/tmp/myGradientBoostingClassificationModel")
sameModel = GradientBoostedTreesModel.load(sc,
                                           "target/tmp/myGradientBoostingClassificationModel")