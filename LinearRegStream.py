import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Vectors 
from pyspark.mllib.regression import LabeledPoint 
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD

conf = (SparkConf()
     .setMaster("local")
     .setAppName("My app")
     .set("spark.executor.memory", "1g"))

sc = SparkContext(conf = conf)
ssc = StreamingContext(sc, 1)

# parsing the data 
lines1 = ssc.textFileStream("file:///mnt/vdatanodea/datasets/creditcards/credit/b")
trainingData = lines1.map(lambda line: LabeledPoint( float(line.split(" ")[0]), Vectors.dense(line.split(" ") [1])) ).cache()
trainingData.pprint()
lines2 = ssc.textFileStream("file:///mnt/vdatanodea/datasets/creditcards/credit/c")
testData = lines2.map(lambda line: LabeledPoint( float(line.split(" ")[0]), Vectors.dense(line.split(" ") [1])) )
testData.pprint()

# training the model
numFeatures = 3 
model = StreamingLinearRegressionWithSGD()
model.setInitialWeights([0.0, 0.0, 0.0])
model.trainOn(trainingData) 

# testing the model 
print(model.predictOnValues(testData.map(lambda lp: (lp.label, lp.features))))

ssc.start() 
ssc.awaitTermination()