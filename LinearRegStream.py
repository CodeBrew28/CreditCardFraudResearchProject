import sys
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.mllib.linalg import Vectors 
from pyspark.mllib.regression import LabeledPoint 
from pyspark.mllib.regression import StreamingLinearRegressionWithSGD


"""
conf = (SparkConf()
     .setMaster("local")
     .setAppName("My app")
     .set("spark.executor.memory", "1g"))

sc = SparkContext(conf = conf)
"""
ssc = StreamingContext(sc, 1)



# parsing the data 
lines1 = ssc.textFileStream("file:///mnt/vdatanodea/datasets/creditcards/credit/b")
trainingData = lines1.map(lambda line: LabeledPoint( float(line.split(" ")[1]), Vectors.dense(line.split(" ") [0])) ).cache()
print(trainingData)
trainingData.pprint()
lines2 = ssc.textFileStream("file:///mnt/vdatanodea/datasets/creditcards/credit/c")
testData = lines2.map(lambda line: LabeledPoint( float(line.split(" ")[1]), Vectors.dense(line.split(" ") [0])) )
testData.pprint()

# training the model
numFeatures = 1
model = StreamingLinearRegressionWithSGD()
model.setInitialWeights([1.0])
model.trainOn(trainingData)

#labelsAndPreds = testData.map(lambda p: (p.label, model.predict(p.features)))
values = lines2.map(lambda line: Vectors.dense(line.split(" ") [0]))
values.pprint()
result = model.predictOn(testData.features)
result.pprint()
result.pprint()
print("hi")

ssc.start() 
#ssc.awaitTermination()