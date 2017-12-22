"""
  A spark 2.2.0 implementation of the word2vec

  Author: {
      name: "Chandan Uppuluri",
      email_id : "chandan.uppuluri@gmail.com"
  }


  reference:
  http://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html#pyspark.ml.feature.Word2Vec



  Internal Working:



  NOTE: When using RDD's , pyspark.mllib.feature.Word2Vec to be used
        When usning DataFrames pyspark.ml.feature.Word2Vec to be used

"""





# Using RDD's
from pyspark import SparkContext
from pyspark.mllib.feature import Word2Vec
sc = SparkContext(master = "local[*]", appName="mlib word similarties")
inp = sc.textFile("./data/text8").map(lambda row: row.split(" "))





word2vec = Word2Vec().setVectorSize(100).setWindowSize(5).setSeed(1)
model = word2vec.fit(inp)

synonyms = model.findSynonyms('complete', 5)

for word, cosine_distance in synonyms:
    print("{}: {}".format(word, cosine_distance))




# using DataFrames

# from pyspark.sql import SparkSession
# from pyspark.conf import SparkConf
# from pyspark.ml.feature import Word2Vec
#
# spark = SparkSession \
#     .builder \
#     .appName("mlib word similarities") \
#     .config(conf = SparkConf().setMaster("local[*]")) \
#     .getOrCreate()

# word2vec = Word2Vec( vectorSize=3, minCount=0, inputCol="Top1", outputCol="result" )
