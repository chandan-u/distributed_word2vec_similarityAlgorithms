"""
  A spark 2.2.0 implementation of the word2vec

  Author: {
      name: "Chandan Uppuluri",
      email_id : "chandan.uppuluri@gmail.com"
  }


  references:
  http://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html#pyspark.ml.feature.Word2Vec
  https://arxiv.org/pdf/1606.08495.pdf


  Internal Working:

    The word2vec implementation in spark does initialize simultaneous execution of Skip-gram model
    on each of the executors for each partition. It parallely computes the gradients on each partition seperately
    and then combines the gradients to perform the descent() at the end. Thsi will take time to converge.

    Another disadvantage of this approach is that it could get really slow if the vocab size increases.
    As the weights must be broadcasted and present in the memory.



  NOTE: When using RDD's , pyspark.mllib.feature.Word2Vec to be used
        When usning DataFrames pyspark.ml.feature.Word2Vec to be used

"""





# Using RDD's

from pyspark import SparkContext
from pyspark.mllib.feature import Word2Vec
sc = SparkContext(master = "local[*]", appName="mlib word similarties")
inp = sc.textFile("./data/sample.txt").map(lambda row: row.split(" "))

#inp = inp.takeSample(True, 1000, 1)


# :: Word2Vec config ::
# vectorSize : sie of the word vector
# Seed: weights can re-initialized using these seeds
# windowSize : Frame size in which words are paried together.

word2vec = Word2Vec().setVectorSize(50).setWindowSize(5).setSeed(1)
model = word2vec.fit(inp)

#synonyms = model.findSynonyms('complete', 5)

#for word, cosine_distance in synonyms:
#    print("{}: {}".format(word, cosine_distance))
#

words_dict = model.getVectors()

print words_dict.keys()
model.save(sc, "./models/mllib_word2vec_v1")

vectors = []
labels = []
for key, value  in words_dict.iteritems():


    vectors.append(value)
    labels.append(value)




import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from sklearn.manifold import TSNE

# to display plotly images in notebook enable this
#init_notebook_mode(connected=True)

# reduce to 2D
tsne = TSNE(n_components=2)

# transform the generated word2vec vectors
X_tsne = tsne.fit_transform(vectors)

N = len(word_weights)
random_x = X_tsne[:, 0]
random_y = X_tsne[:, 1]



# labels:

labels = [word for word,_ in word_tuples]



plot([go.Scatter(x=list(X_tsne[:, 0]), y=list(X_tsne[:, 1]), text= labels, mode='markers', marker=go.Marker(color='red', size=3, opacity=0.3))])


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
