# Measuring the similarity of words from a corpus



## Dataset:

The text corpus can be downloaded from here: http://mattmahoney.net/dc/text8.zip



## Methods:

There are many ways to measure similarity between a pair of words.
In this project I have decided to explore and use word2vec  to compute similarities between,
words.




## A solution using Apache Spark Mlib

Spark Mllib uses skip-gram model of Word2Vec, with Hierarchial Softmax in the last layer of the model.




## A distributed solution using Spark DeepDist
