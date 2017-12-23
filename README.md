# Measuring the similarity of words from a corpus



## Dataset:

The text corpus can be downloaded from here: http://mattmahoney.net/dc/text8.zip



## Methods:

There are many ways to measure similarity between words:

### Word2vec

In this project I have decided to explore and use word2vec  to compute similarities between,
words. The Idea behind word2vec is similar words have similar vectors in the high dimensional space.

Example:
   Consider two sentences:    

      a. Mike is a good man
      b. John is a good man too


      In the above sentences both Mike and John occur in same context of words. i.e
      (Mike, is),(Mike, a), (Mike, good), (Mike, man)
      (John, is), (John, a), (John, good), (John, man), (John, too)    

      So the way word2vec works it gives similar/closest vectors for the both mike and john in the euclidian space.

There are two ways that I know of we can generate word2vec :

  1. Skip-gram model
  2. CBOW model



### Other  

1. Jacard Similarity    



## A solution using Apache Spark Mlib

Spark Mllib uses skip-gram model of Word2Vec, with Hierarchial Softmax in the last layer of the model.

In each iteration the Spark driver sends the latest vectors to all Spark executors.
Each executor modifies its local copy of vectors based on its partition of the training data set, and the driver then combines local vector modifications to update the global vectors.
It requires all vectors to be stored in the memory of all Spark executors, and, similarly to its single machine counterparts, is thus unsuitable for large vocabularies.    


ref: https://arxiv.org/pdf/1606.08495.pdf

## A distributed solution using Spark DeepDist (It uses Downpour SGD to compute gradients)

It uses the concept of Downpour SGD in which:

    a. The master node has the main model: with the parameters.
    b. Each of the slaves have a replica of the master and work on a partition of the data
    c. At the end of the computational step, gradient updates are collected and transfered to the master.

    d. The master and slave work independently.
    e. The difference between mllib implementation and deepdist is Model replica requests an updated copy of model parameters from master server.


ref: http://deepdist.com/
