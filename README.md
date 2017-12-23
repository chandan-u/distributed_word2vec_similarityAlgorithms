# Measuring the similarity of words from a corpus



## Dataset:

The text corpus can be downloaded from here: http://mattmahoney.net/dc/text8.zip



## Methods:

There are many ways to measure similarity between words:

### Word2vec

In this project I have decided to explore word2vec  to compute similarities between words. Implemented word2vec using Spark.Mllib package. The Idea behind word2vec is similar words have similar vectors in the high dimensional space.

Example:
   Consider two sentences:    

      a. Mike is a good man
      b. John is a good man too


      In the above sentences both Mike and John occur in same context of words. i.e
      (Mike, is),(Mike, a), (Mike, good), (Mike, man)
      (John, is), (John, a), (John, good), (John, man), (John, too)    

      So the way word2vec works it gives similar/closest vectors for the both mike and john in the euclidian space.

There are two ways that I know of we can generate vectors using word2vec  :

  1. Skip-gram model
  2. CBOW model



### Other  

1. Jacard Similarity    




## Setup:

please use pip to setup the env from requirements.txt files

```
pip install -r requirements.txt
```


Also to use deepdist please install the deepdist library from third_party folders:

```
python setup.py build
python setup.py install
```


## Result:

Please find the plotly visualiztion below. The model can also be visualized using the script: mllib_word2vec_visualization.ipynb in the repo. The model is generated using the spark.mllib.feature.word2vec implementation.

https://plot.ly/create/?fid=chandan-u:5


## Using Single Instance vs Distributed:

Stochastic Gradient Descent is a sequential process. Every gradient is dependent on the previous updates. Hence Single Instance can give good results as the Stochasitic Gradient Descent (SGD) will converge better, but the resources wont be sufficient for larger datasets. For very large corpus and vocabularies it's logical to use distributed computing. But with distributed computing if the communication between the subproblems ( i.e gradient on each partition of data) is not  optimal then it will result in poor convergence.

ref : http://ruder.io/optimizing-gradient-descent/index.html#downpoursgd


### Using TensorFlow:
#### Please refer to this blog I have written (it shows how to use tensorflow core to build word2vec skip gram model) http://chandan-u.github.io/implementing-and-visualizing-word2vec-using-tensorflow.html#implementing-and-visualizing-word2vec-using-tensorflow


### Using python Gensim

Gensim can parallelize within a node. It runs on multiple cores. It uses C under the hood.

```
import gensim
model = gensim.models.Word2Vec(sg=1, sentences= <list of tokenized sentences>, size=100, window=5, min_count= 1)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))
```


## A distributed solution using Apache Spark Mlib

Spark Mllib uses skip-gram model of Word2Vec, with Hierarchical Softmax in the last layer of the model.


### Scalability of the Solution:

Disadvantage: In each iteration the Spark driver sends the latest vectors to all Spark executors. Each executor modifies its local copy of vectors based on its partition of the training data set, and the driver then combines local vector modifications to update the global vectors.It requires all vectors to be stored in the memory of all Spark executors, and, similarly to its single machine counterparts, is thus unsuitable for large vocabularies.    


ref: https://arxiv.org/pdf/1606.08495.pdf

## Alternative for better and faster Convergence: A distributed solution using Spark DeepDist (It uses Downpour SGD to compute gradients)

It uses the concept of Downpour SGD in which:


    a. Each of the slaves have a replica of the master and work on a partition of the data
    b. At the end of the computational step, gradient updates are collected and transfered to the master.
    c. The master and slave work independently.
    d. The difference between mllib implementation and deepdist is Model replica requests an updated copy of model parameters from master server. Gradient updates are frequent and hence the model will converge faster and better. Please refer the below link for more details.

ref: http://deepdist.com/






## Productionizing the model to handle streaming scenarios:

For production scenarios we can use Lambda Architecture. In a lambda architecture,
we have two processes running simultaneously.
1. One that handles batch data, and
2. one that handles streaming data    


### Batch processor:

The batch handler does the actual model training of word2vec. It takes the entire batch of text (collected so far from the stream ), processes it and gives us the trained model, which can be deployed to production. This should be done iteratively.



### Stream processor:

All the streaming text is processed by the streaming Handle. In our case the streaming handle is supposed to give similarities using the trained model. And the stream text should be saved, for further batch processing.


### The above model can be extended to lambda architecture by using Spark Streaming:

In spark streaming we get DStreams, which are blocks or RDD collected for a certain window of time. We can use traied model to predict the similarities on the DStreams and at the same time save it to a DB/filesystem for batch training the word2vec model.


### Improving Production scenarios:

Some libraries give the capability of using GPU for computation. The spark Mllib does'nt use the GPU. Computing gradients for skip-gram models is faster using GPU. If we can use a combination of distributed and gpu , it could speed up training much much better. (tensorflow distributed needs to be explored).

Even spark can be used to parallelize to run replicas of tensorflow models locally on nodes and merge the results. This can done too.


### How to tackle memory limits:

Spark offers disk spillout but this will slow down the training.  Here is an interesting a paper I found, that speaks about using parametric servers for this problem (deepdist approach uses parametric server, but the implementation explained in this paper is quite different. It uses column based rather than word based split). Yet to explore.
