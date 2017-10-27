import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import timeit

mnist = read_data_sets("data", one_hot=False)

print(type(mnist.train.labels[0]))
a = mnist.train.labels.astype(np.int32, copy = False)
mnist.train.labels = a
print(type(mnist.train))