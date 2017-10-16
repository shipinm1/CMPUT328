import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import timeit


def run(x_test,mnist,y_test):
    n_inputs = 28 * 28
    predicted_y_test = []
    mnist = read_data_sets("data", one_hot=True)
    Xtr, Ytr = mnist.train.next_batch(20000)
    #Xte, Yte = mnist.test.next_batch(1000)
    
    k = 4
    
    x_train = tf.placeholder("float", [None, n_inputs])
    x_te = tf.placeholder("float", [n_inputs])
    
    distance = tf.reduce_sum(tf.abs(tf.add(x_train,tf.negative(x_te))), reduction_indices = 1)
    predicted = tf.argmin(distance, 0)
    
    init = tf.global_variables_initializer()
    accuracy = 0.
    with tf.Session() as sess:
        sess.run(init)
        
        for _ in range(len(x_test)):
            #x_batch, y_batch = mnist.train.next_batch(100)
            #x_batch_test, y_batch_test = mnist.test.next_batch(100)
            
            pos = sess.run(predicted, feed_dict={x_train: Xtr, x_te:x_test[_, :]})
            
            if (_%500 == 0):
                print("running")
                #print("test" , _, "prediction: ", np.argmax(Ytr[pos]), "True class", np.argmax(Yte[_]))
            
            predicted_y_test.append(np.argmax(Ytr[pos]))
            
            
            
    
    
    #implemented from Aymeric Damien's Github--Tensorflow examples
    return predicted_y_test


def hyperparameters_search():
    raise NotImplementedError


if __name__ == '__main__':
    hyperparameters_search()
