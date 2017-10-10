import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import timeit


def run(x_test): 
    print(x_test)
    n_inputs = 28 * 28

    x = tf.placeholder(tf.float32,[None, n_inputs])
    y = tf.placeholder(tf.float32,[None, 1])
    W = tf.Variable(tf.random_uniform([n_inputs,1],minval=-1,maxval=1,seed=1))
    b = tf.Variable(0.0)
    
    #C hyperparameter
    C = 0.5 
    #C_range = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    score = tf.matmul(tf.transpose(W),x) + b
    lr = 0.001 

    
    #hinge loss function
    regularization_loss = 0.5 * tf.reduce_sum(tf.square(W))
    
    hinge_loss = tf.reduce_sum(tf.maximum(0.0, 1.0 - y*score))
    
    svm = regularization_loss + C*hinge_loss
    optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    training = optimizer.minimize(svm)
    
    #init node for variable
    init = tf.global_variables_initializer()
    
    #y1 = tf.placeholder(tf.float32,[None, 1])
    #y2 = tf.placeholder(tf.float32,[None, 1])
    
    n_epochs = 100
    batch_size = 200
    
    with tf.Session() as sess:
        sess.run(init)
        print("In Session")
        print(sess.run(training))
            
        
        
        
        
        
        
    
    predicted_y_test = tf.sign(score)
    return predicted_y_test


def hyperparameters_search():
    raise NotImplementedError


if __name__ == '__main__':
    hyperparameters_search()
