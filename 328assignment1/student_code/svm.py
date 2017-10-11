import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import timeit


def run(x_test,mnist,y_test): 
    
    n_inputs = 28 * 28
    x = tf.placeholder(tf.float32,[None, n_inputs])         #? x 784
    W = tf.Variable(tf.zeros([n_inputs,10]))                #784 x 10
    b = tf.Variable(tf.zeros([10]))                         #10    1d array
    y = tf.placeholder(tf.float32,[None, 10])               #? x 10
    
    
    
    #C hyperparameter
    C = 0.5 
    #C_range = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    
    #print("shape of W: ", W.shape)
    print("shape of b: ", b.shape)
    
    score = tf.matmul(x,W) + b
    #score = tf.transpose(tf.matmul(tf.transpose(W),x)) + b
    #score = tf.transpose(score)
    #print("shape of score: ", score.shape)
    #print("shape of trans score: ",tf.transpose(score))
    lr = 0.001
    
    #hinge loss function
    regularization_loss = 0.5 * tf.reduce_sum(tf.square(W))
    hinge_loss = tf.reduce_sum(tf.maximum(0.0, 1.0 - y*score))
    
    svm = regularization_loss + C*hinge_loss
    
    optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    training = optimizer.minimize(svm)
    
    #init node for variable
    init = tf.global_variables_initializer()
    
    n_samples = 55000
    n_epochs = 100
    batch_size = 200
    
    with tf.Session() as sess:
        sess.run(init) 
        print("In Session")
        
        #implementation from lecture slides
        
        '''for epoch in range(n_epochs):
            n_batches = mnist.train.num_examples // batch_size
            for iteration in range(n_batches):              
                x_batch,y_batch = mnist.train.next_batch(batch_size)
                y_batch = y_batch.reshape(batch_size,1)
                sess.run(training, feed_dict={x:x_batch, y:y_batch})
            if epoch%10 == 0:
                1'''
        #implementation from https://www.tensorflow.org/get_started/mnist/beginners
        for i in range(1000):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            #batch_y = batch_y.reshape(batch_size,10)
            sess.run(training, feed_dict={x:batch_x, y:batch_y})
        
        
        
    #correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print(sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
    predicted_y_test = tf.sign(y)
    return predicted_y_test


def hyperparameters_search():
    raise NotImplementedError


if __name__ == '__main__':
    hyperparameters_search()
