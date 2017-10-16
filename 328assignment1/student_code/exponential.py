import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import timeit


def run(x_test,mnist,y_test):
    n_inputs = 28 * 28
    C = 0.5
    lr = 0.0001
    mnist = read_data_sets("data", one_hot=True)
    predicted_y_test = []
    
    x = tf.placeholder("float", [None, n_inputs])
    y = tf.placeholder("float", [None, 10])
    w = tf.Variable(tf.zeros([n_inputs, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    score = tf.matmul(x,w) + b
    exp_loss = tf.reduce_sum(tf.exp(tf.negative(y) * score))
    regularization_loss = 0.5 * tf.reduce_sum(tf.square(w))
    loss = 0.5 * regularization_loss + C * exp_loss
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)
    
    correct_predict = tf.equal(tf.argmax(score,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predict, "float"))
    
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        init.run()
        for _ in range(1000):
            #offset = (_ * 200) % 55000
            x_batch, y_batch = mnist.train.next_batch(100)
            #x_batch = mnist.train.images[offset:(offset + 200),:]
            #y_batch = mnist.train.labels[offset:(offset + 200)]
            train_step.run(feed_dict={x:x_batch, y:y_batch})
            #print("loss: ", loss.eval(feed_dict={x:x_batch, y:y_batch}))
            sess.run(train_step, feed_dict={x: x_batch, y: y_batch})
            #print(tf.argmax(score,1))
         
        
        print("accurcy: ", accuracy.eval(feed_dict={x:mnist.train.images, y:mnist.train.labels}))
        print("accurcy: ", accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels}))
    predicted_y_test.append(tf.argmax(score,1))
    print(tf.argmax(score,1))
    
    
    
    
    
    
    
    
    
    
    
    
    
    return predicted_y_test


def hyperparameters_search():
    raise NotImplementedError


if __name__ == '__main__':
    hyperparameters_search()
