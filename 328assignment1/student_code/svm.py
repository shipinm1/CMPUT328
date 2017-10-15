import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import timeit


def run(x_test,mnist,y_test): 
    
    n_inputs = 28 * 28
   
    mnist1 = mnist
    mnist1.train.labels.setflags(write = 1)
    mnist2 = mnist
    mnist2.train.labels.setflags(write = 1)
    mnist3 = mnist
    mnist3.train.labels.setflags(write = 1)
    mnist4 = mnist
    mnist4.train.labels.setflags(write = 1)
    mnist5 = mnist
    mnist5.train.labels.setflags(write = 1)
    mnist6 = mnist
    mnist6.train.labels.setflags(write = 1)
    mnist7 = mnist
    mnist7.train.labels.setflags(write = 1)
    mnist8 = mnist
    mnist8.train.labels.setflags(write = 1)
    mnist9 = mnist
    mnist9.train.labels.setflags(write = 1)
    mnist10 = mnist
    mnist10.train.labels.setflags(write = 1)
    
    y_labels1 = mnist1.train.labels.astype(np.int32)
    y_labels2 = mnist2.train.labels.astype(np.int32)
    y_labels3 = mnist3.train.labels.astype(np.int32)
    y_labels4 = mnist4.train.labels.astype(np.int32)
    y_labels5 = mnist5.train.labels.astype(np.int32)
    y_labels6 = mnist6.train.labels.astype(np.int32)
    y_labels7 = mnist7.train.labels.astype(np.int32)
    y_labels8 = mnist8.train.labels.astype(np.int32)
    y_labels9 = mnist9.train.labels.astype(np.int32)
    y_labels10 = mnist10.train.labels.astype(np.int32)
    
    #mnist1.train.labels[mnist1.train.labels != 1] = -1
    #mnist2.train.labels[mnist1.train.labels != 2] = -1
    #mnist3.train.labels[mnist1.train.labels != 3] = -1
    #mnist4.train.labels[mnist1.train.labels != 4] = -1
    #mnist5.train.labels[mnist1.train.labels != 5] = -1
    #mnist6.train.labels[mnist1.train.labels != 6] = -1
    #mnist7.train.labels[mnist1.train.labels != 7] = -1
    #mnist8.train.labels[mnist1.train.labels != 8] = -1
    #mnist9.train.labels[mnist1.train.labels != 9] = -1
    #mnist10.train.labels[mnist1.train.labels != 0] = -1
    
    
    x1 = tf.placeholder(tf.float32,[None, n_inputs])      
    W1 = tf.Variable(tf.zeros([n_inputs,1]))                
    b1 = tf.Variable(tf.zeros([1]))                         
    y1 = tf.placeholder(tf.float32,[None,])    
    
    x2 = tf.placeholder(tf.float32,[None, n_inputs])      
    W2 = tf.Variable(tf.zeros([n_inputs,1]))                
    b2 = tf.Variable(tf.zeros([1]))                         
    y2 = tf.placeholder(tf.float32,[None, ])
    
    x3 = tf.placeholder(tf.float32,[None, n_inputs])      
    W3 = tf.Variable(tf.zeros([n_inputs,1]))                
    b3 = tf.Variable(tf.zeros([1]))                         
    y3 = tf.placeholder(tf.float32,[None, ])
    
    x4 = tf.placeholder(tf.float32,[None, n_inputs])      
    W4 = tf.Variable(tf.zeros([n_inputs,1]))                
    b4 = tf.Variable(tf.zeros([1]))                         
    y4 = tf.placeholder(tf.float32,[None, ])
    
    x5 = tf.placeholder(tf.float32,[None, n_inputs])      
    W5 = tf.Variable(tf.zeros([n_inputs,1]))                
    b5 = tf.Variable(tf.zeros([1]))                         
    y5 = tf.placeholder(tf.float32,[None, ])
    
    x6 = tf.placeholder(tf.float32,[None, n_inputs])      
    W6 = tf.Variable(tf.zeros([n_inputs,1]))                
    b6 = tf.Variable(tf.zeros([1]))                         
    y6 = tf.placeholder(tf.float32,[None, ])
    
    x7 = tf.placeholder(tf.float32,[None, n_inputs])      
    W7 = tf.Variable(tf.zeros([n_inputs,1]))                
    b7 = tf.Variable(tf.zeros([1]))                         
    y7 = tf.placeholder(tf.float32,[None, ])
    
    x8 = tf.placeholder(tf.float32,[None, n_inputs])      
    W8 = tf.Variable(tf.zeros([n_inputs,1]))                
    b8 = tf.Variable(tf.zeros([1]))                         
    y8 = tf.placeholder(tf.float32,[None, ])
    
    x9 = tf.placeholder(tf.float32,[None, n_inputs])      
    W9 = tf.Variable(tf.zeros([n_inputs,1]))                
    b9 = tf.Variable(tf.zeros([1]))                         
    y9 = tf.placeholder(tf.float32,[None, ])
    
    x10 = tf.placeholder(tf.float32,[None, n_inputs])      
    W10 = tf.Variable(tf.zeros([n_inputs,1]))                
    b10 = tf.Variable(tf.zeros([1]))                         
    y10 = tf.placeholder(tf.float32,[None, ])    
    
    
    
    #C hyperparameter
    C = 0.5 
    #C_range = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    
    
    score1 = tf.matmul(x1,W1) + b1
    score2 = tf.matmul(x2,W2) + b2
    score3 = tf.matmul(x3,W3) + b3
    score4 = tf.matmul(x4,W4) + b4
    score5 = tf.matmul(x5,W5) + b5
    score6 = tf.matmul(x6,W6) + b6
    score7 = tf.matmul(x7,W7) + b7
    score8 = tf.matmul(x8,W8) + b8
    score9 = tf.matmul(x9,W9) + b9
    score10 = tf.matmul(x10,W10) + b10
    
    lr = 0.001
    
    #hinge loss function
    regularization_loss1 = 0.5 * tf.reduce_sum(tf.square(W1))
    regularization_loss2 = 0.5 * tf.reduce_sum(tf.square(W2))
    regularization_loss3 = 0.5 * tf.reduce_sum(tf.square(W3))
    regularization_loss4 = 0.5 * tf.reduce_sum(tf.square(W4))
    regularization_loss5 = 0.5 * tf.reduce_sum(tf.square(W5))
    regularization_loss6 = 0.5 * tf.reduce_sum(tf.square(W6))
    regularization_loss7 = 0.5 * tf.reduce_sum(tf.square(W7))
    regularization_loss8 = 0.5 * tf.reduce_sum(tf.square(W8))
    regularization_loss9 = 0.5 * tf.reduce_sum(tf.square(W9))
    regularization_loss10 = 0.5 * tf.reduce_sum(tf.square(W10))
    
    hinge_loss1 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y1*score1))
    hinge_loss2 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y2*score2))
    hinge_loss3 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y3*score3))
    hinge_loss4 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y4*score4))
    hinge_loss5 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y5*score5))
    hinge_loss6 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y6*score6))
    hinge_loss7 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y7*score7))
    hinge_loss8 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y8*score8))
    hinge_loss9 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y9*score9))
    hinge_loss10 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y10*score10))
    
    
    svm1 = regularization_loss1 + C*hinge_loss1
    svm2 = regularization_loss2 + C*hinge_loss2
    svm3 = regularization_loss3 + C*hinge_loss3
    svm4 = regularization_loss4 + C*hinge_loss4
    svm5 = regularization_loss5 + C*hinge_loss5
    svm6 = regularization_loss6 + C*hinge_loss6
    svm7 = regularization_loss7 + C*hinge_loss7
    svm8 = regularization_loss8 + C*hinge_loss8
    svm9 = regularization_loss9 + C*hinge_loss9
    svm10 = regularization_loss10 + C*hinge_loss10
    
    optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    
    training1 = optimizer.minimize(svm1)
    training2 = optimizer.minimize(svm2)
    training3 = optimizer.minimize(svm3)
    training4 = optimizer.minimize(svm4)
    training5 = optimizer.minimize(svm5)
    training6 = optimizer.minimize(svm6)
    training7 = optimizer.minimize(svm7)
    training8 = optimizer.minimize(svm8)
    training9 = optimizer.minimize(svm9)
    training10 = optimizer.minimize(svm10)
    
    
    #init node for variable
    init = tf.global_variables_initializer()
    
    n_samples = 55000
    n_epochs = 100
    batch_size = 200
    
    predicted = tf.sign(score1)
    correct_predicted = tf.equal(y1,predicted)
    accuracy = tf.reduce_mean(tf.cast(correct_predicted, tf.float32))
    
    with tf.Session() as sess:
        sess.run(init) 
        print("In Session")
        
        for i in range(1000):
            batch_x1, batch_y1 = mnist1.train.next_batch(batch_size)
            batch_x2, batch_y2 = mnist2.train.next_batch(batch_size)
            batch_x3, batch_y3 = mnist3.train.next_batch(batch_size)
            batch_x4, batch_y4 = mnist4.train.next_batch(batch_size)
            batch_x5, batch_y5 = mnist5.train.next_batch(batch_size)
            batch_x6, batch_y6 = mnist6.train.next_batch(batch_size)
            batch_x7, batch_y7 = mnist7.train.next_batch(batch_size)
            batch_x8, batch_y8 = mnist8.train.next_batch(batch_size)
            batch_x9, batch_y9 = mnist9.train.next_batch(batch_size)
            batch_x10, batch_y10 = mnist10.train.next_batch(batch_size)
            
            #batch_y = batch_y.reshape(batch_size,1)
            
            sess.run(training1, feed_dict={x1:batch_x1, y1:batch_y1})
            sess.run(training2, feed_dict={x2:batch_x2, y2:batch_y2})
            sess.run(training3, feed_dict={x3:batch_x3, y3:batch_y3})
            sess.run(training4, feed_dict={x4:batch_x4, y4:batch_y4})
            sess.run(training5, feed_dict={x5:batch_x5, y5:batch_y5})
            sess.run(training6, feed_dict={x6:batch_x6, y6:batch_y6})
            sess.run(training7, feed_dict={x7:batch_x7, y7:batch_y7})
            sess.run(training8, feed_dict={x8:batch_x8, y8:batch_y8})
            sess.run(training9, feed_dict={x9:batch_x9, y9:batch_y9})
            sess.run(training10, feed_dict={x10:batch_x10, y10:batch_y10})
        
        
        print("This is the accuracy ", (accuracy.eval(feed_dict={x1:mnist1.train.images, y1:y_labels1})))
        #print("This is the accuracy ", (accuracy.eval(feed_dict={x2:mnist2.train.images, y2:y_labels2})))
        #print("This is the accuracy ", (accuracy.eval(feed_dict={x3:mnist3.train.images, y3:y_labels3})))
        #print("This is the accuracy ", (accuracy.eval(feed_dict={x4:mnist4.train.images, y4:y_labels4})))
        #print("This is the accuracy ", (accuracy.eval(feed_dict={x5:mnist5.train.images, y5:y_labels5})))
        #print("This is the accuracy ", (accuracy.eval(feed_dict={x6:mnist6.train.images, y6:y_labels6})))
        #print("This is the accuracy ", (accuracy.eval(feed_dict={x7:mnist7.train.images, y7:y_labels7})))
        #print("This is the accuracy ", (accuracy.eval(feed_dict={x8:mnist8.train.images, y8:y_labels8})))
        #print("This is the accuracy ", (accuracy.eval(feed_dict={x9:mnist9.train.images, y9:y_labels9})))
        #print("This is the accuracy ", (accuracy.eval(feed_dict={x10:mnist10.train.images, y10:y_labels10})))
    
    
    
    predicted_y_test = tf.sign(score1)
    return predicted_y_test


def hyperparameters_search():
    raise NotImplementedError


if __name__ == '__main__':
    hyperparameters_search()
