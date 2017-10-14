import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import timeit


def run(x_test,mnist,y_test): 
    
    n_inputs = 28 * 28
    
    x1 = tf.placeholder(tf.float32,[None, n_inputs])      
    W1 = tf.Variable(tf.zeros([n_inputs,1]))                
    b1 = tf.Variable(tf.zeros([1]))                         
    y1 = tf.placeholder(tf.float32,[None, 1])    
    
    x2 = tf.placeholder(tf.float32,[None, n_inputs])      
    W2 = tf.Variable(tf.zeros([n_inputs,1]))                
    b2 = tf.Variable(tf.zeros([1]))                         
    y2 = tf.placeholder(tf.float32,[None, 1])
    
    x3 = tf.placeholder(tf.float32,[None, n_inputs])      
    W3 = tf.Variable(tf.zeros([n_inputs,1]))                
    b3 = tf.Variable(tf.zeros([1]))                         
    y3 = tf.placeholder(tf.float32,[None, 1])
    
    x4 = tf.placeholder(tf.float32,[None, n_inputs])      
    W4 = tf.Variable(tf.zeros([n_inputs,1]))                
    b4 = tf.Variable(tf.zeros([1]))                         
    y4 = tf.placeholder(tf.float32,[None, 1])
    
    x5 = tf.placeholder(tf.float32,[None, n_inputs])      
    W5 = tf.Variable(tf.zeros([n_inputs,1]))                
    b5 = tf.Variable(tf.zeros([1]))                         
    y5 = tf.placeholder(tf.float32,[None, 1])
    
    x6 = tf.placeholder(tf.float32,[None, n_inputs])      
    W6 = tf.Variable(tf.zeros([n_inputs,1]))                
    b6 = tf.Variable(tf.zeros([1]))                         
    y6 = tf.placeholder(tf.float32,[None, 1])
    
    x7 = tf.placeholder(tf.float32,[None, n_inputs])      
    W7 = tf.Variable(tf.zeros([n_inputs,1]))                
    b7 = tf.Variable(tf.zeros([1]))                         
    y7 = tf.placeholder(tf.float32,[None, 1])
    
    x8 = tf.placeholder(tf.float32,[None, n_inputs])      
    W8 = tf.Variable(tf.zeros([n_inputs,1]))                
    b8 = tf.Variable(tf.zeros([1]))                         
    y8 = tf.placeholder(tf.float32,[None, 1])
    
    x9 = tf.placeholder(tf.float32,[None, n_inputs])      
    W9 = tf.Variable(tf.zeros([n_inputs,1]))                
    b9 = tf.Variable(tf.zeros([1]))                         
    y9 = tf.placeholder(tf.float32,[None, 1])
    
    x10 = tf.placeholder(tf.float32,[None, n_inputs])      
    W10 = tf.Variable(tf.zeros([n_inputs,1]))                
    b10 = tf.Variable(tf.zeros([1]))                         
    y10 = tf.placeholder(tf.float32,[None, 1])    
    
    
    
    #C hyperparameter
    C = 0.5 
    #C_range = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7]
    
    #print("shape of W: ", W.shape)
    #print("shape of b: ", b.shape)
    
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
    
    #score = tf.transpose(tf.matmul(tf.transpose(W),x)) + b
    #score = tf.transpose(score)
    #print("shape of score: ", score.shape)
    #print("shape of trans score: ",tf.transpose(score))
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
            batch_y = batch_y.reshape(batch_size,1)
            
            sess.run(training1, feed_dict={x1:batch_x, y1:batch_y})
        
        
        
    #correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #print(sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
    predicted_y_test = tf.sign(svm1)
    
    return predicted_y_test


def hyperparameters_search():
    raise NotImplementedError


if __name__ == '__main__':
    hyperparameters_search()
