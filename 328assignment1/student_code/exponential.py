import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import timeit


def run(x_test,mnist,y_test):
    n_inputs = 28 * 28
    C = 20
    lr = 0.0002    
    predicted_y_test = []
    
    #parameter initializing
    x = tf.placeholder(tf.float32,[None, n_inputs])      
    W1 = tf.Variable(tf.random_normal([n_inputs,1], stddev = 0.01))                
    b1 = tf.Variable(tf.zeros([1]))                         
    y = tf.placeholder(tf.float32,[None, 1])    

    W2 = tf.Variable(tf.random_normal([n_inputs,1], stddev = 0.01))                
    b2 = tf.Variable(tf.zeros([1]))                         
   
    W3 = tf.Variable(tf.random_normal([n_inputs,1], stddev = 0.01))                
    b3 = tf.Variable(tf.zeros([1]))                         
  
    W4 = tf.Variable(tf.random_normal([n_inputs,1], stddev = 0.01))                
    b4 = tf.Variable(tf.zeros([1]))                         
   
    W5 = tf.Variable(tf.random_normal([n_inputs,1], stddev = 0.01))                
    b5 = tf.Variable(tf.zeros([1]))                         

    W6 = tf.Variable(tf.random_normal([n_inputs,1], stddev = 0.01))                
    b6 = tf.Variable(tf.zeros([1]))                         
  
    W7 = tf.Variable(tf.random_normal([n_inputs,1], stddev = 0.01))                
    b7 = tf.Variable(tf.zeros([1]))                         
  
    W8 = tf.Variable(tf.random_normal([n_inputs,1], stddev = 0.01))                
    b8 = tf.Variable(tf.zeros([1]))                         
   
    W9 = tf.Variable(tf.random_normal([n_inputs,1], stddev = 0.01))                
    b9 = tf.Variable(tf.zeros([1]))                         

    W10 = tf.Variable(tf.random_normal([n_inputs,1], stddev = 0.01))                
    b10 = tf.Variable(tf.zeros([1]))                         

    
    #score function
    score1 = tf.matmul(x,W1) + b1
    score2 = tf.matmul(x,W2) + b2
    score3 = tf.matmul(x,W3) + b3
    score4 = tf.matmul(x,W4) + b4
    score5 = tf.matmul(x,W5) + b5
    score6 = tf.matmul(x,W6) + b6
    score7 = tf.matmul(x,W7) + b7
    score8 = tf.matmul(x,W8) + b8
    score9 = tf.matmul(x,W9) + b9
    score10 = tf.matmul(x,W10) + b10
    
    #loss function implementation
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
    
    exp_loss1 = tf.reduce_sum(tf.exp(-y * score1))
    exp_loss2 = tf.reduce_sum(tf.exp(-y * score2))
    exp_loss3 = tf.reduce_sum(tf.exp(-y * score3))
    exp_loss4 = tf.reduce_sum(tf.exp(-y * score4))
    exp_loss5 = tf.reduce_sum(tf.exp(-y * score5))
    exp_loss6 = tf.reduce_sum(tf.exp(-y * score6))
    exp_loss7 = tf.reduce_sum(tf.exp(-y * score7))
    exp_loss8 = tf.reduce_sum(tf.exp(-y * score8))
    exp_loss9 = tf.reduce_sum(tf.exp(-y * score9))
    exp_loss10 = tf.reduce_sum(tf.exp(-y * score10))
    
    #hinge_loss1 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y*score1))
    
    
    loss1 = regularization_loss1 + C*exp_loss1
    loss2 = regularization_loss2 + C*exp_loss2
    loss3 = regularization_loss3 + C*exp_loss3
    loss4 = regularization_loss4 + C*exp_loss4
    loss5 = regularization_loss5 + C*exp_loss5
    loss6 = regularization_loss6 + C*exp_loss6
    loss7 = regularization_loss7 + C*exp_loss7
    loss8 = regularization_loss8 + C*exp_loss8
    loss9 = regularization_loss9 + C*exp_loss9
    loss10 = regularization_loss10 + C*exp_loss10
    
    optimizer = tf.train.AdamOptimizer(learning_rate = lr)
    
    training1 = optimizer.minimize(loss1)
    training2 = optimizer.minimize(loss2)
    training3 = optimizer.minimize(loss3)
    training4 = optimizer.minimize(loss4)
    training5 = optimizer.minimize(loss5)
    training6 = optimizer.minimize(loss6)
    training7 = optimizer.minimize(loss7)
    training8 = optimizer.minimize(loss8)
    training9 = optimizer.minimize(loss9)
    training10 = optimizer.minimize(loss10)
    
    
    #init node for variable
    init = tf.global_variables_initializer()
    
    n_samples = 55000
    n_epochs = 100
    batch_size = 400
    
    
    with tf.Session() as sess:
        init.run()
        #sess.run(init) 
        print("In Session")
        
        for i in range(8000):
    
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_y = batch_y.reshape(batch_size,1)
            
            temp1 = batch_y.astype(np.int32)
            temp1[temp1 !=1] = -1
            
            temp2 = batch_y.astype(np.int32)
            temp2[temp2 !=2] = -1          
            
            temp3 = batch_y.astype(np.int32)
            temp3[temp3 !=3] = -1   
    
            temp4 = batch_y.astype(np.int32)
            temp4[temp4 !=4] = -1
            
            temp5 = batch_y.astype(np.int32)
            temp5[temp5 !=5] = -1
            
            temp6 = batch_y.astype(np.int32)
            temp6[temp6 !=6] = -1
            
            temp7 = batch_y.astype(np.int32)
            temp7[temp7 !=7] = -1
            
            temp8 = batch_y.astype(np.int32)
            temp8[temp8 !=8] = -1
            
            temp9 = batch_y.astype(np.int32)
            temp9[temp9 !=9] = -1
        
            temp10 = batch_y.astype(np.int32)
            temp10[temp10 !=0] = -1
            
            training1.run( feed_dict={x:batch_x, y:temp1})
            training2.run( feed_dict={x:batch_x, y:temp2})
            training3.run( feed_dict={x:batch_x, y:temp3})
            training4.run( feed_dict={x:batch_x, y:temp4})
            training5.run( feed_dict={x:batch_x, y:temp5})
            training6.run( feed_dict={x:batch_x, y:temp6})
            training7.run( feed_dict={x:batch_x, y:temp7})
            training8.run( feed_dict={x:batch_x, y:temp8})
            training9.run( feed_dict={x:batch_x, y:temp9})
            training10.run( feed_dict={x:batch_x, y:temp10})
            if (i % 400 == 0):
                print("in step*: ", i, "target 8000")
                #print("loss", score1.eval(feed_dict={x:batch_x, y:temp1}))
        s1=score1.eval(feed_dict = {x:x_test})
        s2=score2.eval(feed_dict = {x:x_test})
        s3=score3.eval(feed_dict = {x:x_test})
        s4=score4.eval(feed_dict = {x:x_test})
        s5=score5.eval(feed_dict = {x:x_test})
        s6=score6.eval(feed_dict = {x:x_test})
        s7=score7.eval(feed_dict = {x:x_test})
        s8=score8.eval(feed_dict = {x:x_test})
        s9=score9.eval(feed_dict = {x:x_test})
        s10=score10.eval(feed_dict = {x:x_test})
        
        for _ in range(len(s1)):
            temp_list = []
            temp_list.extend([s10[_],s1[_],s2[_],s3[_],s4[_],s5[_],s6[_],s7[_],s8[_],s9[_]])
            predicted_y_test.append(temp_list.index(max(temp_list)))
        
        #print("Accuracy ", (accuracy1.eval(feed_dict={x1:mnist.test.images, y1:mnist.test.labels})))
    
    #print(predicted_y_test)
    #print(predicted_y_test.count(1))
   
    
    #predicted_y_test.append(np.random.randint(10, size = 5000))    
    
    
    
    return predicted_y_test


def hyperparameters_search():
    raise NotImplementedError


if __name__ == '__main__':
    hyperparameters_search()
