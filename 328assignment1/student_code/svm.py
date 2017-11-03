import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import timeit

def run(x_test,mnist,y_test): 
    predicted_y_test = []
    n_inputs = 28 * 28
    
    #parameter initializing
    x = tf.placeholder(tf.float32,[None, n_inputs])      
    W1 = tf.Variable(tf.random_normal([n_inputs,1]))                
    b1 = tf.Variable(tf.zeros([1]))                         
    y = tf.placeholder(tf.float32,[None, 1])    

    W2 = tf.Variable(tf.random_normal([n_inputs,1]))                
    b2 = tf.Variable(tf.zeros([1]))                         
   
    W3 = tf.Variable(tf.random_normal([n_inputs,1]))                
    b3 = tf.Variable(tf.zeros([1]))                         
  
    W4 = tf.Variable(tf.random_normal([n_inputs,1]))                
    b4 = tf.Variable(tf.zeros([1]))                         
   
    W5 = tf.Variable(tf.random_normal([n_inputs,1]))                
    b5 = tf.Variable(tf.zeros([1]))                         

    W6 = tf.Variable(tf.random_normal([n_inputs,1]))                
    b6 = tf.Variable(tf.zeros([1]))                         
  
    W7 = tf.Variable(tf.random_normal([n_inputs,1]))                
    b7 = tf.Variable(tf.zeros([1]))                         
  
    W8 = tf.Variable(tf.random_normal([n_inputs,1]))                
    b8 = tf.Variable(tf.zeros([1]))                         
   
    W9 = tf.Variable(tf.random_normal([n_inputs,1]))                
    b9 = tf.Variable(tf.zeros([1]))                         

    W10 = tf.Variable(tf.random_normal([n_inputs,1]))                
    b10 = tf.Variable(tf.zeros([1]))                         
  
    #C hyperparameter
    C = 1
    
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
    lr = 0.001
    
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
    
    hinge_loss1 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y*score1))
    hinge_loss2 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y*score2))
    hinge_loss3 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y*score3))
    hinge_loss4 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y*score4))
    hinge_loss5 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y*score5))
    hinge_loss6 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y*score6))
    hinge_loss7 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y*score7))
    hinge_loss8 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y*score8))
    hinge_loss9 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y*score9))
    hinge_loss10 = tf.reduce_sum(tf.maximum(0.0, 1.0 - y*score10))
    
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
        init.run()
        #sess.run(init) 
        print("In Session")
        
        for i in range(40000):
            if (i % 2000 == 0):
                print("In step: ", i)
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
            #print("loss", svm1.eval(feed_dict={x1:batch_x1, y1:batch_y1}))
        
        
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
        
        for _ in range(5000):
            temp_list = []
            temp_list.extend([s10[_],s1[_],s2[_],s3[_],s4[_],s5[_],s6[_],s7[_],s8[_],s9[_]])
            predicted_y_test.append(temp_list.index(max(temp_list)))
        
        #print("Accuracy ", (accuracy1.eval(feed_dict={x1:mnist.test.images, y1:mnist.test.labels})))
    
    print(predicted_y_test)
    #predicted_y_test.append(np.random.randint(10, size = 5000))
    return predicted_y_test



def hyperparameters_search():
	raise NotImplementedError


if __name__ == '__main__':
	hyperparameters_search()