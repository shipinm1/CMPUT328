import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import timeit

def run(x_test,mnist,y_test): 
    predicted_y_test = []
    n_inputs = 28 * 28
   
    mnist1 = mnist
    mnist2 = mnist
    mnist3 = mnist
    mnist4 = mnist
    mnist5 = mnist
    mnist6 = mnist
    mnist7 = mnist
    mnist8 = mnist
    mnist9 = mnist
    mnist10 = mnist
    
    #y_labels1 = mnist1.train.labels.astype(np.int32)
    #y_labels2 = mnist2.train.labels.astype(np.int32)
    #y_labels3 = mnist3.train.labels.astype(np.int32)
    #y_labels4 = mnist4.train.labels.astype(np.int32)
    #y_labels5 = mnist5.train.labels.astype(np.int32)
    #y_labels6 = mnist6.train.labels.astype(np.int32)
    #y_labels7 = mnist7.train.labels.astype(np.int32)
    #y_labels8 = mnist8.train.labels.astype(np.int32)
    #y_labels9 = mnist9.train.labels.astype(np.int32)
    #y_labels10 = mnist10.train.labels.astype(np.int32)
    
    
    #y_labels1[y_labels1 !=1] = -1
    #y_labels2[y_labels2 !=2] = -1
    #y_labels3[y_labels3 !=3] = -1
    #y_labels4[y_labels4 !=4] = -1
    #y_labels5[y_labels5 !=5] = -1
    #y_labels6[y_labels6 !=6] = -1
    #y_labels7[y_labels7 !=7] = -1
    #y_labels8[y_labels8 !=8] = -1
    #y_labels9[y_labels9 !=9] = -1
    #y_labels10[y_labels10 !=0] = -1
    
    
    x1 = tf.placeholder(tf.float32,[None, n_inputs])      
    W1 = tf.Variable(tf.zeros([n_inputs,1]))                
    b1 = tf.Variable(tf.zeros([1]))                         
    y1 = tf.placeholder(tf.float32,[None, ])    
    
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
    
    predicted1 = tf.sign(score1[0])
    correct_predicted1 = tf.equal(y1,predicted1)
    accuracy1 = tf.reduce_mean(tf.cast(correct_predicted1, tf.float32)) 

    predicted2 = tf.sign(score2[0])
    correct_predicted2 = tf.equal(y2,predicted2)
    accuracy2 = tf.reduce_mean(tf.cast(correct_predicted2, tf.float32))    

    predicted3 = tf.sign(score3[0])
    correct_predicted3 = tf.equal(y3,predicted3)
    accuracy3 = tf.reduce_mean(tf.cast(correct_predicted3, tf.float32)) 
    
    predicted4 = tf.sign(score4[0])
    correct_predicted4 = tf.equal(y4,predicted4)
    accuracy4 = tf.reduce_mean(tf.cast(correct_predicted4, tf.float32))    
    
    accuracy5 = tf.reduce_mean(tf.cast(tf.equal(y5,tf.sign(score5[0])), tf.float32))
    accuracy6 = tf.reduce_mean(tf.cast(tf.equal(y6,tf.sign(score6[0])), tf.float32))
    accuracy7 = tf.reduce_mean(tf.cast(tf.equal(y7,tf.sign(score7[0])), tf.float32))
    accuracy8 = tf.reduce_mean(tf.cast(tf.equal(y8,tf.sign(score8[0])), tf.float32))
    accuracy9 = tf.reduce_mean(tf.cast(tf.equal(y9,tf.sign(score9[0])), tf.float32))
    accuracy10 = tf.reduce_mean(tf.cast(tf.equal(y10,tf.sign(score10[0])), tf.float32))
    
    with tf.Session() as sess:
        init.run()
        #sess.run(init) 
        print("In Session")
        
        for i in range(10):
            batch_x, batch_y = mnist1.train.next_batch(batch_size)
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
            
            #batch_y = batch_y.reshape(batch_size,1)
            
            training1.run( feed_dict={x1:batch_x, y1:temp1})
            training2.run( feed_dict={x2:batch_x, y2:temp2})
            training3.run( feed_dict={x3:batch_x, y3:temp3})
            training4.run( feed_dict={x4:batch_x, y4:temp4})
            training5.run( feed_dict={x5:batch_x, y5:temp5})
            training6.run( feed_dict={x6:batch_x, y6:temp6})
            training7.run( feed_dict={x7:batch_x, y7:temp7})
            training8.run( feed_dict={x8:batch_x, y8:temp8})
            training9.run( feed_dict={x9:batch_x, y9:temp9})
            training10.run( feed_dict={x10:batch_x, y10:temp10})
            #print("loss", svm1.eval(feed_dict={x1:batch_x1, y1:batch_y1}))
        
        #print("Accuracy ", (accuracy1.eval(feed_dict={x1:mnist1.test.images, y1:y_labels1})))
        #print("Accuracy ", (accuracy2.eval(feed_dict={x2:mnist2.test.images, y2:y_labels2})))
        #print("Accuracy ", (accuracy3.eval(feed_dict={x3:mnist3.test.images, y3:y_labels3})))
        #print("Accuracy ", (accuracy4.eval(feed_dict={x4:mnist4.test.images, y4:y_labels4})))
        #print("Accuracy ", (accuracy5.eval(feed_dict={x5:mnist5.test.images, y5:y_labels5})))
        #print("Accuracy ", (accuracy6.eval(feed_dict={x6:mnist6.test.images, y6:y_labels6})))
        #print("Accuracy ", (accuracy7.eval(feed_dict={x7:mnist7.test.images, y7:y_labels7})))
        #print("Accuracy ", (accuracy8.eval(feed_dict={x8:mnist8.test.images, y8:y_labels8})))
        #print("Accuracy ", (accuracy9.eval(feed_dict={x9:mnist9.test.images, y9:y_labels9})))
        #print("Accuracy ", (accuracy10.eval(feed_dict={x10:mnist10.test.images, y10:y_labels10})))
        
        
        s1=score1.eval(feed_dict = {x1:x_test})
        s2=score2.eval(feed_dict = {x2:x_test})
        s3=score3.eval(feed_dict = {x3:x_test})
        s4=score4.eval(feed_dict = {x4:x_test})
        s5=score5.eval(feed_dict = {x5:x_test})
        s6=score6.eval(feed_dict = {x6:x_test})
        s7=score7.eval(feed_dict = {x7:x_test})
        s8=score8.eval(feed_dict = {x8:x_test})
        s9=score9.eval(feed_dict = {x9:x_test})
        s10=score10.eval(feed_dict = {x10:x_test})
        
        
        print(average(s1))
        for i in range(5000):
            temp_list = []
            temp_list.extend([s10[i],s1[i],s2[i],s3[i],s4[i],s5[i],s6[i],s7[i],s8[i],s9[i]])
            predicted_y_test.append(temp_list.index(max(temp_list)))
        
        #print("Accuracy ", (accuracy1.eval(feed_dict={x1:mnist.test.images, y1:mnist.test.labels})))
    
    
    #predicted_y_test.append(np.random.randint(10, size = 5000))
    print(predicted_y_test)
    return predicted_y_test


def hyperparameters_search():
    raise NotImplementedError


if __name__ == '__main__':
    hyperparameters_search()
