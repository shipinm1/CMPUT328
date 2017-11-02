from ops import *
import timeit
import time
from ops import batch_norm
from cifar10 import Cifar10
from tensorflow.contrib.layers import flatten
BATCH_SIZE = 100

def net(input, is_training, dropout_kept_prob):
  # TODO: Write your network architecture here
  # Or you can write it inside train() function
  # Requirements:
  # - At least 5 layers in total
  # - At least 1 fully connected and 1 convolutional layer
  # - At least one maxpool layers
  # - At least one batch norm
  # - At least one skip connection
  # - Use dropout
  mu = 0
  sigma = 0.1
  
  '''first convolutional layer'''
  conv1_w = tf.Variable(tf.truncated_normal(shape = (5, 5, 3, 64), mean = mu, stddev = sigma), name = 'conv1_w')
  conv1_b = tf.Variable(tf.zeros(64), name = 'conv1_b')
  conv1 = tf.nn.conv2d(input, conv1_w, strides = [1, 1, 1, 1], padding = 'VALID') + conv1_b #(28,28,64)
  #activation function relu
  conv1 = tf.nn.relu(conv1)
  #norm1 = tf.nn.lrn(conv1, 4, bias = 1.0, alpha = 0.001/9, beta = 0.75, name = 'norm1')
  #max pooling
  conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')#(14,14,64)
  print(conv1)
  
  '''second convolutional layer'''
  conv2_w = tf.Variable(tf.truncated_normal(shape = (5, 5, 64, 128), mean = mu, stddev = sigma), name = 'conv2_w')
  conv2_b = tf.Variable(tf.zeros(128), name = 'conv2_b')
  conv2 = tf.nn.conv2d(conv1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b #(10,10,128)
  #batch normalization
  conv2 = batch_norm(conv2, is_training)
  #activation function relu
  conv2 = tf.nn.relu(conv2) #(10,10,128)
  conv2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID') #(5, 5, 128)
  print(conv2)
  
  '''third fully connected layer'''
  fl1 = flatten(conv2) #(400)
  print(fl1)
  fl1_w = tf.Variable(tf.truncated_normal(shape = (3200, 400), mean = mu, stddev = sigma), name = 'fl1_w')
  fl1_b = tf.Variable(tf.zeros(400), name = 'fl1_b')
  fl2 = tf.matmul(fl1, fl1_w) + fl1_b
  #activation function relu
  fl2 = tf.nn.relu(fl2) #(400)
  print(fl2)
  
  '''forth fully connected layer'''
  fl2_w = tf.Variable(tf.truncated_normal(shape = (400, 84), mean = mu, stddev = sigma), name = 'fl2_w') 
  fl2_b = tf.Variable(tf.zeros(84), name = 'fl2_b')
  fl3 = tf.matmul(fl2, fl2_w) + fl2_b
  #activation function relu
  fl3 = tf.nn.relu(fl3)
  #Dropout
  fl3_drop = tf.nn.dropout(fl3, dropout_kept_prob)
  print(fl3)
  
  '''fifth fully connected layer'''
  fl3_w = tf.Variable(tf.truncated_normal(shape = (84, 10), mean = mu, stddev = sigma), name = 'fl3_w')
  fl3_b = tf.Variable(tf.zeros(10), name = 'fl3_b')
  flf = tf.matmul(fl3, fl3_w) + fl3_b
  print(flf)
  
  '''
  #third convolutional layer
  conv3_w = tf.Variable(tf.truncated_normal(shape = (5, 5, 16, 32), mean = mu, stddev = sigma), name = 'conv3_w')
  conv3_b = tf.Variable(tf.zeros(32), name = 'conv2_b')
  conv3 = tf.nn.conv2d(conv2, conv3_w, strides = [1,1,1,1], padding = 'VALID') + conv3_b #(3,3,32)  
  #activation function relu
  conv3 = tf.nn.relu(conv3)
  #second max pooling
  conv3 = tf.nn.max_pool(conv3, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID') 
  #print('this is conv3', conv3)
  
  #forth convolutional layer
  conv4_w = tf.Variable(tf.truncated_normal(shape = (3, 3, 32, 64), mean = mu, stddev = sigma), name = 'conv4_w')
  conv4_b = tf.Variable(tf.zeros(64), name = 'conv4_b')
  conv4 = tf.nn.conv2d(conv3, conv4_w, strides = [1,1,1,1], padding = 'VALID') + conv4_b #(1,1,64)
  #activation function relu
  conv4 = tf.nn.relu(conv4)
  #print('this is conv4', conv4)
  
  #fifth fully connected layer
  fl = flatten(conv4) #(256)
  #print(fl)
  fl_w = tf.Variable(tf.truncated_normal(shape = (64, 10), mean = mu, stddev = sigma), name = 'fl_w')
  fl_b = tf.Variable(tf.zeros(10), name = 'fl_b')
  fl1 = tf.matmul(fl, fl_w) + fl_b
  #activation function relu
  return logits
  '''
  
  return flf



def train():
  # Always use tf.reset_default_graph() to avoid error
  tf.reset_default_graph()
  # TODO: Write your training code here
  # - Create placeholder for inputs, training boolean, dropout keep probablity
  # - Construct your model
  # - Create loss and training op
  # - Run training
  # AS IT WILL TAKE VERY LONG ON CIFAR10 DATASET TO TRAIN
  # YOU SHOULD USE tf.train.Saver() TO SAVE YOUR MODEL AFTER TRAINING
  # AT TEST TIME, LOAD THE MODEL AND RUN TEST ON THE TEST SET
  
  x = tf.placeholder(tf.float32, (None, 32, 32, 3))
  y = tf.placeholder(tf.int32, (None))
  #one_hot_y = tf.one_hot(y,10)
  
  '''data preparation'''
  cifar10_train = Cifar10(batch_size = 100, one_hot = True, test = False, shuffle = True)
  cifar10_test = Cifar10(batch_size = 100, one_hot = False, test = True, shuffle = False)
  test_images, test_labels = cifar10_test.images, cifar10_test.labels
  
  #print(batch_y[0])
  
  
  
  lr = 0.000001
  logits = net(x, True, 0.8)
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = y)
  loss = tf.reduce_mean(cross_entropy)
  optimizer = tf.train.AdamOptimizer(learning_rate = lr)
  grads_and_vars = optimizer.compute_gradients(loss, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
  training_operation = optimizer.apply_gradients(grads_and_vars)
  #training_operation = optimizer.minimize(grads_and_vars)
  #train = tf.train.AdamOptimizer.minimize(loss)
  
  '''create summary'''
  for var in tf.trainable_variables():
    tf.summary.histogram(var.op.name + "/histogram", var)

  add_gradient_summaries(grads_and_vars)
  tf.summary.scalar('loss_operation', loss)
  merged_summary_op= tf.summary.merge_all()
  summary_writer = tf.summary.FileWriter('logs/')
  
  correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y,1))
  accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  saver = tf.train.Saver(max_to_keep = 5)
  sess = tf.Session()
  
  try:
    print("================Trying to restore last checkpoint ...================")
    saver.restore(sess, tf.train.latest_checkpoint('ckpt'))
    print("================Checkpoint restored .================")
  except:
    print("================Failed to find last check point================")
    sess.run(tf.global_variables_initializer())
  
  
  '''training'''
  start_time = time.time()
  with sess:
    
    print("Training")
    global_step = 0
    for i in range(1000):
      for offset in range(0, 500):
        batch_x, batch_y = cifar10_train.get_next_batch()
        _, summaries = sess.run([training_operation, merged_summary_op], feed_dict = {x: batch_x, y: batch_y})
        if global_step % 100 == 1:
          print("steps: ", global_step, "|time consume: %.2f" % ((time.time() - start_time)/3600), "hours.")
          summary_writer.add_summary(summaries, global_step = global_step)
        if global_step % 500 == 0:
          _loss = sess.run(loss, feed_dict = {x:batch_x, y:batch_y})
          print("current loss: ", _loss, "Step: ", global_step)
        global_step += 1
      ''' 
      #validation_accuracy = evaluate(test_images, test_labels, accuracy_operation)
      total_accuracy = 0
      test_x, test_y = cifar10_test.get_next_batch()
      accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
      total_accuracy += (accuracy * len(batch_x))     
      '''
      training_accuracy = evaluate(batch_x, batch_y, accuracy_operation, x, y)
      print("sets: ", i)
      print("accuracy: ", training_accuracy * 100, " % ")
      print()
      saver.save(sess, 'ckpt/netCheckpoint', global_step = i)
  
  
  
  
def evaluate(X_data, y_data, accuracy_operation, x, y):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples  
  
  
def add_gradient_summaries(grads_and_vars):
    for grad, var in grads_and_vars:
        if grad is not None:
            tf.summary.histogram(var.op.name + "/gradient", grad)



def test(cifar10_test_images):
  # Always use tf.reset_default_graph() to avoid error
  tf.reset_default_graph()
  # TODO: Write your testing code here
  # - Create placeholder for inputs, training boolean, dropout keep probablity
  # - Construct your model
  # (Above 2 steps should be the same as in train function)
  # - Create label prediction tensor
  # - Run testing
  # DO NOT RUN TRAINING HERE!
  # LOAD THE MODEL AND RUN TEST ON THE TEST SET
  
  x = tf.placeholder(tf.float32, (None, 32, 32, 3))
  saver = tf.train.Saver()
  with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('ckpt'))
    test_accuracy = evaluate(cifar10_test_images)
  
