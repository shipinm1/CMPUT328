import tensorflow as tf
import numpy as np
from input_helper import TextureImages


def SemSeg(input_tensor, is_training):
    from tensorflow.contrib.layers import flatten
    # TODO: Implement Semantic Segmentation network here
    # Returned logits must be a tensor of size:
    # (None, image_height, image_width, num_classes + 1)
    # 1st dimension is batch dimension
    # image_height and image_width are the height and width of input_tensor
    # last dimension is the softmax dimension. There are 4 texture classes plus 1 background class
    # therefore last dimension will be 5
    # Hint: To make your output tensor has the same height and width with input_tensor,
    # you can use tf.image.resize_bilinear
    
        
    #conv1
    print(input_tensor)
    conv1 = tf.layers.conv2d(inputs = input_tensor, filters = 64, kernel_size = [3,3], padding = "valid", activation = tf.nn.relu)
    print("shape of conv1: ", conv1) #192, 192, 2
    
    #pool1
    pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [4,4], strides = 4)
    print("shape of pool1: ", pool1) #96, 96, 2
    
    #conv2
    conv2 = tf.layers.conv2d(inputs = pool1, filters = 128, kernel_size = [3, 3], padding = "valid", activation = tf.nn.relu)
    print("shape of conv2: ", conv2)
    
    #pool2
    pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [4, 4], strides = 4)
    print("shape of pool2: ", pool2) 
    
    #conv3
    conv3 = tf.layers.conv2d(inputs = pool2, filters = 256, kernel_size = [3, 3], padding = "valid", activation = tf.nn.relu)
    print("shape of conv3: ", conv3)
        
    #pool3
    pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = [4, 4], strides = 4)
    print("shape of pool3: ", pool3)     
    
    print("")
    print("fully connected")
    
    #fc
    fc = tf.layers.conv2d(inputs = pool3, filters = 1024, kernel_size = [2, 2], padding = "valid", activation = tf.nn.relu)
    print("fc shape: ", fc)
    fc2 = tf.layers.conv2d(inputs = fc, filters = 1024, kernel_size = [1, 1], padding = "valid", activation = tf.nn.relu)
    print("fc2 shape: ", fc2)    
    print("")
    
    #deconv1
    upsample1 = tf.layers.conv2d_transpose(inputs = fc2, filters = 256, kernel_size = [2, 2], strides = 1, padding = "valid", activation = tf.nn.relu)
    print("deconv1 upsample1: ", upsample1)
    
    #deconv2
    upsample11 = tf.layers.conv2d_transpose(inputs = upsample1, filters = 256, kernel_size = [1, 1], strides = 1, padding = "valid", activation = tf.nn.relu)
    print("deconv2 upsample11: ", upsample11)    
    
    ##unpool1
    #unpool1 = tf.image.resize_bilinear(upsample11, [9,9])
    #print("unpooled unpool1: ", unpool1)
    
    def UnPooling(x):
        # https://github.com/tensorflow/tensorflow/issues/2169
        out = tf.concat([x, tf.zeros_like(x)], 3)
        out = tf.concat([out, tf.zeros_like(out)], 2)
    
        sh = x.get_shape().as_list()
        if None not in sh[1:]:
            out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
            return tf.reshape(out, out_size)
        else:
            shv = tf.shape(x)
            ret = tf.reshape(out, tf.stack([-1, shv[1] * 2, shv[2] * 2, sh[3]]))
            return ret    
    unpooltest = UnPooling(UnPooling(upsample11))
    print("unpooltest: ", unpooltest)
    
    
    #deconv2
    upsample2 = tf.layers.conv2d_transpose(inputs = unpooltest, filters = 128, kernel_size = [4, 4], strides = 1, padding = "valid", activation = tf.nn.relu)
    print("deconv2 upsample2: ", upsample2)
    
    #deconv2-1
    upsample21 = tf.layers.conv2d_transpose(inputs = upsample2, filters = 128, kernel_size = [1, 1], strides = 1, padding = "valid", activation = tf.nn.relu)
    
    ##unpool2
    #unpool2 = tf.image.resize_bilinear(upsample21, [46,46]) 
    #print("unpooled unpool2: ", unpool2 )
    
    unpooltest2 = UnPooling(UnPooling(upsample21))
    print("unpooltest2: ",unpooltest2)
    
    #deconv3
    upsample3 = tf.layers.conv2d_transpose(inputs = unpooltest2, filters = 64, kernel_size = [3, 3], strides = 1, padding = "valid", activation = tf.nn.relu)
    print("deconv3 upsample3: ", upsample3)  
    
    #deconv3-1
    upsample4 = tf.layers.conv2d_transpose(inputs = upsample3, filters = 64, kernel_size = [3, 3], strides = 1, padding = "valid", activation = tf.nn.relu)
    print("deconv4 upsample3: ", upsample4)  
    
    ##unpool3
    #unpool3 = tf.image.resize_bilinear(upsample4, [194,194]) 
    #print("unpooled unpool3: ", unpool3 )
    
    unpooltest3 = UnPooling(UnPooling(upsample4))
    print("unpooltest3: ",unpooltest3)    
    
    #deconv4
    upsample5 = tf.layers.conv2d_transpose(inputs = unpooltest3, filters = 20, kernel_size = [3, 3], strides = 1, padding = "valid", activation = tf.nn.relu)
    print("deconv5 : ", upsample5)    
    
    #deconv4-1
    upsample6 = tf.layers.conv2d_transpose(inputs = upsample5, filters = 5, kernel_size = [3, 3], strides = 1, padding = "valid", activation = tf.nn.relu)
    print("deconv6 : ", upsample6)     
    
    logits = tf.nn.softmax(upsample6)
    print(logits)
    
    
    
    return logits

def run():
    # You can tune the hyperparameters here.
    EPOCHS = 10
    BATCH_SIZE = 64
    NUM_ITERS = int(2000 / BATCH_SIZE * EPOCHS)

    train_set = TextureImages('train', batch_size=BATCH_SIZE)
    test_set = TextureImages('test', shuffle=False)

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, (None, 196, 196, 3))
    y = tf.placeholder(tf.int32, (None, 196, 196, 1))
    one_hot_y = tf.one_hot(y, 5)
    is_training = tf.placeholder(tf.bool, ())

    rate = 0.00005
    logits = SemSeg(x, is_training=is_training)
    prediction = tf.argmax(logits, axis=-1)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y))
    loss_operation = cross_entropy
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    training_operation = optimizer.apply_gradients(grads_and_vars)

    def evaluation(images, true_labels):
        eval_batch_size = 100
        predicted_labels = []
        for start_index in range(0, len(images), eval_batch_size):
            end_index = start_index + eval_batch_size
            batch_x = images[start_index: end_index]
            batch_predicted_labels = sess.run(prediction, feed_dict={x: batch_x, is_training: False})
            predicted_labels += list(batch_predicted_labels)
        predicted_labels = np.vstack(predicted_labels).flatten()
        true_labels = true_labels.flatten()
        accuracy = float((predicted_labels == true_labels).astype(np.int32).sum()) / true_labels.size
        return predicted_labels, accuracy

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("Training...")
    for i in range(NUM_ITERS):
        batch_x, batch_y = train_set.get_next_batch()
        _ = sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, is_training: True})
        if (i + 1) % 50 == 0 or i == NUM_ITERS - 1:
            _, test_accuracy = evaluation(test_set._images, test_set._masks)
            print("Iter {}: Test Pixel Accuracy = {:.3f}".format(i, test_accuracy))

    print('Evaluating on test set')
    _, test_accuracy = evaluation(test_set._images, test_set._masks)
    print("Test Pixel Accuracy = {:.3f}".format(test_accuracy))
    sess.close()
    return test_accuracy


if __name__ == '__main__':
    run()
