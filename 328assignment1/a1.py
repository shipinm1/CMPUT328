from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import timeit

mnist      = input_data.read_data_sets("/tmp/data/", one_hot = False)
batch_size = 100
n_epoch    = 50
FLAGS      = None


x = tf.placeholder( 'float', [None, 784] )
y = tf.placeholder( 'float' )

def main( _ ):

	svm = { 'weights': tf.Variable( tf.random_normal( [784, 10] ) ),
	        'biases':  tf.Variable( tf.random_normal( [10] ) ) }

	prediction = tf.matmul( x, svm['weights'] ) + svm['biases']

	loss      = tf.losses.hinge_loss( labels = y, logits = prediction )
	optimizer = tf.train.AdamOptimizer().minimize(loss)
	
	with tf.Session() as sess:

		sess.run( tf.global_variables_initializer() )

		for epoch in range( n_epoch ):

			epoch_loss = 0

			for _ in range( int( mnist.train.num_examples / batch_size ) ):

				epoch_x, epoch_y = mnist.train.next_batch( batch_size )

				one_hot_epoch_y = np.zeros( [ len( epoch_y ), 10 ] )

				for i in range( len( epoch_y ) ):
					one_hot_epoch_y[i][epoch_y[i]] = 1

				_, c = sess.run( [optimizer, loss], feed_dict = { x: epoch_x, \
				                                                  y: one_hot_epoch_y } )
				epoch_loss += c

			print( "\x1b[1A\x1b[2K\x1b[1A\x1b[2KPercentage:",
			       ( ( epoch + 1 ) * 100 / n_epoch ),
			       "\nloss:", epoch_loss )

		correct   = tf.equal( tf.argmax( prediction, 1 ), tf.argmax( y, 1 ) )

		accuracy  = tf.reduce_mean( tf.cast( correct, 'float' ) )

		yt        = mnist.test.labels
		one_hot_y = np.zeros( [ len( yt ), 10 ] )

		for i in range( len( yt ) ):
			one_hot_y[i][yt[i]] = 1

		print( 'Accuracy:', accuracy.eval( { x: mnist.test.images, \
		                                     y: one_hot_y } ) )

		



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
											help='Directory for storing input data')
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
