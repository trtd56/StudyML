#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Try MNIST for beginner.
'''

#%matplotlib inline

import tensorflow as tf
# predict "input_data" from https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/examples/tutorials/mnist/input_data.py
import input_data
import time, random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm

def _to_number(label):
    for index, n in enumerate(label):
        if n != 0:
            return index

def plot_mnist_data(samples):
    for index, (data, label) in enumerate(samples):
        plt.subplot(5, 5, index + 1)
        plt.axis('off')
        plt.imshow(data.reshape(28, 28), cmap=cm.gray_r, interpolation='nearest')
        n = _to_number(label)
        plt.title(n, color='red')
    plt.show()

def show_shaped_mnist(x,y):
    print "Number:",str(xrange(9)[y.tolist().index(1)])
    print np.array(map(np.sign,x),dtype=np.int32).reshape((28,28))

def random_sampling(mnist_data, num):
    index = random.sample(range(0, len(mnist_data.train.labels)), num)
    label = mnist_data.train.labels[index]
    data = mnist_data.train.images[index]
    return data, label

if __name__=="__main__":
    start_time = time.time()

    print "--- start read MNIST dataset ---"
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    print "--- finish read MNIST dataset ---"

    # sampling 25 data for confirm
    #X = mnist.train.images
    #y = mnist.train.labels
    #p = np.random.random_integers(0, len(X), 25)
    #samples = np.array(list(zip(X, y)))[p]
    #plot_mnist_data(samples)

    # check mnist data: the number is "3"
    #batch_xs, batch_ys = mnist.train.next_batch(100)
    #show_shaped_mnist(batch_xs[1],batch_ys[1])

    # predict 784(= 28 * 28) list for input data 
    x = tf.placeholder(tf.float32, [None, 784])
    # Weigth, initial value is 0
    W = tf.Variable(tf.zeros([784, 10]))
    # bias, initial value is 0
    b = tf.Variable(tf.zeros([10]))

    # softmax regression
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    # closs enrtopy
    correct_data = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(correct_data*tf.log(y))
    # SGD
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    # initilarize variable
    init = tf.initialize_all_variables()
    # start session
    sess = tf.Session()
    sess.run(init)

    print "--- start training ---"
    # 1000 trainig for training
    for i in range(1000):
        # get random 100 sample
        #batch_xs, batch_ys = mnist.train.next_batch(100)
        batch_xs, batch_ys = random_sampling(mnist, 100)
        # passing arguments 
        sess.run(train_step, feed_dict={x: batch_xs, correct_data:batch_ys})
    print "--- finish training ---"

    # check the answer
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(correct_data,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print "accuracy:"
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, correct_data: mnist.test.labels}))
    end_time = time.time()
    print "time: " + str(end_time - start_time)

    # check one predict
    #i = 0
    #f = tf.argmax(y, 1)
    #predict = sess.run(f, feed_dict={x: [mnist.test.images[i]]})[0]
    #print "predict", predict
    #show_shaped_mnist(mnist.test.images[i],mnist.test.labels[i])
