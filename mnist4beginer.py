#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Try MNIST for beginner.
'''

import tensorflow as tf
# predict "input_data" from https://github.com/tensorflow/tensorflow/blob/r0.9/tensorflow/examples/tutorials/mnist/input_data.py
import input_data
import time

if __name__=="__main__":
    start_time = time.time()
    print start_time
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
    
