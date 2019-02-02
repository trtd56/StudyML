#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
This prigram is same behaver as sample 1.
The calculate of this program is used Tensor Flow.
'''

import tensorflow as tf

if __name__=="__main__":
    # make constant value "x", and input "1" to "x".
    x = tf.constant(1, name='x')
    # make "y" variable, and "y" is defined "x + 2" 
    y = tf.Variable(x + 2, name='y')
    # initialize all variable
    model = tf.initialize_all_variables()
    # make session to run calculate
    with tf.Session() as session:
        # run model made line 16
        session.run(model)
        # run y
        result = session.run(y)
        # print result
        print result
