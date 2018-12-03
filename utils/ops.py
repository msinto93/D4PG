'''
## Ops ##
# Common ops for the networks
@author: Mark Sinton (msinto93@gmail.com) 
'''

import tensorflow as tf

def conv2d(inputs, kernel_size, filters, stride, activation=None, use_bias=True, weight_init=tf.contrib.layers.xavier_initializer(), bias_init=tf.zeros_initializer(), scope='conv'):
    with tf.variable_scope(scope):
        if use_bias:
            return tf.layers.conv2d(inputs, filters, kernel_size, stride, 'valid', activation=activation, 
                                    use_bias=use_bias, kernel_initializer=weight_init)
        else:
            return tf.layers.conv2d(inputs, filters, kernel_size, stride, 'valid', activation=activation, 
                                    use_bias=use_bias, kernel_initializer=weight_init,
                                    bias_initializer=bias_init)
            
def batchnorm(inputs, is_training, momentum=0.9, scope='batch_norm'):
    with tf.variable_scope(scope):
        return tf.layers.batch_normalization(inputs, momentum=momentum, training=is_training, fused=True)

def dense(inputs, output_size, activation=None, weight_init=tf.contrib.layers.xavier_initializer(), bias_init=tf.zeros_initializer(), scope='dense'):
    with tf.variable_scope(scope):
        return tf.layers.dense(inputs, output_size, activation=activation, kernel_initializer=weight_init, bias_initializer=bias_init)

def flatten(inputs, scope='flatten'):
    with tf.variable_scope(scope):
        return tf.layers.flatten(inputs)
    
def relu(inputs, scope='relu'):
    with tf.variable_scope(scope):
        return tf.nn.relu(inputs)
    
def tanh(inputs, scope='tanh'):
    with tf.variable_scope(scope):
        return tf.nn.tanh(inputs)
    
def softmax(inputs, scope='softmax'):
    with tf.variable_scope(scope):
        return tf.nn.softmax(inputs)
