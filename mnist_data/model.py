# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2018/7/25 下午9:37"
import tensorflow as tf


# y = W*x+b
def regression(x):
    W = tf.Variable(tf.zeros([784, 10]), name="W")
    b = tf.Variable(tf.zeros([10]), name="b")
    y = tf.nn.softmax(tf.matmul(x, W)+b)
    return y, [W, b]


# 定义卷积模型
def convlutional(x ,keep_prob):

    def con2d(x, W):
        return tf.nn.conv2d([1, 1, 1, 1], padding='SAME')

    def max_pool_2X2(x): #池化
        return tf.nn.max_pool(x,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    x_image = tf.reshape(x, [-1, 28, 28, 1]) # 输入
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(con2d(x_image, W_conv1)+b_conv1)
    h_pool1 = max_pool_2X2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(con2d(h_pool1, W_conv2)+b_conv2)
    h_pool2 = max_pool_2X2(h_conv2)

    # 全连接
    W_fc1 = weight_variable([7*7*64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) # 防止过拟合

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2)+b_fc2)

    return y, [W_conv1, b_conv1, W_conv2, b_conv2, W_fc1, b_fc1, W_fc2, b_fc2]
