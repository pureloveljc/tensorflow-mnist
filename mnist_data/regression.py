# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2018/7/25 下午8:28"
import tensorflow as tf
from mnist_data import model
from mnist_data import input_data
import os
data = input_data.read_data_sets('MNIST_DATA', one_hot=True)
# 建立线性模型
with tf.variable_scope("regression"):
    x = tf.placeholder(tf.float32, [None, 784])
    y, variables = model.regression(x)

# train
y_ = tf.placeholder("float", [None, 10])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 全局初始化
    for _ in range(1000):
        batch_xs, batch_ys = data.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print((sess.run(accuracy, feed_dict={x: data.test.images, y_: data.test.labels})))

    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'regression.ckpt'),
        write_meta_graph=False, write_state=False)
    print("Saved:", path)

