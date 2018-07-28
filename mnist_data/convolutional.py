# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2018/7/28 下午10:36"
import os
from mnist_data import model
import tensorflow as tf
from mnist_data import input_data
import os
data = input_data.read_data_sets('MNIST_DATA', one_hot=True)

# model
with tf.variable_scope("convolutional"):
    x = tf.placeholder(tf.float32, [None, 784], name="x")
    keep_prob = tf.placeholder(tf.float32)
    y, variables = model.convlutional(x, keep_prob)

# train
y_ = tf.placeholder(tf.float32, [None, 10], name="y")
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver(variables)

with tf.Session() as sess:
    merged_summary_op = tf.summary.merge_all()
    summay_writer = tf.summary.FileWriter('/tmp/mnist_log/1', sess.graph)
    summay_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    # 卷积训练一般都需要1万 到2万次训练
    for i in range(1000):
        batch = data.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_:batch[1], keep_prob: 1.0})
            print("step %d, tranining accuracy %g" %(i, train_accuracy))
        sess.run(train_step, feed_dict={x: batch[0], y_:batch[1], keep_prob: 0.5})
    print(sess.run(accuracy, feed_dict={x:data.test.images, y_: data.test.labels, keep_prob: 1.0}))
    path = saver.save(
        sess, os.path.join(os.path.dirname(__file__), 'data', 'convalutional,ckpt'),
        write_meta_graph=False, write_state=False
    )
    print("saved:", path)
