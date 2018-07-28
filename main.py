# -*- coding: utf-8 -*-
__author__ = "purelove"
__date__ = "2018/7/28 下午11:02"
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify ,render_template, request
from mnist_data import model
x = tf.placeholder("float", [None, 784])
sess = tf.Session()

with tf.variable_scope("regression"):
    y1, variables = model.regression(x)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist_data/data/regression.ckpt")

with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convlutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist_data/data/convalutional.ckpt")


def regression(input):
    return sess.run(y1, feed_dict={x: input}).flatten().tolist()


def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


app = Flask(__name__)


@app.route('/api/mnist', methods=['post'])
def mnist():
    input = ((255-np.array(request.json, dtype=uint8)) / 255.0).reshape(1, 784)
    output1 = regression(input)
    output2 = convolutional(input)
    return jsonify(results=[output1, output2])


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=8000)
