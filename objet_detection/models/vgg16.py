#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class VGG16(object):
    """
    VGGNet with 16 weight layers. Images will be cropped to a resolution of 640x480 pixels.
    Images are RGB (3 channels)
    """

    def __init__(self):
        self.width = 640
        self.height = 480
        self.channels = 3
        self.x = tf.placeholder(shape=[None, self.height, self.width, self.channels], dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, self.height, self.width, self.channels], dtype=tf.float32)

    @staticmethod
    def weights(dims, stdev=0.1):
        return tf.truncated_normal(shape=dims, stddev=stdev, name='weights')

    @staticmethod
    def biases(dims):
        return tf.constant(shape=dims, name='biases')

    def conv2d(self, input, dims):
        return tf.nn.relu(tf.nn.conv2d(input, filter=self.weights(dims), strides=[1, 1, 1, 1])
                          + self.biases(dims), name='conv2d')

    def pooling(self, input):
        return tf.nn.max_pool(input, strides=[1, 2, 2, 1], name='max_pool2x2')

    def loss(self, logits):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits), name='xentropy')

    def train(self, loss, learning_rate=0.001):
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def accuracy(self, y, logits):
        # TODO
        pass

    def build(self):

        # TODO, Add summaries to work with TensorBoard

        with tf.name_scope('block1'):
            conv = self.conv2d(self.x, dims=[3, 3, 64])
            conv = self.conv2d(conv, dims=[3, 3, 64])
            pool = self.pooling(conv)

        with tf.name_scope('block2'):
            conv = self.conv2d(pool, dims=[3, 3, 128])
            conv = self.conv2d(conv, dims=[3, 3, 128])
            pool = self.pooling(conv)

        with tf.name_scope('block3'):
            conv = self.conv2d(pool, dims=[3, 3, 256])
            conv = self.conv2d(conv, dims=[3, 3, 256])
            conv = self.conv2d(conv, dims=[3, 3, 256])
            pool = self.pooling(conv)

        with tf.name_scope('block4'):
            conv = self.conv2d(pool, dims=[3, 3, 512])
            conv = self.conv2d(conv, dims=[3, 3, 512])
            pool = self.pooling(conv)

        with tf.name_scope('block5'):
            conv = self.conv2d(pool, dims=[3, 3, 512])
            conv = self.conv2d(conv, dims=[3, 3, 512])
            pool = self.pooling(conv)

        with tf.name_scope('fc'):
            fc = tf.reshape(pool, [-1, 40 * 15 * 512])
            fc = tf.layers.dense(fc, units=4096, activation=tf.nn.relu)
            fc = tf.layers.dense(fc, units=4096, activation=tf.nn.relu)
            fc = tf.layers.dense(fc, units=1000, activation=tf.nn.relu)

        with tf.name_scope('loss'):
            loss = self.loss(fc)

        return loss
