#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf


class SSDVGG16(object):
    """
    SSD with VGGNet with 16 weight layers. Images will be cropped to a resolution of 640x480 pixels.
    Images are RGB (3 channels)
    """

    def __init__(self):
        self.width = 640
        self.height = 480
        self.channels = 3
        self.num_classes = 2
        self.x = tf.placeholder(shape=[None, self.height, self.width, self.channels], dtype=tf.float32)
        tf.Summary.image('input', self.x)

        self.y = tf.placeholder(shape=[None, self.height, self.width, self.channels], dtype=tf.float32)

    @staticmethod
    def weights(dims, stdev=0.1):
        weights = tf.truncated_normal(shape=dims, stddev=stdev, name='weights')
        tf.summary.histogram('weights', weights)
        return weights

    @staticmethod
    def biases(dims):
        return tf.constant(0., shape=[dims], name='biases', dtype=tf.float32)

    def conv2d(self, input, dims):
        conv2d = tf.nn.conv2d(input, filter=self.weights(dims),
                              strides=[1, 1, 1, 1], padding='SAME') + self.biases(dims[-1])

        """
        Notes on Batch Norm for tensorflow
        
        From documentation : https://www.tensorflow.org/api_docs/python/tf/nn/moments
        
        When using these moments for batch normalization (see tf.nn.batch_normalization):

        for so-called "global normalization", used with convolutional filters with shape [batch, height, width, depth], 
        pass axes=[0, 1, 2]. for simple batch normalization pass axes=[0] (batch only).
        """
        mean, variance = tf.nn.moments(conv2d, axes=[0, 1, 2])
        bn = tf.nn.batch_normalization(conv2d, mean, variance, offset=None, scale=None, variance_epsilon=0.001)
        return tf.nn.relu(bn, name='relu')

    def pooling(self, input):
        return tf.nn.max_pool(input, strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1], padding='SAME', name='max_pool2x2')

    def loss(self, _in):
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self._in), name='xentropy')
            tf.summary.scalar('loss', loss)
        return loss

    def optimizer(self, loss, learning_rate=0.001):
        return tf.train.AdamOptimizer(learning_rate).minimize(loss)

    def accuracy(self, y, logits):
        accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
        tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def build(self):

        with tf.name_scope('block1'):
            conv = self.conv2d(self.x, dims=[3, 3, 3, 64])
            conv = self.conv2d(conv, dims=[3, 3, 64, 64])
            pool = self.pooling(conv)

        with tf.name_scope('block2'):
            conv = self.conv2d(pool, dims=[3, 3, 64, 128])
            conv = self.conv2d(conv, dims=[3, 3, 128, 128])
            pool = self.pooling(conv)

        with tf.name_scope('block3'):
            conv = self.conv2d(pool, dims=[3, 3, 128, 256])
            conv = self.conv2d(conv, dims=[3, 3, 256, 256])
            conv = self.conv2d(conv, dims=[3, 3, 256, 256])
            pool = self.pooling(conv)

        with tf.name_scope('block4'):
            conv = self.conv2d(pool, dims=[3, 3, 256, 512])
            conv = self.conv2d(conv, dims=[3, 3, 512, 512])
            pool = self.pooling(conv)

        with tf.name_scope('block5'):
            conv = self.conv2d(pool, dims=[3, 3, 512, 512])
            conv = self.conv2d(conv, dims=[3, 3, 512, 512])
            pool = self.pooling(conv)

        # We remove the full connection
        with tf.name_scope('fc'):
            fc = tf.reshape(pool, [-1, 40 * 15 * 512])
            fc = tf.layers.dense(fc, units=4096, activation=tf.nn.relu)
            fc = tf.layers.dense(fc, units=4096, activation=tf.nn.relu)
            fc = tf.layers.dense(fc, units=self.num_classes, activation=None)

        return fc
