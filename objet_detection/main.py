#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from objet_detection.models.vgg16 import SSDVGG16

if __name__ == '__main__':

    model = SSDVGG16()

    nn = model.build()

    trainable = model.loss(nn)
    testable = model.accuracy()
