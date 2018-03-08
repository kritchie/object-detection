#!/usr/bin/env python
# -*- coding: utf-8 -*-

from objet_detection.models.vgg16 import VGG16

if __name__ == '__main__':
    model = VGG16()
    nn = model.network()
    trainable = model.loss
    testable = model.accuracy
