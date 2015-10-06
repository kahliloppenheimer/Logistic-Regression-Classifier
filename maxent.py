# -*- mode: Python; coding: utf-8 -*-

from __future__ import division

from classifier import Classifier
import numpy as np
import math


class MaxEnt(Classifier):

    def __init__(self, model=None):
        super(MaxEnt, self).__init__(model=None)
        self.weights = None

    def get_model(self): return self.weights;

    def set_model(self, model): pass

    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""
        self.train_sgd(instances, dev_instances, 0.0001, 30)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient 
        """
        pass

    def classify(self, instance):
        pass


def posterior(y, w, labelsToFeatures):
    dotProds = {}
    # Calculate each posterior once
    for label, featureVec in labelsToFeatures.iteritems():
        print('expDotProd ' + str(featureVec) + ', ' + str(w) + ' = ' + str(expDotProd(featureVec, w)))
        dotProds[label] = expDotProd(featureVec, w)

    print str(dotProds)
    return dotProds[y] / sum(dotProds.itervalues())

def expDotProd(w, x):
    return math.exp(np.dot(w, x))
