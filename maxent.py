# -*- mode: Python; coding: utf-8 -*-
from __future__ import division
from collections import defaultdict
from classifier import Classifier
import numpy as np
import sys
import math

class MaxEnt(Classifier):

    def __init__(self, model=None):
        super(MaxEnt, self).__init__(model=None)
        self.labelsToWeights = None
        self.NUM_ITERATIONS = 10 # Fixed number of iterations of SGD

    def get_model(self): return self.labelsToWeights;
    def set_model(self, model): self.labelsToWeights = model
    model = property(get_model, set_model)

    def train(self, instances, dev_instances=None):
        self.train_sgd(instances, dev_instances, 0.001, 30)

    # Trains this classifier using stochastic gradient descent
    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        self.labelsToWeights = self.initializeWeights(train_instances)
        for j in range(self.NUM_ITERATIONS):
            for i in range(0, len(train_instances), batch_size):
                batch = train_instances[i : (i + batch_size)]
                gradient = self.gradient(batch)
                for label in self.labelsToWeights:
                    self.labelsToWeights[label] += learning_rate * gradient[label]
            print 'negLogLikelihood = ',self.negLogLikelihood(dev_instances)

    # Classifies the given instance as the most likely label from the dataset,
    # given the current model
    def classify(self, instance):
        posteriors = {}
        for label in self.labelsToWeights:
            posteriors[label] = self.posterior(label, instance.features())
        return max(posteriors, key=posteriors.get)

    # Initializes model parameter weights to zero
    def initializeWeights(self, train_instances):
        labels = {}
        numFeatures = len(train_instances[0].features())
        for instance in train_instances:
            if instance.label not in labels:
                labels[instance.label] = np.zeros(numFeatures)
        return labels

    # Returns the posterior probability P(label | featureVec)
    def posterior(self, label, featureVec):
        dotProds = {}
        # Calculate each posterior once
        for l, w in self.labelsToWeights.iteritems():
            dotProds[l] = expDotProd(w, featureVec)
        return dotProds[label] / sum(dotProds.itervalues())

    # Returns the observed counts for each feature in the passed mini-batch
    def observedCounts(self, instances):
        observedCounts = defaultdict(lambda: np.zeros(len(instances[0].features())))
        for instance in instances:
            observedCounts[instance.label] += instance.features()
        return observedCounts

    # Returns the expected model counts (right hand sand of gradient difference)
    # given a mini batch of instances
    def expectedModelCounts(self, instances):
        expectedCounts = defaultdict(lambda: np.zeros(len(instances[0].features())))
        for instance in instances:
            for label, w in self.labelsToWeights.iteritems():
                posterior = self.posterior(label, instance.features())
                expectedCounts[label] += instance.features() * posterior
        return expectedCounts

    # Computes the gradient over the given instances
    def gradient(self, instances):
        expected = self.expectedModelCounts(instances)
        observed = self.observedCounts(instances)
        gradient = defaultdict(lambda: np.zeros(len(instances[0].features())))
        for label in self.labelsToWeights:
            gradient[label] = observed[label] - expected[label]
        return gradient

    # Computes the negative log-likelihood over a set of instances
    def negLogLikelihood(self, instances):
        return -1 * sum([math.log(self.posterior(instance.label, instance.features())) for instance in instances])

    # Computes the accuracy of the classifier over a set of instances
    def accuracy(self, instances):
        return sum([instance.label == self.classify(instance) for instance in instances]) / len(instances)

# Returns e^(w dot x)
def expDotProd(w, x):
    return math.exp(np.dot(w, x))
