# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier

class MaxEnt(Classifier):

    def get_model(self): return None

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

    
