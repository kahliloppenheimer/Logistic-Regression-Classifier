import numpy as np
from corpus import Document, NamesCorpus, ReviewCorpus, Name, Review
from maxent import MaxEnt
from unittest import TestCase, main
from random import shuffle, seed
import sys


class BagOfWords(Document):
    def features(self):
        """Trivially tokenized words."""
        return self.data.split()

class Animal(Document):
    def features(self):
        return self.data

def accuracy(classifier, test, verbose=sys.stderr):
    correct = [classifier.classify(x) == x.label for x in test]
    if verbose:
        print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
    return float(sum(correct)) / len(correct)

class MaxEntTest(TestCase):
    u"""Tests for the MaxEnt classifier."""

    # Tests posterior calculations against calculations in PS2
    def test_posterior(self):
        classifier = MaxEnt()
        classifier.labelsToWeights = {'cat': [2, -3], 'dog': [-2, 3]}
        self.assertAlmostEqual(classifier.posterior('cat', np.array([0, 1])), .00247, 4)
        self.assertAlmostEqual(classifier.posterior('cat', np.array([1, 0])), .98201, 4)
        self.assertAlmostEqual(classifier.posterior('dog', np.array([0, 1])), .99753, 4)
        self.assertAlmostEqual(classifier.posterior('dog', np.array([1, 0])), .01799, 4)
        self.assertAlmostEqual(classifier.posterior('cat', np.array([1, 1])), .11920, 4)
        self.assertAlmostEqual(classifier.posterior('dog', np.array([1, 1])), .88080, 4)

    # Tests expected count calculations against calculations in PS2
    def test_expected_counts(self):
        classifier = MaxEnt()
        classifier.labelsToWeights = {'cat': [2, -3], 'dog': [-2, 3]}
        dataset = ([Animal(np.array([1, 0]), 'cat'), Animal(np.array([0, 1]), 'cat'), Animal(np.array([0, 1]), 'dog'),
                    Animal(np.array([0, 1]), 'dog'), Animal(np.array([1, 1]), 'cat')])
        expectedCounts = classifier.expectedModelCounts(dataset)
        self.assertAlmostEqual(expectedCounts['cat'][0], 1.10121, 4)
        self.assertAlmostEqual(expectedCounts['cat'][1], .12661, 4)
        self.assertAlmostEqual(expectedCounts['dog'][0], .89879, 4)
        self.assertAlmostEqual(expectedCounts['dog'][1], 3.87339, 4)

    # Tests observed count calculations against calculations in PS2
    def test_observed_counts(self):
        classifier = MaxEnt()
        classifier.labelsToWeights = {'cat': [2, -3], 'dog': [-2, 3]}
        dataset = ([Animal(np.array([1, 0]), 'cat'), Animal(np.array([0, 1]), 'cat'), Animal(np.array([0, 1]), 'dog'),
                    Animal(np.array([0, 1]), 'dog'), Animal(np.array([1, 1]), 'cat')])
        observedCounts = classifier.observedCounts(dataset)
        self.assertEqual(observedCounts['cat'][0], 2)
        self.assertEqual(observedCounts['cat'][1], 2)
        self.assertEqual(observedCounts['dog'][0], 0)
        self.assertEqual(observedCounts['dog'][1], 2)

    # Tests gradient calculations against calculations in PS2
    def test_gradient(self):
        classifier = MaxEnt()
        classifier.labelsToWeights = {'cat': [2, -3], 'dog': [-2, 3]}
        dataset = ([Animal(np.array([1, 0]), 'cat'), Animal(np.array([0, 1]), 'cat'), Animal(np.array([0, 1]), 'dog'),
                    Animal(np.array([0, 1]), 'dog'), Animal(np.array([1, 1]), 'cat')])
        gradient = classifier.gradient(dataset)
        self.assertAlmostEqual(gradient['cat'][0], .89879, 4)
        self.assertAlmostEqual(gradient['cat'][1], 1.87339, 4)
        self.assertAlmostEqual(gradient['dog'][0], -.89879, 4)
        self.assertAlmostEqual(gradient['dog'][1], -1.87339, 4)

    def split_names_corpus(self, names):
        """Split the names corpus into training, dev, and test sets"""
        self.assertEqual(len(names), 5001 + 2943) # see names/README
        seed(hash("names"))
        shuffle(names)
        return (names[:5000], names[5000:6000], names[6000:])

    def test_names_nltk(self):
        """Classify names using NLTK features"""
        names = NamesCorpus(document_class=Name)
        train, dev, test = self.split_names_corpus(names)
        classifier = MaxEnt()
        classifier.train(train, dev)
        acc = accuracy(classifier, test)
        self.assertGreater(acc, 0.70)

    def split_review_corpus(self, reviews):
        """Split the yelp review corpus into training, dev, and test sets"""
        seed(hash("reviews"))
        shuffle(reviews)
        n = len(reviews)
        return (reviews[:n * 7 / 10], reviews[n * 7 / 10:n * 8 / 10], reviews[n * 8 / 10 : n])

    def test_reviews(self):
        """Classify sentiment using bag-of-words"""
        reviews = ReviewCorpus('yelp_reviews.json', document_class=Review, numLines=15000)
        train, dev, test = self.split_review_corpus(reviews)
        print 'train length = ', len(train), ' dev length = ', len(dev), ' test length = ', len(test)
        print 'number of features = ', len(train[0].features())
        classifier = MaxEnt()
        classifier.train(train, dev)
        self.assertGreater(accuracy(classifier, test), 0.55)

if __name__ == '__main__':
    # Run all of the tests, print the results, and exit.
    main(verbosity=2)

