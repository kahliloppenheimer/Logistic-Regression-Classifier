# -*- mode: Python; coding: utf-8 -*-

"""For the purposes of classification, a corpus is defined as a collection
of labeled documents. Such documents might actually represent words, images,
etc.; to the classifier they are merely instances with features."""

from abc import ABCMeta, abstractmethod
from collections import Counter
from glob import glob
import os
from os.path import basename, dirname, split, splitext
from itertools import islice, izip
from cPickle import dump, load, HIGHEST_PROTOCOL as HIGHEST_PICKLE_PROTOCOL
import numpy
import json
import re
import time


class Document(object):
    """A document completely characterized by its features."""

    max_display_data = 10 # limit for data abbreviation

    def __init__(self, data, label=None, source=None):
        self.data = data
        self.label = label
        self.source = source
        self.feature_vector = []

    def __repr__(self):
        return ("<%s: %s>" % (self.label, self.abbrev()) if self.label else
                "%s" % self.abbrev())

    def abbrev(self):
        return (self.data if len(self.data) < self.max_display_data else
                self.data[0:self.max_display_data] + "...")

    def features(self):
        """A list of features that characterize this document."""
        return [self.data]


class Name(Document):

    def features(self):
        text = self.removeNonAlphabetic(self.data)
        return  (self.oneHotCharEncoding(text[0])
            +   self.oneHotCharEncoding(text[-1])
            +   self.bagOfChars(text)
            +   self.setOfChars(text))

    def removeNonAlphabetic(self, str):
        regex = re.compile('[^a-zA-Z]')
        return regex.sub('', str.lower())

    def oneHotCharEncoding(self, char):
        arr = [0] * 26
        arr[ord(char) - ord('a')] = 1
        return arr

    def bagOfChars(self, text):
        arr = [0] * 26
        for char in text:
            arr[ord(char) - ord('a')] += 1
        return arr

    def setOfChars(self, text):
        arr = [0] * 26
        for char in text:
            arr[ord(char) - ord('a')] = 1
        return arr

class Review(Document):

    def features(self):
        normalized = self.normalize(self.data)
        return self.unigrams(normalized)

    def unigrams(self, normalized):
        return Counter(normalized.split())

    def bigrams(self, normalized):
        words = [word for word in normalized.split()]
        return Counter(izip(words, islice(words, 1, None)))

    def normalize(self, text):
        return self.stripNonAlphaNum(text.lower())

    def stripNonAlphaNum(self, text):
        from unicodedata import category
        return ''.join(ch for ch in text if category(ch)[0] != 'P')

class Corpus(object):
    """An abstract collection of documents."""

    __metaclass__ = ABCMeta

    def __init__(self, datafiles, document_class=Document):
        self.documents = []
        self.datafiles = glob(datafiles)
        for datafile in self.datafiles:
            self.load(datafile, document_class)

    # Act as a mutable container for documents.
    def __len__(self): return len(self.documents)
    def __iter__(self): return iter(self.documents)
    def __getitem__(self, key): return self.documents[key]
    def __setitem__(self, key, value): self.documents[key] = value
    def __delitem__(self, key): del self.documents[key]

    @abstractmethod
    def load(self, datafile, document_class):
        """Make labeled document instances for the data in a file."""
        pass

class PlainTextFiles(Corpus):
    """A corpus contained in a collection of plain-text files."""

    def load(self, datafile, document_class):
        """Make a document from a plain-text datafile. The document is labeled
        using the last component of the datafile's directory."""
        label = split(dirname(datafile))[-1]
        with open(datafile, "r") as file:
            data = file.read()
            self.documents.append(document_class(data, label, datafile))

class PlainTextLines(Corpus):
    """A corpus in which each document is a line in a datafile."""

    def load(self, datafile, document_class):
        """Make a document from each line of a plain text datafile.
        The document is labeled using the datafile name, sans directory
        and extension."""
        label = splitext(basename(datafile))[0]
        with open(datafile, "r") as file:
            for line in file:
                data = line.strip()
                self.documents.append(document_class(data, label, datafile))


class NamesCorpus(PlainTextLines):
    """A collection of names, labeled by gender. See names/README for
    copyright and license."""

    def __init__(self, datafiles="names/*.txt", document_class=Name):
        super(NamesCorpus, self).__init__(datafiles, document_class)

    def load(self, datafile, document_class):
        label = splitext(basename(datafile))[0]
        with open(datafile, "r") as file:
            for line in file:
                data = line.strip()
                self.documents.append(document_class(data, label, datafile).features())

class ReviewCorpus(Corpus):
    """Yelp dataset challenge. A collection of business reviews. 
    """
    def __init__(self, datafiles, document_class=Review):
        self.featureIdxMap = {}
        self.idxFeatureMap = {}
        self.SAVED_VECTOR_EXTENSION = "p"
        super(ReviewCorpus, self).__init__(datafiles, document_class)

    def load(self, jsonFile, document_class=Review):
        vectorFile = jsonFile.split('.')[0] + '.' + self.SAVED_VECTOR_EXTENSION
        if os.path.isfile(vectorFile):
            self.documents = self.loadVectorsFromFile(vectorFile)
        else:
            self.documents = self.readAndConvertJsonToVectors(jsonFile)
            self.saveVectorsToFile(vectorFile)

        """Make an unencoded document from each row of a json-formatted Yelp reviews
        """

    # Convert JSON to vectors
    def readAndConvertJsonToVectors(self, jsonFile, document_class=Review):
        unencoded = []
        counter = 1
        start = time.time()
        with open(jsonFile, "r") as vectorFile:
            for line in vectorFile:
                review = json.loads(line)
                label = review['sentiment']
                data = review['text']
                instance = document_class(data, label, jsonFile)
                unencoded.append(instance)
                self.encodeFeatureIdxs(instance)
                if counter % 20000 == 0:
                    print 'Loaded ', counter, ' instances'
                    print 'Unique # features = ', len(self.featureIdxMap)
                counter += 1
        end = time.time()
        print 'finished loading in ', end - start
        start = time.time()
        encoded = [self.encodeAsVec(instance.features()) for instance in unencoded]
        end = time.time()
        print 'finished encoding as featureVectors in ', end - start
        return encoded

    # Saves vectors along with feature index map to given file
    def saveVectorsToFile(self, file):
        """Save the current model to the given file."""
        if isinstance(file, basestring):
            with open(file, "wb") as file:
                self.saveVectorsToFile(file)
        else:
            print 'saving vectors to ', file.name
            start = time.time()
            dump([self.featureIdxMap, self.documents], file, HIGHEST_PICKLE_PROTOCOL)
            end = time.time()
            print 'Finished saving vectors to file in ', end - start, 's'

    # Loads vectors along with feature index map from file
    def loadVectorsFromFile(self, file):
        """Load a saved model from the given file."""
        if isinstance(file, basestring):
            with open(file, "rb") as file:
                self.loadVectorsFromFile(file)
        else:
            print 'loading vectors from ', file
            start = time.time()
            loaded = load(file)
            self.featureIdxMap = loaded[0]
            self.documents = loaded[1]
            end = time.time()
            print 'Finished loading vectors from file in ', end - start, 's'


    # Appends unseen features of this instance to the map of
    # {feature_name -> feature_idx_in_vector}
    def encodeFeatureIdxs(self, instance):
        for feature in instance.features():
            if feature not in self.featureIdxMap:
                nextIdx = len(self.featureIdxMap)
                self.featureIdxMap[feature] = nextIdx
                self.idxFeatureMap[nextIdx] = feature

    # Returns map of {feature_name -> feature_idx_in_vectpr}
    def getFeatureToIdxMap(self):
        return self.featureIdxMap

    # Returns map of {feature_idx_in_vector -> feature_name}
    def getIdxToFeatureMap(self):
        return self.idxFeatureMap

    # Takes in an encoding as a dict (i.e. {('The', 'Boy') -> 3, ...}
    # and returns encoding as vec [0, 0, ..., 3, ..., 0] based on mapping
    # of features to their indexes in the featureVector
    def encodeAsVec(self, featureDict):
        vec = numpy.zeros(len(self.featureIdxMap))
        for feature in featureDict:
            if feature in self.featureIdxMap:
                idx = self.featureIdxMap[feature]
                vec[idx] = featureDict[feature]
        return vec

    # Takes in an instance encoded as a feature vector (i.e. [0, 1, ... , 3, 0]
    # and returns it encoded as a dictionary (i.e. {'Girl' -> 5, ..., 'What' -> 7}
    def decodeVec(self, featureVec):
        dict = {}
        for idx, val in enumerate(featureVec):
            dict[self.idxFeatureMap[idx]] = val
        return dict

nc = NamesCorpus()
print 'sample vec = ', nc.documents[0]


#rc = ReviewCorpus('yelp_reviews.json', Review)
#print 'sample featureVec = ', rc.documents[0]
#print(rc.decodeVec(rc.encodeAsVec({u'the': 2, u'and': 3, u'forrealz': 1})) == {u'the': 2, u'and': 3, u'forrealz': 1})
#print(rc.encodeAsVec({u'the': 3, u'and': 3, u'forrealz': 1}) != rc.encodeAsVec({u'the': 2, u'and': 3, u'forrealz': 1}))
#print(rc.encodeAsVec({u'ther': 2, u'and': 3, u'forrealz': 1}) != rc.encodeAsVec({u'the': 2, u'and': 3, u'forrealz': 1}))
