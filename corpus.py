# -*- mode: Python; coding: utf-8 -*-

"""For the purposes of classification, a corpus is defined as a collection
of labeled documents. Such documents might actually represent words, images,
etc.; to the classifier they are merely instances with features."""

from abc import ABCMeta, abstractmethod
from collections import Counter
from glob import glob
from os.path import basename, dirname, split, splitext
from itertools import islice, izip
import numpy as np
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
        return np.array(arr)

    def bagOfChars(self, text):
        arr = [0] * 26
        for char in text:
            arr[ord(char) - ord('a')] += 1
        return np.array(arr)

    def setOfChars(self, text):
        arr = [0] * 26
        for char in text:
            arr[ord(char) - ord('a')] = 1
        return np.array(arr)

class Review(Document):

    def __init__(self, data, label=None, source=None):
        super(Review, self).__init__(data, label, source)
        # Encoded feature vector cached so we do not recompute
        self.encoded = None

    def features(self):
        return self.encoded if self.encoded is not None else self.unencodedFeatures()

    def unencodedFeatures(self):
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

    def setEncoded(self, encoded):
        self.encoded = encoded

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
                self.documents.append(document_class(data, label, datafile))

class ReviewCorpus(Corpus):
    """Yelp dataset challenge. A collection of business reviews. 
    """
    def __init__(self, datafiles, document_class=Review):
        self.featureIdxMap = {}
        self.labelIdxMap = {}
        self.labels = []
        super(ReviewCorpus, self).__init__(datafiles, document_class)

    def load(self, jsonFile, document_class=Review):
        self.documents = self.readDocumentsFromJson(jsonFile)

    # Convert JSON to vectors
    def readDocumentsFromJson(self, jsonFile, document_class=Review):
        reviews = []
        counter = 1
        start = time.time()
        with open(jsonFile, "r") as vectorFile:
            for line in vectorFile:
                review = json.loads(line)
                label = review['sentiment']
                if label not in self.labels:
                    self.labels.append(label)
                data = review['text']
                instance = document_class(data, label, jsonFile)
                reviews.append(instance)
                self.encodeFeatureIdxs(instance)
                if counter % 20000 == 0:
                    print 'Loaded ', counter, ' instances'
                    print 'Unique # features = ', len(self.featureIdxMap)
                counter += 1
        end = time.time()
        # Initialize map of labels -> idxs
        for label in self.labels:
            self.labelIdxMap[label] = len(self.labelIdxMap) + len(self.featureIdxMap)
        print 'finished loading reviews in ', end - start
        start = time.time()
        for instance in reviews:
            instance.setEncoded(encodeAsVec(instance, self.featureIdxMap, self.labelIdxMap))
        end = time.time()
        print 'finished encoding as featureVectors in ', end - start
        return reviews

    # Appends unseen features of this instance to the map of
    # {feature_name -> feature_idx_in_vector}
    def encodeFeatureIdxs(self, instance):
        for feature in instance.unencodedFeatures():
            if feature not in self.featureIdxMap:
                self.featureIdxMap[feature] = len(self.featureIdxMap)

    # Returns map of {feature_name -> feature_idx_in_vectpr}
    def getFeatureToIdxMap(self):
        return self.featureIdxMap

# Takes in an encoding as a dict (i.e. {('The', 'Boy') -> 3, ...}
# and returns encoding as vec [0, 0, ..., 3, ..., 0] based on mapping
# of features to their indexes in the featureVector
def encodeAsVec(instance, featureIdxMap, labelIdxMap):
    unencodedFeatureDict = instance.unencodedFeatures()
    vec = np.zeros(len(featureIdxMap) + len(labelIdxMap))
    for feature in unencodedFeatureDict:
        if feature in featureIdxMap:
            idx = featureIdxMap[feature]
            vec[idx] = unencodedFeatureDict[feature]
    # Encode prior/bias probability
    vec[labelIdxMap[instance.label]] = 1
    return vec

# Takes in an instance encoded as a feature vector (i.e. [0, 1, ... , 3, 0]
# and returns it encoded as a dictionary (i.e. {'Girl' -> 5, ..., 'What' -> 7}
def decodeVec(featureVec, idxFeatureMap):
    decoded = Counter()
    for idx, val in enumerate(featureVec):
        if val != 0:
            decoded[idxFeatureMap[idx]] = int(val)
    return decoded
