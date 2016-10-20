import os
from collections import defaultdict
from random import shuffle
import math
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class CorpusLoader:
    '''loads and distributes data'''

    def __init__(self, file=None, min = None, max = None):
        self.data = defaultdict(list)
        self.target_names = []
        self.containing = []
        if file != None:
            self.load(file, min, max)


    def add_Corpus(self, path, min, max):
        #combines another corpus with the current
        temp = self.target_names
        self.load(path, min, max)
        newTargets = set(temp + self.target_names)
        self.target_names = list(newTargets)

    def _tokenize(self, sentList):

        tokenized = []

        for sent in sentList:

            if "meta" in self.containing:
                sent = sent.lower()

            words = word_tokenize(sent, "english")

            stopwordList = set(stopwords.words("english"))
            stopwordList.add("'s")
            filtered = [w for w in words if w not in stopwordList]

            joined = ' '.join(str(elem) for elem in filtered)

            tokenized.append(joined)

        return tokenized

    def tokenize(self):

        for label in self.data.keys():

            self.data[label] = [self._tokenize(sentPair) for sentPair in self.data[label]]

        return self


    def load(self, path, min = None, max = None):
        #loads corpus into self.data
        #min = minLength of phrase
        #max = MaxLength of phrase

        file = open(os.getcwd()+"/" + path)

        for line in file:
            line = line.split("\t")
            text = line[-2:]
            textLength = (text[0] + text[1]).split()

            if min != None and len(textLength) < min:
                continue

            if max != None and len(textLength) > max:
                continue

            self.data[line[0]].append(text)
        self.target_names = list(self.data.keys())



    def stats(self):
        #returns the distribution of data
        stats = {}
        stats["total"] = 0
        for key in self.data.keys():
            stats[key] = len(self.data[key])
            stats["total"] += len(self.data[key])
            print(str(key) + " : " + str(len(self.data[key])))

        print(str("total") + " : " + str(stats["total"]))

        return stats


    def grab(self, label, n, shuffled=True):
        #grabs n elements with this label
        #returns list of sentence pairs with length n

        sentences = self.data[label]
        if n > len(sentences):
            raise ValueError("The requested number is larger than the existing data.")

        if shuffled:
            shuffle(sentences)

        grabbed = sentences[:n]
        return grabbed

    def partition(self, labels, partitionList, shuffled=True):
        #same as grab() but with balanced labels
        #partitionList: list of percentages
        #returns a list of partitions

        #partitioned = [ defaultdict(list) for x in partitionList]
        partitioned = [CorpusLoader() for x in partitionList]

        if sum(partitionList) != 100:
            raise ValueError("The given percentages do not sum to 100.")

        else:

            for label in labels:

                total = len(self.data[label])
                temp = self.data[label][:]
                partitions = [math.floor(part*(total/100)) for part in partitionList]
                start = 0
                i = 0

                for part in partitions:

                    partitioned[i].data[label] += temp[start:(start + part)]
                    start = start + part
                    i += 1

            return partitioned

    def get_smallest(self, listOfLabels):
        #returns the smallest label (number,label)
        smallest = len(self.data[listOfLabels[0]]), listOfLabels[0]
        for label in listOfLabels:
            if len(self.data[label]) < smallest[0]:
                smallest = len(self.data[label]),label
        #print(smallest)
        return smallest



    def balance(self, labels = None):
        #balance the corpus with respect to the smallest class
        #returns dict with label:[[sent1,sent2],...]
        if labels is None:
            labels = self.target_names
        corpus = defaultdict(list)
        #numOfClasses = len(labels)
        #percentage = math.trunc(100/numOfClasses)
        #print(percentage)
        min, smallest = self.get_smallest(labels)
        for label in labels:
            corpus[label] = self.grab(label, min)
        return corpus

    def mergeLabel(self, labels, newLabel):
        #merges labels into a new one
        merged = []
        for label in labels:
            if label in self.data.keys():
                merged += self.data[label]
        self.data[newLabel] = merged

        if newLabel not in self.target_names:
            self.target_names.append(newLabel)

    def mergeData(self):
        #merges utterances into String [sent1,sent2] -> sent1 + sent2
        #print(self.data["negative"][1])
        #print(type(self.data["negative"][1]))
        for label in self.data:
            i = 0
            for elem in self.data[label]:
                newStr = elem[0] + " " + elem[1]
                self.data[label][i] = newStr
                i += 1
        #print(self.data["negative"][1])
        #print(type(self.data["negative"][1]))

    def mergeWithCorpus(self, corpora):
        '''
        merges self.data with the data of another CL object or list of CLs
        :param corpora: list of CorpusLoader objects
        :return: self
        '''

        for corpus in corpora:

            for label in corpus.data.keys():
                if label in self.data.keys():
                    self.data[label] += corpus.data[label]
                else:
                    self.data[label] = corpus.data[label]

        return self


    def toLists(self, labels, corpus = None):
        '''
        converts CL object into samples, y and a mapping
        :param labels: labels which should be used
        :return: samples: list of instances; y: target list; mapping: mapping between y and labels
        '''
        #samples: list of strings
        #y: target
        #mapping: mapping labels to int

        if corpus == None:
            corpus = self.data

        samples = []
        y = []
        i = 0
        mapping = []
        for label in labels:
            mapping.append(label)
            for elem in corpus[label]:
                samples.append(elem)
                y.append(i)
            i += 1

        return samples, np.array(y), mapping

