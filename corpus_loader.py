import os
from collections import defaultdict
from random import shuffle
import math, random
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
        '''
        comines self with another corpus
        :param path: path to file
        :param min: min length of sentence
        :param max: max length of sentence
        :return: self
        '''
        temp = self.target_names
        self.load(path, min, max)
        newTargets = set(temp + self.target_names)
        self.target_names = list(newTargets)

    def clone(self):
        '''
        deep copy of CL object
        :return: new CL object
        '''

        CL = CorpusLoader()
        for key in self.data.keys():
            CL.data[key] = []
            for sentPair in self.data[key]:
                CL.data[key].append(sentPair[:])

        CL.target_names = self.target_names[:]
        CL.containing = self.containing[:]

        return CL

    def _tokenize(self, sentList):
        '''
        helper function for tokenize. does all the dirty work
        :param sentList: sentence pair [sent1, sent2]
        :return: list of tokenized sentence pairs
        '''

        tokenized = []

        for sent in sentList:

            if "meta" in self.containing:
                sent = sent.lower()

            words = word_tokenize(sent, "english")

            #stopwordList = set(stopwords.words("english"))
            #stopwordList.add("'s")
            stopwordList = set("'s")

            filtered = [w.replace("n't", "not") for w in words if w not in stopwordList]

            joined = ' '.join(str(elem) for elem in filtered)

            tokenized.append(joined)

        return tokenized

    def tokenize(self):
        '''
        tokenize words
        :return:
        '''
        for label in self.data.keys():

            self.data[label] = [self._tokenize(sentPair) for sentPair in self.data[label]]

        return self


    def load(self, path, min = None, max = None):
        '''
        load labeled sentence in self.data as dictionary {label:sentence}
        :param path: path to file
        :param min: min length of sentence
        :param max: max length of sentence
        :return: self
        '''

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
        '''
        print the number of instances per label
        :return:
        '''
        stats = {}
        stats["total"] = 0
        for key in self.data.keys():
            stats[key] = len(self.data[key])
            stats["total"] += len(self.data[key])
            print(str(key) + " : " + str(len(self.data[key])))

        print(str("total") + " : " + str(stats["total"]))

        return stats


    def grab(self, label, n, shuffled=True):
        '''
        grab n elements with label
        :param label: target label
        :param n: number of phrases which should be grabbed
        :return: list of n sentence pairs
        '''

        sentences = self.data[label]
        if n > len(sentences):
            raise ValueError("The requested number is larger than the existing data.")

        if shuffled:
            shuffle(sentences)

        grabbed = sentences[:n]
        return grabbed

    def partition(self, labels, partitionList, shuffled=False):
        '''
        partition the labels of a corpus into balanced partitions
        :param labels: list of labels which should be partitioned
        :param partitionList: list of percentages
        :param shuffled: if the data should be shuffled
        :return: list of partitions [ [part1], [part2]...]
        '''

        partitioned = [CorpusLoader() for x in partitionList]

        if sum(partitionList) != 100:
            raise ValueError("The given percentages do not sum to 100.")

        else:

            for label in labels:

                total = len(self.data[label])

                if shuffled:
                    temp = self.data[label][:]
                    random.shuffle(temp)
                else:
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
        '''
        get the label with the smallest amount of phrases
        :param listOfLabels: list of labels which should be considered
        :return: name of the smallest label
        '''
        smallest = len(self.data[listOfLabels[0]]), listOfLabels[0]
        for label in listOfLabels:
            if len(self.data[label]) < smallest[0]:
                smallest = len(self.data[label]),label
        return smallest



    def balance(self, labels = None):
        '''
        balance the labels/corpus
        :param labels: list of labels which should be balanced
        :return: dict{label:[ [sent1, sent2] ,..., [sent3, sent4] ]}
        '''

        if labels is None:
            labels = self.target_names
        corpus = defaultdict(list)
        min, smallest = self.get_smallest(labels)
        for label in labels:
            corpus[label] = self.grab(label, min)
        return corpus

    def mergeLabel(self, labels, newLabel):
        '''
        maps all data of labels to newLabel
        :param labels: list of labels which should be merged
        :param newLabel: new label of the merged data
        :return: self
        '''
        merged = []
        for label in labels:
            if label in self.data.keys():
                merged += self.data[label]
        self.data[newLabel] = merged

        if newLabel not in self.target_names:
            self.target_names.append(newLabel)

    def mergeData(self):
        '''
        merge the two phrases into one if the data consists of two phrases per label [sent1, sent2] -> [sent1+2]
        :return: self
        '''

        for label in self.data:
            i = 0
            for elem in self.data[label]:
                newStr = elem[0] + " " + elem[1]
                self.data[label][i] = newStr
                i += 1

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
        :param labels: list of labels which should be used
        :return: samples: list of instances; y: target list; mapping: mapping between y and labels
        '''

        if corpus == None:
            corpus = self.data

        samples = []
        y = []
        i = 0
        mapping = []

        #restricting the biggest class to same size as the second biggest class
        biggest = (0, "")
        secondBiggest = (0, "")
        for label in labels:

            if len(corpus[label]) > biggest[0]:
                secondBiggest = biggest
                biggest = (len(corpus[label]), label)
            elif len(corpus[label]) > secondBiggest[0]:
                secondBiggest = (len(corpus[label]), label)
        #print(self.stats())
        #print(biggest)
        #print(secondBiggest)

        #convert to list format
        for label in labels:

            if label == biggest[1]:
                max = secondBiggest[0]
                newCorp = corpus[label][:max]
                #print(len(newCorp))
            else:
                newCorp = corpus[label]

            mapping.append(label)
            for elem in newCorp:
                samples.append(elem)
                y.append(i)
            i += 1
        return samples, np.array(y), mapping

