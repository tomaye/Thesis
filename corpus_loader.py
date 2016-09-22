import os
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from random import shuffle
import math
import numpy as np

class CorpusLoader:
    """loads and distributes training and test data"""

    def __init__(self, file=None):
        self.data = defaultdict(list)
        self.target_names = []
        if file != None:
            self.load(file, False)

    def add_Corpus(self,path):
        #combines another corpus with the current
        temp = self.target_names
        self.load(path)
        newTargets = set(temp + self.target_names)
        self.target_names = list(newTargets)

    def load(self, path, lemmatize=False):
        #loads corpus into self.data
        file = open(os.getcwd()+"/"+ path)
        if lemmatize:
            lemmatizer = WordNetLemmatizer()
        for line in file:
            line = line.split("\t")
            text = line[-2:]
            if lemmatize:
                text = []
                for sentence in line[-2:]:
                    temp = []
                    for word in sentence.split():
                        temp.append(lemmatizer.lemmatize(word.lower(), "v"))
                    temp = " ".join(temp)
                    text.append(temp)
            self.data[line[0]].append(text)
        self.target_names = list(self.data.keys())

    def stats(self, data):
        #returns the distribution of data
        stats = {}
        stats["total"] = 0
        for key in data.keys():
            stats[key] = len(data[key])
            stats["total"] += len(data[key])
            print(str(key) + " : " + str(len(data[key])))

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

    def get_smallest(self,listOfLabels):
        #returns the smallest label (number,label)
        smallest = len(self.data[listOfLabels[0]]),listOfLabels[0]
        for label in listOfLabels:
            if len(self.data[label]) < smallest[0]:
                smallest = len(self.data[label]),label
        #print(smallest)
        return smallest

    def balance(self,labels = None):
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

    def distribute(self,labels,percentages):
        #[l1,l2,l3],[50,30,20]
        None
        #TODO

    def mergeLabel(self,label1, label2, newLabel):
        #merges two labels into a new one
        self.data[newLabel] = self.data[label1] + (self.data[label2])
        #self.target_names.remove(label1)
        #self.target_names.remove(label2)
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

    def toLists(self, corpus, labels):
        #samples: list of strings
        #y: target
        #mapping: mapping labels to int

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


def main():
    corpus = CorpusLoader()
    file = "data/corpus/Metalogue_extractedLinks_fullCorpus.txt"
    corpus.load(file, False)
    #print(corpus.grab("negative",1))
    #corpus.stats(corpus.data)
    #corp = corpus.balance(["evidence","negative","justification"])
    #corpus.stats(corp)
    corpus.mergeData()
main()