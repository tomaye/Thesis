from itertools import chain, combinations
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
from sklearn.feature_selection import SelectKBest, chi2


class SkipgramVectorizer(DictVectorizer):

    def __init__(self):

        self.vectorizer = DictVectorizer()
        super(DictVectorizer, self).__init__()

    def pad_sequence(self, sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
        if pad_left:
            sequence = chain((pad_symbol,) * (n-1), sequence)
        if pad_right:
            sequence = chain(sequence, (pad_symbol,) * (n-1))
        return sequence

    def skipgrams(self, sequence, n, k, pad_left=False, pad_right=False, pad_symbol=None):
        sequence_length = len(sequence)
        sequence = iter(sequence)
        sequence = self.pad_sequence(sequence, n, pad_left, pad_right, pad_symbol)

        if sequence_length + pad_left + pad_right < k:
            raise Exception("The length of sentence + padding(s) < skip")

        if n < k:
            raise Exception("Degree of Ngrams (n) needs to be bigger than skip (k)")

        history = []
        nk = n+k

        # Return point for recursion.
        if nk < 1:
            return
        # If n+k longer than sequence, reduce k by 1 and recur
        elif nk > sequence_length:
            for ng in self.skipgrams(list(sequence), n, k-1):
                yield ng

        while nk > 1: # Collects the first instance of n+k length history
            history.append(next(sequence))
            nk -= 1

        # Iterative drop first item in history and picks up the next
        # while yielding skipgrams for each iteration.
        for item in sequence:
            history.append(item)
            current_token = history.pop(0)
            # Iterates through the rest of the history and
            # pick out all combinations the n-1grams
            for idx in list(combinations(range(len(history)), n-1)):
                ng = [current_token]
                for _id in idx:
                    ng.append(history[_id])
                yield tuple(ng)

        # Recursively yield the skigrams for the rest of sequence where
        # len(sequence) < n+k
        for ng in list(self.skipgrams(history, n, k-1)):
            yield ng

    def getSkipgrams(self, text, n = 2, k = 2):
        """
        :param text: formatted text data
        :type text : list of strings
        :param n: n in k-skip-n-grams
        :param k: k in k-skip-n-grams
        :return: list of dicts of skipgrams [instance1{skip1:1,skip2:1},instance2{...}]
        """

        skips = []

        for sent in text:

            dict = defaultdict(int)

            skipgramList = list(self.skipgrams(sent.split(), n, k))

            for skipgram in skipgramList:
                dict[skipgram] += 1
            skips.append(dict)

        return skips

    def get_best_features(self, X, y, max = 10):
        '''
        :param X: list of skipgrams
        :type X: list of dicts
        :param y: target values
        :type y: list
        :param max: return the max most frequent
        :return: DictVectorizer with k best as features
        '''

        vec = DictVectorizer()
        matrix = vec.fit_transform(X)

        #select k best
        support = SelectKBest(chi2, k=max).fit(matrix, y)
        vec.restrict(support.get_support())

        #transform to k best features
        #matrix = vec.transform(X)
        #print(matrix.shape)

        return vec

    def fit(self, X, y=None):

        skipgrams = self.getSkipgrams(X)
        self.vectorizer.fit(skipgrams)
        self.feature_names_ = self.vectorizer.feature_names_
        self.vocabulary_ = self.vectorizer.vocabulary_

        return self

    def transform(self, X, y=None):

        skipgrams = self.getSkipgrams(X)
        matrix = self.vectorizer.transform(skipgrams)
        self.feature_names_ = self.vectorizer.feature_names_
        self.vocabulary_ = self.vectorizer.vocabulary_

        return matrix

    def fit_transform(self, X, y=None):

        skipgrams = self.getSkipgrams(X)
        matrix = self.vectorizer.fit_transform(skipgrams)
        self.feature_names_ = self.vectorizer.feature_names_
        self.vocabulary_ = self.vectorizer.vocabulary_

        return matrix

    def restrict(self, support, indices=False):

        self.vectorizer.restrict(support, indices)

        return self



#testing

text = ["killed by my husband", "in the by house in the my household", "the household killed my husband"]
y = [0, 1, 1]

vec = SkipgramVectorizer()

matrix = vec.fit_transform(text)

support = SelectKBest(chi2, 10).fit(matrix, y)
vec.restrict(support.get_support())

matrix = vec.transform(text)

print(vec.get_feature_names())
print(matrix.shape)