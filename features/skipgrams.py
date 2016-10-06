from itertools import chain, combinations
from sklearn.feature_extraction import DictVectorizer
from collections import defaultdict
from sklearn.feature_selection import SelectKBest, chi2

def pad_sequence(sequence, n, pad_left=False, pad_right=False, pad_symbol=None):
    if pad_left:
        sequence = chain((pad_symbol,) * (n-1), sequence)
    if pad_right:
        sequence = chain(sequence, (pad_symbol,) * (n-1))
    return sequence

def skipgrams(sequence, n, k, pad_left=False, pad_right=False, pad_symbol=None):
    sequence_length = len(sequence)
    sequence = iter(sequence)
    sequence = pad_sequence(sequence, n, pad_left, pad_right, pad_symbol)

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
        for ng in skipgrams(list(sequence), n, k-1):
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
    for ng in list(skipgrams(history, n, k-1)):
        yield ng

def getSkipgrams(text, n, k):
    """
    :param text: formatted text data
    :type text : list of strings
    :param n: n in k-skip-n-grams
    :param k: k in k-skip-n-grams
    :return: list of dicts of skipgrams [instance1{skip1:1,skip2:1},instance2{...}]
    """

    X = []

    for sent in text:

        dict= defaultdict(int)

        skipgramList = list(skipgrams(sent.split(),n, k))

        for skipgram in skipgramList:
            dict[skipgram] += 1
        X.append(dict)

    return X

def get_best_features(X,y, max = 10):
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


    return vec


def main():
    text = ["killed by my husband", "in the by house in the my household"]


    X = getSkipgrams(text,2,2)
    print(X)
    vec = get_best_features(X,[0, 1])
    matrix = vec.transform(X)

#main()