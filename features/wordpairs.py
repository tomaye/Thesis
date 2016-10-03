import itertools
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2


def get_word_pairs(data,y, max = 100, n = 2):


    X = []

    for sent in data:
        dict = {}
        wordList = sent.split()

        pairs = list(itertools.permutations(wordList, n))

        for pair in pairs:
            dict[pair] = 1
        X.append(dict)

    vec = DictVectorizer()

    matrix = vec.fit_transform(X)

    print("WORDPAIRS")
    print(len(vec.get_feature_names()))

    support = SelectKBest(chi2, k=max).fit(matrix, y)
    vec.restrict(support.get_support())

    print(len(vec.get_feature_names()))

    matrix = vec.transform(X)


    #print(matrix.shape)

    return matrix

def main():
    text = ["killed by my husband", "in the by house in the my household"]

    get_word_pairs(text,[0,1],10)

#main()