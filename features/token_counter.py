from sklearn.feature_extraction import DictVectorizer



def countTokens(sentList):
    X = []
    #
    i = 0

    for elem in sentList:

        sentLength = len(elem.split())

        X.append({"#Token": sentLength})

    vec = DictVectorizer()
    matrix = vec.fit_transform(X)

    return matrix