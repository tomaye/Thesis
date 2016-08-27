from sklearn.feature_extraction import DictVectorizer


modals = ["must","can","may","need","shall","should"]

def countTokens(data):
    X = []
    i = 0
    for elem in data.data:
        X.append({"#Token":len(elem),"pos":i})
        i += 1


    vec = DictVectorizer()
    matrix = vec.fit_transform(X)

    return matrix