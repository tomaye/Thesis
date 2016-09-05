from sklearn.feature_extraction import DictVectorizer


modals = ["must","can","may","need","shall","should"]

def checkModality(data):
    X_mod = []
    i = 0
    #print(data.data[i])
    for elem in data:
        dic = {}
        for modal in modals:
            if modal in elem.split():
                dic = {"#modal": 1}
                continue
            else:
                dic = {"#modal": 0}
        X_mod.append(dic)
        i += 1



    vec = DictVectorizer()
    matrix = vec.fit_transform(X_mod)

    return matrix