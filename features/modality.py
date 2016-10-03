from sklearn.feature_extraction import DictVectorizer


modals = ["must","can","may","need","shall","should"]

def checkModality(data):
    """
    :param data: list of string
    :return: scipy.sparse.csr.csr_matrix
    """
    X = []

    for elem in data:
        dic = {}
        for modal in modals:
            if modal in elem.split():
                dic = {"#modal": 1}
                continue
            else:
                dic = {"#modal": 0}
        X.append(dic)




    #print(X)
    vec = DictVectorizer()
    matrix = vec.fit_transform(X)

    print("MODALITY")
    print(type(matrix))
    print(matrix.size)

    return matrix


#checkModality(["killed by my husband", "in the by house in the my household"])