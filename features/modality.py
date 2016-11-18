from sklearn.feature_extraction import DictVectorizer



class ModelVectozier():

    def __init__(self):

        self.vectorizer = DictVectorizer()

    def check_modality(self, text):
        """
        checks if a modal verb is present
        :param data: list of string
        :return: scipy.sparse.csr.csr_matrix
        """

        modals = ['can', 'could', 'may', 'might', 'must', 'will', 'would', 'shall', 'should']

        mods = []

        for sentpair in text:

            dic = {}

            for sent in sentpair:
                for modal in modals:
                    if modal in sent.split():
                        dic = {"#modal": 1}
                        continue
                    #dic = {modal: 1}
                    else:
                        dic = {"#modal": 0}
                    #dic = {modal: 0}
            mods.append(dic)

        matrix = self.vectorizer.fit_transform(mods)

        return matrix

