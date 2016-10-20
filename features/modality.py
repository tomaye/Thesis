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

        for elem in text:

            dic = {}

            for modal in modals:
                if modal in elem.split():
                    dic = {"#modal": 1}
                    continue
                    #dic = {modal: 1}
                else:
                    dic = {"#modal": 1}
                    #dic = {modal: 0}
            mods.append(dic)

        matrix = self.vectorizer.fit_transform(mods)

        return matrix

