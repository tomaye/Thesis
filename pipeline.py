from corpus_loader import CorpusLoader
from features import modality, token_counter, skipgrams, wordpairs
from sklearn import svm, cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import taxonomie
import scipy.sparse as sp
import numpy as np


class Pipeline:


    def __init__(self):
        self.corpora = {}
        self.tax = taxonomie.Taxonomie()
        self.train = []
        self.train_unified = []
        self.test = []
        self.test_unified = []
        self.classifier =  svm.SVC(kernel='linear', C=1)
        self.feature_models = {}
        self.feature_list = []
        self.X_train = -1
        self.X_test = -1
        self.y_train = -1
        self.y_test = -1
        self.max_features = {
                            "ngrams": 500,
                            "skipgrams": 500,
                            "wordpairs": 500
                             }


    def assignAsTest(self, corpus):
        '''
        assigns a corpus for testing
        :param corpus: key of a corpus in self.corpora
        :return: None
        '''
        self.test.append(corpus)

    def assignAsTrain(self, corpus):
        '''
        assigns a corpus for training
        :param corpus: key of a corpus in self.corpora
        :return: None
        '''
        self.train.append(corpus)

    def load_corpus(self, name, files, min=15, max= 100, merge=False):
        '''
        :param name: key for dictionary entry in self.corpora
        :param files: list of files
        :param min, max: min and max length of sentences
        :param merge: one or two text elements. one if true
        :return: None
        '''

        CL = CorpusLoader(files[0], min, max)

        if len(files) > 1:
            iterfiles = iter(files)
            next(iterfiles)
            for file in iterfiles:
                CL.add_Corpus(file, min, max)

        if merge:
            CL.mergeData()

        CL.containing.append(name)
        CL.tokenize()

        corpus = self.tax.expandTax(CL)

        self.corpora[name] = corpus

        print(name + " loaded...")

    def mergeCorpora(self, corpora):
        '''
        merges the corpora into one new CL object
        :param corpora: list of self.corpora keys
        :return: CL
        '''

        merge = []
        CL = CorpusLoader()

        for corpus in corpora:

            merge.append(self.corpora[corpus])
            CL.containing.append(corpus)

        CL.mergeWithCorpus(merge)

        return CL


    def set_features(self, featureList):
        self.feature_list = featureList
        for feature in featureList:
            self.feature_models[feature] = -1

    def get_labels(self, corpus):

        if type(corpus) == str:
            self.corpora[corpus].stats()
        else:
            None
            #TODO

    def _unify_data(self, samples):
        '''
        convertes [ [pre,suc], ...] in [ [unified], ... ]
        :param samples: list of instances
        :return: unified data
        '''

        unified = [pre + " " + suc for [pre,suc] in samples]

        return unified

    def _get_model(self, feature):
        '''
        computes the vector/matrix for feature and returns a DictVectorizer
        :param feature: feature name
        :return: vec: DictVectorzier, train/test_matrix: matrix from self.train/self.test fitted on vec
        '''

        if feature == "skipgrams":

            vec = skipgrams.SkipgramVectorizer()
            matrix = vec.fit_transform(self.train_unified)

            support = SelectKBest(chi2, self.max_features[feature]).fit(matrix, self.y_train)
            vec.restrict(support.get_support())

            train_matrix = vec.transform(self.train_unified)
            test_matrix = vec.transform(self.test_unified)

            return vec, train_matrix, test_matrix

        if feature == "#tokens":

            train_matrix = token_counter.countTokens(self.train_unified)
            test_matrix = token_counter.countTokens(self.test_unified)

            return None, train_matrix, test_matrix

        if feature == "wordpairs":

            vec = wordpairs.WordpairVectorizer()
            matrix = vec.fit_transform(self.train)

            support = SelectKBest(chi2, self.max_features[feature]).fit(matrix, self.y_train)
            vec.restrict(support.get_support())

            train_matrix = vec.transform(self.train)
            test_matrix = vec.transform(self.test)

            return vec, train_matrix, test_matrix

        if feature == "modals":

            vec = modality.ModelVectozier()

            train_matrix = vec.check_modality(self.train_unified)
            test_matrix = vec.check_modality(self.test_unified)

            return None, train_matrix, test_matrix

        if feature == "ngrams":

            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=self.max_features[feature])

            train_matrix = vec.fit_transform(self.train_unified)
            test_matrix = vec.transform(self.test_unified)

            return vec, train_matrix, test_matrix


    def train_model(self):
        '''
        calls the computation of each feature in self.feature_list
        builds self.X_train, self.X_test matrices and fits classifier on the trainings data
        :return: None
        '''

        self.train_unified = self._unify_data(self.train)
        self.test_unified = self._unify_data(self.test)

        for feature in self.feature_list:
            model, train, test = self._get_model(feature)
            self.feature_models[feature] = model

            if type(self.X_train) == int:
                self.X_train = train
            else:
                self.X_train = sp.hstack((self.X_train, train), format="csr")

            if type(self.X_test) == int:
                self.X_test = test
            else:
                self.X_test = sp.hstack((self.X_test, test), format="csr")

        self.classifier.fit(self.X_train, self.y_train)


    def predict(self):
        '''
        predicts the test data on the trained model
        :return:
        '''
        predicted = self.classifier.predict(self.X_test)
        print(np.mean(predicted == self.y_test))

    def classify(self):
        None

    def cross_validation(self):

        scores = cross_validation.cross_val_score(self.classifier, self.X_train, self.y_train, cv=5)
        print("\n")
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("The following features have been used: " + str(self.feature_list))

    def set_classifier(self, classifier):
        self.classifier = classifier

