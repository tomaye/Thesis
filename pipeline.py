from corpus_loader import CorpusLoader
from features import modality, token_counter, skipgrams, wordpairs, doc2vec, chunk_counter
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
import taxonomie
import scipy.sparse as sp
import numpy as np
from nltk.corpus import stopwords


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
        print(corpus + " added to Test")
        self.test.append(corpus)

    def assignAsTrain(self, corpus):
        '''
        assigns a corpus for training
        :param corpus: key of a corpus in self.corpora
        :return: None
        '''
        print(corpus +" added to Train")
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

        #print(name + " loaded...")

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

        unified = [pre + " " + suc for [pre, suc] in samples]

        return unified

    def _filter(self, samples):
        '''
        filter stopwords from samples
        :param samples: list of instances
        :return: filtered data
        '''


        stopwordList = set(stopwords.words("english"))
        stopwordList.add("'s")
        filtered = []

        for sentpair in samples:
            temp = []
            for sent in sentpair:
                sent = " ".join([w for w in sent.split() if w not in stopwordList])
                temp.append(sent)
            filtered.append(temp)

        return filtered

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

            train_matrix = vec.check_modality(self.train_raw)
            test_matrix = vec.check_modality(self.test_raw)

            return None, train_matrix, test_matrix

        if feature == "ngrams":

            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=self.max_features[feature])

            train_matrix = vec.fit_transform(self.train_unified)
            test_matrix = vec.transform(self.test_unified)

            return vec, train_matrix, test_matrix

        if feature == "doc2vec":

            #load existing model
            #model = Doc2Vec.load(fname)

            #train model
            model = doc2vec.train_model(doc2vec.prep_data(self.train_unified))

            #save model
            #model.save(fname)

            train_matrix = doc2vec.get_train_X(model, len(self.train_unified))
            test_matrix = doc2vec.transform(model, self.test_unified)

            return model, train_matrix, test_matrix

        if feature == "#chunks":

            vec = chunk_counter.ChunkcountVectorizer()

            train_matrix = vec.count_chunks(self.train_raw)
            test_matrix = vec.count_chunks(self.test_raw)

            return None, train_matrix, test_matrix

        if feature == "#args":

            vec = chunk_counter.ChunkcountVectorizer()

            train_matrix = vec.count_args(self.train_raw)
            test_matrix = vec.count_args(self.test_raw)

            return None, train_matrix, test_matrix


    def train_model(self):
        '''
        calls the computation of each feature in self.feature_list
        builds self.X_train, self.X_test matrices and fits classifier on the trainings data
        :return: None
        '''

        self.train_raw = self.train
        self.train = self._filter(self.train)
        self.test_raw = self.test
        self.test = self._filter(self.test)

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

        return np.mean(predicted == self.y_test)

    def classify(self):
        None

    def cross_validation(self):

        cv = StratifiedKFold(5)
        scores = cross_val_score(self.classifier, self.X_train, self.y_train, cv=cv)
        print("\n")
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("The following features have been used: " + str(self.feature_list))

    def set_classifier(self, classifier):
        self.classifier = classifier

