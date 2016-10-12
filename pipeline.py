from corpus_loader import CorpusLoader
from features import modality, token_counter, skipgrams, wordpairs
from sklearn import svm
from sklearn import cross_validation

class Pipeline:


    def __init__(self):
        self.corpora = {}
        self.classifier =  svm.SVC(kernel='linear', C=1)
        self.features = {}
        self.X_train = -1
        self.X_test = -1
        self.y_train = -1
        self.y_test = -1

        #TODO
        None


    def load_corpus(self, name, files, min=15, max= 100, merge=True):
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

        self.corpora[name] = CL

        print(name + " loaded...")


    def set_features(self, featureList):
        for feature in featureList:
            self.features[feature] = -1

    def get_labels(self, corpus):
        corpus.stats()

    def preprocessing(self, corpus, labels, balance = True):
        '''
        :param corpus: dict key for CL object in self.corpora
        :param labels: list of used labels from the corpus
        :param balance: states if the classes should be balanced
        :return: TODO
        '''

        corpus_raw = self.corpora[corpus]
        matrix = -1

        if balance:
            corpus_final = corpus_raw.balance(labels)
        else:
            corpus_final = corpus_raw.data

        X_train, y_train, train_mapping = corpus_raw.toLists(corpus_final, labels)

        for feature in self.features.keys():
            None

        return matrix

    def set_test(self, corpus):
        None

    def classify(self):
        None

    def cross_validation(self):
        scores = cross_validation.cross_val_score(self.classifier, self.X_train, self.y_train, cv=5)

        print("\n")
        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("The following features have been used: " + str(self.features.keys()))

    def set_classifier(self, classifier):
        self.classifier = classifier

