import pipeline
import csv
from sklearn.feature_selection import SelectKBest, chi2
from features import skipgrams


corpusMapping = {
    "ibm": ["data/corpus/IBM_extracted_raw.txt", "data/corpus/IBM_extracted_raw_negatives.txt"],
    "meta": ["data/corpus/metalogue_corpus.txt", "data/corpus/Metalogue_Corpus_NegativePhrases.txt"],
    "forum": ["data/corpus/forum_corpus.txt"]
}

taxonomyMapping = {
    "coarse": ["support","non-supportive"],
    "fine": ["contingency","evidence","no Label"],
    "relations": ["justify", "motivate", "expert","exemplification", "explain", "study","noLabel"]

}

sentenceLength = [15, 100]
partitioning = [75, 25]


with open('config.csv', newline="") as csvfile:

    expreader = csv.reader(csvfile, delimiter=";")

    for exp in expreader:
        [train, test, features, level, hiera] = exp
        train = train.split(",")
        test = test.split(",")
        features = features.split(",")


        pip = pipeline.Pipeline()

        #Loading and assigning training data
        for corpus in train:
            pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
            pip.assignAsTrain(corpus)


        #Loading and assigning test data
        for corpus in test:

            if test == train:
                cv = True
                pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
                pip.assignAsTest(corpus)
                break

            if corpus in list(pip.corpora.keys()):
                cv = False
                [train_part, test_part] = pip.corpora[corpus].partition(taxonomyMapping[level], partitioning)
                pip.corpora[corpus+"_train"]  = train_part
                pip.corpora[corpus+"_test"] = test_part
                del pip.corpora[corpus]
                pip.train.remove(corpus)
                pip.assignAsTrain(corpus+"_train")
                pip.assignAsTest(corpus+"_test")
                continue

            else:
                pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
                pip.assignAsTest(corpus)

        #merge corpora to one CL object each
        pip.train = pip.mergeCorpora(pip.train)
        pip.test = pip.mergeCorpora(pip.test)

        #to lists
        pip.train, pip.y_train, mapping_train = pip.train.toLists(taxonomyMapping[level])
        pip.test, pip.y_test, mapping_test = pip.test.toLists(taxonomyMapping[level])

        #print(pip.train[0])

        #pip.train:
        #  [ ["pre","suc"], ..., ["pre","suc"] ]
        #y_train:
        #  [1, 2, ..., 3, 2]

        #set max_feature:
        pip.max_features = {
                            "ngrams": 100,
                            "skipgrams": 500,
                            "wordpairs": 500
                             }

        #set features
        pip.set_features(features)
        pip.train_model()

        if cv:
            pip.cross_validation()
        else:
            pip.predict()

        #print(pip.train_unified)
        #print(pip.feature_models)
        print(pip.X_train.shape)
        print(pip.X_test.shape)

