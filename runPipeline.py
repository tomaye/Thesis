import pipeline
import csv
import numpy as np


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
partitioning = [20, 20, 20, 20, 20]
partitioning = [70, 30]


with open('config.csv', newline="") as csvfile:

    expreader = csv.reader(csvfile, delimiter=";")

    for exp in expreader:
        [train, test, features, level, hiera] = exp
        train = train.split(",")
        test = test.split(",")
        features = features.split(",")


        pip = pipeline.Pipeline()
        parts = []

        #Loading and assigning training data
        for corpus in train:
            pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
            pip.assignAsTrain(corpus)


        #Loading and assigning test data
        for corpus in test:

            #corpus already loaded in train
            if test == train:
                cv = True
                #pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
                pip.assignAsTest(corpus)
                break

            #if test corpus not in train
            else:
                pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
                pip.assignAsTest(corpus)

        scores = []

        #merge corpora to one CL object each
        pip.train = pip.mergeCorpora(pip.train)
        pip.test = pip.mergeCorpora(pip.test)

        #to lists
        pip.train, pip.y_train, mapping_train = pip.train.toLists(taxonomyMapping[level])
        pip.test, pip.y_test, mapping_test = pip.test.toLists(taxonomyMapping[level])

            #pip.train:
            #  [ ["pre","suc"], ..., ["pre","suc"] ]
            #y_train:
            #  [1, 2, ..., 3, 2]

        #set max_feature:
        pip.max_features = {
                                "unigrams": 500,
                                "bigrams": 500,
                                "skipgrams": 500,
                                "wordpairs": 200
                                 }

        #set features
        pip.set_features(features)
        pip.train_model()

        if cv:
            #pip.test_significance()
            pip.cross_validation()
            #pip.average_precision()
            pip.confusion_matrix(mapping_train)

        else:
            predicted = pip.predict()
            scores.append(predicted)

        if not cv:
            scores = np.array(scores)
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            print("The following features have been used: " + str(features))


