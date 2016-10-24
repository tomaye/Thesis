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

            if test == train:
                cv = True
                pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
                pip.assignAsTest(corpus)
                break

            if corpus in list(pip.corpora.keys()):
                cv = False
                if len(partitioning) == 2:
                    [train_part, test_part] = pip.corpora[corpus].partition(taxonomyMapping[level], partitioning)
                    pip.corpora[corpus + "_train"] = train_part
                    pip.corpora[corpus + "_test"] = test_part
                    del pip.corpora[corpus]
                    pip.train.remove(corpus)
                    pip.assignAsTrain(corpus + "_train")
                    pip.assignAsTest(corpus + "_test")


                if len(partitioning) == 5:
                    parts = pip.corpora[corpus].partition(taxonomyMapping[level], partitioning)
                    #part_list = parts
                    part_list = [cl.clone() for cl in parts]
                    train_part = part_list[0].mergeWithCorpus([part_list[1], part_list[2], part_list[3]])
                    pip.corpora[corpus + "_train"] = train_part
                    pip.corpora[corpus + "_test"] = part_list[4]
                    del pip.corpora[corpus]
                    pip.train.remove(corpus)
                    pip.assignAsTrain(corpus + "_train")
                    pip.assignAsTest(corpus + "_test")

                continue

            else:
                pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
                pip.assignAsTest(corpus)

        scores = []

        for i in range(1, len(partitioning)+1):
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
                predicted = pip.predict()
                scores.append(predicted)


            #own cv
            if len(partitioning) < 5:
                break
            elif parts != []:

                pip = pipeline.Pipeline()

                for corpus in train:
                    if corpus in test:
                        continue
                    pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
                    pip.assignAsTrain(corpus)

                for corpus in test:
                    if corpus in train:
                        part_list = [cl.clone() for cl in parts]
                        indices = [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0], [2, 3, 4, 0, 1], [3, 4, 0, 1, 2],[4, 0, 1, 2, 3]]
                        train_part = part_list[indices[i-1][1]].mergeWithCorpus([part_list[indices[i-1][2]],part_list[indices[i-1][3]],part_list[indices[i-1][4]]])
                        test_part = part_list[indices[i-1][0]]
                        pip.corpora[corpus + "_train"] = train_part
                        pip.corpora[corpus + "_test"] = test_part
                        pip.assignAsTrain(corpus + "_train")
                        pip.assignAsTest(corpus + "_test")
                    else:
                        pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
                        pip.assignAsTest(corpus)

        if len(partitioning) == 5:
            scores = np.array(scores)
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            print("The following features have been used: " + str(features))





                #print(len(partitioning))

        #print(pip.train_unified)
        #print(pip.feature_models)
        #print(pip.X_train.shape)
        #print(pip.X_test.shape)

