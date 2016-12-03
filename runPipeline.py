import pipeline
import csv
import numpy as np


corpusMapping = {
    "ibm": ["data/corpus/IBM_extracted_raw.txt", "data/corpus/IBM_extracted_raw_negatives.txt"],
    "meta": ["data/corpus/metalogue_corpus.txt", "data/corpus/Metalogue_Corpus_NegativePhrases.txt"],
    "forum": ["data/corpus/forum_corpus.txt"]
}

taxonomyMapping = {
    "coarse": ["support", "non-supportive"],
    "fine": ["contingency", "evidence", "no Label"],
    "relations": ["justification", "motivation", "expert", "exemplification", "explain", "study"]
    #, "noLabel"

}

sentenceLength = [15, 100]
partitioning = [20, 20, 20, 20, 20]
partitioning = [70, 30]
cv = False
testSignificance = True

overallScores = []

with open('config.csv', newline="") as csvfile:

    expreader = csv.reader(csvfile, delimiter=";")

    #for i in range(0,1):
    #    exp = next(expreader)
    for exp in expreader:

        toFile = {
            "p" : 1,
            "CV" : 0,
            "matrix" : None,
            "AP" : -1,
            "accuracy" : -1,
            "features": None,
            "corpora": None,
            "level" : None
        }

        for i in range(0, 2):
            [train, test, features, level, hierarchical] = exp
            toFile["corpora"] = train + "-" + test
            toFile["features"] = features
            toFile["level"] = level
            train = train.split(",")
            test = test.split(",")
            features = features.split(",")

            if hierarchical == "true":
                features.append("coarse_predictions")
                toFile["features"] += " coarse_predictions"



            pip = pipeline.Pipeline()
            parts = []

            #use cv as evaluation
            if train == test and cv:
                cv = True
                AP = False
                for corpus in train:
                    pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
                    pip.assignAsTrain(corpus)
                    pip.assignAsTest(corpus)

            #split into train and test
            else:
                #Average precision only for binary classification
                if len(taxonomyMapping[level]) == 2:
                    AP = True
                else:
                    AP = False
                #Loadin.6g and assigning training data
                for corpus in train:
                    if corpus in test:
                        pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
                        [train_part, test_part] = pip.corpora[corpus].partition(taxonomyMapping[level], partitioning, True)
                        pip.corpora[corpus + "_train"] = train_part
                        pip.corpora[corpus + "_test"] = test_part
                        del pip.corpora[corpus]
                        pip.assignAsTrain(corpus + "_train")
                        pip.assignAsTest(corpus + "_test")
                    else:
                        pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
                        pip.assignAsTrain(corpus)


                #Loading and assigning test data
                for corpus in test:
                    if test == train:
                        break
                    elif corpus in train:
                        continue
                    #if test corpus not in train
                    else:
                        pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
                        pip.assignAsTest(corpus)

            scores = []

            #merge corpora to one CL object each
            pip.train = pip.mergeCorpora(pip.train)
            pip.test = pip.mergeCorpora(pip.test)

            #to lists
            pip.train, pip.y_train, pip.mapping = pip.train.toLists(taxonomyMapping[level])
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
                pip.cross_validation()
                pip.confusion_matrix(pip.mapping)

            if AP:
                averagePrecision = pip.average_precision()
                confusionMatrix = pip.confusion_matrix(pip.mapping)
                toFile["AP"] = averagePrecision

            if testSignificance:
                score, pvalue = pip.test_significance(False)
                confusionMatrix = pip.confusion_matrix(pip.mapping)
                if pvalue < toFile["p"]:
                    toFile["p"] = pvalue
                    toFile["CV"] = str(score)
                    toFile["matrix"] = (confusionMatrix, pip.mapping)



            if not cv:
                predicted = pip.predict()
                scores = [predicted[i] == pip.y_test[i] for i in range(0, len(predicted))]
                scores = np.array(scores)
                print("Accuracy: " + str(scores.mean()))
                print("The following features have been used: " + str(features))
                overallScores.append(scores.mean())
                pip.confusion_matrix(pip.mapping)
                print("\n")

        if len(overallScores) > 1:
            final_accuracy = np.array(overallScores).mean()
            toFile["accuracy"] = str(final_accuracy)[:5]
            print("Accuracy: %0.2f (+/- %0.2f)" % (final_accuracy, np.array(overallScores).std() * 2))
            print("\n")

        pip.save_as_file(toFile)
