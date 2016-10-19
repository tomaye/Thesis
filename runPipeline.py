import pipeline
import csv
from sklearn.feature_selection import SelectKBest, chi2
from features import skipgrams


#pipe = pipeline.Pipeline()
#pipe = pipeline.Pipeline()
#tax = taxonomie.Taxonomie()


#metalogue = ["data/corpus/Metalogue_extractedLinks_fullCorpus.txt","data/corpus/Metalogue_Corpus_NegativePhrases.txt"]
#IBM = ["data/corpus/IBM_extracted_raw.txt", "data/corpus/IBM_extracted_raw_negatives.txt"]

#corp1 = "meta"
#corp2 = "ibm"

#pipe.load_corpus(corp1, metalogue)
#pipe.load_corpus(corp2,IBM)

#pipe.get_labels(corp1)

#pipe.preprocessing(corp1,["NoLabel", "justification", "evidence"])


#text = ["killed by my husband", "in the by house in the my household", "the household killed my husband"]
#y = [0, 1, 1]

#vec = skipgrams.SkipgramVectorizer()

#matrix = vec.fit_transform(text)

#support = SelectKBest(chi2, 10).fit(matrix, y)
#vec.restrict(support.get_support())

#matrix = vec.transform(text)


##################################################


#tax = taxonomie.Taxonomie()


#metalogue = ["data/corpus/Metalogue_extractedLinks_fullCorpus.txt","data/corpus/Metalogue_Corpus_NegativePhrases.txt"]
#IBM = ["data/corpus/IBM_extracted_raw.txt", "data/corpus/IBM_extracted_raw_negatives.txt"]

#corp1 = "meta"
#corp2 = "ibm"

#pipe.load_corpus(corp1, metalogue)
#pipe.load_corpus(corp2,IBM)

#pipe.get_labels(corp1)

#pipe.preprocessing(corp1,["NoLabel", "justification", "evidence"])


#text = ["killed by my husband", "in the by house in the my household", "the household killed my husband"]
#y = [0, 1, 1]

#vec = skipgrams.SkipgramVectorizer()

#matrix = vec.fit_transform(text)

#support = SelectKBest(chi2, 10).fit(matrix, y)
#vec.restrict(support.get_support())

#matrix = vec.transform(text)


##################################################


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


        pip = pipeline.Pipeline()

        #Loading and assigning training data
        for corpus in train:
            pip.load_corpus(corpus, corpusMapping[corpus], sentenceLength[0], sentenceLength[1])
            pip.assignAsTrain(corpus)


        #Loading and assigning test data
        for corpus in test:

            if corpus in list(pip.corpora.keys()):
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

        pip.train = pip.mergeCorpora(pip.train)
        pip.test = pip.mergeCorpora(pip.test)


        pip_train, pip.y_train, mapping_train = pip.train.toLists(taxonomyMapping[level])
        pip_test, pip.y_test, mapping_test = pip.test.toLists(taxonomyMapping[level])


        print(pip.train.containing)
        pip.train.stats()
        print("\n")
        print(pip.test.containing)
        pip.test.stats()