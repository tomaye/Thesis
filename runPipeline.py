import pipeline
from sklearn.feature_selection import SelectKBest, chi2
from features import skipgrams

pipe = pipeline.Pipeline()

metalogue = ["data/corpus/Metalogue_extractedLinks_fullCorpus.txt","data/corpus/Metalogue_Corpus_NegativePhrases.txt"]
IBM = ["data/corpus/IBM_extracted_raw.txt", "data/corpus/IBM_extracted_raw_negatives.txt"]

corp1 = "meta"
corp2 = "ibm"

pipe.load_corpus(corp1, metalogue)
pipe.load_corpus(corp2,IBM)

pipe.get_labels(corp1)

pipe.preprocessing(corp1,["NoLabel", "justification", "evidence"])


#text = ["killed by my husband", "in the by house in the my household", "the household killed my husband"]
#y = [0, 1, 1]

vec = skipgrams.SkipgramVectorizer()

#matrix = vec.fit_transform(text)

#support = SelectKBest(chi2, 10).fit(matrix, y)
#vec.restrict(support.get_support())

#matrix = vec.transform(text)