from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tag import StanfordNERTagger
from nltk.stem import WordNetLemmatizer


def ner_tags(words):
    '''
    :param words: list of words
    :type words: list of strings
    :return: TODO
    '''

    ner = st = StanfordNERTagger('/home/frigori/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz','/home/frigori/stanford-ner-2014-06-16/stanford-ner-3.4.jar')

    print(ner.tag(words))


def stem(words):

    stemmed = []
    stemmer = PorterStemmer()

    for w in words:
        stemmed.append((stemmer.stem(w)))

    return stemmed

def lemmatize(words):

    lemmatized = []

    lemmatizer = WordNetLemmatizer()


    lemmas = [lemmatizer.lemmatize(w.lower(), "v") for w in words ]

    return lemmas

def filter_stopwords(words):

    stopwordList = set(stopwords.words("english"))
    stopwordList.add("'s")
    filtered = [w for w in words if not w in stopwordList]

    return filtered

def tokenize(text):

    words = word_tokenize(text, "english")

    return words


def main():

    text = "Praveen Attri claims genetic reasons to be largely responsible for social deviance	Praveen Attri claims genetic reasons to be largely responsible for social deviance"
    text = "a 1984 Supreme Court decision in City Council of Los Angeles v. Taxpayers for Vincent, where the majority stated that, the First Amendment does not guarantee the right to employ every"

    words= tokenize(text)
    print(ner_tags(lemmatize((words))))

main()