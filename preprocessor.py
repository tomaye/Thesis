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

    #print("above" in stopwordList)

    filtered = [w for w in words if w not in stopwordList]

    filtered = ' '.join(str(elem) for elem in filtered)
    return filtered

def tokenize(text):

    words = word_tokenize(text, "english")

    return words


def main():

    text = "Praveen Attri claims genetic reasons to be largely responsible for social deviance	Praveen Attri claims genetic reasons to be largely responsible for social deviance"
    text = "a 1984 Supreme Court decision in City Council of Los Angeles v. Taxpayers for Vincent, where the majority stated that, the First Amendment does not guarantee the right to employ every"
    text = "Los Gatos has already banned it and SO has the county.	I don't think that our businesses are in as a competitive disadvantage as they may think."
    text = "AND LET ME ANNOUNCE MYSELF IN TELLING TO YOU HOW IS THIS SO BAD AND HOW THIS CAN LEAD TO DEFLOODKY'S ARGUMENT"


    text = text.lower()
    words= tokenize(text)
    #print(words)
    print(filter_stopwords(words))

main()