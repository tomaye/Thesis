import corpus_loader


class Taxonomie:

    def __init__(self, corpus = None):
        if corpus != None:
            if type(corpus) == corpus_loader.CorpusLoader:
                self.corpus = corpus
            else:
                self.corpus = corpus_loader.CorpusLoader(corpus)

        self.tax = {"support":
                            {"contingency":

                                ["justification", "cause", "reason", "motivation", "EXPERT"],

                            "evidence":

                                ["exemplification", "exception", "explain", "STUDY", "STUDY, EXPERT"]},

                    "non-supportive":
                            {"no Label":

                                ["noLabel"]}

                    }


    def _mergeTax(self, tax = None, label = None):
        '''
        expands the labels of the corpus in self.corpus to cover the whole taxonomie
        :param tax: Taxonomie (needed for recursion)
        :param label: needed for recursion only
        :return: None
        '''

        if tax == None:
            tax = self.tax

        if type(tax) == dict:

            mergeList = []

            for key in tax.keys():
                    mergeList.append(key)
                    self._mergeTax(tax[key], key)

        else:
                mergeList = tax

        if label != None:

            self.corpus.mergeLabel(mergeList, label)


    def expandTax(self, corpus):

        self.corpus = corpus
        self._mergeTax()

        return self.corpus



#data = {"study":["hello","world"],"explain": ["explain"], "justify":["just","fication"], "expert":["ex", "per smpre"]}

#t =Taxonomie("data/corpus/Metalogue_extractedLinks_AllSessions.txt")
#t= Taxonomie()
#t.corpus.data = data
#print(t.corpus.data["reason"])
#t.corpus.stats()
#print("XXXXXX")
#print("\n")

#t.mergeTax(t.tax)
#print("XXXXXXXXXXXx")
#t.corpus.stats()


#partitions =t.corpus.partition(["exception"],[50,50])
#for part in partitions:
#    x,y,m = t.corpus.toLists(["exception"],part)
#    print(x)
#    print(y)
