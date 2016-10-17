import corpus_loader


class Taxonomie:

    def __init__(self, corpus = None):
        if corpus != None:
            if type(corpus) == corpus_loader.CorpusLoader:
                self.corpus = corpus
            else:
                self.corpus = corpus_loader.CorpusLoader(corpus)
        else:
            print("No valid corpus format.")
        self.tax = {"support":
                            {"contingency":

                                ["justify", "motivate", "expert"],

                            "evidence":

                                ["exemplification", "explain", "study"]},

                    "non-supportive":
                            {"no Label":

                                ["noLabel"]}

                    }


    def mergeTax(self, tax = None, label = None):
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

                    self.mergeTax(tax[key], key)

        else:
                mergeList = tax

        self.corpus.mergeLabel(mergeList, label)


#data = {"study":["hello","world"],"explain": ["explain"], "justification":["just","fication"], "expert":["ex", "per smpre"]}

#t =Taxonomie("data/corpus/Metalogue_extractedLinks_AllSessions.txt")
#t= Taxonomie()
#t.corpus.data = data
#print(t.corpus.data["reason"])
#t.corpus.stats()
#print("XXXXXX")
#print("\n")

#t.mergeTax(t.tax)
#t.corpus.stats()


#partitions =t.corpus.partition(["exception"],[50,50])
#for part in partitions:
#    x,y,m = t.corpus.toLists(["exception"],part)
#    print(x)
#    print(y)
