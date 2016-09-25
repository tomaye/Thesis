import nltk,re, collections
from nltk.tokenize import sent_tokenize



connectives ={
            #"although":"contrast",
            "because":"justification",
            #"but":"contrast",
            #"by comparison":"contrast",
            "especially":"reason",
            "for example":"exemplification",
            #"however":"contrast",
            #"#in contrast":"contrast",
            "indeed":"justification",
            "insofar as":"reason",
            #"nonetheless":"contrast",
            #"nor":"contrast",
            "now that":"reason",
            #"on the contrary":"contrast"
            "since":"justification",
            #"so ":"reason",
            "ultimately":"reason"
            #,"whereas":"contrast"
              }
extracted = []

def readFile(data):
    #read file sentence per sentence
    relPath = "data/"
    file= open(data+".txt")
    text = file.read().replace("\n", "")
    text = re.sub(r"\.", ". ", text)
    text = re.sub(r"  ", " ", text)
    tokenize_list = sent_tokenize(text)

    return tokenize_list

def print2File(filename, length = 4,  putLabelFirst = True):
        file = open('./'+filename, 'w+', encoding='utf-8')

        if length == 4:
            for elem in extracted:
                file.write(elem[0]+"\t"+elem[1]+"\t"+elem[2]+"\t"+elem[3]+"\t"+"\n")

        if length == 3:
            if putLabelFirst:
                for elem in extracted:
                    file.write(elem[2]+"\t"+elem[1]+"\t"+elem[1]+"\n")
            else:
                for elem in extracted:
                    file.write(elem[0]+"\t"+elem[1]+"\t"+elem[2]+"\n")
        else:
            print("Wrong length information.")


def extractor(text):
    counter = 0
    pred = "None"
    succ = "None"
    #for sentences in text:
    for i in range (0,len(text)):
        if i-1 > 0 and i+1<len(text):
            pred = text[i-1]
            succ = text[i+1]
        for connective in connectives.keys():
            if connective in text[i]:
                text[i] = text[i].replace(connective, "<connective>"+connective.upper()+"</connective>")
                counter += 1
                extracted.append([pred,text[i],succ,connectives[connective]])

    print(str(counter)+" connectives extracted.")

def checkIBM(searchmode = "connectives"):
    #counts the connectives or labels
    # CE-EMNLP-2015 format

    corpus ={}
    index = 0
    file = open("data/corpus/IBM_evidence.txt",encoding='utf-8')

    for line in file:
    #saves the last three columns in dict with line index
        line = line.split("\t")
        corpus[index] = line[-3:]
        index += 1
        #print(corpus[1])

        #if searchmode == "connectives":
        #search for connectives

    hasConnective ={"yes":0,"no":0}
    extracable = collections.defaultdict(int)
    unique = collections.defaultdict(int)
    numConnective = collections.defaultdict(int)
    labels = collections.defaultdict(int)

    for instance in corpus.keys():
        #instance[0] for claim 1 for evidence
        unique[corpus[instance][1]] += 1
        labels[corpus[instance][2]] += 1
    for instance in unique.keys():
        for connective in connectives.keys():
            #print(corpus[instance][1])
            if connective in instance:
                numConnective[connective] += 1
                hasConnective["yes"] += 1
                extracable[instance] += 1
            else:
                hasConnective["no"] += 1
    found = extracable.values()
    print(hasConnective)
    print("Unique claims: "+ str(len(unique.keys())))
    print("Total connectives: " + str(len(found)))
    print(labels)
    print(numConnective)

def extractIBM():

    toExtract = ["[STUDY, EXPERT]\n", "[STUDY]\n", "[EXPERT]\n"]
    file = open("data/corpus/IBM_evidence.txt",encoding='utf-8')

    for line in file:
    #saves the last three columns in dict with line index
        line = line.split("\t")
        coreInfo = line[-3:]
        if coreInfo[2] in toExtract:
            coreInfo[2] = coreInfo[2][1:-2]
            str = coreInfo[0].strip()
            str = str.replace("[REF]", "")
            coreInfo[0] = str.replace("[REF", "")
            str = coreInfo[1].strip()
            str = str.replace("[REF]", "")
            coreInfo[1] = str.replace("[REF", "")

            extracted.append(coreInfo)

    print2File("IBM_extracted_by_label.txt",3)

def main():
    file="IBM_evidence"
    sentences = readFile(file)
    extractor(sentences)
    print2File("extracted_"+file+".txt")

if __name__ == "__main__":
    #main()
    #checkIBM()
    extractIBM()