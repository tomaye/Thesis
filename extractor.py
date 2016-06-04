import nltk,re
from nltk.tokenize import sent_tokenize



connectives ={
            "although":"contrast",
            #"because":"justification",
            "but":"contrast",
            "by comparison":"contrast",
            #"especially":"reason",
            #"for example":"exemplification",
            "however":"contrast",
            "in contrast":"contrast",
            #"indeed":"justification",
            #"insofar as":"reason",
            "nonetheless":"contrast",
            "nor":"contrast",
            #"now that":"reason",
            "on the contrary":"contrast"
            #"since":"justification",
            #"so ":"reason",
            #"ultimately":"reason"
            ,"whereas":"contrast"
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

def print2File(filename):
        file = open('./'+filename, 'w+')
        for elem in extracted:
            file.write(elem[0]+"\t"+elem[1]+"\t"+elem[2]+"\t"+elem[3]+"\t"+"\n")

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

def main():
    file="smoking ban debate"
    sentences = readFile(file)
    extractor(sentences)
    print2File("extracted_"+file+".txt")

if __name__ == "__main__":
    main()