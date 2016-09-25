import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tkinter.filedialog import askopenfilename
import os
from corpus_loader import CorpusLoader






#datapath = askopenfilename(initialdir= os.getcwd(), title="Select ASR file")

def cutData(data, min, max):

    newData = []

    for line in data:
        p1 = line[0].split()
        p2 = line[0 + 1].split()
        sentLength = len(p1) + len(p2)

        if sentLength > min and sentLength < max:
            newData.append(line)

    return newData


def getDistribution(data, withLabel= True):

    if withLabel:
        i = 1
    else:
        i = 0

    freq = defaultdict(int)

    for line in data:
        p1 = line[i].split()
        p2 = line[i+1].split()
        sentLength = len(p1) + len(p2)
        freq[sentLength] += 1

    for key in freq:

       freq[key] = freq[key]/len(data)


    return freq

def getRawNumbers(data):

    length = []

    for line in data:
        p1 = line[0].split()
        p2 = line[0+1].split()
        sentLength = len(p1) + len(p2)
        length.append(sentLength)

    return length


def checkDist(dict):
    total = 0
    for key in dict:
        total += dict[key]

    print("Total sum: " +str(total))

def plotDict(d, color = "r", label ="noLabel"):

    x = list(d.keys())
    y = list(d.values())
    #plt.scatter(x, y, color=color,label=label)
    #plt.plot(x, y, color=color,label=label)
    #plt.bar(x, y, color=color, label=label)


    #plt.plot(x, y, '-o',color=color)

def plotHisto(data,binRange , color = "r", label ="noLabel"):
    #binRange like a range() with 3 paras: min,max,step

    bins = []
    for i in range(binRange[0],binRange[1],binRange[2]):
        bins.append(i)


    plt.hist(data, bins=bins,normed=1,histtype='bar', rwidth=1,color=color, label=label)


def loadData():
    file = "data/corpus/Metalogue_extractedLinks_fullCorpus.txt"
    file2 = "data/corpus/Metalogue_Corpus_NegativePhrases.txt"
    file3 = "data/corpus/IBM_extracted_raw.txt"

    CL = CorpusLoader()
    CL.load(file3)

    #CL.add_Corpus(file2)
    #CL.mergeLabel("justification", "evidence", "contingency")
    CL.stats(CL.data)
    print("DONE")

    return CL.data


def main():

    f1 = "data/corpus/Metalogue_Corpus_NegativePhrases.txt"
    f2 = "data/corpus/Metalogue_extractedLinks_elaboration.txt"
    f3 = "data/corpus/Metalogue_extractedLinks_contrast.txt"
    f4 = "data/corpus/Metalogue_extractedLinks_fullCorpus.txt_justification.txt"
    f5 = "data/corpus/Metalogue_extractedLinks_fullCorpus.txt_evidence.txt"
    f6 = "data/corpus/IBM_extracted_raw.txt"


    corpus = loadData()


    classes = [corpus["negative"],corpus["noLabel"],corpus["contingency"]]
    classes = [corpus["STUDY"], corpus["EXPERT"], corpus["STUDY, EXPERT"]]
    #classes = [corpus["negative"]]
    labels = ["Contrast","Elaboration","PhrasePairs"]
    labels = ["contingency","PhrasePairs","negative"]
    labels = ["Study/Expert","Expert","Study"]
    #labels = ["negative"]
    colors = list("rgb")
    min = 10
    max = 120
    bins = 2

    histoData = []

    for label in classes:

        #f = open(file)
        #lines = [line.strip().split('\t') for line in f]
        cutD = cutData(label,min, max)
        #d = getDistribution(cutD,False)
        histoData.append(getRawNumbers(cutD))
        #print(len(d.keys()))
        #checkDist(d)
        #plotDict(d,colors.pop(),labels.pop())
    plotHisto(histoData,[min,max,bins],color=["r","g","b"],label=labels)

    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()



main()