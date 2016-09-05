import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from tkinter.filedialog import askopenfilename
import os







#datapath = askopenfilename(initialdir= os.getcwd(), title="Select ASR file")



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

def checkDist(dict):
    total = 0
    for key in dict:
        total += dict[key]



def plotDict(d, color = "r", label ="noLabel"):

    x = list(d.keys())
    y = list(d.values())
    plt.scatter(x, y, color=color,label=label)

    #plt.plot(x, y, '-o',color=color)

def main():

    f1 = "data/corpus/Metalogue_Corpus_NegativePhrases.txt"
    f2 = "data/corpus/Metalogue_extractedLinks_elaboration.txt"
    f3 = "data/corpus/Metalogue_extractedLinks_contrast.txt"
    f4 = "data/corpus/Metalogue_extractedLinks_fullCorpus.txt_justification.txt"
    f5 = "data/corpus/Metalogue_extractedLinks_fullCorpus.txt_evidence.txt"

    files = [f4,f5]
    labels = ["Contrast","Elaboration","PhrasePairs"]
    colors = list("rgb")

    for file in files:

        f = open(file)
        lines = [line.strip().split('\t') for line in f]
        d = getDistribution(lines,False)
        plotDict(d,colors.pop(),labels.pop())

    legend = plt.legend(loc='upper right', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    plt.show()

main()