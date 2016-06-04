import os, re, sys
import collections
from tkinter.filedialog import askopenfilename

#labels which one wants to extract
#extractionLabels = ["exemplification","exception","explain","specification","justify","motivate","cause","purpose","reason","contrast"]
extractionLabels = ["elaboration"]
corpus = []
outputFile = "/output/extractedLinks.txt"

def readFile(path):
    file = open(path)
    lines = [line.strip().split('\t') for line in file]
    return lines

def extractLinks(data, debug = False):

    #link counter
    d = collections.defaultdict(lambda: [0,0])

    lineNumber = -1

    #if extracted with/without extra columns for start/endtime
    if len(data[0])== 23:
        rhetRole = 22
        rhetLink = 14
        verbal = 5
    elif len(data[0])== 21:
        rhetRole = 20
        rhetLink = 12
        verbal = 3
    else:
            sys.exit("Unknown format. Please adapt the script.")

    for line in data:
        lineNumber += 1

        if len(line)>=rhetRole and line[rhetRole] != "" and line[rhetRole] in extractionLabels:
            label = line[rhetRole]
            d[label][1] += 1
            d["In total"][1] += 1

            #if link directly to verbal
            if "inform" not in data[lineNumber][rhetLink] and "agreement" not in str(data[lineNumber][rhetLink]):
                corpus.append([data[lineNumber][rhetLink], data[lineNumber][verbal], label])
                d[label][0] += 1
                d["In total"][0] += 1
            #if link to other role
            else:
                rhetLinks = line[rhetLink].strip().replace("[", "").replace("]", "").split('_')
                for elem in rhetLinks:
                    try:
                        if (lineNumber -1) >=0 and not data[lineNumber-1][rhetRole] == "" and elem in data[lineNumber-1][rhetRole]:
                                corpus.append([data[lineNumber-1][verbal], data[lineNumber][verbal], label])
                                d[label][0] += 1
                                d["In total"][0] += 1
                    except IndexError:
                        continue


    print("Extracted links: ")
    for elem in d:
        if elem != "In total":
            print("\t"+elem + ": " + str(d[elem][0]) + " of " + str(d[elem][1]))
    print("In total" + ": " + str(d["In total"][0]) + " of " + str(d["In total"][1]))

    return d

def print2File(filename):
    if os.path.isfile('./'+filename):
        file = open('./'+filename, 'a')
        for elem in corpus:
            file.write(elem[2]+"\t"+elem[0]+"\t"+elem[1]+"\n")
    else:
        file = open('./'+filename, 'w+')
        for elem in corpus:
            file.write(elem[2]+"\t"+elem[0]+"\t"+elem[1]+"\n")

def countRoles(data, filename):
    d = collections.defaultdict(int)
    for line in data:
        if len(line)>=18:
            if line[23] in extractionLabels:
                role = line[23]
                d[role] += 1
    print (filename + ": " + str(d))

def runOnMultipleFiles():
    path = os.getcwd()+"/data/annotated/"
    extracted =[0,0]
    print(os.listdir(path))
    for file in os.listdir(path):
        print("Reading file "+file+" ...")
        f = open(path+file)
        data = [line.strip().split('\t') for line in f]
        d = extractLinks(data)
        extracted[0]+= d["In total"][0]
        extracted[1]+= d["In total"][1]
        print("Done processing file "+file+ "\n")
    print2File(outputFile)
    print("Extracted: "+ str(extracted[0])+ " of "+ str(extracted[1])+ " possible links in all files.")

def main(debug):

    datapath = askopenfilename(initialdir= os.getcwd(), title="Select ASR file for Speaker 1")
    data = readFile(datapath)
    extractLinks(data)
    print2File(outputFile)

if __name__ == "__main__":
    runOnMultipleFiles()
    #main(True)

