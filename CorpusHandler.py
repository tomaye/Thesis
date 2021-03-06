import os
from tkinter.filedialog import askopenfilename

file = askopenfilename(initialdir=os.getcwd(), title="Select source file!")
fileIn= open(file, encoding="utf-8")



switchColumns = False
cutOffLabels = False
changeLabels = False

def main():

    refactor()
    #cleaner()
    #splitter()
    #tokenizer()

def cleaner():

    fileOut1 = open(file + "_justification.txt", 'w+')  # justification
    fileOut2 = open(file + "_evidence.txt", 'w+')  # evidence
    fileOut3 = open(file + "_negative.txt", 'w+')  # negative


    justification = ["justification", "reason", "justify", "motivate", "cause"]
    evidence = ["explain","exemplification","exemplify", "exception","evidence"]
    negative = ["contrast", "elaboration","negative"]

    for string in fileIn:

        string = string.strip()
        string = string.replace("</connective>", "")
        string = string.replace("<connective>", "")
        string = string.replace("_", " ")
        string = string.replace("[", "")
        string = string.replace("]", "")

        if switchColumns:
            line = string.split("\t")
            line = line[2]+"\t"+line[0]+"\t"+line[1]+"\n"

        if changeLabels:
            line = string.split("\t")
            if line[0] in justification:
                line[0] = "justification"
                fileOut = fileOut1
            elif line[0] in evidence:
                line[0] = "evidence"
                fileOut = fileOut2
            elif line[0] in negative:
                line[0] = "negative"
                fileOut = fileOut3

            if cutOffLabels:
                line = line[1] + "\t" + line[2] + "\n"
                #fileOut = fileOut3
            else:
                line = line[0]+"\t"+line[1]+"\t"+line[2]+"\n"



        fileOut.write(line)

    print("DONE")

def splitter():
    i = 1
    for line in fileIn:
        fileOut = open(file +str(i)+".txt", 'w+')
        fileOut.write(line)
        i += 1
    print("Done splitting!")

def tokenizer():
    fileOut = open(file + "(sentences)" + ".txt", 'w+')
    for line in fileIn:
        line = line.split("\t")
        fileOut.write(line[1]+"\n")
        fileOut.write(line[2])
    print("Done!")

def refactor():

    mapping = {
        "justify":"justification",
        "motivate":"motivation",
        "exemplify":"exemplification"
    }

    fileOut = open("forum_corpus.txt", 'w+', encoding="utf-8")
    for line in fileIn:
        line = line.split("\t")

        if line[0] in mapping.keys():
            line[0] = mapping[line[0]]

        fileOut.write(line[0]+"\t"+line[1]+"\t"+line[2])
    print("DONE")

main()